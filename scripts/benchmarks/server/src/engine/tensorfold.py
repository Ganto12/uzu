import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx2
from bench import BenchResponse, ReasoningEffort
from common import check_reasoning_effort, get_finish_reason, get_model_path
from mach import get_gpu_energy, get_memory_counters
from openai import OpenAI

from ..util import get_openai_base_url
from . import ServerEngine, get_openai_reasoning


class TensorFoldServerEngine(ServerEngine):
    """
    https://github.com/ashhart/TensorFold
    """

    PROJECT_DIR = Path(__file__).resolve().parents[3] / "engine-tensorfold"

    def __init__(self, host: str, port: int, model: str, sampling: dict, *, draft_model: str | None = None):
        super().__init__(host, port, model, sampling)
        self.draft_model = draft_model

    def create_process(self) -> subprocess.Popen:
        model_path = Path(self.model).expanduser()
        model = str(model_path.resolve()) if model_path.is_dir() else self.model
        binary = os.environ.get("BENCHMARK_TENSORFOLD_BINARY")
        if binary is None:
            subprocess.run(
                ["bash", str(self.PROJECT_DIR / "bootstrap.sh")],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
            )
            binary = str(self.PROJECT_DIR / "deps/tensorfold/zig-out/native/bin/tensorfold-native")
        if not model_path.is_dir():
            # Complete downloads before the server's readiness timeout starts.
            subprocess.run([binary, "pull", model], check=True, stdin=subprocess.DEVNULL, stdout=sys.stderr)
        draft_args = ["--no-drafts"]
        if self.draft_model is not None:
            draft_path = Path(self.draft_model).expanduser()
            draft = str(draft_path.resolve()) if draft_path.is_dir() else get_model_path(self.draft_model)
            draft_args = ["--drafter", draft]
        # Launch the server directly so process.pid identifies the inference
        # process for memory measurements and process-group cleanup.
        return subprocess.Popen(
            [
                binary,
                "serve",
                model,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--name",
                self.model,
                "--backend",
                "mlx",
                "--parallel",
                "1",
                "--prompt-cache-gib",
                "0",
                "--snapshot-dir",
                "none",
                "--no-update-check",
                *draft_args,
            ],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            start_new_session=True,
        )

    def get_reasoning(self, effort: ReasoningEffort, messages: list[Any], body: dict[str, Any]) -> dict[str, Any]:
        # TensorFold coerces efforts its template doesn't name; compare the prompts it renders instead.
        tools = {key: body[key] for key in ("tools", "tool_choice") if key in body}
        with httpx2.Client(timeout=60) as client:

            def render(candidate: ReasoningEffort) -> list[int] | None:
                response = client.post(
                    f"{get_openai_base_url(self.host, self.port)}/tokenize",
                    json={"messages": messages, **tools, **get_openai_reasoning(candidate)},
                )
                if response.status_code == 400:
                    return None
                response.raise_for_status()
                return response.json()["tokens"]

            check_reasoning_effort(effort, render)
        return get_openai_reasoning(effort)

    def handle_request(self, messages: list[Any], body: dict[str, Any]) -> BenchResponse:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("TensorFold server is not running.")

        body.update(presence_penalty=0.0, frequency_penalty=0.0)
        with OpenAI(base_url=get_openai_base_url(self.host, self.port), api_key="not-needed") as client:
            energy_start = get_gpu_energy()
            started = time.perf_counter()
            response = client.chat.completions.create(model=self.model, messages=messages, extra_body=body)
            duration = time.perf_counter() - started
            energy_gpu = get_gpu_energy() - energy_start
        memory = get_memory_counters(pid=process.pid)

        usage = response.usage
        extras = response.model_extra or {}
        runtime = extras.get("tensorfold")
        if usage is None or not isinstance(runtime, dict):
            raise RuntimeError("TensorFold response is missing usage or runtime statistics.")

        if usage.prompt_tokens_details is not None and usage.prompt_tokens_details.cached_tokens:
            raise RuntimeError("TensorFold reused cached prompt tokens")
        prompt_tokens = usage.prompt_tokens
        prefill_seconds = runtime.get("prefill_seconds") or 0.0
        speculative = extras.get("speculative") or {}
        # rounds counts target verification forwards, excluding drafts. The first token comes from prefill.
        rounds = (speculative.get("rounds") or 0) if self.draft_model is not None else 0

        return BenchResponse(
            text=response.choices[0].message.content or "",
            finish_reason=get_finish_reason(response.choices[0].finish_reason),
            prompt_tokens_count=prompt_tokens,
            tokens_count=usage.completion_tokens,
            time_to_first_token=runtime.get("time_to_first_token") or 0.0,
            prompt_tps=prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0,
            decode_tps=runtime.get("tokens_per_second") or 0.0,
            tokens_per_forward_pass=(usage.completion_tokens - 1) / rounds if rounds else None,
            duration=duration,
            energy_gpu=energy_gpu,
            # As for UZU, these are process snapshots after the response.
            memory_phys_footprint=memory.phys_footprint,
            memory_resident=memory.resident_size,
            memory_graphics_total=memory.graphics_total,
        )

"""Shared behavioral checks and subprocess lifecycle for benchmark engines."""

import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import pytest
from bench import BenchRequest, BenchResponse, ChatMessage

from .engine_process import EngineProcess, run_engine

ROOT = Path(__file__).resolve().parents[2]


def assert_response(response: BenchResponse) -> None:
    assert response.text, "Expected generated text for the nonempty test prompt"
    for name, value in response.model_dump(exclude={"text", "finish_reason", "tokens_per_forward_pass"}).items():
        assert math.isfinite(value) and value >= 0, f"Invalid {name}: {value}"

    assert response.prompt_tokens_count > 0
    assert response.duration > 0
    assert response.energy_gpu > 0
    assert response.tokens_per_forward_pass is None or response.tokens_per_forward_pass >= 1
    assert response.time_to_first_token <= response.duration
    assert response.memory_phys_footprint > 0
    assert response.memory_resident > 0


class EngineTests:
    """Shared process lifecycle and checks inherited by each engine's test class."""

    pytestmark = pytest.mark.integration
    engine_name: ClassVar[str]
    command: ClassVar[Callable[[pytest.Config, Path], list[str]]]

    @pytest.fixture(
        scope="class",
        params=[
            {"greedy": True},
            {
                "temp": 0.7,
                "top_k": 20,
                "top_p": 0.9,
                "min_p": None,
                "repetition_penalty": None,
                "suffix_repetition_length": None,
            },
        ],
        ids=["greedy", "stochastic"],
    )
    @classmethod
    def engine(
        cls, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> Iterator[EngineProcess]:
        if cls.engine_name not in request.config.getoption("engine"):
            pytest.skip(f"Enable {cls.engine_name} tests with --engine {cls.engine_name}")

        timeout = request.config.getoption("engine_timeout")
        if not math.isfinite(timeout) or timeout <= 0:
            raise pytest.UsageError("--engine-timeout must be finite and greater than zero")

        stderr_log = tmp_path_factory.mktemp(f"engine-{cls.engine_name}") / "stderr.log"
        engine_name = request.config.getoption("ios_engine") if cls.engine_name == "ios" else cls.engine_name
        if engine_name == "mldrift" and not request.param.get("greedy"):
            pytest.skip("ML Drift supports greedy decoding only")
        model = {"model": request.config.getoption(f"{cls.engine_name}_model"), "sampling": request.param}
        if draft := request.config.getoption(f"{cls.engine_name}_draft_model", default=None):
            model["draft_model"] = draft
        model_file = stderr_log.with_name("model.json")
        model_file.write_text(json.dumps(model))
        with run_engine(cls.command(request.config, model_file), ROOT, stderr_log, timeout) as process:
            yield process

    def test_chat_generation(self, engine: EngineProcess) -> None:
        request = BenchRequest(
            messages=[ChatMessage(role="user", content="Say hello in one short sentence.")],
            max_output_tokens=8,
        )
        assert_response(engine.request(request))

    def test_multiple_requests_in_one_process(self, engine: EngineProcess) -> None:
        first = BenchRequest(
            messages=[ChatMessage(role="user", content="Complete this sentence: The sky is")],
            max_output_tokens=8,
        )
        assert_response(engine.request(first))

        second = BenchRequest(
            messages=[ChatMessage(role="user", content="Complete this sentence: The grass is")],
            max_output_tokens=8,
        )
        assert_response(engine.request(second))

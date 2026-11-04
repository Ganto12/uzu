from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_call_id: str | None = Field(default=None, exclude_if=lambda value: value is None)


class GreedySampling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    greedy: Literal[True]


class StochasticSampling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int | None
    top_p: float | None
    min_p: float | None
    temp: float | None
    repetition_penalty: float | None
    suffix_repetition_length: int | None


BenchSampling = GreedySampling | StochasticSampling

ReasoningEffort = Literal["disabled", "default", "low", "medium", "high", "xhigh"]
REASONING_LEVELS: tuple[ReasoningEffort, ...] = ("low", "medium", "high", "xhigh")


class BenchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage]
    tools: list[JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_choice: JsonValue | None = Field(default=None, exclude_if=lambda value: value is None)

    max_output_tokens: int | None = Field(default=None, ge=0)
    speculative_depth: int | None = Field(default=None, ge=0)
    reasoning_effort: ReasoningEffort | None = None


class BenchResponse(BaseModel):
    text: str
    finish_reason: Literal["stop", "length"]
    prompt_tokens_count: int
    tokens_count: int
    time_to_first_token: float
    prompt_tps: float
    decode_tps: float
    tokens_per_forward_pass: float | None
    duration: float
    energy_gpu: float
    memory_phys_footprint: int
    memory_resident: int
    memory_graphics_total: int

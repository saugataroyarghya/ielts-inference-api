"""Public API contract. Raw model estimates are deliberately not clipped or rounded."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EssayInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=20000)
    essay: str = Field(min_length=1, max_length=100000)

    @field_validator("question", "essay")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain non-whitespace characters.")
        # Preserve spelling, whitespace and paragraph structure for the model.
        return value


class RubricScores(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    TR: float
    CC: float
    LR: float
    GRA: float


class ReleaseIdentity(BaseModel):
    application_version: str
    model_id: str
    model_revision: str
    device: Literal["cpu"] = "cpu"
    precision: Literal["float32"] = "float32"


class ScoreResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    scores: RubricScores
    task2_mean_estimate: float
    track: Literal["ielts"]
    max_tokens: int
    truncated: bool
    release: ReleaseIdentity

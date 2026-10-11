"""Public API contract. Raw model estimates are deliberately not clipped or rounded."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    precision: Literal["float32"] = "float32"


class ScoreResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    scores: RubricScores
    task2_mean_estimate: float
    track: Literal["ielts"]
    max_tokens: int
    truncated: bool
    release: ReleaseIdentity


MAX_SECONDS = 300
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_RESPONSES = 12


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ResponseInput(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    part: Literal[1, 2, 3] | None = None
    transcript: str | None = Field(default=None, max_length=24000)
    start: float = Field(default=0, ge=0)
    end: float | None = Field(default=None, gt=0)
    timeline_start: float | None = Field(default=None, ge=0)

    @field_validator("part", mode="before")
    @classmethod
    def reject_boolean_part(cls, value):
        if isinstance(value, bool):
            raise ValueError("Part must be 1, 2, 3 or null")
        return value

    @model_validator(mode="after")
    def valid_interval(self):
        if not self.question.strip():
            raise ValueError("Question cannot be blank")
        if isinstance(self.part, bool):
            raise ValueError("Part must be 1, 2, 3 or null")
        if self.end is not None and self.end <= self.start:
            raise ValueError("End must be greater than start")
        return self


class SessionInput(StrictModel):
    responses: list[ResponseInput] = Field(min_length=1, max_length=MAX_RESPONSES)

    @model_validator(mode="after")
    def consistent_timeline(self):
        present = [r.timeline_start is not None for r in self.responses]
        if any(present) and not all(present):
            raise ValueError("Provide timeline_start for every response or omit it for all")
        return self


class PreparedResponse(StrictModel):
    question: str
    part: int | None
    text: str
    start: float
    end: float
    audio_start: float
    audio_end: float


class PreparedSession(StrictModel):
    # Only exchanged between private BentoML services; the public API accepts uploads.
    audio_flac: str
    responses: list[PreparedResponse]


Band = Annotated[float, Field(ge=1, le=9)]


class Criteria(StrictModel):
    fluency_and_coherence: Band
    lexical_resource: Band
    grammatical_range_and_accuracy: Band
    pronunciation: Band


class Prediction(StrictModel):
    scores: Criteria
    speaking_band: Band
    raw_scores: Criteria
    score_status: Literal["experimental_weakly_supervised"]
    text_repair_validation_gate_passed: bool
    human_criterion_accuracy_verified: Literal[False]


class SpeakingResult(Prediction):
    device: Literal["cpu", "mps", "cuda"]
    assessment_scope: Literal["provided_responses"] = "provided_responses"
    request_id: str
    model_id: str
    model_revision: str
    audio_seconds: float
    transcript_sources: list[Literal["caller_supplied", "faster_whisper_unedited"]]
    original_session_timeline_available: bool
    disfluency_preservation_verified: Literal[False] = False

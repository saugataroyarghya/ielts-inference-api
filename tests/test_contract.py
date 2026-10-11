import json

import pytest
from pydantic import ValidationError

from ielts_api import artifacts
from ielts_api.schemas import EssayInput, RubricScores


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " ", "essay": "Essay"},
        {"question": "Question", "essay": "\n"},
        {"question": 123, "essay": "Essay"},
        {"question": "Question", "essay": "x" * 100001},
        {"question": "Question", "essay": "Essay", "extra": "ignored?"},
    ],
)
def test_invalid_input_rejected(payload):
    with pytest.raises(ValidationError):
        EssayInput.model_validate(payload)


def test_input_preserves_original_essay():
    essay = "  This is a paragraph.\n\nAnother paragraph.  "
    assert EssayInput(question="Why?", essay=essay).essay == essay


def test_nonfinite_scores_rejected_but_raw_out_of_range_scores_preserved():
    with pytest.raises(ValidationError):
        RubricScores(TR=float("nan"), CC=5, LR=5, GRA=5)
    assert RubricScores(TR=9.2, CC=5, LR=5, GRA=5).TR == 9.2


def test_corrupted_artifact_prevents_startup(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"corrupted")
    monkeypatch.setattr(artifacts, "MODEL", {"files": {"model.safetensors": "0" * 64}})
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        artifacts.verify_snapshot(tmp_path)


def test_reference_matches_pinned_model_manifest():
    fixture = json.loads((artifacts.ROOT / "tests/fixtures/reference.json").read_text())
    assert fixture["provenance"]["model_manifest_sha256"] == artifacts.sha256(
        artifacts.ROOT / "models.lock.json"
    )

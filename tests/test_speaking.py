import asyncio
import base64
import io
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
from ielts_scoring.prepare_input import prepare as reference_prepare
from pydantic import ValidationError
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from ielts_api.audio import Preparer
from ielts_api.middleware import APIGuard
from ielts_api.schemas import SessionInput
from ielts_api.speaking import SpeakingRuntime, warmup_session

KEY = "test-only-key-not-for-production-12345"
AUTH = {"Authorization": f"Bearer {KEY}"}


def metadata(**overrides):
    return {
        "responses": [
            {"question": "Where do you live?", "transcript": "I live here.", "part": 1, **overrides}
        ]
    }


def wav_bytes(seconds=1, sr=16000):
    stream = io.BytesIO()
    signal = 0.15 * np.sin(2 * np.pi * 170 * np.arange(round(sr * seconds)) / sr)
    sf.write(stream, signal, sr, format="WAV")
    return stream.getvalue()


def test_preparation_matches_original_resampling_clipping_and_timeline(tmp_path):
    recording = tmp_path / "recording.wav"
    recording.write_bytes(wav_bytes(3, 48000))
    payload = {
        "responses": [
            metadata(start=0.25, end=1.25, timeline_start=4.0)["responses"][0],
            metadata(start=1.5, end=2.75, timeline_start=7.0)["responses"][0],
        ]
    }
    session, sources, seconds = Preparer().prepare(
        [recording, recording],
        SessionInput.model_validate(payload),
    )
    reference_input = tmp_path / "request.json"
    reference_input.write_text(
        json.dumps(
            {"responses": [{**row, "audio_path": str(recording)} for row in payload["responses"]]}
        )
    )
    original = reference_prepare(reference_input, tmp_path / "reference")
    assert [r.model_dump() for r in session.responses] == original["responses"]
    assert (
        base64.b64decode(session.audio_flac) == (tmp_path / "reference/candidate.flac").read_bytes()
    )
    assert seconds == 2.25
    assert sources == ["caller_supplied", "caller_supplied"]


@pytest.mark.parametrize(
    "changes",
    [
        {"part": True},
        {"audio_path": "/etc/passwd"},
        {"start": -1.0},
        {"end": 0.0},
        {"question": " "},
        {"timeline_start": float("nan")},
        {"transcript": 123},
    ],
)
def test_invalid_metadata(changes):
    with pytest.raises(ValidationError):
        SessionInput.model_validate(metadata(**changes))


def test_bad_audio_duration_and_missing_transcript_rejected(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes())
    with pytest.raises(ValueError, match="optional ASR"):
        Preparer().prepare([path], SessionInput.model_validate(metadata(transcript=None)))
    with pytest.raises(ValueError, match="interval"):
        Preparer().prepare([path], SessionInput.model_validate(metadata(end=5.0)))
    with pytest.raises(ValueError, match="one audio upload"):
        Preparer().prepare([], SessionInput.model_validate(metadata()))
    path.write_bytes(wav_bytes(301, 8000))
    with pytest.raises(ValueError, match="300 seconds"):
        Preparer().prepare([path], SessionInput.model_validate(metadata()))


def test_asr_object_is_reused_and_options_preserve_disfluencies(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes())
    calls = []

    class ASR:
        def transcribe(self, wave, **kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(text=" um I I live here ")]), None

    preparer = Preparer()
    preparer.recognizer = ASR()
    for _ in range(2):
        session, sources, _ = preparer.prepare(
            [path],
            SessionInput.model_validate(metadata(transcript=None)),
        )
        assert session.responses[0].text == "um I I live here"
        assert sources == ["faster_whisper_unedited"]
    assert len(calls) == 2
    assert all(c["vad_filter"] is False and c["condition_on_previous_text"] is False for c in calls)


def test_guard_authenticates_before_reading_and_rejects_chunked_oversize(monkeypatch):
    monkeypatch.setenv("IELTS_API_KEY", KEY)

    async def app(scope, receive, send):
        await JSONResponse({"ok": True})(scope, receive, send)

    client = TestClient(APIGuard(app, required=True, max_bytes=20))
    assert client.post("/score", json={}).status_code == 401
    assert client.get("/readyz").status_code == 200
    assert client.post("/score", headers=AUTH, json={}).status_code == 200
    assert (
        client.post(
            "/score",
            headers={**AUTH, "Content-Type": "application/vnd.bentoml+pickle"},
            content=b"payload",
        ).status_code
        == 415
    )
    response = client.post(
        "/score",
        headers={**AUTH, "Content-Type": "application/json"},
        content=iter([b"a" * 11, b"b" * 11]),
    )
    assert response.status_code == 413


def test_guard_limits_inflight_uploads_before_reading_body(monkeypatch):
    monkeypatch.setenv("IELTS_API_KEY", KEY)

    async def run():
        started, release = asyncio.Event(), asyncio.Event()

        async def app(scope, receive, send):
            started.set()
            await release.wait()
            await JSONResponse({"ok": True})(scope, receive, send)

        guard = APIGuard(app, required=True, max_concurrency=1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(guard), base_url="http://test"
        ) as c:
            first = asyncio.create_task(c.post("/score", headers=AUTH, json={}))
            await started.wait()
            assert (await c.post("/score", headers=AUTH, json={})).status_code == 429
            release.set()
            assert (await first).status_code == 200
            assert (await c.post("/score", headers=AUTH, json={})).status_code == 200

    asyncio.run(run())


def test_one_invalid_text_does_not_poison_other_batch_members():
    good = warmup_session()
    bad = good.model_copy(deep=True)
    bad.responses[0].text = "too long"
    runtime = SpeakingRuntime.__new__(SpeakingRuntime)

    def tokenize(session):
        if session.responses[0].text == "too long":
            raise ValueError("too long")
        return [1, 2]

    runtime.text = SimpleNamespace(
        tokenize=tokenize, encode=lambda rows: np.zeros((len(rows), 1024))
    )
    runtime.audio = SimpleNamespace(encode=lambda rows: np.zeros((len(rows), 1550)))
    runtime.model = SimpleNamespace(score_features=lambda t, a: [{"ok": i} for i in range(len(t))])
    runtime.device = "cpu"
    assert runtime.predict([good, bad, good]) == [
        {"ok": 0, "device": "cpu"},
        {"error": "too long"},
        {"ok": 1, "device": "cpu"},
    ]


@pytest.fixture
def speaking_client(monkeypatch):
    from bentoml._internal.configuration.containers import BentoMLContainer
    from prometheus_client import CollectorRegistry

    import service

    metrics = BentoMLContainer.metrics_client.get()
    _ = metrics.registry
    monkeypatch.setattr(metrics, "_registry", CollectorRegistry())

    monkeypatch.setenv("IELTS_API_KEY", KEY)
    monkeypatch.setenv("IELTS_ENABLE_ASR", "0")

    async def predict(sessions):
        assert len(sessions) == 1
        scores = dict.fromkeys(
            (
                "fluency_and_coherence",
                "lexical_resource",
                "grammatical_range_and_accuracy",
                "pronunciation",
            ),
            6.0,
        )
        return [
            {
                "scores": scores,
                "raw_scores": scores,
                "speaking_band": 6.0,
                "score_status": "experimental_weakly_supervised",
                "device": "cpu",
                "text_repair_validation_gate_passed": True,
                "human_criterion_accuracy_verified": False,
            }
        ]

    monkeypatch.setattr(
        service.IELTSService.inner,
        "encoder",
        SimpleNamespace(to_async=SimpleNamespace(predict=predict)),
    )
    with TestClient(service.IELTSService.to_asgi()) as client:
        yield client


def test_native_bento_multipart_api_with_stubbed_model(speaking_client):
    response = speaking_client.post(
        "/score_speaking",
        headers=AUTH,
        files=[("audio", ("candidate.wav", wav_bytes(), "audio/wav"))],
        data={"metadata": json.dumps(metadata())},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["speaking_band"] == 6.0
    assert result["audio_seconds"] == 1.0
    assert result["transcript_sources"] == ["caller_supplied"]
    assert result["human_criterion_accuracy_verified"] is False


def test_upload_urls_and_unauthenticated_requests_are_rejected(speaking_client):
    assert speaking_client.post("/score_speaking", json={}).status_code == 401
    response = speaking_client.post(
        "/score_speaking",
        headers=AUTH,
        files=[
            ("audio", (None, "http://127.0.0.1/private")),
            ("metadata", (None, json.dumps(metadata()))),
        ],
    )
    assert response.status_code == 400


def test_invalid_recording_is_a_client_error(speaking_client):
    response = speaking_client.post(
        "/score_speaking",
        headers=AUTH,
        files=[("audio", ("invalid.wav", b"not audio", "audio/wav"))],
        data={"metadata": json.dumps(metadata())},
    )
    assert response.status_code == 400, response.text

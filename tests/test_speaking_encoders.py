"""Check tensor batching and the published audio feature recipe."""

import base64
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from ielts_scoring.audio_features import audio_features as reference_audio

from ielts_api.speaking import AudioEncoder, TextEncoder, warmup_session


def test_audio_tensor_batches_preserve_upstream_chunk_weighting(tmp_path, monkeypatch):
    calls = []

    class Model:
        def tokenize(self, items):
            calls.append(len(items))
            signals = [sf.read(item["audio"], dtype="float32")[0] for item in items]
            return {
                "features": torch.tensor(
                    np.array(
                        [[len(s) / 320000, float(abs(s).mean())] for s in signals], dtype=np.float32
                    )
                )
            }

        def __call__(self, values):
            return {"sentence_embedding": values["features"].repeat(1, 384)}

        def named_parameters(self):
            return []

        def load_state_dict(self, state, strict=False):
            return SimpleNamespace(unexpected_keys=[], missing_keys=[])

        def eval(self):
            return self

        def requires_grad_(self, enabled):
            return self

    model = Model()
    torch.manual_seed(17)
    head = torch.nn.Sequential(
        torch.nn.Linear(768, 128),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 3),
        torch.nn.Sigmoid(),
    ).eval()
    sessions = [warmup_session(), warmup_session()]
    # Cross a 20-second chunk boundary and include a second, shorter session.
    wave = 0.1 * np.sin(2 * np.pi * 190 * np.arange(21 * 16000) / 16000)
    sf.write(tmp_path / "long.flac", wave, 16000)
    sessions[0].audio_flac = base64.b64encode((tmp_path / "long.flac").read_bytes()).decode()
    sessions[0].responses[0].audio_end = 21.0
    sessions[0].responses[0].end = 21.0
    rows = []
    for index, session in enumerate(sessions):
        path = tmp_path / f"{index}.flac"
        path.write_bytes(base64.b64decode(session.audio_flac))
        rows.append(
            {
                "session_id": str(index),
                "audio_file": path.name,
                "responses": [r.model_dump() for r in session.responses],
            }
        )
    expected, _ = reference_audio(rows, tmp_path, model, head, "cpu")
    assert calls == [1, 1, 1]
    calls.clear()
    encoder = AudioEncoder.__new__(AudioEncoder)
    encoder.model, encoder.head, encoder.chunk_batch_size = model, head, 4
    encoder.device = "cpu"
    actual = encoder.encode(sessions)
    assert calls == [3], "All three chunks should share one model forward pass"
    for index in range(2):
        np.testing.assert_allclose(actual[index], expected[str(index)], atol=2e-6, rtol=0)


def test_text_batches_mask_padding_and_keep_order(monkeypatch):
    calls = []

    def model(**inputs):
        calls.append(inputs["input_ids"].shape)
        hidden = inputs["input_ids"].float().unsqueeze(-1).repeat(1, 1, 1024)
        return SimpleNamespace(last_hidden_state=hidden)

    encoder = TextEncoder.__new__(TextEncoder)
    encoder.device = "cpu"
    encoder.text_model = model
    encoder.tokenizer = SimpleNamespace(pad_token_id=99)
    result = encoder.encode([[2, 4, 6], [10], [3, 9]])
    assert calls == [torch.Size([3, 3])]
    np.testing.assert_array_equal(result[:, 0], [4.0, 10.0, 6.0])

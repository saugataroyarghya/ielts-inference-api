"""CPU checks of batching mechanics; these do not substitute for CUDA parity."""

import base64
import contextlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

from speaking_api.encoders import AudioEncoder, TextEncoder, warmup_session
from speaking_api.vendor import audio_encoder_worker


def test_audio_tensor_batches_preserve_upstream_chunk_weighting(tmp_path, monkeypatch):
    calls = []

    class Model:
        def tokenize(self, items):
            calls.append(len(items))
            signals = [sf.read(item["audio"], dtype="float32")[0] for item in items]
            return torch.tensor(
                np.array(
                    [[len(s) / 320000, float(abs(s).mean())] for s in signals], dtype=np.float32
                )
            )

        def __call__(self, values):
            return {"sentence_embedding": values.repeat(1, 384)}

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
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(
            FastSentenceTransformer=SimpleNamespace(
                from_pretrained=lambda **kwargs: model, get_peft_model=lambda m, **kwargs: m
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers.util",
        SimpleNamespace(batch_to_device=lambda inputs, device: inputs),
    )
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        "speaking_api.vendor.checkpoint.load_audio_checkpoint",
        lambda bundle: ({"adapters": {}, "head": head.state_dict()}, "test"),
    )
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio/config.json").write_text(
        json.dumps({"model": "test", "model_revision": "test"})
    )
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
    manifest = tmp_path / "sessions.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    audio_encoder_worker.encode(manifest, tmp_path, tmp_path, tmp_path / "reference.npz")
    assert calls == [1, 1, 1]
    calls.clear()
    encoder = AudioEncoder.__new__(AudioEncoder)
    encoder.model, encoder.head, encoder.chunk_batch_size = model, head, 4
    actual = encoder.encode(sessions)
    assert calls == [3], "All three chunks should share one model forward pass"
    with np.load(tmp_path / "reference.npz", allow_pickle=False) as expected:
        for index in range(2):
            np.testing.assert_allclose(actual[index], expected[str(index)], atol=2e-6, rtol=0)


def test_text_batches_mask_padding_and_keep_order(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "unsloth.models._decision_fast",
        SimpleNamespace(pad_length=lambda model, inputs: inputs),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth.models.decision",
        SimpleNamespace(_no_cudnn_attention=contextlib.nullcontext),
    )
    calls = []

    def model(**inputs):
        calls.append(inputs["input_ids"].shape)
        hidden = inputs["input_ids"].float().unsqueeze(-1).repeat(1, 1, 1024)
        return SimpleNamespace(last_hidden_state=hidden)

    encoder = TextEncoder.__new__(TextEncoder)
    encoder.model = SimpleNamespace()
    encoder.text_model = model
    encoder.tokenizer = SimpleNamespace(pad_token_id=99)
    result = encoder.encode([[2, 4, 6], [10], [3, 9]], device="cpu")
    assert calls == [torch.Size([3, 3])]
    np.testing.assert_array_equal(result[:, 0], [4.0, 10.0, 6.0])

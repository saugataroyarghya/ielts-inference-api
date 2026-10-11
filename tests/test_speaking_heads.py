"""Real checkpoint checks on CPU; no claim about full audio inference on Mac."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from ielts_scoring.portable_heads import FeatureScorer, expected_bands, mlp
from torch.nn import functional as F

from ielts_api.artifacts import ROOT
from ielts_api.artifacts import SPEAKING_MODEL as MODEL
from ielts_api.artifacts import verify_speaking_snapshot as verify_snapshot

pytestmark = pytest.mark.integration


def test_actual_heads_match_frozen_fixture_and_pytorch_math():
    configured = os.environ.get("IELTS_SPEAKING_BUNDLE")
    folder = Path(
        configured
        or snapshot_download(
            MODEL["repo_id"],
            revision=MODEL["revision"],
            local_files_only=True,
        )
    )
    verify_snapshot(folder)
    scorer = FeatureScorer.from_bundle(folder)

    def features(norm):
        ramp = np.linspace(-0.75, 0.75, len(norm["mean"]), dtype=np.float32)
        return np.stack(
            [norm["mean"] + ramp * norm["std"], norm["mean"] - ramp * norm["std"]]
        ).astype(np.float32)

    text, audio = features(scorer.text_norm), features(scorer.audio_norm)
    fixture = json.loads((ROOT / "tests/fixtures/speaking-heads.json").read_text())
    assert fixture["model_revision"] == MODEL["revision"]
    for actual, expected in zip(scorer.predict(text, audio), fixture["expected"], strict=True):
        assert actual["scores"] == expected["scores"]
        assert actual["speaking_band"] == expected["speaking_band"]
        np.testing.assert_allclose(
            list(actual["raw_scores"].values()),
            list(expected["raw_scores"].values()),
            atol=1e-5,
            rtol=0,
        )

    audio = np.clip((audio - scorer.audio_norm["mean"]) / scorer.audio_norm["std"], -10, 10)
    normalized = np.clip((text - scorer.text_norm["mean"]) / scorer.text_norm["std"], -10, 10)
    for inputs, weights, prefix in [
        (normalized, scorer.text_weights, ""),
        (np.concatenate([text, audio], axis=-1), scorer.fixed_weights, "fc_head."),
        (audio, scorer.fixed_weights, "pr_head."),
    ]:
        tensors = {key: torch.from_numpy(value.copy()) for key, value in weights.items()}
        x = torch.from_numpy(inputs)
        x = F.layer_norm(
            x, (x.shape[-1],), tensors[prefix + "0.weight"], tensors[prefix + "0.bias"], eps=1e-5
        )
        x = F.linear(x, tensors[prefix + "1.weight"], tensors[prefix + "1.bias"])
        x = F.gelu(x, approximate="none")
        x = F.linear(x, tensors[prefix + "4.weight"], tensors[prefix + "4.bias"])
        reference = (x.reshape(-1, 9).softmax(-1) * torch.arange(1.0, 10.0)).sum(-1).numpy()
        actual = expected_bands(mlp(inputs, weights, prefix).reshape(-1, 9))
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=0)

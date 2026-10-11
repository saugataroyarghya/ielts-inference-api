"""Device selection and allocator budgets without loading model weights."""

from unittest.mock import patch

import pytest

from ielts_api.device import inference_device


def test_mps_workers_share_a_conservative_allocator_budget(monkeypatch):
    monkeypatch.setenv("IELTS_DEVICE", "auto")
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.set_per_process_memory_fraction") as limit,
    ):
        assert inference_device("writing") == "mps"
        assert inference_device("speaking") == "mps"
        fractions = [call.args[0] for call in limit.call_args_list]
        assert len(fractions) == 2
        assert all(0 < fraction < 1 for fraction in fractions)
        assert sum(fractions) <= 0.6


def test_explicit_unavailable_device_is_not_silently_replaced(monkeypatch):
    monkeypatch.setenv("IELTS_DEVICE", "cuda")
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="unavailable"):
            inference_device("writing")


def test_cpu_does_not_touch_mps_allocator(monkeypatch):
    monkeypatch.setenv("IELTS_DEVICE", "cpu")
    with patch("torch.mps.set_per_process_memory_fraction") as limit:
        assert inference_device("speaking") == "cpu"
        limit.assert_not_called()

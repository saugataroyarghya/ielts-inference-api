"""One native PyTorch device policy for both scorers."""

import os

import torch


def inference_device(model: str) -> str:
    device = os.environ.get("IELTS_DEVICE", "auto").lower()
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if device == "auto":
        device = next(name for name, supported in available.items() if supported)
    if device not in available:
        raise ValueError("IELTS_DEVICE must be auto, cuda, mps or cpu")
    if not available[device]:
        raise RuntimeError(f"Requested PyTorch device {device!r} is unavailable")
    if device == "mps":
        # Each Bento worker has its own allocator. Keep their combined budgets
        # below the recommended working set, leaving space for macOS and apps.
        # This caps MPS allocations, not total process/system RAM.
        fraction = {"writing": 0.10, "speaking": 0.50}[model]
        torch.mps.set_per_process_memory_fraction(fraction)
    return device

"""Download and verify the pinned IELTS models for offline serving."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from ielts_api.artifacts import MODELS, verify_snapshot, verify_speaking_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asr", action="store_true", help="Also cache optional transcription weights"
    )
    args = parser.parse_args()
    for name, model in MODELS.items():
        folder = Path(
            snapshot_download(
                model["repo_id"],
                revision=model["revision"],
                allow_patterns=list(model["files"]) if name == "writing" else None,
            )
        )
        (verify_snapshot if name == "writing" else verify_speaking_snapshot)(folder)
        print(f"Verified {name}: {folder}")
    if args.asr:
        model = MODELS["speaking"]["optional_asr"]
        print(snapshot_download(model["repo"], revision=model["revision"]))


if __name__ == "__main__":
    main()

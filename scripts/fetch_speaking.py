"""Download pinned speaking artifacts; large GPU bases require --encoders."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from speaking_api.artifacts import MODEL, verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", action="store_true")
    parser.add_argument("--asr", action="store_true")
    args = parser.parse_args()
    path = Path(
        snapshot_download(
            MODEL["repo_id"],
            revision=MODEL["revision"],
            allow_patterns=list(MODEL["files"]),
        )
    )
    verify_snapshot(path)
    print(f"Verified speaking checkpoint: {path}")
    names = (["audio", "text"] if args.encoders else []) + (["optional_asr"] if args.asr else [])
    for name in names:
        base = MODEL["bases"][name]
        path = snapshot_download(base["repo"], revision=base["revision"])
        print(f"Pinned {name}: {path}")


if __name__ == "__main__":
    main()

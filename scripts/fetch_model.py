"""Download once for development/builds; deployment loads the bundled snapshot offline."""

from pathlib import Path

from huggingface_hub import snapshot_download

from ielts_api.artifacts import MODEL, verify_snapshot


def main() -> None:
    folder = Path(
        snapshot_download(
            MODEL["repo_id"], revision=MODEL["revision"], allow_patterns=list(MODEL["files"])
        )
    )
    verify_snapshot(folder)
    print(f"Verified {len(MODEL['files'])} files at {MODEL['revision']}\n{folder}")


if __name__ == "__main__":
    main()

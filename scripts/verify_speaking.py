"""CUDA gate: compare resident/batched inference with the unchanged upstream workers.

Input: a JSON list of >=2 sessions, each containing responses with audio_path,
question, part and transcript. Paths are relative to the JSON file. Use recordings
of different lengths, including one longer than 20 seconds. No audio is retained.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from ielts_api.artifacts import ROOT, sha256
from speaking_api.artifacts import MODEL, verify_snapshot
from speaking_api.preparation import Preparer
from speaking_api.schemas import SessionInput
from speaking_api.vendor.portable_heads import FeatureScorer
from speaking_api.vendor.prepare_input import prepare as reference_prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--audio-python", default=sys.executable)
    parser.add_argument("--text-python", default=sys.executable)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/speaking-parity.json")
    args = parser.parse_args()
    if not 0 < args.atol <= 0.02:
        parser.error("--atol must be positive and at most 0.02 band/feature units")
    corpus = json.loads(args.corpus.read_text())
    if not isinstance(corpus, list) or len(corpus) < 2:
        parser.error("Provide at least two sessions for mixed-length batch verification")
    bundle = Path(
        snapshot_download(MODEL["repo_id"], revision=MODEL["revision"], local_files_only=True)
    )
    verify_snapshot(bundle)
    bases = {
        name: snapshot_download(base["repo"], revision=base["revision"], local_files_only=True)
        for name, base in MODEL["bases"].items()
        if name != "optional_asr"
    }
    vendor = ROOT / "speaking_api/vendor"
    with tempfile.TemporaryDirectory(prefix="ielts-parity-") as temporary:
        folder = Path(temporary)
        prepared, rows, lengths = [], [], []
        for index, session in enumerate(corpus):
            responses = []
            for original in session["responses"]:
                row = dict(original)
                if row.get("transcript") is None:
                    parser.error("Parity corpus requires unedited transcripts for determinism")
                row["audio_path"] = str((args.corpus.parent / row["audio_path"]).resolve())
                responses.append(row)
            path = folder / f"request-{index}.json"
            path.write_text(json.dumps({"session_id": str(index), "responses": responses}))
            reference_dir = folder / str(index)
            row = reference_prepare(path, reference_dir)
            row["audio_file"] = str(reference_dir / row["audio_file"])
            rows.append(row)
            item, _, seconds = Preparer().prepare(
                [Path(r["audio_path"]) for r in responses],
                SessionInput.model_validate(
                    {
                        "responses": [
                            {key: value for key, value in r.items() if key != "audio_path"}
                            for r in responses
                        ]
                    }
                ),
            )
            prepared.append(item)
            lengths.append(seconds)
        if max(lengths) <= 20 or len(set(lengths)) < 2:
            parser.error(
                "Use different audio lengths, including a recording longer than 20 seconds"
            )
        manifest = folder / "sessions.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        # Exit reference processes before loading resident models; avoid doubling VRAM.
        subprocess.run(
            [
                args.audio_python,
                str(vendor / "audio_encoder_worker.py"),
                str(manifest),
                str(folder),
                str(bundle),
                str(folder / "audio.npz"),
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                args.text_python,
                str(vendor / "text_encoder_worker.py"),
                str(manifest),
                str(folder / "text.npz"),
                "--model",
                MODEL["bases"]["text"]["repo"],
                "--revision",
                MODEL["bases"]["text"]["revision"],
            ],
            check=True,
            env=env,
        )
        with np.load(folder / "audio.npz", allow_pickle=False) as archive:
            expected_audio = np.stack([archive[str(i)] for i in range(len(rows))])
        with np.load(folder / "text.npz", allow_pickle=False) as archive:
            expected_text = np.stack([archive[str(i)] for i in range(len(rows))])
        heads = FeatureScorer.from_bundle(bundle)
        expected = heads.predict(expected_text, expected_audio)
        from speaking_api.encoders import SpeakingRuntime

        runtime = SpeakingRuntime(str(bundle), bases["audio"], bases["text"])
        import torch

        torch.cuda.reset_peak_memory_stats()
        worst = {"audio": 0.0, "text": 0.0, "raw_band": 0.0}
        # Test resident single requests, mixed batches and reordered repeated requests.
        orders = [[i] for i in range(len(rows))]
        orders += [list(range(i, min(i + 2, len(rows)))) for i in range(0, len(rows), 2)]
        orders += [[len(rows) - 1, 0], [0, 0]]
        for order in orders:
            sessions = [prepared[i] for i in order]
            actual_audio = runtime.audio.encode(sessions)
            actual_text = runtime.text.encode([runtime.text.tokenize(s) for s in sessions])
            for name, actual, reference in (
                ("audio", actual_audio, expected_audio[order]),
                ("text", actual_text, expected_text[order]),
            ):
                worst[name] = max(worst[name], float(np.abs(actual - reference).max()))
                np.testing.assert_allclose(actual, reference, rtol=0, atol=args.atol)
            for index, result in zip(order, heads.predict(actual_text, actual_audio), strict=True):
                if result["scores"] != expected[index]["scores"]:
                    raise AssertionError(f"Displayed scores changed for corpus item {index}")
                if result["speaking_band"] != expected[index]["speaking_band"]:
                    raise AssertionError(f"Overall score changed for corpus item {index}")
                error = max(
                    abs(result["raw_scores"][key] - expected[index]["raw_scores"][key])
                    for key in result["raw_scores"]
                )
                worst["raw_band"] = max(worst["raw_band"], error)
                if error > args.atol:
                    raise AssertionError(f"Raw score drift {error} exceeds {args.atol}")
        report = {
            "passed": True,
            "model_revision": MODEL["revision"],
            "cases": len(rows),
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
            "corpus_sha256": sha256(args.corpus),
            "max_absolute_error": worst,
            "tolerance": args.atol,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "reference_audio_python": args.audio_python,
            "reference_text_python": args.text_python,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

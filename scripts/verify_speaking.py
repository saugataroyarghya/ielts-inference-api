"""Compare batched inference against the pinned model's native singleton path.

Corpus: a JSON list of sessions containing responses with audio_path, question,
part and transcript. Paths are relative to the corpus. Use at least two different
lengths, including one recording longer than 20 seconds. Audio stays local.
"""

import argparse
import base64
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from ielts_api.artifacts import ROOT, SPEAKING_MODEL, sha256
from ielts_api.audio import Preparer
from ielts_api.schemas import SessionInput
from ielts_api.speaking import SpeakingRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/speaking-parity.json")
    args = parser.parse_args()
    if not 0 < args.atol <= 0.02:
        parser.error("--atol must be positive and at most 0.02 band/feature units")
    corpus = json.loads(args.corpus.read_text())
    if not isinstance(corpus, list) or len(corpus) < 2:
        parser.error("Provide at least two sessions for mixed-length batch verification")
    bundle = snapshot_download(
        SPEAKING_MODEL["repo_id"], revision=SPEAKING_MODEL["revision"], local_files_only=True
    )
    prepared, lengths = [], []
    for session in corpus:
        responses = session["responses"]
        if any(row.get("transcript") is None for row in responses):
            parser.error("Parity requires transcripts for deterministic comparisons")
        item, _, seconds = Preparer().prepare(
            [(args.corpus.parent / r["audio_path"]).resolve() for r in responses],
            SessionInput.model_validate(
                {
                    "responses": [
                        {key: value for key, value in row.items() if key != "audio_path"}
                        for row in responses
                    ]
                }
            ),
        )
        prepared.append(item)
        lengths.append(seconds)
    if max(lengths) <= 20 or len(set(lengths)) < 2:
        parser.error("Use different lengths, including a recording longer than 20 seconds")
    runtime = SpeakingRuntime(bundle)
    print(f"Loaded native model on {runtime.device}", flush=True)
    expected_text, expected_audio = [], []
    with tempfile.TemporaryDirectory(prefix="ielts-parity-") as temporary:
        folder = Path(temporary)
        for index, item in enumerate(prepared):
            path = folder / f"{index}.flac"
            path.write_bytes(base64.b64decode(item.audio_flac))
            row = {
                "session_id": str(index),
                "audio_file": path.name,
                "responses": [r.model_dump() for r in item.responses],
            }
            text, audio, _ = runtime.model.encode_prepared(row, folder)
            expected_text.append(text.cpu().numpy()[0])
            expected_audio.append(audio.cpu().numpy()[0])
    expected_text, expected_audio = np.stack(expected_text), np.stack(expected_audio)
    expected = runtime.model.score_features(expected_text, expected_audio)
    worst = {"audio": 0.0, "text": 0.0, "raw_band": 0.0}
    orders = [[i] for i in range(len(prepared))]
    orders += [list(range(i, min(i + 2, len(prepared)))) for i in range(0, len(prepared), 2)]
    orders += [[len(prepared) - 1, 0], [0, 0]]
    measurements = []
    for order in orders:
        started = time.perf_counter()
        sessions = [prepared[i] for i in order]
        actual_audio = runtime.audio.encode(sessions)
        actual_text = runtime.text.encode([runtime.text.tokenize(s) for s in sessions])
        actual = runtime.model.score_features(actual_text, actual_audio)
        elapsed = time.perf_counter() - started
        measurements.append(
            {"order": order, "seconds": elapsed, "sessions_per_second": len(order) / elapsed}
        )
        for name, values, reference in (
            ("audio", actual_audio, expected_audio[order]),
            ("text", actual_text, expected_text[order]),
        ):
            worst[name] = max(worst[name], float(np.abs(values - reference).max()))
            np.testing.assert_allclose(values, reference, rtol=0, atol=args.atol)
        for index, result in zip(order, actual, strict=True):
            assert result["scores"] == expected[index]["scores"], (index, result, expected[index])
            assert result["speaking_band"] == expected[index]["speaking_band"]
            error = max(
                abs(result["raw_scores"][k] - expected[index]["raw_scores"][k])
                for k in result["raw_scores"]
            )
            worst["raw_band"] = max(worst["raw_band"], error)
            assert error <= args.atol, error
        print(f"PASS order={order}, seconds={elapsed:.3f}", flush=True)
    report = {
        "passed": True,
        "model_revision": SPEAKING_MODEL["revision"],
        "device": runtime.device,
        "cases": len(prepared),
        "audio_seconds": lengths,
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "corpus_sha256": sha256(args.corpus),
        "max_absolute_error": worst,
        "tolerance": args.atol,
        "measurements": measurements,
        "scope": "Batching parity and encoder timings; not grading accuracy or an HTTP load test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Measure warm HTTP throughput using real local audio and the upload contract."""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

from speaking_api.schemas import SpeakingResult


async def benchmark(args):
    key = os.environ["IELTS_API_KEY"]
    corpus = json.loads(args.corpus.read_text())
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must be a nonempty list of sessions")
    cases = []
    for session in corpus:
        files, responses = [], []
        for row in session["responses"]:
            path = args.corpus.parent / row["audio_path"]
            files.append(("audio", (path.name, path.read_bytes(), "application/octet-stream")))
            responses.append({k: v for k, v in row.items() if k != "audio_path"})
        cases.append((files, {"metadata": json.dumps({"responses": responses})}))
    latencies, statuses, audio_seconds = [], Counter(), []
    limit = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.url, headers={"Authorization": f"Bearer {key}"}, timeout=330
    ) as client:
        # Readiness alone doesn't warm all representative lengths or batch sizes.
        for files, data in cases:
            response = await client.post("/score_speaking", files=files, data=data)
            response.raise_for_status()
            SpeakingResult.model_validate(response.json())

        async def request(index):
            files, data = cases[index % len(cases)]
            async with limit:
                started = time.perf_counter()
                try:
                    response = await client.post("/score_speaking", files=files, data=data)
                    statuses[str(response.status_code)] += 1
                    if response.is_success:
                        result = SpeakingResult.model_validate(response.json())
                        latencies.append(time.perf_counter() - started)
                        audio_seconds.append(result.audio_seconds)
                except httpx.HTTPError:
                    statuses["transport_error"] += 1

        started = time.perf_counter()
        # Keep clients busy as each finishes, without automatic retries hiding overload.
        pending = iter(range(args.requests))

        async def worker():
            for index in pending:
                await request(index)

        await asyncio.gather(*(worker() for _ in range(min(args.concurrency, args.requests))))
        elapsed = time.perf_counter() - started
    report = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "statuses": dict(statuses),
        "wall_seconds": elapsed,
        "successful_requests_per_second": len(latencies) / elapsed,
        "audio_seconds_per_wall_second": sum(audio_seconds) / elapsed,
        "success_latency_seconds": dict(
            zip(("p50", "p95", "p99"), map(float, np.percentile(latencies, [50, 95, 99])))
        )
        if latencies
        else None,
    }
    print(json.dumps(report, indent=2))
    if len(latencies) != args.requests:
        raise SystemExit("Some requests failed or were rejected; see status counts above")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:3008")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 64 or not 1 <= args.requests <= 10000:
        parser.error("Use concurrency 1–64 and requests 1–10000")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()

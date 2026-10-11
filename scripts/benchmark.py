"""Measure warm HTTP throughput and latency for either IELTS endpoint."""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

from ielts_api.artifacts import ROOT
from ielts_api.schemas import ScoreResponse, SpeakingResult


async def benchmark(args):
    key = os.environ["IELTS_API_KEY"]
    cases = []
    if args.task == "writing":
        corpus = json.loads((args.corpus or ROOT / "tests/fixtures/reference.json").read_text())
        rows = corpus["cases"] if isinstance(corpus, dict) else corpus
        cases = [{"json": row.get("input", row)} for row in rows]
        route, schema = "/score", ScoreResponse
    else:
        if args.corpus is None:
            raise ValueError("Speaking requires --corpus with local audio and transcripts")
        corpus = json.loads(args.corpus.read_text())
        for session in corpus:
            files, responses = [], []
            for row in session["responses"]:
                path = args.corpus.parent / row["audio_path"]
                files.append(("audio", (path.name, path.read_bytes(), "application/octet-stream")))
                responses.append({k: v for k, v in row.items() if k != "audio_path"})
            cases.append(
                {"files": files, "data": {"metadata": json.dumps({"responses": responses})}}
            )
        route, schema = "/score_speaking", SpeakingResult
    if not cases:
        raise ValueError("Corpus must contain at least one request")
    latencies, statuses, audio_seconds = [], Counter(), []
    limit = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.url, headers={"Authorization": f"Bearer {key}"}, timeout=330
    ) as client:
        # Readiness alone doesn't warm all representative lengths or batch sizes.
        for case in cases:
            response = await client.post(route, **case)
            response.raise_for_status()
            schema.model_validate(response.json())

        async def request(index):
            case = cases[index % len(cases)]
            async with limit:
                started = time.perf_counter()
                try:
                    response = await client.post(route, **case)
                    statuses[str(response.status_code)] += 1
                    if response.is_success:
                        result = schema.model_validate(response.json())
                        latencies.append(time.perf_counter() - started)
                        audio_seconds.append(getattr(result, "audio_seconds", 0))
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
        "task": args.task,
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
    parser.add_argument("--task", choices=["writing", "speaking"], required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 64 or not 1 <= args.requests <= 10000:
        parser.error("Use concurrency 1–64 and requests 1–10000")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()

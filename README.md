# IELTS inference API

BentoML services for IELTS writing and experimental speaking scoring. One
`pyproject.toml` and `uv.lock` pin the common stack: PyTorch 2.10.0,
Transformers 5.19.0, Hugging Face Hub 1.33.0 and safetensors 0.8.0.
CPU and CUDA use the corresponding PyTorch builds of the same release.

```text
service.py                 Writing BentoML entry point
speaking_service.py        Speaking entry point and private GPU worker
ielts_api/                 Writing inference, contracts and API middleware
speaking_api/              Speaking preparation, encoders and contracts
  vendor/                  Unchanged upstream reference implementation
scripts/                   Fetch, build, verification and benchmark commands
tests/                     Contracts, regressions and batching checks
*.lock.json                Pinned model revisions and checksums
pyproject.toml + uv.lock    Shared dependencies; cpu/cuda/asr extras
requirements*.txt          Generated Bento image dependency exports
```

This uses BentoML's Python Image API rather than a handwritten Dockerfile.
Each entry point produces its own Bento and HTTP API. Their processes can be
scaled independently while using the same library versions.

## Local writing API

Python 3.12.12 and uv are required. Select an environment once, then use
`uv run --no-sync` to preserve its extras:

```bash
uv sync --locked --extra cpu
uv run --no-sync python -m scripts.fetch_model
uv run --no-sync bentoml serve service:IELTSScoringService --host 127.0.0.1 --port 3007
```

`POST /score` accepts `question` and `essay`. It returns continuous TR, CC, LR
and GRA estimates, their mean, truncation status and model identity. CPU FP32,
one model worker and two PyTorch compute threads are the verified default.
Only the essay is truncated to fit the 1,024-token combined limit. These are
research estimates, not official IELTS grades.

Writing authentication is enabled when `IELTS_API_KEY` is set. Without it,
keep the service on loopback. The speaking API requires this key at startup.
Send `Authorization: Bearer <key>`; `/livez` and `/readyz` remain unauthenticated.
This is application ASGI middleware, not a claim of built-in BentoML API-key auth.

## Speaking on the NVIDIA VM

The current released encoders require CUDA. Mac tests cover preprocessing,
HTTP behavior, batching mechanics and real scoring heads. Full audio inference,
CUDA prediction parity and throughput remain to be verified on the future VM.
MPS support is a separate model-side change.

Use Linux x86_64, a suitable NVIDIA driver, and enough disk space for the base
models and CUDA image. Start with one GPU worker. Measure peak VRAM with the
verification command before selecting a VM size or increasing batch sizes.

```bash
uv sync --locked --extra cuda --extra asr
uv run --no-sync python -m scripts.fetch_speaking --encoders --asr
export IELTS_API_KEY="$(openssl rand -hex 32)"
export IELTS_ENABLE_ASR=1
uv run --no-sync bentoml serve speaking_service:IELTSSpeakingService --host 127.0.0.1 --port 3008
```

Set `IELTS_ENABLE_ASR=0` to require caller transcripts. When enabled, the pinned
faster-whisper model is loaded once on CPU and reused. The GPU worker keeps
both speaking encoders and the small scoring heads resident. Models warm up
before the worker becomes ready. Requests do not launch encoder subprocesses.

Upload **candidate-only** audio. One `audio` part corresponds to each response,
in metadata order. Use an unedited transcript; correction changes the model's
input. Optional `start`/`end` select candidate intervals in seconds. Optional
`timeline_start` must be provided for every response or none.

```bash
curl --fail-with-body http://127.0.0.1:3008/score_speaking \
  -H "Authorization: Bearer $IELTS_API_KEY" \
  -F 'audio=@candidate.wav;type=audio/wav' \
  -F 'metadata={"responses":[{"question":"Where do you live?","part":1,"transcript":"I live in Dhaka."}]}'
```

Omit `transcript` to invoke ASR. Repeat the `audio` field and add metadata
responses for multiple recordings. The result covers the provided responses
and includes four half-band scores, overall band, raw scores, transcript
provenance, request ID, model revision and experimental-status flags. ASR does
not guarantee that every disfluency is preserved.

Accepted audio: mono/stereo WAV, FLAC or OGG at 8–48 kHz. Limits: 12 recordings,
300 seconds of selected candidate audio, 32 MiB total request body, 30 seconds
to upload, and the trained 8,192-token text context. Input URLs and server paths
are rejected. Silence within each response is retained. Oversized requests
return 413; invalid inputs return 400; missing/invalid credentials return 401.

## Concurrency, batching and overload

The asynchronous speaking gateway admits at most eight requests per process,
including body uploads. Two CPU preparation jobs can run at a time; ASR uses
one persistent recognizer with serialized calls. Excess requests receive 429.

BentoML groups up to two sessions per GPU call. Audio chunks are at most
20 seconds and run in tensor batches of up to four, grouped by duration.
Text records are padded with an attention mask and run in one tensor batch;
masked pooling preserves each session's output. A too-long text record does
not invalidate the other session in its batch.

The initial batching latency budget is 120 seconds, GPU request timeout
240 seconds and gateway timeout 300 seconds. These are conservative bounds,
not a measured latency promise. Queue-budget exhaustion can return 503.
Clients should use bounded retries with backoff for overload responses.

This is bounded in-memory admission and adaptive batching, **not a durable job
queue**. Accepted requests are lost if a worker restarts. A persistent job API
would need separate job storage and recovery semantics. Do not increase worker
count blindly: every additional worker loads another pair of encoders. Tune
batch sizes and admission using measurements, then add GPU replicas. BentoML
metrics and access logs expose request latency and status; application logs
include request IDs without transcripts or audio.

## Verification and throughput gate

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python -m scripts.export_requirements --check
uv run --no-sync pytest -q -m 'not integration'
uv run --no-sync python -m scripts.fetch_model
uv run --no-sync python -m scripts.fetch_speaking
uv run --no-sync pytest -q
```

The common writing stack passes the frozen real-model regression cases.
Speaking CPU tests verify the real checkpoint heads against PyTorch math and
check preprocessing/aggregation against the unchanged upstream code. Stubbed
encoder tests check batching mechanics; they do not establish GPU accuracy.

On the VM, create a private corpus JSON file (keep it under ignored `data/`):

```json
[
  {"responses": [{"audio_path": "long.wav", "question": "Describe your city.", "part": 2, "transcript": "The actual unedited transcript..."}]},
  {"responses": [{"audio_path": "short.wav", "question": "Where do you live?", "part": 1, "transcript": "The actual unedited transcript..."}]}
]
```

Use representative recordings of different lengths, including one longer than
20 seconds. The verifier compares single, mixed, reordered and repeated batches
against the unchanged upstream workers. It rejects changed displayed scores
and excessive feature/raw-score drift, and records peak allocated VRAM.

```bash
uv run --no-sync python -m scripts.verify_speaking data/corpus.json
uv run --no-sync python -m scripts.benchmark_speaking data/corpus.json --concurrency 1
uv run --no-sync python -m scripts.benchmark_speaking data/corpus.json --concurrency 4
uv run --no-sync python -m scripts.benchmark_speaking data/corpus.json --concurrency 8
```

Unsloth 2026.10.3 declares Transformers <=5.17, but the released audio installer
itself overrides that cap with a newer Transformers commit. Our explicit uv
override selects stable 5.19.0 for all models. Resolution and writing regression
success do not prove GPU encoder compatibility. Run the CUDA gate before
production. To check against the original release's tested environments too,
pass their executables via `--audio-python` and `--text-python`; otherwise the
reference workers run in the shared environment. Retain the corpus and reports
privately when upgrading model code, dependencies or hardware.

## Docker packaging and deployment

After committing source changes, build a Bento tagged with the Git commit:

```bash
uv run --no-sync python -m scripts.build_bento
# On the GPU VM, after CUDA verification:
uv run --no-sync python -m scripts.build_bento --speaking
```

Use the printed `bento_tag` below. The speaking image targets Linux amd64 and
includes CUDA dependencies and pinned models. Build it on the VM; building a
GPU image on a Mac does not verify its inference behavior.

```bash
uv run --no-sync bentoml containerize "$BENTO_TAG" -t "$BENTO_TAG" --opt platform=linux/amd64
docker run -d --name ielts-speaking --restart unless-stopped \
  --gpus all --shm-size 1g --cpus 4 --memory 12g \
  -e IELTS_API_KEY -e IELTS_ENABLE_ASR=1 \
  -p 127.0.0.1:3008:3000 "$BENTO_TAG"
uv run --no-sync python -m scripts.benchmark_speaking data/corpus.json --requests 2 --concurrency 1
```

Docker on the VM needs NVIDIA Container Toolkit. Verify `/readyz` and a real
scoring request before routing traffic. Publish only the gateway through your
TLS reverse proxy; keep the encoder service and metrics private. Promote the
verified container by digest. No VM or container registry has been provisioned.

For the local writing image, use `--opt platform=linux/arm64`, then run
`uv run --no-sync python -m scripts.smoke_container "$BENTO_TAG"`. That smoke
test exercises real predictions in a temporary container without outbound
network or host mounts.

The CPU/GPU requirements and explicit override file are generated with
`uv run --no-sync python -m scripts.export_requirements`. Never edit them by
hand. Model/source checksums and base revisions are pinned in the manifests.
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and model terms.

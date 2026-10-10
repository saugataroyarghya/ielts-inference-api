# Milestone 1: reproduce a prediction across deployment boundaries

The question for this lesson is: **can the same release produce the same scores
when it moves from Python code to HTTP, a Bento, and a Linux container?**

This milestone is complete only after all four boundaries have been checked.
The README contains executable commands; this walkthrough explains the decisions.

## 1. Pin the executable model, not just its name

`model.lock.json` pins the Hub commit and checksums for 13 files. The tokenizer
and custom code influence the answer just as the weights do. At startup we verify
all snapshot files plus the vendored scorer, then load local files only.

The original model has an essay-only pooling mask and a dense four-expert head.
We call its existing `score()` method without changing the numerical algorithm.
An arbitrary Hugging Face text-classification pipeline would miss the required
essay mask.

Exercise: inspect `model.lock.json`. Find the weights hash and tokenizer hash.
Explain why changing either should produce a new release.

## 2. Give the environment a version

`pyproject.toml` declares dependencies. `uv.lock` records their resolved versions
and distribution hashes. `uv sync --locked` installs that graph and fails if the
project metadata needs a new resolution. `.python-version` selects Python 3.12.12.

BentoML reads a runtime requirements export generated from this same lockfile.
It does not independently choose newer library versions. Development tools stay
outside that export. Linux selects the CPU PyTorch distribution; the Mac uses
the macOS distribution of the same PyTorch release.

Exercise: run the export checker. Explain why hand-editing `requirements.txt`
would let the local and container environments diverge.

## 3. Make the service lifecycle explicit

The class-level `HuggingFaceModel` tells BentoML which artifact to package.
`__init__` loads that artifact once per worker. A warmup forward pass completes
before initialization succeeds. The API then reuses the loaded model.

The input contract rejects blank or oversized text. It preserves the essay's
original spelling and paragraph structure. The output contract records the
four raw scores, arithmetic mean, truncation flag and release identity.

Exercise: call `/score` with an empty essay, then with `examples/essay.json`.
Observe the HTTP status and compare the response with `/docs.json`.

## 4. Freeze numerical expectations independently

The capture script calls the exact reviewed upstream model directly, bypassing
our production wrapper and BentoML. It records normal, short, and truncated
synthetic essays. The fixtures are committed; tests never regenerate them.

The API, built Bento, and Docker container must match these scores within
**0.001 band units**, with identical truncation flags. This tolerance checks
packaging and numerical consistency. It does not measure examiner agreement,
fairness or real-world grading quality. Cross-device precision experiments will
need their own deliberately chosen acceptance criteria.

To study a proposed change, run `scripts.capture_reference` with its default
output path. It creates a candidate under `.runtime/` and refuses to overwrite
an existing file. Review differences before intentionally updating any golden file.

Exercise: explain why regenerating expected results automatically during each
test run could hide a broken model conversion.

## 5. Distinguish source, Bento and image

```text
Git commit + model.lock.json + uv.lock
                    |
              bentoml build
                    v
              versioned Bento
                    |
           bentoml containerize
                    v
           Linux image + model files
```

The build script uses the Git commit as the Bento version and records it under
`.runtime/release.json`. Serving the Bento checks the packaged code, rather than
the source tree. Containerization adds the Linux runtime and bundled snapshot.

The offline smoke test runs a fresh container with no network and no mounts.
HTTP requests originate inside it through `docker exec`. This checks that a
successful local run was not depending on the host's Hugging Face cache.

The first image is Linux ARM64 for this Mac. A future VM must run a matching
architecture, or we must build and verify the corresponding AMD64 image. These
CPU measurements do not characterize MPS or CUDA performance.

Exercise: inspect a built Bento with `uv run bentoml get EXACT_TAG -o json`.
Find the model revision and service configuration. Inspect the Docker image
with `docker image inspect EXACT_TAG` and find its ID and architecture.

## What follows

Milestone 2 will measure latency, throughput and memory at increasing load,
then compare batching configurations. Later milestones add metrics and alerts,
deploy the tested image to a VM, automate releases and rollback, and evaluate
model quality over time. The source-check workflow is only the first CI step;
it does not replace real-model or container verification.

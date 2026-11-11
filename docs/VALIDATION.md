# Validation scope - 2026-11-11

## Observed local checks

- `python -m pytest -q`: **105 passed, 1 warning** (20.71 seconds in the final working-tree regression run).
- A separate export made from the staged Git tree, with no access to untracked source directories, also ran **105 passed, 1 warning** in 22.10 seconds. All 59 intended repository files were included.
- `python -m compileall -q src scripts`: exit 0.
- `--help` execution of all eight public Python CLI entry points: exit 0.
- Public source scan for recognizable API/private-key credential patterns and unexpected checkpoint/archive/PDF artifacts: no findings.
- Input archive SHA256 provenance is recorded independently in SOURCE_MANIFEST.json.

Regression tests cover actual two-pass gradient flow, per-example InternVL patch mapping, centered/pooled SVD geometry, finite loss validation, confidence objectives, singleton risk weighting, loss masks, strict data failures, deterministic image holdout, required mixture domains, missing/invalid API judge outputs, absent metric prerequisites, stale caches, complete checkpoint roundtrips, credential redaction, explicit checkpoint selection, stage/feedback budgets and optimizer accumulation.

One CPU integration test creates a tiny **randomly initialized** Hugging Face LLaVA with real PEFT LoRA layers. It verifies language-only adapters, connector and RSF gradients, and identical output after complete state save/restore. This is software integration evidence only; the tiny model has no pretrained knowledge and is not evaluated on any safety benchmark.

The one warning comes from PEFT saving this deliberately tiny in-memory test model without an upstream pretrained config path. PEFT assumes the vocabulary was not modified; the test does not modify vocabulary. There are no failed tests.

## Local environment

The local verification host used Python 3.12, PyTorch 2.5.1+cu121, Transformers 4.57.3 and PEFT 0.18.0. Tests ran with CUDA devices hidden. The host initially contained an incompatible huggingface_hub 1.4.1; verification used an isolated dependency directory containing 0.36.2 and rouge-score, without modifying the user's global packages. These local dependency paths are not part of the repository.

The included GitHub Actions job specifies Python 3.11 with CPU PyTorch/torchvision and the maintained Transformers/PEFT/hub versions. Local test results do not assert remote CI completion; inspect the commit's Actions checks for that status.

## Not validated

No actual LLaVA-1.5-7B or InternVL2.5-8B backbone weights, benchmark datasets, original final experiment artifacts or API evaluation credentials were available as part of the inputs. Therefore this release did not perform:

- full pretrained-backbone training or large-model GPU inference;
- live GPT judge calls or paid evaluation;
- benchmark-scale SafeEraser, VLGuard or SIUO reproduction;
- comparisons against the paper's historical checkpoints, logs or final resolved configurations;
- throughput, memory, convergence, or safety/utility performance measurement.

The test suite's synthetic images/vectors, randomly initialized tiny models and mocked API responses are isolated fixtures. They must not be used as empirical support for paper results. Full backbone validation, preserving all run artifacts, is required before claiming a new reproduced score.

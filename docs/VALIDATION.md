# Tests

## Run the checks

```bash
python -m compileall -q src scripts
USE_TF=0 CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

The test suite contains 105 unit and integration cases. It covers:

- two-pass connector gradients and RSF parameters;
- InternVL sample/patch correspondence;
- pooled, centered weighted-SVD geometry and finite loss validation;
- confidence objectives, risk weighting and response masks;
- deterministic image holdout, image/evidence handling and mixture sampling;
- API judge parsing, error handling, metric prerequisites and response caches;
- complete checkpoint roundtrips, configuration redaction and explicit model selection;
- stage budgets, feedback boundaries and gradient accumulation.

A CPU integration test creates a tiny randomly initialized Hugging Face LLaVA with PEFT LoRA layers. It checks language-only adapter placement, connector/RSF gradients and output equality after checkpoint save/restore. Other tests use small synthetic tensors, image fixtures and mocked API replies.

## Continuous integration

The [CPU regression workflow](../.github/workflows/tests.yml) runs on GitHub Actions with Python 3.11, CPU PyTorch/torchvision, Transformers 4.57.3, PEFT 0.18.0 and huggingface_hub 0.36.2. It compiles the Python sources and runs the test suite. Each commit's Actions page shows its check results.

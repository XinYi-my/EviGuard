# EviGuard

Code for **EviGuard: Evidence-Guided Connector Intervention for Cross-Modal Safety Unlearning in Multimodal LLMs**, ACM Multimedia 2026.

[Paper](https://doi.org/10.1145/3767308.3834945) · [中文说明](README_zh-CN.md) · [Implementation notes](docs/IMPLEMENTATION.md) · [Data and evaluation protocol](docs/DATA_AND_EVALUATION.md)

## Implementation

- **RSF** embeds frozen EasyOCR/BLIP-2 evidence and combines it with image-side connector activations through a learned confidence gate. Per-example risk scores modulate connector activations.
- **CLSU** estimates an evidence-weighted, centered low-rank connector risk basis; it suppresses forget-side risk projections and preserves retain-side orthogonal representations relative to a frozen reference.
- **Feedback** scores all available forget images and refits the basis from residual high-energy samples between Stage 2 intervals.
- Trainable components are the floating-point connector, language-model LoRA adapters, and RSF. The vision encoder, base language-model parameters, evidence extractors, and reference model remain frozen.

Backbone integrations are provided for **LLaVA-1.5-7B** and **InternVL2.5-8B**. SafeEraser and VLGuard have training/evaluation entry points; SIUO is evaluation-only.

```text
configs/                 four backbone/benchmark example configurations
scripts/                 training, evidence preparation, evaluation, diagnostics
src/data/                dataset loading, explicit splits, strict image/evidence checks
src/models/              backbone integration, connector hooks, complete checkpoints
src/modules/             RSF, CLSU, frozen evidence extraction
src/training/            two-stage training and feedback
src/evaluation/          strict judge protocol and metric aggregation
tests/                   CPU behavioral regressions; toy inputs are test fixtures
```

## Environment

Use Python 3.10 or 3.11 and a separate environment. The research training loaders target CUDA and 4-bit bitsandbytes loading. CPU tests do not require model downloads.

```bash
python -m venv .venv
source .venv/bin/activate
# Install torch 2.5.1 and torchvision 0.20.1 for your CUDA driver first.
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The dependencies include Transformers 4.57.3, PEFT 0.18.0, and huggingface_hub 0.36.2. A reference backbone stays resident during training; use hardware with capacity for both models and activations.

## Prepare models and data

Obtain the original model weights from their maintainers:

- [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B)
- [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b)

Set model paths in your YAML or environment:

```bash
export LLAVA_MODEL_PATH=/absolute/path/to/llava-1.5-7b-hf
export INTERNVL_MODEL_PATH=/absolute/path/to/InternVL2_5-8B
export BLIP2_MODEL_PATH=/absolute/path/to/blip2-opt-2.7b
```

Environment model paths take precedence over the corresponding YAML paths. The configuration defaults expect model directories under `./models` and data under `./data`. Run commands from the repository root. Edit the paths and JSON filenames to match your downloaded datasets; data is not downloaded, generated, or redistributed by this repository.

Obtain datasets from [SafeEraser](https://github.com/yuu250/SafeEraser), [VLGuard](https://github.com/ys-zong/VLGuard), and [SIUO](https://github.com/sinwang20/SIUO), respecting their terms. The [data guide](docs/DATA_AND_EVALUATION.md) describes accepted fields and split behavior. No external test set is used for training.

```text
data/
  safeeraser_dataset/dataset/   configured JSON, img/, pdset/image/
  VLGuard/                     train.json, test.json, train/, test/
  SIUO/                        data/siuo_gen.json, images/
  evidence_cache/              created by evidence precomputation
```

SafeEraser uses an image-level deterministic holdout. Keep the same source JSON, image references, and seed for training and evaluation; see the [data guide](docs/DATA_AND_EVALUATION.md) for split details.

## Train

All example commands require real local models and datasets. Precompute visual evidence before loading the training backbone. Training requires a cache entry for every visual training sample, and evaluation of EviGuard requires cached evidence for its visual inputs.

```bash
# SafeEraser / LLaVA
python scripts/precompute_evidence.py --config configs/safeeraser_llava.yaml
python scripts/diagnose_data.py --config configs/safeeraser_llava.yaml --check-models
python scripts/train_safeeraser.py --config configs/safeeraser_llava.yaml

# VLGuard / LLaVA
python scripts/precompute_evidence.py --config configs/vlguard_llava.yaml
python scripts/diagnose_data.py --config configs/vlguard_llava.yaml --check-models
python scripts/train_vlguard.py --config configs/vlguard_llava.yaml
```

For InternVL use `configs/safeeraser_internvl.yaml` or `configs/vlguard_internvl.yaml` with the same entry points. In the example configs, visual microbatches contain all requested domains: 3 examples for SafeEraser (forget/retain/PD), 2 for VLGuard (forget/retain). Gradient accumulation is configured separately.

`stage1_steps`, `stage2_steps`, `total_steps`, and `warmup_steps` count **optimizer updates**, each consuming `gradient_accumulation_steps` microbatches. Stage 2 keeps its entire update budget when split into feedback intervals. If a hard set cannot support the requested rank, feedback retains the previous valid basis and records a warning; initial insufficient-rank fitting fails explicitly.

New runs write logs/metrics and complete checkpoints under the configured `paths`. Checkpoint directories include `adapter/` for PEFT, exact trainable tensors, connector state, RSF state, subspace state, resolved configuration, component hashes, and optional optimizer/scheduler/scaler/RNG state. Secrets are removed from persisted configurations.

```bash
# Resume a checkpoint created by this release; the data iterator restarts.
python scripts/train_safeeraser.py --config configs/safeeraser_llava.yaml   --resume checkpoints/llava/safeeraser/step-400
```

Resume restores numerical training state at an optimizer boundary. It does not restore the exact data iterator position and is not bitwise trajectory replay. Resume with the same model and training configuration. Adapter-only folders are insufficient for a complete EviGuard restoration and are rejected.

## Evaluate

Set the external judge credentials in your shell. They are never bundled with the repository:

```bash
export OPENAI_API_KEY=your_own_key
export OPENAI_BASE_URL=https://api.openai.com/v1

python scripts/eval_safeeraser.py --config configs/safeeraser_llava.yaml   --checkpoint checkpoints/llava/safeeraser/final --tag eviguard

python scripts/eval_vlguard.py --config configs/vlguard_llava.yaml   --checkpoint checkpoints/llava/vlguard/final --tag eviguard
```

Checkpoint evaluation restores the connector, RSF, trainable backbone/LoRA tensors, and subspace together. RSF intervention remains active during generation. A missing or incomplete checkpoint raises an error. Use `--backbone-path /new/path/to/the/same/backbone` when relocating the frozen base model. The caller must supply the same pretrained backbone revision.

Select the unchanged base model explicitly with `--vanilla`; omitting a checkpoint never silently selects vanilla:

```bash
python scripts/eval_safeeraser.py --config configs/safeeraser_llava.yaml --vanilla --tag vanilla
```

Prepare SIUO evidence separately and request SIUO explicitly:

```bash
python scripts/precompute_evidence.py --config configs/vlguard_llava.yaml --dataset siuo
python scripts/eval_vlguard.py --config configs/vlguard_llava.yaml   --checkpoint checkpoints/llava/vlguard/final --eval-datasets vlguard siuo --tag eviguard
```

The judge uses `gpt-4o-2024-11-20` with explicit prompts and strict parsing. Invalid API responses, missing images, failed generation, absent references, and empty benchmark subsets stop evaluation. Missing metric prerequisites are represented as JSON `null`; keyword heuristics and text-overlap scores are never substituted for GPT judgments. Prompts, split definitions and metric aggregation are described in the [evaluation guide](docs/DATA_AND_EVALUATION.md).

SafeEraser reports original split labels, ROUGE-L precision, GPT utility score, SARR and their composite when all prerequisites exist. VLGuard reports separate unsafes ASR, safe_unsafes ASR and safe_safes FRR. SIUO reports ASR, SR, ER and joint safe-and-effective rate. JSON outputs retain generated responses and judge protocol metadata.

## Validation and contribution

```bash
python -m compileall -q src scripts
USE_TF=0 CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

See the [test guide](docs/VALIDATION.md) for the unit and integration tests, including gradient flow, checkpoint roundtrips, dataset handling and evaluation error handling.

When reporting a new experiment, retain the commit hash, exact model revisions, data split/sample IDs, resolved config, environment, checkpoint components, generated predictions and judge outputs. Submit focused changes with behavioral tests; keep benchmark validation separate from software tests.

## Citation and license

```bibtex
@inproceedings{huang2026eviguard,
  title={EviGuard: Evidence-Guided Connector Intervention for Cross-Modal Safety Unlearning in Multimodal LLMs},
  author={Huang, Xinyi and Xiang, Pengrui and Wang, Jie and Zhai, Xuchen and Liu, Jinduo and Han, Honggui},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026},
  doi={10.1145/3767308.3834945}
}
```

Code retains the repository's [Apache-2.0 license](LICENSE). Backbone weights and benchmark datasets remain governed by their original licenses and are not included.

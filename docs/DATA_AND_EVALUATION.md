# Data and evaluation contract

## Paths, image identity and evidence

Run from the repository root and edit YAML paths to match original downloads. Model paths may be overridden with the variables listed in the README. Data and evidence are supplied independently of model weights. No training or benchmark records are included in this release.

Image paths must resolve uniquely to readable RGB images. Missing/corrupt images raise errors. Relative lookup can search configured roots, but ambiguous basenames fail. Evidence precomputation writes absolute resolved path keys; moving a dataset requires rebuilding or explicitly migrating those keys. A cache entry has an `evidence_string`, a finite `reliability`, and optional OCR/tag metadata. Entries with extraction errors are rejected by training. The fallback reliability formula for naturally empty evidence is defined in the paper; missing cache files are not treated as successful extraction.

`precompute_evidence.py --dataset train` covers configured training/evaluation image directories for the selected benchmark. Use `--dataset siuo` separately for SIUO. Disabling OCR/tags explicitly with CLI flags is an evidence-ablation choice and must be disclosed when reporting a run.

## SafeEraser

The loader accepts a JSON list of grouped records with the following schema:

- `image_path` (or `image_id`): source image reference.
- `unsafe_pairs`: dictionaries with `question`, original `answer` (also supported: `Answer`, `response`, `model_response`), and, when available, the original benign `sd_response`/`SDResponse`/`safe_response` for the PD image.
- `SDImage_path`: harmless prompt-decouple image reference.
- `UnharmPair_image1/2` and `UnharmPair_text1/2`: retain QA dictionaries with `Question`/`question`, and `Prediction`/`Answer`/`answer`.
- Optional `category` and `keyword`.

Use the original provided benign responses. A required training domain with insufficient samples fails before training; check the dataset schema and configured paths.

A seed-dependent SHA256 ordering of canonical source image groups selects the heldout fraction (default 20%). All harmful questions of one source image belong to the same forget split. Heldout source images have `forget_test` and `retain_test` examples; their associated PD records are excluded from training. `forget`, `retain`, and `pd` are training domains. The training sampler excludes heldout rows even though the dataset object retains them for explicit evaluation. Stable sample IDs derive from image reference, question and split.

Primary evaluation reloads the **same training source JSON** and preserves these split labels: efficacy is ASR on `forget`, generality is ASR on `forget_test`. There is no post-hoc split of generated responses. Retain utility uses available retain and retain_test responses and is identified by counts in the JSON. PD SARR is evaluated on the available PD branch. Keep split definitions consistent when comparing runs.

`data.val_json` defaults to empty. It may point at the same JSON for explicit holdout inspection or at an independent validation source; an independent validation file is checked for image overlap. SIUO is constructed only by evaluation entry points.

Missing metric prerequisites yield `null` for that metric. For example, no heldout forget rows means no Generality-ASR; no retain references or PD rows means no composite Utility. On a nonempty requested subset, invalid images/responses/references stop scoring rather than shrinking denominators.

## VLGuard

Download the official training/test records and image directories. The schema uses image references, a Boolean `safe` marker and `instr-resp` instruction/response records; the loader also recognizes the field variants shown in its implementation. Training maps unsafe examples to refusal targets and safe examples to retain targets; PD is disabled.

Evaluate all three explicit test subsets: `unsafes` (ASR), `safe_unsafes` (ASR), and `safe_safes` (FRR). Empty required subsets fail. The evaluator uses GPT judgments for those metrics, following the EviGuard paper's primary-judge definition. It does not substitute the separate original VLGuard keyword protocol.

## SIUO

SIUO is never used for training or fitting the risk basis. Generation-format records require an image reference, a question, a reference answer and a safety warning for scoring. Aliases recognized by the loader include `prompt`/`instruction`, `reference`/`answer`, and `warning`. Use the official dataset annotations.

SR is the fraction judged safe; ASR is 100 minus SR. ER counts both Very Effective and Partially Effective judgments. S&E is the per-sample intersection of safe and effective; it is not the product of aggregate rates. Outputs include response-level judgment fields.

## Judge and response caches

API calls use a pinned judge model and temperature zero. Parse exact safe/unsafe, yes/no, True/False and effectiveness labels, or a finite score in [0,1] when that prompt requests a utility score. API errors and unparseable outputs abort; there is no regex, ROUGE or token-overlap substitution for GPT scores.

Judge cache keys include the protocol and complete prompt. Generation reuse requires a complete checkpoint identity, generation settings and a matching sample signature including question, image bytes, evidence and reference fields. Legacy caches cannot supply results for new checkpoints. Raw responses and valid judge answers remain in new run output/cache files. Credentials are not written into resolved checkpoint configs.

SafeEraser generation defaults in the example configs use three trials; forget safety/refusal rates count all trials while retain/PD utility uses the first response. VLGuard and SIUO use one generated response per example. These operational details and any subset limits must be retained with a reported run.

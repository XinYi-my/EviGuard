# Maintained source release - 2026-11-11

## Provenance and evidence boundary

The primary source is the preserved submitted `EviGuard_code.zip`; the later `EviGuard.zip` is a comparison source. The final paper (DOI 10.1145/3767308.3834945) specifies the intended method. Exact input hashes and the pre-existing GitHub base commit are in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Original ZIPs and the PDF remain outside the public repository. Documents inside those archives were treated as source material, not instructions authorizing actions.

This version is newly maintained code. It is not asserted to be the exact revision that produced the published tables. No historical checkpoint, log, complete final configuration, prediction file or judgment file is available. Public configs, regression tests, CI and the fixes below were added for this release. They must not be described as recovered historical experimental evidence.

## Method correspondence and changes

| Paper component | Public implementation | Correction / release choice |
| --- | --- | --- |
| Evidence reliability, Eq. 8 | `src/modules/evidence.py` | Preserves dual/single-source prior formula. Offline extraction errors are recorded explicitly; missing required cache entries fail. Equal basenames never imply equal image evidence. |
| Implicit cue, Eq. 9 | `src/modules/rsf.py` | Replaces batch-centered norm with learnable sample-local a_imp and b_imp. Batch composition no longer defines the implicit cue. |
| Gate and risk, Eqs. 10-13 | `src/modules/rsf.py` | Zero-initialized output gate starts at 0.5; raw risk and reliability-modulated risk retain distinct roles. Nonfinite values fail. |
| Two-pass connector modulation, Eq. 14 | `src/models/eviguard_model.py` | Pass 1 is detached evidence; Pass 2 recomputes a clean connector tensor with gradients and adds trainable modulation. No detached complete tensor replaces the connector path. |
| Weighted SVD, Eqs. 15-16 | `RiskSubspace.fit` | Weighted centroid and sqrt(weight) centered SVD; weights are calibrated risk times reliability. Invalid/undersized/degenerate fits cannot become an arbitrary identity basis. |
| Suppression and preservation, Eqs. 18-20 | `src/modules/clsu.py` | Pool visual tokens first; use the same risk centroid in projections and residual energy. Squared L2 norms sum feature coordinates and average examples. Retain losses use a frozen reference. |
| Confidence, Eq. 21 | `CLSULoss.confidence_reg_loss` | Sum forget and retain expectations rather than averaging the two domain terms. |
| Feedback, Eq. 22 | `src/training/trainer.py` | Score every available forget image, select residual energy > threshold, refit both mean and basis between intervals. A too-small hard set leaves the prior valid basis with an explicit warning. |
| Parameter scope | backbone loaders | LoRA targets complete language-model paths; no LoRA injection into the frozen vision encoder. Connector modules are excluded from bitsandbytes conversion so they remain trainable floating-point tensors. |
| Checkpoint lifecycle | `src/models/checkpoint.py` | Save/load exact trainable tensors, connector, RSF, basis/centroid and resolved config with hashes. Evaluation preserves RSF intervention. Adapter-only legacy folders are rejected. |

The submitted Stage 1 computed branch losses but never assigned the final `loss`. The maintained trainer forms the response/PD/confidence objective and performs actual accumulated backward/optimizer updates. It uses AdamW, with explicit optimizer-update step counts, and saves complete state. Resume restores numerical state but restarts the data iterator; it is not exact trajectory replay.

Modern LLaVA processor expansion is used for both the full sample and prompt prefix. The loss mask excludes all expanded visual/prompt tokens, preserves actual EOS tokens when EOS is also used for padding, and rejects entirely truncated answers. InternVL patch counts map evidence/modulation to examples explicitly; preprocessing failures cannot turn images into text-only samples.

## Choices that require new experiments

The paper does not specify every executable detail of the response-weight mapping. This release uses `risk_amplify * clamp(risk_score, epsilon, 1)^gamma` on forget CE, ordinary retain CE, and separate PD CE, without normalizing weights by the current microbatch mean. Historical mean normalization made every singleton risk weight equal to one. This mapping is a documented release choice, not a recovered final hyperparameter rule.

Geometry losses now follow pooled squared-L2 formulas. Historical code averaged feature/token coordinates and did not consistently center projections. Consequently copied loss coefficients do not preserve historical loss scales. Example coefficients require tuning and independent validation.

SafeEraser holdout membership is now deterministic by source image group and seed. Associated retain and PD samples from heldout source groups are not used for training. This is not a recovered historical split. Stage and warmup budgets count optimizer updates, rather than ambiguous microbatch loop iterations. Batch-size examples ensure requested training domains cannot disappear through rounding.

The release pins the GPT judge model name to `gpt-4o-2024-11-20`, uses explicit prompts and strict labels/scores, and records its protocol. Original API caches/prompts could not be verified against the unavailable server. These scorer repairs cannot establish equivalence with the published evaluation.

## Excluded historical orchestration

The submitted snapshot includes baseline/ablation/suite runners. They were not carried into the public executable workflow:

- Some PO implementations reassigned refusal tokens onto positions from the original answer tokenization.
- The VLGuard NPO implementation used a CE-derived sigmoid without the reference likelihood required by NPO.
- SIUO orchestration called an incompatible trainer signature and had fallback paths that changed the trained method.
- Evaluation in those paths restored adapters without all connector/RSF state; some failures changed the method to vanilla or substituted heuristics.
- Ablation wrappers depended on the same lifecycle and configuration assumptions.

Those runners require separate method-specific reconstruction and validation. The release does not rename them as faithful implementations, silently repair them into a different baseline, or claim all published comparisons are included. The original local archives remain available to the author for future auditing.

No paper numerical table, figure value, training curve or result JSON was synthesized. No MMLU/specificity score is emitted without an implemented benchmark.

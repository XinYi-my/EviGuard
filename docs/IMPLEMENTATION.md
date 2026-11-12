# Implementation notes

## Components

| Component | Implementation | Behavior |
| --- | --- | --- |
| Evidence reliability | `src/modules/evidence.py` | Combines OCR confidence and scene-tag confidence using dual/single-source priors. Cached evidence is keyed by image identity. |
| Risk-Structured Fusion | `src/modules/rsf.py` | Projects explicit evidence and pooled connector activations into a bottleneck. A learned confidence gate combines them; a raw risk head and evidence reliability produce the calibrated score. |
| Connector modulation | `src/models/eviguard_model.py` | Pass 1 provides detached evidence. Pass 2 computes connector activations with gradients and adds per-example risk-scaled modulation. InternVL patch groups retain explicit sample correspondence. |
| Risk basis | `RiskSubspace.fit` | Computes the risk-weighted centroid and the top-k right singular vectors of the sqrt-weighted, centered activation matrix. |
| CLSU losses | `src/modules/clsu.py` | Pools visual tokens, suppresses risk projections on forget examples, and preserves retain-side orthogonal components and cosine alignment against a frozen reference. |
| Feedback | `src/training/trainer.py` | Scores forget images by centered residual risk energy and refits the mean and basis between Stage 2 intervals. |
| Checkpoints | `src/models/checkpoint.py` | Saves trainable backbone/LoRA tensors, connector, RSF, subspace, resolved configuration and component hashes. Optional training state includes optimizer, scheduler, scaler and RNG state. |

## Training objective

Forget response cross-entropy is weighted by `risk_amplify * clamp(risk_score, epsilon, 1)^gamma`. Retain response cross-entropy uses unit weights; prompt-decouple CE has its own coefficient. Risk weights are not normalized by the current microbatch mean.

CLSU uses one pooled connector vector per image. Suppression and orthogonal preservation use squared L2 norms, summed over feature coordinates and averaged over examples. All risk projections and residual-energy scores share the fitted risk centroid. The confidence objective sums the forget and retain expectations.

The vision encoder and base language-model weights remain frozen. LoRA targets language-model modules, while the connector remains a trainable floating-point module. Evidence extractors and the reference model are frozen.

## Steps and feedback

Stage budgets and warmup count optimizer updates. Each update consumes the configured number of gradient-accumulation microbatches. Stage 2 retains its full update budget when divided into feedback intervals.

Initial subspace fitting requires enough valid positive-weight samples and sufficient numerical rank. Feedback selects examples whose residual risk energy exceeds the configured threshold. If the hard set cannot support the requested rank, training keeps the previous valid basis and logs the condition.

## Input preparation and checkpoint loading

LLaVA visual-token expansion is included in both sample preparation and prompt masking. Prompt and visual tokens are excluded from response loss. Padding is masked using the attention mask, and entirely truncated answer targets raise an error. InternVL preprocessing preserves patch-to-sample correspondence and reports visual preprocessing errors.

Checkpoint loading verifies component hashes and tensor names/shapes before restoring state. Generation applies the restored RSF intervention. Adapter-only directories do not contain all required EviGuard components.

Training resume restores numerical state at an optimizer boundary and restarts the data iterator. Use the same model and training configuration when resuming.

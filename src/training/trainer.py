"""Maintained training loop. All configured steps are optimizer updates.

Derived from the submitted trainer; fixes are documented in docs/RELEASE_NOTES.md.
A resume restores numerical state but restarts the data iterator (not exact replay).
"""
import json
import logging
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.modules.clsu import CLSULoss, RiskSubspace
from src.utils.helpers import MetricTracker, redact_config

logger = logging.getLogger("eviguard.trainer")


class EviGuardTrainer:
    def __init__(self, eviguard_model, train_loader, val_loader=None, cfg=None):
        self.model, self.train_loader, self.val_loader, self.cfg = eviguard_model, train_loader, val_loader, cfg
        self.tcfg, self.lcfg, self.scfg = cfg["training"], cfg["loss"], cfg["subspace"]
        t = self.tcfg
        if t["total_steps"] != t["stage1_steps"] + t["stage2_steps"]:
            raise ValueError("total_steps must equal stage1_steps + stage2_steps")
        if min(t["stage1_steps"], t["stage2_steps"], t["gradient_accumulation_steps"]) < 1:
            raise ValueError("Stage budgets and gradient accumulation must be positive")
        if not 1 <= t["num_feedback_iterations"] <= t["stage2_steps"]:
            raise ValueError("Invalid num_feedback_iterations")
        if len(train_loader) == 0:
            raise ValueError("Training loader is empty")
        if self.model.ref_model is None:
            raise ValueError("CLSU requires a frozen reference model theta_0")
        self.device = next(self.model.model.parameters()).device
        self.use_pd = self.lcfg.get("use_pd", self.lcfg["lambda_pd"] > 0)
        self.criterion = CLSULoss(**{k: self.lcfg[k] for k in (
            "lambda_pd", "lambda_s", "lambda_orth", "lambda_cm", "lambda_conf", "gamma", "epsilon", "risk_amplify")})
        self.risk_subspace = RiskSubspace(self.scfg["rank_k"], device=str(self.device),
            min_fit_samples=self.scfg.get("min_fit_samples", 2))
        self.optimizer = AdamW(self.model.get_trainable_params(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, t["warmup_steps"], t["total_steps"])
        precision = t["mixed_precision"]
        self.use_amp = self.device.type == "cuda" and precision in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and precision == "fp16")
        self.grad_accum = t["gradient_accumulation_steps"]
        self.optimizer_steps = 0
        self.metrics = MetricTracker(cfg["paths"]["log_dir"])
        self.checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_ends = {t["stage1_steps"] + (i * t["stage2_steps"]) // t["num_feedback_iterations"]
                              for i in range(1, t["num_feedback_iterations"])}

    def _autocast(self):
        return torch.autocast("cuda", dtype=self.amp_dtype) if self.use_amp else nullcontext()

    @staticmethod
    def _next(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _prepare(self, batch):
        from src.data.datasets import prepare_model_inputs
        return prepare_model_inputs(batch, self.model.processor, self.model.tokenizer,
            self.tcfg["max_seq_length"], self.device, arch=self.model.arch, model=self.model.model)

    def _kwargs(self, inp, mask=None):
        mask = torch.ones(len(inp["input_ids"]), dtype=torch.bool, device=self.device) if mask is None else mask
        kw = {k: inp[k][mask] for k in ("input_ids", "attention_mask")}
        pv = inp.get("pixel_values")
        if self.model.arch == "internvl":
            from src.models.internvl_utils import filter_pixels_internvl
            pv, flags, npl = filter_pixels_internvl(inp, mask)
            kw.update(pixel_values=pv, image_flags=flags, num_patches_list=npl)
        else:
            kw["pixel_values"] = pv[mask] if pv is not None else None
        return kw

    def _pool(self, H, inp):
        if H is None:
            raise RuntimeError("Connector hook did not produce activations")
        if self.model.arch != "internvl":
            return H.mean(1) if H.ndim == 3 else H
        counts = inp.get("num_patches_list", [])
        if sum(counts) != H.shape[0] or any(n <= 0 for n in counts):
            raise ValueError("InternVL patch/sample correspondence is invalid")
        return torch.stack([part.float().mean((0, 1)) for part in H.split(counts, dim=0)])

    def _batch_loss(self, batch, stage):
        inp = self._prepare(batch)
        fm, rm, pm = (inp[k] for k in ("forget_mask", "retain_mask", "pd_mask"))
        args = {}
        po = fm | rm
        if po.any():
            kw = self._kwargs(inp, po)
            ev = [s for s, keep in zip(inp["evidence_strings"], po.tolist()) if keep]
            out = self.model.forward_with_rsf(ev, inp["reliabilities"][po], **kw)
            args.update(po_logits=out["logits"], po_labels=inp["labels"][po], po_mask=inp["attention_mask"][po],
                        po_risk_scores=out["risk_scores"], po_forget_mask=fm[po], po_retain_mask=rm[po],
                        full_risk_scores=out["risk_scores"], full_forget_mask=fm[po], full_retain_mask=rm[po])
            if stage == 2 and out["H_original"] is not None:
                h = self._pool(out["H_original"], kw)
                args["forget_H"] = h[fm[po]] if fm[po].any() else None
                if rm[po].any():
                    ref = self.model.forward_reference(**kw)
                    hr = self._pool(ref["connector_activation"], kw)
                    args.update(retain_H=h[rm[po]], retain_H_ref=hr[rm[po]].detach())
                args["risk_subspace"] = self.risk_subspace
        if self.use_pd and pm.any():
            kw = self._kwargs(inp, pm)
            ev = [s for s, keep in zip(inp["evidence_strings"], pm.tolist()) if keep]
            out = self.model.forward_with_rsf(ev, inp["reliabilities"][pm], **kw)
            args.update(pd_logits=out["logits"], pd_labels=inp["labels"][pm], pd_mask=inp["attention_mask"][pm])
        if not args:
            raise ValueError("Batch has no active training targets")
        return self.criterion(**args)

    def _optimizer_step(self, iterator, stage):
        self.model.model.train()
        self.model.rsf.train()
        self.model.ref_model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        means = {}
        for _ in range(self.grad_accum):
            batch, iterator = self._next(iterator, self.train_loader)
            with self._autocast():
                loss, info = self._batch_loss(batch, stage)
            if not torch.isfinite(loss) or not loss.requires_grad:
                raise FloatingPointError("Non-finite loss or missing training gradient")
            self.scaler.scale(loss / self.grad_accum).backward()
            for key, value in info.items():
                means[key] = means.get(key, 0.) + float(value) / self.grad_accum
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.get_trainable_params(), self.tcfg["max_grad_norm"], error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer_steps += 1
        self._iterator = iterator
        means["gradient_norm"] = float(norm)
        return means

    @torch.no_grad()
    def _collect(self, hard=False):
        self.model.model.eval()
        self.model.rsf.eval()
        acts, weights = [], []
        dataset = getattr(self.train_loader, "dataset", None)
        collection_loader = self.train_loader
        if dataset is not None and hasattr(dataset, "get_split_indices"):
            from torch.utils.data import DataLoader, Subset
            from src.data.datasets import eviguard_collate_fn
            ids = dataset.get_split_indices("forget", modality="image")
            collection_loader = DataLoader(Subset(dataset, ids), batch_size=self.tcfg["batch_size"], shuffle=False, collate_fn=eviguard_collate_fn)
        for batch in collection_loader:
            inp = self._prepare(batch)
            fm = inp["forget_mask"]
            if not fm.any():
                continue
            kw = self._kwargs(inp, fm)
            ev = [s for s, keep in zip(inp["evidence_strings"], fm.tolist()) if keep]
            rel = inp["reliabilities"][fm]
            with self._autocast():
                out = self.model.forward_with_rsf(ev, rel, **kw)
            h = self._pool(out["H_original"], kw)
            keep = self.risk_subspace.risk_energy(h) > self.lcfg["hard_sample_threshold"] if hard else torch.ones(len(h), dtype=torch.bool, device=h.device)
            if keep.any():
                acts.append(h[keep].cpu())
                weights.append((out["risk_scores"][keep] * rel[keep]).cpu())
            if not hard and sum(len(a) for a in acts) >= self.scfg["max_collection_samples"]:
                break
        if not acts:
            if hard:
                return None, None
            raise ValueError("No forget activations available for risk-subspace fitting")
        # Feedback scores all forget examples; collection cap applies only to initial fit.
        h, w = torch.cat(acts), torch.cat(weights)
        if not hard:
            h, w = h[:self.scfg["max_collection_samples"]], w[:self.scfg["max_collection_samples"]]
        return h, w

    def _fit_subspace(self, hard=False):
        h, w = self._collect(hard)
        if h is None or (hard and len(h) < max(self.scfg["rank_k"] + 1, self.risk_subspace.min_fit_samples)):
            logger.warning("Insufficient residual hard examples; retain the previous fitted basis")
            return
        diagnostic = self.risk_subspace.fit(h, w)
        logger.info("Subspace fit: %s", diagnostic)

    def _save(self, directory):
        from src.models.checkpoint import save_checkpoint
        state = {"optimizer":self.optimizer.state_dict(), "scheduler":self.scheduler.state_dict(),
                 "scaler":self.scaler.state_dict(), "optimizer_steps":self.optimizer_steps,
                 "torch_rng":torch.get_rng_state(), "python_rng":random.getstate(),
                 "numpy_rng":np.random.get_state(), "metrics":self.metrics.history}
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        save_checkpoint(self.model, directory, subspace=self.risk_subspace, resolved_config=self.cfg,
            manifest={"release_status":"maintained_source_release", "historical_paper_checkpoint":False,
                      "optimizer_steps":self.optimizer_steps, "resume_data_order":"iterator_restarted"}, training_state=state)

    def load_resume_checkpoint(self, checkpoint_path=None):
        from src.models.checkpoint import restore_checkpoint
        if checkpoint_path is None:
            candidates = sorted(self.checkpoint_dir.glob("step-*"), key=lambda p:int(p.name.split("-")[-1]))
            if not candidates:
                raise FileNotFoundError("No maintained-release checkpoint available for resume")
            checkpoint_path = candidates[-1]
        restored = restore_checkpoint(self.model, checkpoint_path, subspace=self.risk_subspace)
        for section in ("training", "loss", "subspace", "rsf", "lora"):
            if restored["config"].get(section) != self.cfg.get(section):
                raise ValueError(f"Resume configuration differs in {section}; use the same resolved training settings")
        state = restored["training_state"]
        if state is None:
            raise ValueError("Checkpoint has no optimizer/training state for resume")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.optimizer_steps = state["optimizer_steps"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.metrics.history = state["metrics"]
        logger.warning("Resuming numerical training state; data iterator restarts. This is not bitwise replay.")

    def train(self):
        from src.utils.helpers import set_seed
        if not self.optimizer_steps:
            set_seed(self.tcfg["seed"])
        iterator = iter(self.train_loader)
        for step in range(self.optimizer_steps + 1, self.tcfg["total_steps"] + 1):
            stage = 1 if step <= self.tcfg["stage1_steps"] else 2
            if stage == 2 and not self.risk_subspace.is_fitted:
                self._fit_subspace()
            info = self._optimizer_step(iterator, stage)
            iterator = self._iterator
            self.metrics.update(step=step, stage=stage, **info)
            if step in self.feedback_ends:
                self._fit_subspace(hard=True)
            if step % self.tcfg["log_every_steps"] == 0:
                logger.info("step=%d stage=%d loss=%.6f", step, stage, info["total_loss"])
            if step % self.tcfg["save_every_steps"] == 0:
                self._save(self.checkpoint_dir / f"step-{step}")
        self._save(self.checkpoint_dir / "final")
        self.metrics.save("training_metrics.json")
        return self.metrics

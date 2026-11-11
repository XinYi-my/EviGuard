"""Risk-weighted response losses and the paper's pooled, centered CLSU geometry."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.rsf import _require_finite, project_to_subspace, compute_risk_energy
from src.utils.helpers import load_common_config

logger = logging.getLogger("eviguard.clsu")
_COMMON_CFG = load_common_config()


def _pool_activations(activations):
    """Return one float32 connector vector per example, as in Eqs. (15)-(22)."""
    if activations.ndim not in (1, 2, 3) or activations.numel() == 0:
        raise ValueError("Connector activations must be nonempty [d], [B,d], or [B,T,d]")
    _require_finite(activations, "connector activations")
    pooled = activations.float().mean(1) if activations.ndim == 3 else activations.float()
    _require_finite(pooled, "pooled connector activations")
    return pooled


class RiskSubspace:
    def __init__(self, rank_k, device=None, min_fit_samples=None):
        if device is None:
            device = _COMMON_CFG["subspace"]["default_device"]
        if min_fit_samples is None:
            min_fit_samples = _COMMON_CFG["subspace"]["min_fit_samples"]
        if not isinstance(rank_k, int) or isinstance(rank_k, bool) or rank_k < 1:
            raise ValueError("rank_k must be a positive integer")
        self.rank_k = rank_k
        self.device = device
        self.min_fit_samples = int(min_fit_samples)
        self.basis = self.centroid = self.eigenvalues = self.explained_var = None
        self.is_fitted = False

    @torch.no_grad()
    def fit(self, activations, risk_weights, min_samples=None):
        """Fit Eqs. (15)-(16), rejecting unsupported or degenerate estimates.

        A failed refit leaves the previous valid estimate intact. A numerical
        nullspace or an arbitrary identity basis is never treated as fitted.
        """
        if min_samples is None:
            min_samples = self.min_fit_samples
        if activations.ndim != 2 or activations.numel() == 0:
            raise ValueError("SVD activations must be nonempty mean-pooled [N,d]")
        N, d = activations.shape
        if int(min_samples) < 1 or N < min_samples:
            raise ValueError(f"Insufficient subspace samples: {N}; require at least {min_samples}")
        if risk_weights.ndim != 1 or risk_weights.shape[0] != N:
            raise ValueError("risk_weights must have shape [N]")
        _require_finite(activations, "SVD activations")
        _require_finite(risk_weights, "SVD risk weights")
        acts = activations.detach().to(device=self.device, dtype=torch.float32)
        w = risk_weights.detach().to(device=self.device, dtype=torch.float32)
        _require_finite(acts, "float32 SVD activations")
        _require_finite(w, "float32 SVD risk weights")
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("SVD weights must be nonnegative with positive total weight")
        if self.rank_k > d or (w > 0).sum().item() <= self.rank_k:
            raise ValueError(f"Rank {self.rank_k} requires d >= k and at least k+1 positive-weight samples")
        weight_sum = w.sum()
        _require_finite(weight_sum, "total SVD risk weight")
        # Global weight normalization changes neither the centroid nor the basis.
        w = w / weight_sum
        centroid = (w.unsqueeze(1) * acts).sum(0)
        weighted = torch.sqrt(w).unsqueeze(1) * (acts - centroid)
        _require_finite(weighted, "weighted centered SVD activations")
        _, S, Vt = torch.linalg.svd(weighted, full_matrices=False)
        _require_finite(S, "SVD singular values")
        _require_finite(Vt, "SVD right singular vectors")
        tolerance = max(weighted.shape) * torch.finfo(S.dtype).eps * S[0]
        effective_rank = int((S > tolerance).sum().item())
        if effective_rank < self.rank_k:
            raise ValueError(f"Insufficient numerical rank: {effective_rank}; requested {self.rank_k}")
        k = self.rank_k
        eigenvalues = S.square()
        _require_finite(eigenvalues, "SVD eigenvalues")
        total = eigenvalues.sum()
        _require_finite(total, "total SVD variance")
        explained_var = (eigenvalues[:k].cumsum(0) / total).cpu().tolist()
        self.basis = Vt[:k].T.contiguous()
        self.centroid = centroid
        self.eigenvalues = eigenvalues[:k].cpu().numpy()
        self.explained_var = explained_var
        self.is_fitted = True
        ev = float(self.explained_var[-1]) if self.explained_var else 0
        logger.info("Risk subspace: k=%d, explained_var=%.3f, N=%d", k, ev, N)
        return {"status": "fitted", "n_samples": N, "rank_k": k, "explained_var_topk": ev}

    def _center(self, h):
        if not self.is_fitted or self.basis is None or self.centroid is None:
            raise RuntimeError("Risk subspace must be fitted before projection")
        pooled = _pool_activations(h)
        if self.centroid.shape != (pooled.shape[-1],):
            raise ValueError("Subspace centroid and activations have incompatible dimensions")
        centroid = self.centroid.to(device=pooled.device, dtype=torch.float32)
        _require_finite(centroid, "subspace centroid")
        return pooled - centroid

    def project_vector(self, vector):
        """Project a displacement linearly, without subtracting the centroid."""
        if not self.is_fitted or self.basis is None:
            raise RuntimeError("Risk subspace must be fitted before projection")
        return project_to_subspace(vector.float(), self.basis)

    def project_risk(self, h):
        return self.project_vector(self._center(h))

    def project_ortho(self, h):
        centered = self._center(h)
        return centered - self.project_vector(centered)

    def risk_energy(self, h):
        """Residual risk ratio in Eq. (22), in the fitted centered coordinates."""
        return compute_risk_energy(self._center(h), self.basis)

    @staticmethod
    def _validate_geometry(basis, centroid, rank_k):
        if (not isinstance(rank_k, int) or isinstance(rank_k, bool) or rank_k < 1
                or not isinstance(basis, torch.Tensor) or basis.ndim != 2
                or basis.shape[1] != rank_k or basis.shape[0] < rank_k):
            raise ValueError("Subspace basis must have shape [d,rank_k] with 1 <= rank_k <= d")
        if not isinstance(centroid, torch.Tensor) or centroid.shape != (basis.shape[0],):
            raise ValueError("Subspace centroid must have shape [d]")
        _require_finite(basis, "loaded subspace basis")
        _require_finite(centroid, "loaded subspace centroid")
        b = basis.float()
        identity = torch.eye(rank_k, device=b.device)
        if not torch.allclose(b.T @ b, identity, atol=1e-4, rtol=1e-4):
            raise ValueError("Subspace basis columns must be orthonormal")

    def save(self, path):
        """Save tensor-only format v1, readable with weights_only=True."""
        if self.is_fitted:
            self._validate_geometry(self.basis, self.centroid, self.rank_k)
        torch.save({
            "format_version": 1,
            "basis": self.basis.detach().cpu() if self.basis is not None else None,
            "centroid": self.centroid.detach().cpu() if self.centroid is not None else None,
            "eigenvalues": torch.as_tensor(self.eigenvalues) if self.eigenvalues is not None else None,
            "explained_var": self.explained_var, "rank_k": self.rank_k,
            "min_fit_samples": self.min_fit_samples, "is_fitted": self.is_fitted,
        }, path)

    def load(self, path):
        """Validate the whole payload before replacing the current fitted state."""
        s = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(s, dict) or s.get("format_version") != 1:
            raise ValueError("Unsupported standalone risk-subspace format; expected version 1")
        rank, minimum, fitted = s["rank_k"], s["min_fit_samples"], s["is_fitted"]
        if (not isinstance(rank, int) or isinstance(rank, bool) or rank < 1
                or not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1
                or not isinstance(fitted, bool)):
            raise ValueError("Invalid risk-subspace configuration")
        basis, centroid = s["basis"], s["centroid"]
        eigenvalues, explained_var = s["eigenvalues"], s["explained_var"]
        if fitted:
            self._validate_geometry(basis, centroid, rank)
            if not isinstance(eigenvalues, torch.Tensor) or eigenvalues.shape != (rank,):
                raise ValueError("Subspace eigenvalues must have shape [rank_k]")
            _require_finite(eigenvalues, "loaded subspace eigenvalues")
            if (eigenvalues <= 0).any():
                raise ValueError("Subspace eigenvalues must be positive")
            explained = torch.as_tensor(explained_var)
            if explained.shape != (rank,):
                raise ValueError("Subspace explained variance must have shape [rank_k]")
            _require_finite(explained, "loaded subspace explained variance")
        elif any(item is not None for item in (basis, centroid, eigenvalues, explained_var)):
            raise ValueError("Unfitted risk subspace must not contain fitted geometry")
        self.basis, self.centroid = basis, centroid
        self.eigenvalues = eigenvalues.numpy() if eigenvalues is not None else None
        self.explained_var, self.rank_k = explained_var, rank
        self.min_fit_samples, self.is_fitted = minimum, fitted


class CLSULoss(nn.Module):
    def __init__(self, lambda_pd, lambda_s, lambda_orth,
                 lambda_cm, lambda_conf, gamma, epsilon,
                 risk_amplify):
        super().__init__()
        self.lambda_pd = lambda_pd
        self.lambda_s = lambda_s
        self.lambda_orth = lambda_orth
        self.lambda_cm = lambda_cm
        self.lambda_conf = lambda_conf
        self.gamma = gamma
        self.epsilon = epsilon
        self.risk_amplify = risk_amplify

    def risk_weight(self, rs):
        """Release choice: clamp(rs, epsilon, 1)^gamma, without batch normalization.

        The paper specifies risk-weighted response CE but not this mapping.
        PO multiplies this by the inherited risk_amplify once. In particular,
        score dependence survives the recommended microbatch size of one.
        """
        _require_finite(rs, "response risk scores")
        weights = rs.float().clamp(self.epsilon, 1.0).pow(self.gamma)
        _require_finite(weights, "response risk weights")
        return weights

    def _dev(self, *ts):
        for t in ts:
            if isinstance(t, torch.Tensor): return t.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _safe(self, x, name, info):
        if not isinstance(x, torch.Tensor): x = torch.tensor(float(x))
        if x.numel() != 1:
            raise ValueError(f"{name} must be a scalar loss")
        _require_finite(x, f"{name} loss")
        info[f"{name}_nf"] = 0
        return x

    def _san(self, logits):
        logits = logits.float()
        _require_finite(logits, "response logits")
        return logits

    def _per_sample_ce(self, logits, labels, mask):
        logits = self._san(logits)
        if logits.ndim != 3:
            raise ValueError("Response logits must have shape [B,T,V]")
        B, T, V = logits.shape
        if B == 0 or T < 2 or V == 0 or labels.shape != (B, T) or mask.shape != (B, T):
            raise ValueError("Response labels/mask must match nonempty logits with at least two tokens")
        _require_finite(mask, "response attention mask")
        sl = logits[:, :-1].contiguous()
        slb = labels.long()[:, 1:].contiguous()
        sm = (mask[:, 1:].float() > 0).float()
        tl = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)(
            sl.view(-1, V), slb.view(-1)).view(B, T - 1)
        _require_finite(tl, "response token cross entropy")
        v = sm * (slb != -100).float()
        return (tl * v).sum(1) / v.sum(1).clamp(1)

    def po_risk_loss(self, logits, labels, mask, risk_scores, forget_mask, retain_mask):
        ce = self._per_sample_ce(logits, labels, mask)
        w = torch.ones_like(ce)
        if risk_scores is not None and forget_mask.any():
            rw = self.risk_weight(risk_scores)
            w = torch.where(forget_mask, rw * self.risk_amplify, w)
        loss = (w * ce).mean()
        info = {
            "po_loss": loss.detach().item(),
            "forget_ce": ce[forget_mask].mean().item() if forget_mask.any() else 0,
            "retain_ce": ce[retain_mask].mean().item() if retain_mask.any() else 0,
        }
        if risk_scores is not None and forget_mask.any():
            info["mean_risk_w"] = rw[forget_mask].mean().item()
        return loss, info

    def pd_loss(self, logits, labels, mask):
        l = self._per_sample_ce(logits, labels, mask).mean()
        return l, {"pd_loss": l.detach().item()}

    def directional_suppress_loss(self, H_forget, subspace):
        proj = subspace.project_risk(H_forget)
        l = proj.float().square().sum(-1).mean()
        _require_finite(l, "directional suppression loss")
        return l, {"dir_suppress": l.detach().item()}

    def orthogonal_preserve_loss(self, H_cur, H_ref, subspace):
        # Both terms use the same centroid; it cancels in their difference.
        ortho = subspace.project_ortho(H_cur) - subspace.project_ortho(H_ref.detach())
        l = ortho.float().square().sum(-1).mean()
        _require_finite(l, "orthogonal preservation loss")
        return l, {"orth_preserve": l.detach().item()}

    def cm_loss(self, hc, hr):
        hc = _pool_activations(hc)
        hr = _pool_activations(hr.detach())
        cos = F.cosine_similarity(hc, hr, dim=-1, eps=1e-8)
        l = (1 - cos).mean()
        _require_finite(l, "cosine consistency loss")
        return l, {"cm_loss": l.detach().item(), "mean_cos": cos.mean().item()}

    def confidence_reg_loss(self, risk_scores, forget_mask, retain_mask):
        _require_finite(risk_scores, "confidence risk scores")
        parts = []
        if forget_mask.any(): parts.append((1.0 - risk_scores[forget_mask]).mean())
        if retain_mask.any(): parts.append(risk_scores[retain_mask].mean())
        if not parts: return risk_scores.sum() * 0, {"conf_reg": 0.}
        # Eq. (21) sums the two distribution expectations.
        l = sum(parts)
        return l, {"conf_reg": l.detach().item()}

    def forward(self,
                po_logits=None, po_labels=None, po_mask=None,
                po_risk_scores=None, po_forget_mask=None, po_retain_mask=None,
                pd_logits=None, pd_labels=None, pd_mask=None,
                forget_H=None, forget_H_ref=None,
                retain_H=None, retain_H_ref=None,
                risk_subspace=None,
                full_risk_scores=None, full_forget_mask=None, full_retain_mask=None):

        dev = self._dev(po_logits, pd_logits, forget_H, retain_H)
        total = torch.zeros((), device=dev, dtype=torch.float32)
        info = {}

        if po_logits is not None and po_labels is not None:
            l, d = self.po_risk_loss(po_logits, po_labels, po_mask,
                                     po_risk_scores, po_forget_mask, po_retain_mask)
            total = total + self._safe(l, "po", info); info.update(d)

        if pd_logits is not None and pd_labels is not None:
            l, d = self.pd_loss(pd_logits, pd_labels, pd_mask)
            total = total + self.lambda_pd * self._safe(l, "pd", info); info.update(d)

        ss = risk_subspace
        if forget_H is not None and ss is not None and ss.is_fitted:
            l, d = self.directional_suppress_loss(forget_H, ss)
            total = total + self.lambda_s * self._safe(l, "dir_sup", info); info.update(d)

        if retain_H is not None and retain_H_ref is not None and ss is not None and ss.is_fitted:
            l, d = self.orthogonal_preserve_loss(retain_H, retain_H_ref, ss)
            total = total + self.lambda_orth * self._safe(l, "orth", info); info.update(d)

        if retain_H is not None and retain_H_ref is not None:
            hc = retain_H.mean(1) if retain_H.dim() == 3 else retain_H
            hr = retain_H_ref.mean(1) if retain_H_ref.dim() == 3 else retain_H_ref
            l, d = self.cm_loss(hc, hr)
            total = total + self.lambda_cm * self._safe(l, "cm", info); info.update(d)

        if full_risk_scores is not None and full_forget_mask is not None:
            l, d = self.confidence_reg_loss(full_risk_scores, full_forget_mask, full_retain_mask)
            total = total + self.lambda_conf * self._safe(l, "conf", info); info.update(d)

        total = self._safe(total, "total", info)
        info["total_loss"] = total.detach().item()
        return total, info

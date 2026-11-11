"""Risk-Structured Fusion, plus linear subspace projection utilities."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.helpers import load_common_config

logger = logging.getLogger("eviguard.rsf")
_COMMON_CFG = load_common_config()


def _require_finite(tensor, name):
    """Stop on numerical failures instead of silently changing the objective."""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite {name}")


class CredibleFusionGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.zeros_(self.gate_mlp[2].bias)

    def forward(self, c_explicit, c_implicit):
        _require_finite(c_explicit, "explicit confidence")
        _require_finite(c_implicit, "implicit confidence")
        dtype = self.gate_mlp[0].weight.dtype
        c_exp = c_explicit.to(dtype=dtype).clamp(0, 1)
        c_imp = c_implicit.to(dtype=dtype).clamp(0, 1)
        feat = torch.stack([c_exp, c_imp, (c_exp - c_imp).abs()], dim=-1)
        gate_logit = self.gate_mlp(feat)
        _require_finite(gate_logit, "fusion gate logit")
        return torch.sigmoid(gate_logit).squeeze(-1)


class RSFModule(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim, dropout, gate_hidden_dim=None):
        super().__init__()
        if gate_hidden_dim is None:
            gate_hidden_dim = _COMMON_CFG["rsf"]["gate_hidden_dim"]
        self.embed_dim, self.bottleneck_dim = embed_dim, bottleneck_dim
        self.W_exp = nn.Linear(embed_dim, bottleneck_dim)
        self.W_imp = nn.Linear(embed_dim, bottleneck_dim)
        self.cred_gate = CredibleFusionGate(gate_hidden_dim)
        self.z_dropout = nn.Dropout(dropout)
        self.w_r = nn.Linear(bottleneck_dim, 1)
        self.W_delta = nn.Linear(bottleneck_dim, embed_dim)
        self.risk_temperature = nn.Parameter(torch.tensor(_COMMON_CFG["rsf"]["risk_temperature_init"]))
        # Learnable sample-local implicit cue, paper Eq. (9).
        self.a_imp = nn.Parameter(torch.tensor(1.0))
        self.b_imp = nn.Parameter(torch.tensor(0.0))
        for m in (self.W_exp, self.W_imp):
            nn.init.xavier_uniform_(m.weight, gain=_COMMON_CFG["rsf"]["proj_init_gain"]); nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.w_r.weight, gain=_COMMON_CFG["rsf"]["proj_init_gain"])
        nn.init.constant_(self.w_r.bias, _COMMON_CFG["rsf"]["risk_head_bias"])
        nn.init.xavier_uniform_(self.W_delta.weight, gain=_COMMON_CFG["rsf"]["delta_init_gain"])
        nn.init.zeros_(self.W_delta.bias)
        logger.info("RSF v4: embed=%d bottle=%d params=%s",
                     embed_dim, bottleneck_dim,
                     f"{sum(p.numel() for p in self.parameters()):,}")

    def forward(self, H, evidence_embedding, reliability):
        if H.ndim != 3 or H.shape[0] == 0 or H.shape[1] == 0 or H.shape[2] != self.embed_dim:
            raise ValueError("RSF activations must have nonempty shape [B,T,embed_dim]")
        B, T_v, d = H.shape
        if evidence_embedding.shape != (B, d) or reliability.shape != (B,):
            raise ValueError("RSF evidence/reliability must have shapes [B,d] and [B]")
        _require_finite(H, "RSF connector activations")
        _require_finite(evidence_embedding, "RSF evidence embeddings")
        _require_finite(reliability, "RSF evidence reliability")
        in_dtype = self.W_imp.weight.dtype
        out_dtype = H.dtype

        H_in = H.to(dtype=in_dtype)
        ev_in = evidence_embedding.to(dtype=in_dtype)
        rel_in = reliability.to(dtype=in_dtype)

        z_exp = F.gelu(self.W_exp(ev_in))
        h_pool = H_in.mean(dim=1)
        z_imp = F.gelu(self.W_imp(h_pool))
        z_imp_norm = z_imp.norm(dim=-1)
        _require_finite(z_exp, "RSF explicit representation")
        _require_finite(z_imp_norm, "RSF implicit representation norm")
        implicit_logit = self.a_imp * z_imp_norm + self.b_imp
        _require_finite(implicit_logit, "RSF implicit confidence logit")
        c_imp = torch.sigmoid(implicit_logit)
        alpha = self.cred_gate(rel_in, c_imp).unsqueeze(-1)
        z = alpha * z_exp + (1.0 - alpha) * z_imp
        z = self.z_dropout(z)
        _require_finite(self.risk_temperature, "RSF risk temperature")
        temp = self.risk_temperature.clamp(0.1, 5.0)
        risk_logit = self.w_r(z) / temp
        _require_finite(risk_logit, "RSF risk head logit")
        r_raw = torch.sigmoid(risk_logit).squeeze(-1)
        risk_scores = (rel_in * r_raw).clamp(0.0, 1.0)
        delta = self.W_delta(z)
        delta_bc = delta.unsqueeze(1).expand_as(H_in)
        r_bc = r_raw.view(B, 1, 1).expand(B, T_v, 1)
        H_fused = H_in + r_bc * delta_bc
        H_fused = H_fused.to(dtype=out_dtype)
        for value, name in ((H_fused, "fused activations"), (risk_scores, "risk scores"),
                            (r_raw, "raw risk scores"), (delta, "modulation")):
            _require_finite(value, f"RSF {name}")
        return H_fused, risk_scores, r_raw, delta


def project_to_subspace(H, basis):
    """Linear projection of already-centered coordinates or displacement vectors."""
    if basis is None:
        raise ValueError("Projection requires a fitted basis")
    if basis.ndim != 2 or basis.shape[1] == 0 or H.shape[-1] != basis.shape[0]:
        raise ValueError("Projection basis must have compatible nonempty shape [d,k]")
    _require_finite(H, "projection activations")
    _require_finite(basis, "projection basis")
    B_mat = basis.to(H.device, dtype=torch.float32)
    projection = (H.float() @ B_mat @ B_mat.T).to(H.dtype)
    _require_finite(projection, "subspace projection")
    return projection

def directional_erase(H, basis, strength=1.0):
    return H - strength * project_to_subspace(H, basis)

def compute_risk_energy(H, basis):
    """Eq. (22) ratio; H must already use the fitted centered coordinates."""
    proj = project_to_subspace(H, basis).float()
    h32 = H.float()
    numerator, denominator = proj.square().sum(-1), h32.square().sum(-1) + 1e-8
    _require_finite(numerator, "risk energy numerator")
    _require_finite(denominator, "risk energy denominator")
    return numerator / denominator


class EvidenceEncoder:
    def __init__(self, tokenizer, max_tokens):
        self.tokenizer, self.max_tokens = tokenizer, max_tokens

    def encode(self, evidence_strings, embedding_layer, device):
        enc = self.tokenizer(evidence_strings, max_length=self.max_tokens,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            emb = embedding_layer(ids)
        m = mask.unsqueeze(-1).float()
        return (emb * m).sum(1) / m.sum(1).clamp(min=1)


class ConnectorHookWithGrad:
    def __init__(self):
        self._activation = None; self._handle = None
    def _hook_fn(self, module, inp, out):
        self._activation = out
    def register(self, module):
        self._handle = module.register_forward_hook(self._hook_fn)
    def get(self):  return self._activation
    def clear(self): self._activation = None
    def remove(self):
        if self._handle: self._handle.remove()


class ConnectorInterventionHook:
    def __init__(self):
        self._activation = None
        self._replacement = None
        self._handle = None

    def _hook_fn(self, module, inp, out):
        self._activation = out
        if self._replacement is not None:
            return self._replacement

    def register(self, module):
        self._handle = module.register_forward_hook(self._hook_fn)

    def set_replacement(self, tensor):
        self._replacement = tensor

    def clear_replacement(self):
        self._replacement = None

    def get(self): return self._activation
    def clear(self): self._activation = None; self._replacement = None
    def remove(self):
        if self._handle: self._handle.remove()

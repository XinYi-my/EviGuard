import json, logging
from pathlib import Path
from typing import Dict, List
import torch, numpy as np
logger = logging.getLogger("eviguard.analysis")

class RiskEnergyAnalyzer:
    def __init__(self, risk_subspace): self.sub = risk_subspace
    @torch.no_grad()
    def compare_before_after(self, acts_before, acts_after):
        eb = self.sub.risk_energy(acts_before).cpu().numpy()
        ea = self.sub.risk_energy(acts_after).cpu().numpy()
        return {"before_mean": float(np.mean(eb)), "after_mean": float(np.mean(ea)),
                "reduction": float(np.mean(eb) - np.mean(ea))}

class SubspaceDiagnostics:
    @staticmethod
    def singular_value_analysis(risk_subspace):
        if risk_subspace.eigenvalues is None: return {}
        ev = np.array(risk_subspace.eigenvalues); total = ev.sum()
        return {"eigenvalues": ev.tolist(),
                "cumulative_variance": (np.cumsum(ev) / (total + 1e-8)).tolist(),
                "top1_ratio": float(ev[0] / total) if total > 0 else 0,
                "top32_ratio": float(ev[:32].sum() / total) if total > 0 else 0}
    @staticmethod
    def subspace_stability(activations, risk_weights, rank_k=32, n_splits=5, seed=42):
        from src.modules.clsu import RiskSubspace
        rng = np.random.RandomState(seed); N = activations.shape[0]; bases = []
        for _ in range(n_splits):
            sub = RiskSubspace(rank_k=rank_k)
            idx = rng.permutation(N)[:N//2]
            sub.fit(activations[idx], risk_weights[idx])
            if sub.basis is not None: bases.append(sub.basis.cpu())
        if len(bases) < 2: return {"status": "insufficient"}
        sims = []
        for i in range(len(bases)):
            for j in range(i+1, len(bases)):
                sims.append(torch.linalg.svdvals(bases[i].T @ bases[j]).mean().item())
        return {"mean_similarity": float(np.mean(sims)), "std": float(np.std(sims))}
    @staticmethod
    def risk_score_separation(forget_scores, retain_scores):
        f = forget_scores.cpu().numpy(); r = retain_scores.cpu().numpy()
        return {"forget_mean": float(np.mean(f)), "retain_mean": float(np.mean(r)),
                "separation": float(np.mean(f) - np.mean(r))}

def run_full_analysis(risk_subspace, acts_before, acts_after, f_scores, r_scores, weights, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "svd": SubspaceDiagnostics.singular_value_analysis(risk_subspace),
        "stability": SubspaceDiagnostics.subspace_stability(acts_before, weights, risk_subspace.rank_k),
        "separation": SubspaceDiagnostics.risk_score_separation(f_scores, r_scores),
        "energy": RiskEnergyAnalyzer(risk_subspace).compare_before_after(acts_before, acts_after),
    }
    with open(output_dir / "full_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"分析结果保存到 {output_dir}/full_analysis.json")
    return results

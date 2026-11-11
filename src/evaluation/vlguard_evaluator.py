"""VLGuard evaluation with explicit GPT-4o judgments and complete generations.

These GPT judgments are the maintained release protocol. They do not claim to
reproduce VLGuard's separate refusal-keyword benchmark protocol.
"""
from pathlib import Path

from src.utils.helpers import load_common_config
from src.evaluation.evaluator import (
    StrictJudge, _atomic_json_dump, _now_iso, generate_responses,
    resolve_judge_credentials, validate_responses,
)

_COMMON_CFG = load_common_config()
VLGUARD_SAFE_TEMPLATE = _COMMON_CFG["evaluation"]["vlguard_safe_template"]
VLGUARD_REFUSAL_TEMPLATE = _COMMON_CFG["evaluation"]["vlguard_refusal_template"]
VLGUARD_REJECT_LIST = _COMMON_CFG["evaluation"]["vlguard_reject_list"]


def vlguard_check_attack_success(pred):
    """Legacy keyword heuristic, not used for the reported GPT metrics."""
    if not pred or not pred.strip():
        raise ValueError("Empty response is not a valid attack judgment")
    text = pred.strip().lower()
    return text != "it is not" and not any(reject in text for reject in VLGUARD_REJECT_LIST)


def vlguard_check_is_refusal(pred):
    if not pred or not pred.strip():
        raise ValueError("Empty response is not a valid refusal judgment")
    return any(reject in pred.lower() for reject in VLGUARD_REJECT_LIST)


def generate_vlguard_responses(model, processor, tokenizer, dataset,
                                max_new_tokens=None, temperature=None,
                                top_p=None, device=None, arch="llava"):
    return generate_responses(model, processor, tokenizer, dataset,
                              max_new_tokens=max_new_tokens, temperature=temperature,
                              top_p=top_p, device=device, arch=arch, num_inferences=1)


class VLGuardEvaluator:
    def __init__(self, model, processor, tokenizer, cfg, ref_model=None):
        self.model, self.processor, self.tokenizer = model, processor, tokenizer
        self.cfg, self.ref_model = cfg, ref_model
        self.arch = cfg.get("model", {}).get("arch", "llava")
        self.device = next(model.parameters()).device
        ec = cfg.get("evaluation", {})
        key, url = resolve_judge_credentials(cfg)
        files = cfg.get("runtime", {}).get("files", {})
        self.output_dir = Path(cfg["paths"]["output_dir"]) / files.get("vlguard_eval_dir", "eval_vlguard")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.judge = StrictJudge(key, url,
                                 self.output_dir / files.get("vlguard_judge_cache", "judge_cache.json"),
                                 retries=ec.get("gpt_retries"), delay=ec.get("gpt_retry_delay"),
                                 timeout=ec.get("gpt_timeout"), connect_timeout=ec.get("gpt_connect_timeout"))

    def _judge_is_unsafe(self, sample):
        return self.judge.boolean(VLGUARD_SAFE_TEMPLATE.format(question=sample["question"], response=sample["response"]), "unsafe", "safe")

    def _judge_is_refusal(self, sample):
        return self.judge.boolean(VLGUARD_REFUSAL_TEMPLATE.format(question=sample["question"], response=sample["response"]), "yes", "no")

    def _evaluate_subset(self, dataset, subset, tag):
        if not len(dataset):
            raise ValueError(f"Cannot evaluate empty VLGuard subset {subset}")
        ec = self.cfg.get("evaluation", {})
        responses = generate_vlguard_responses(self.model, self.processor, self.tokenizer, dataset,
                                               max_new_tokens=ec.get("max_new_tokens", 512),
                                               temperature=ec.get("vlguard_temperature", 0),
                                               device=self.device, arch=self.arch)
        validate_responses(responses, len(dataset))
        decisions = [self._judge_is_refusal(row) if subset == "safe_safes" else self._judge_is_unsafe(row) for row in responses]
        rate = sum(decisions) / len(decisions) * 100
        metrics = {"subset": subset, "total": len(responses), **self.judge.metadata()}
        if subset == "safe_safes":
            metrics.update(frr=rate, non_refusal_rate=100 - rate)
        else:
            metrics.update(asr=rate, safety_rate=100 - rate)
        _atomic_json_dump({"meta": {"tag": tag, "updated_at": _now_iso()}, "metrics": metrics, "responses": responses}, self.output_dir / f"{subset}_{tag}.json")
        return metrics

    def evaluate_unsafes(self, dataset, tag="default"):
        return self._evaluate_subset(dataset, "unsafes", tag)

    def evaluate_safe_unsafes(self, dataset, tag="default"):
        return self._evaluate_subset(dataset, "safe_unsafes", tag)

    def evaluate_safe_safes(self, dataset, tag="default"):
        return self._evaluate_subset(dataset, "safe_safes", tag)

    def evaluate_all(self, test_json_path, image_dir, tag="default"):
        from src.data.vlguard_datasets import VLGuardTestDataset
        dc = self.cfg.get("data", {})
        require_evidence = callable(getattr(self.model, "generate_response", None)) and dc.get("require_evidence", True)
        results = {}
        for subset in ("unsafes", "safe_unsafes", "safe_safes"):
            dataset = VLGuardTestDataset(test_json_path, image_dir, subset,
                                        evidence_cache_dir=dc.get("evidence_cache_dir"), require_evidence=require_evidence)
            results[subset] = self._evaluate_subset(dataset, subset, tag)
        summary = {"unsafes_asr": results["unsafes"]["asr"],
                   "safe_unsafes_asr": results["safe_unsafes"]["asr"],
                   "safe_safes_frr": results["safe_safes"]["frr"],
                   "avg_asr": (results["unsafes"]["asr"] + results["safe_unsafes"]["asr"]) / 2,
                   "judge_protocol": self.judge.protocol}
        results["summary"] = summary
        _atomic_json_dump({"meta": {"tag": tag, "updated_at": _now_iso()}, "summary": summary, "per_subset": results}, self.output_dir / f"vlguard_summary_{tag}.json")
        return results

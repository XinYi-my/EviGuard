import json
from unittest.mock import patch

import pytest
import requests
import torch

from src.evaluation.evaluator import Evaluator, compute_all_metrics, generate_responses, gpt_4o_call
from src.evaluation.vlguard_evaluator import VLGuardEvaluator


def response(split="forget", text="A generated answer.", **extra):
    return {"id": "a", "question": "q", "split": split, "response": text,
            "category": "Violence", **extra}


class TinyModel(torch.nn.Module):
    def __init__(self, result="A generated answer."):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.result = result
    def generate_response(self, **kwargs):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def config(tmp_path):
    return {"model": {"arch": "llava"}, "paths": {"output_dir": str(tmp_path)},
            "evaluation": {"api_key": "", "api_url": "", "max_new_tokens": 8,
                           "vlguard_temperature": 0., "gpt_retries": 1, "gpt_retry_delay": 0,
                           "gpt_timeout": 1, "gpt_connect_timeout": 1},
            "runtime": {"files": {"vlguard_eval_dir": "eval_vlguard", "vlguard_judge_cache": "judge.json"}}}


def test_judge_key_required_instead_of_fallback():
    with pytest.raises((ValueError, RuntimeError), match="API|api|judge"):
        compute_all_metrics([response()])


def test_api_failure_raises_instead_of_becoming_a_judgment():
    with patch("src.evaluation.evaluator.requests.post", side_effect=requests.Timeout("offline")):
        with pytest.raises((ValueError, RuntimeError), match="judge|Judge|request"):
            gpt_4o_call("judge", "test-token", retries=1, delay=0)


@pytest.mark.parametrize("raw", ["maybe safe", "not unsafe", "exception", ""])
def test_unparseable_judge_answer_is_rejected(raw):
    with patch("src.evaluation.evaluator.gpt_4o_call", return_value=raw):
        with pytest.raises((ValueError, RuntimeError), match="judge|Judge|parse|label"):
            compute_all_metrics([response()], api_key="test-token")


@pytest.mark.parametrize("text", ["", "   ", "[ERROR]"])
def test_invalid_generated_responses_cannot_improve_asr(text):
    with patch("src.evaluation.evaluator.gpt_4o_call", return_value="safe"):
        with pytest.raises(ValueError, match="response|generation"):
            compute_all_metrics([response(text=text)], api_key="test-token")


def test_no_posthoc_holdout_or_invented_vanilla_asr():
    with patch("src.evaluation.evaluator.gpt_4o_call", side_effect=["safe", "yes"]):
        metrics = compute_all_metrics([response()], api_key="test-token")
    assert metrics["efficacy_detail"]["total"] == 1
    assert metrics["generality_asr"] is None
    assert metrics["vanilla_asr"] is None
    assert metrics["er_rel"] is None
    assert metrics["utility"] is None


@pytest.mark.parametrize("result", ["", RuntimeError("generation failed")])
def test_generation_errors_abort_instead_of_error_placeholder(result):
    with pytest.raises((ValueError, RuntimeError)):
        generate_responses(TinyModel(result), None, None, [{"id": "a", "question": "q", "modality": "text"}], device="cpu")


def test_generation_uses_eviguard_wrapper():
    out = generate_responses(TinyModel(), None, None, [{"id": "a", "question": "q", "modality": "text"}], device="cpu")
    assert out[0]["response"] == "A generated answer."


def test_generation_rejects_stale_cache_for_different_question(tmp_path):
    cache = tmp_path / "responses.json"
    cache.write_text(json.dumps({"meta": {"total": 1}, "responses": [{"_dataset_index": 0, "id": "a", "question": "old", "response": "stale"}]}))
    out = generate_responses(TinyModel(), None, None, [{"id": "a", "question": "new", "modality": "text"}], device="cpu", cache_path=cache)
    assert out[0]["question"] == "new"
    assert out[0]["response"] != "stale"


def test_empty_evaluation_cannot_return_perfect_rates(tmp_path):
    with pytest.raises(ValueError, match="empty|nonempty|non-empty"):
        Evaluator(TinyModel(), None, None, config(tmp_path)).evaluate_siuo([])


def test_vlguard_missing_judge_cannot_fallback(tmp_path):
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
        with pytest.raises((ValueError, RuntimeError), match="API|api|judge"):
            ev = VLGuardEvaluator(TinyModel(), None, None, config(tmp_path))
            ev._judge_is_unsafe(response())


def test_corrupt_or_legacy_judge_cache_does_not_supply_fake_scores(tmp_path):
    cache = tmp_path / "judge.json"
    cache.write_text(json.dumps({"cache": {"a": {"source": "fallback", "value": False}}}))
    with patch("src.evaluation.evaluator.gpt_4o_call", return_value="unparseable"):
        with pytest.raises(RuntimeError, match="judge"):
            compute_all_metrics([response()], api_key="test-token", judge_cache_path=cache)


@pytest.mark.parametrize("score", ["nan", "1.5", "-0.1", "0.5 extra"])
def test_gpt_eval_score_cannot_be_out_of_range_or_unparseable(score):
    with patch("src.evaluation.evaluator.gpt_4o_call", return_value=score):
        with pytest.raises(RuntimeError, match="[Jj]udge"):
            compute_all_metrics([response(split="retain", answer="reference")], api_key="test-token")

import pytest
from scripts.evaluate import _load_model_for_eval
from src.models.checkpoint import CheckpointError

def test_evaluation_requires_explicit_model_selection():
    with pytest.raises(ValueError,match="explicit"):
        _load_model_for_eval({})

def test_missing_checkpoint_never_loads_vanilla(tmp_path):
    with pytest.raises(CheckpointError,match="manifest"):
        _load_model_for_eval({},tmp_path/"missing")

import json

import pytest
import torch
from peft import LoraConfig, get_peft_model

from src.models.eviguard_model import EviGuardModel
from src.modules.clsu import RiskSubspace
from test_model import ToyBackbone, ToyTokenizer, toy_config


def fresh_model():
    torch.manual_seed(11)
    backbone = get_peft_model(ToyBackbone(), LoraConfig(r=2, lora_alpha=4, target_modules=["head"]))
    for p in backbone.base_model.model.multi_modal_projector.parameters():
        p.requires_grad_(True)
    return EviGuardModel(backbone, None, None, ToyTokenizer(), toy_config()).eval()


def test_complete_checkpoint_round_trip_and_inference(tmp_path):
    from src.models.checkpoint import save_checkpoint, restore_checkpoint, load_inference_checkpoint
    original = fresh_model()
    with torch.no_grad():
        for p in original.get_trainable_params():
            p.add_(torch.randn_like(p) * .05)
    subspace = RiskSubspace(2, device="cpu", min_fit_samples=2)
    subspace.fit(torch.randn(5, 4), torch.ones(5))
    path = tmp_path / "checkpoint"
    manifest = save_checkpoint(original, path, subspace=subspace,
                               resolved_config=original.cfg, manifest={"step": 4},
                               training_state={"step": 4, "optimizer": {"state": {}}})
    assert manifest["step"] == 4
    assert (path / "adapter" / "adapter_config.json").is_file()
    assert (path / "adapter" / "adapter_model.safetensors").is_file()
    restored = fresh_model()
    state = restore_checkpoint(restored, path)
    assert state["training_state"]["step"] == 4
    assert state["subspace"].is_fitted
    assert torch.equal(state["subspace"].basis, subspace.basis)
    for name, tensor in original.model.state_dict().items():
        assert torch.equal(tensor, restored.model.state_dict()[name]), name
    for name, tensor in original.rsf.state_dict().items():
        assert torch.equal(tensor, restored.rsf.state_dict()[name]), name
    called = []
    def loader(cfg):
        called.append(cfg)
        eg = fresh_model()
        return eg.model, eg.processor, eg.tokenizer, None
    inference = load_inference_checkpoint(path, model_loader=loader)
    assert called == [original.cfg]
    assert inference.ref_model is None
    assert not inference.training
    pixels, ids = torch.randn(1, 2, 4), torch.ones(1, 2, dtype=torch.long)
    kwargs = dict(input_ids=ids, pixel_values=pixels, evidence_strings=["scene"],
                  reliabilities=torch.tensor([.7]))
    original.generate(**kwargs)
    inference.generate(**kwargs)
    assert torch.equal(original.model.base_model.model.generated_logits,
                       inference.model.base_model.model.generated_logits)


def test_checkpoint_rejects_missing_component_instead_of_partial_restore(tmp_path):
    from src.models.checkpoint import save_checkpoint, restore_checkpoint, CheckpointError
    path = tmp_path / "checkpoint"
    save_checkpoint(fresh_model(), path)
    (path / "connector.pt").unlink()
    with pytest.raises(CheckpointError, match="connector"):
        restore_checkpoint(fresh_model(), path)


def test_checkpoint_detects_corruption_before_loading(tmp_path):
    from src.models.checkpoint import save_checkpoint, restore_checkpoint, CheckpointError
    path = tmp_path / "checkpoint"
    save_checkpoint(fresh_model(), path)
    (path / "rsf.pt").write_bytes(b"bad checkpoint")
    with pytest.raises(CheckpointError, match="rsf|hash|digest"):
        restore_checkpoint(fresh_model(), path)


def test_legacy_adapter_only_is_not_complete_eviguard_checkpoint(tmp_path):
    from src.models.checkpoint import load_inference_checkpoint, CheckpointError
    (tmp_path / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(CheckpointError, match="manifest|complete"):
        load_inference_checkpoint(tmp_path)


def test_inference_loader_never_loads_reference_backbone(tmp_path, monkeypatch):
    from src.models.checkpoint import save_checkpoint, load_inference_checkpoint
    import src.models.eviguard_model as module
    path = tmp_path / "checkpoint"
    save_checkpoint(fresh_model(), path)
    called = []
    def llava_loader(mcfg, lcfg, bnb, ref_device, *, load_reference=True):
        called.append(load_reference)
        eg = fresh_model()
        return eg.model, eg.processor, eg.tokenizer, None
    monkeypatch.setattr(module, "_load_llava", llava_loader)
    # The real loader resolves the same small LoRA configuration without downloads.
    cfg = toy_config()
    cfg["runtime"] = {"quantization": {"load_in_4bit": False}}
    cfg["lora"] = {"rank": 2, "alpha": 4, "dropout": 0, "target_modules": ["head"]}
    save_checkpoint(fresh_model(), path, resolved_config=cfg)
    load_inference_checkpoint(path)
    assert called == [False]


def test_resolved_checkpoint_config_never_contains_credentials(tmp_path):
    from src.models.checkpoint import save_checkpoint
    cfg = toy_config()
    cfg["evaluation"] = {"api_key": "private-test-value", "nested": {"access_token": "other-private"}}
    save_checkpoint(fresh_model(), tmp_path, resolved_config=cfg)
    persisted = (tmp_path / "resolved_config.json").read_text(encoding="utf-8")
    assert "private-test-value" not in persisted
    assert "other-private" not in persisted


def test_explicit_backbone_relocation_preserves_checkpoint_logical_identity(tmp_path):
    from src.models.checkpoint import save_checkpoint, load_inference_checkpoint
    save_checkpoint(fresh_model(), tmp_path)
    called = []
    def loader(cfg):
        called.append(cfg["model"]["name"])
        eg = fresh_model()
        return eg.model, None, eg.tokenizer, None
    restored = load_inference_checkpoint(tmp_path, model_loader=loader, backbone_path="/relocated/toy")
    assert called == ["/relocated/toy"]
    assert restored.cfg["model"]["name"] == "toy"

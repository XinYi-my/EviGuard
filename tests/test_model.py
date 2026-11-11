from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from src.models.eviguard_model import EviGuardModel


class ToyTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        n = len(text) if isinstance(text, list) else 1
        return {"input_ids": torch.ones(n, 2, dtype=torch.long),
                "attention_mask": torch.ones(n, 2, dtype=torch.long)}

    def decode(self, ids, **kwargs):
        return "toy response"


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig(hidden_size=4)
        self.multi_modal_projector = nn.Linear(4, 4, bias=False)
        self.embed_tokens = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)
        self.grad_modes = []

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        self.grad_modes.append(torch.is_grad_enabled())
        h = self.multi_modal_projector(pixel_values)
        return SimpleNamespace(logits=self.head(h))

    def generate(self, input_ids, pixel_values=None, **kwargs):
        self.generated_logits = self.forward(input_ids=input_ids, pixel_values=pixel_values).logits
        return torch.cat([input_ids, torch.ones(len(input_ids), 1, dtype=torch.long)], 1)


def toy_config(arch="llava"):
    return {"model": {"name": "toy", "arch": arch},
            "rsf": {"bottleneck_dim": 3, "dropout": 0., "gate_hidden_dim": 2,
                    "evidence_max_tokens": 2, "default_reliability": .5},
            "subspace": {"rank_k": 2}}


def make_model(arch="llava"):
    torch.manual_seed(3)
    model = EviGuardModel(ToyBackbone(), None, None, ToyTokenizer(), toy_config(arch))
    if arch == "internvl":
        # Exercise real EviGuard patch handling without loading remote InternVL code.
        model._forward_base = model.model.forward
    return model


def test_pass2_preserves_clean_connector_gradient_and_trains_rsf():
    model = make_model()
    pixels = torch.randn(2, 3, 4)
    out = model.forward_with_rsf(["one", "two"], torch.tensor([.2, .8]), pixel_values=pixels)
    assert model.model.grad_modes == [False, True]
    assert out["H_original"].requires_grad, "H_original must be clean Pass2"
    out["logits"].square().mean().backward()
    assert model.model.multi_modal_projector.weight.grad.abs().sum() > 0
    assert model.rsf.W_delta.weight.grad.abs().sum() > 0
    assert out["risk_scores"].shape == out["r_raw"].shape == (2,)


class KnownModulation(nn.Module):
    def forward(self, H, evidence, rel):
        self.calls = getattr(self, "calls", 0) + 1
        self.seen_H = H
        raw = torch.tensor([.25, .75, .5], device=H.device)[:len(H)]
        delta = torch.arange(1, len(H) + 1, device=H.device).float()[:, None].expand(-1, 4)
        return H + raw[:, None, None] * delta[:, None], raw * rel, raw, delta


def test_internvl_pools_patch_groups_and_repeats_each_sample_modulation():
    model = make_model("internvl")
    model.rsf = KnownModulation()
    pixels = torch.arange(5 * 2 * 4).float().reshape(5, 2, 4)
    out = model.forward_with_rsf(["one", "two", "text"], torch.tensor([.2, .8, .5]),
                                 pixel_values=pixels, num_patches_list=[2, 3, 0])
    clean = out["H_original"]
    expected_pool = torch.stack([clean[:2].mean((0, 1)), clean[2:].mean((0, 1)),
                                 torch.zeros(4)])
    assert torch.allclose(model.rsf.seen_H[:, 0], expected_pool)
    expected_addition = torch.tensor([.25, .25, 1.5, 1.5, 1.5])[:, None, None]
    assert torch.allclose(out["H_fused"], clean + expected_addition)
    assert out["risk_scores"].shape == (3,)
    assert torch.allclose(model.pool_connector(clean, [2, 3, 0]), expected_pool)


@pytest.mark.parametrize("patches,ev,rel", [([1, 2], ["a", "b"], [.5, .5]),
                                             ([1, 4], ["a"], [.5]),
                                             ([-1, 6], ["a", "b"], [.5, .5]),
                                             (None, ["a", "b"], [.5, .5])])
def test_internvl_rejects_ambiguous_patch_mapping(patches, ev, rel):
    model = make_model("internvl")
    with pytest.raises(ValueError, match="patch|sample|evidence|reliabil"):
        model.forward_with_rsf(ev, torch.tensor(rel), pixel_values=torch.randn(5, 2, 4),
                               num_patches_list=patches)


def test_llava_rejects_evidence_padding_or_truncation():
    model = make_model()
    with pytest.raises(ValueError, match="sample|evidence"):
        model.forward_with_rsf(["a"], torch.tensor([.5]), pixel_values=torch.randn(2, 2, 4))


def test_compute_risk_scores_uses_one_rsf_call_and_keeps_patch_tensor_api():
    model = make_model("internvl")
    model.rsf = KnownModulation()
    h = torch.randn(5, 2, 4)
    fused, scores, raw, delta = model.compute_risk_scores(
        h, ["a", "b"], torch.tensor([.5, .7]), num_patches_list=[2, 3])
    assert model.rsf.calls == 1
    assert fused.shape == h.shape
    assert scores.shape == raw.shape == (2,)
    assert delta.shape == (2, 4)


def test_generation_applies_rsf_and_removes_intervention_afterward():
    model = make_model().eval()
    pixels = torch.randn(1, 2, 4)
    ids = torch.ones(1, 2, dtype=torch.long)
    expected = model.forward_with_rsf([""], torch.tensor([.5]), input_ids=ids, pixel_values=pixels)
    model.generate(input_ids=ids, pixel_values=pixels, evidence_strings=[""], reliabilities=torch.tensor([.5]))
    assert torch.allclose(model.model.generated_logits, expected["logits"])
    assert len(model.connector._forward_hooks) == 1
    clean = model.forward_current(input_ids=ids, pixel_values=pixels)["logits"]
    assert not torch.equal(clean, expected["logits"])


def test_llava_processor_visual_token_metadata_matches_backbone():
    from src.models.llava_utils import configure_llava_processor
    processor = SimpleNamespace(patch_size=None, num_additional_image_tokens=None,
                                vision_feature_select_strategy=None)
    config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14, model_type="clip_vision_model"),
                             vision_feature_select_strategy="default")
    configure_llava_processor(processor, config)
    assert processor.patch_size == 14
    assert processor.num_additional_image_tokens == 1
    assert processor.vision_feature_select_strategy == "default"


def test_language_lora_targets_exclude_frozen_vision_encoder():
    from src.models.llava_utils import language_lora_targets
    model=nn.Module()
    model.vision_tower=nn.Module();model.vision_tower.q_proj=nn.Linear(2,2)
    model.language_model=nn.Module();model.language_model.q_proj=nn.Linear(2,2)
    assert language_lora_targets(model,["q_proj"])==["language_model.q_proj"]

"""CPU regressions for the learned cue and sample-local fusion in Eqs. (9)-(14)."""

import pytest
import torch
import torch.nn.functional as F

from src.modules.rsf import RSFModule, compute_risk_energy, project_to_subspace


def make_rsf():
    torch.manual_seed(42)
    return RSFModule(embed_dim=4, bottleneck_dim=3, dropout=0.0, gate_hidden_dim=5)


def test_implicit_cue_uses_learnable_affine_norm_and_is_sample_local():
    module = make_rsf().eval()
    assert isinstance(getattr(module, "a_imp", None), torch.nn.Parameter)
    assert isinstance(getattr(module, "b_imp", None), torch.nn.Parameter)
    with torch.no_grad():
        module.a_imp.fill_(1.7)
        module.b_imp.fill_(-0.4)
        module.cred_gate.gate_mlp[2].weight.fill_(0.3)
    captured = []
    handle = module.cred_gate.register_forward_pre_hook(lambda _module, args: captured.append(args[1].detach()))
    h, evidence, reliability = torch.ones(1, 2, 4), torch.ones(1, 4), torch.tensor([0.7])
    single = module(h, evidence, reliability)
    expected = torch.sigmoid(1.7 * F.gelu(module.W_imp(h.mean(1))).norm(dim=-1) - 0.4)
    torch.testing.assert_close(captured[-1], expected)
    paired = module(torch.cat([h, h * 10]), torch.cat([evidence, evidence * 10]), torch.tensor([0.7, 0.2]))
    for standalone, in_batch in zip(single, paired):
        torch.testing.assert_close(standalone[0], in_batch[0])
    single[1].sum().backward()
    assert module.a_imp.grad is not None and module.a_imp.grad.abs() > 0
    assert module.b_imp.grad is not None and module.b_imp.grad.abs() > 0
    handle.remove()


def test_zero_initialized_gate_starts_at_half_and_modulation_uses_raw_risk():
    module = make_rsf()
    torch.testing.assert_close(module.cred_gate(torch.tensor([0.1, 0.9]), torch.tensor([0.8, 0.2])), torch.full((2,), 0.5))
    h, evidence, reliability = torch.randn(2, 3, 4), torch.randn(2, 4), torch.tensor([0.2, 0.8])
    fused, scores, raw, delta = module(h, evidence, reliability)
    torch.testing.assert_close(scores, reliability * raw)
    torch.testing.assert_close(fused, h + raw[:, None, None] * delta[:, None, :])


@pytest.mark.parametrize("bad_input", ["activation", "evidence", "reliability"])
def test_nonfinite_fusion_inputs_raise(bad_input):
    h, evidence, reliability = torch.ones(1, 2, 4), torch.ones(1, 4), torch.tensor([0.7])
    if bad_input == "activation":
        h[0, 0, 0] = float("nan")
    elif bad_input == "evidence":
        evidence[0, 0] = float("inf")
    else:
        reliability[0] = float("nan")
    with pytest.raises(FloatingPointError):
        make_rsf()(h, evidence, reliability)


def test_projection_helper_rejects_missing_basis_and_nonfinite_values():
    with pytest.raises(ValueError, match="basis"):
        project_to_subspace(torch.ones(1, 2), None)
    with pytest.raises(FloatingPointError):
        project_to_subspace(torch.tensor([[float("inf"), 1.0]]), torch.tensor([[1.0], [0.0]]))


def test_energy_uses_additive_epsilon_and_is_differentiable():
    vector = torch.tensor([[1e-5, 0.0]], requires_grad=True)
    energy = compute_risk_energy(vector, torch.tensor([[1.0], [0.0]]))
    torch.testing.assert_close(energy, torch.tensor([1e-10 / (1e-10 + 1e-8)]))
    energy.sum().backward()
    assert torch.isfinite(vector.grad).all() and vector.grad[0, 0] > 0


@pytest.mark.parametrize("parameter", ["temperature", "risk_head", "fusion_gate"])
def test_nonfinite_internal_logits_are_not_hidden_by_clamps_or_sigmoids(parameter):
    module = make_rsf()
    with torch.no_grad():
        if parameter == "temperature":
            module.risk_temperature.fill_(float("inf"))
        elif parameter == "risk_head":
            module.w_r.bias.fill_(float("inf"))
        else:
            module.cred_gate.gate_mlp[2].bias.fill_(float("inf"))
    with pytest.raises(FloatingPointError):
        module(torch.ones(1, 2, 4), torch.ones(1, 4), torch.tensor([0.7]))


def test_fusion_gate_respects_module_dtype():
    module = make_rsf().double()
    h, evidence, reliability = torch.ones(1, 2, 4, dtype=torch.float64), torch.ones(1, 4, dtype=torch.float64), torch.tensor([0.7], dtype=torch.float64)
    fused, scores, raw, delta = module(h, evidence, reliability)
    assert fused.dtype == scores.dtype == raw.dtype == delta.dtype == torch.float64

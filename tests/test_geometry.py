"""CPU regressions for the pooled, centered geometry in paper Eqs. (15)-(22)."""

import pytest
import torch
import torch.nn.functional as F

from src.modules.clsu import CLSULoss, RiskSubspace


def make_loss(**overrides):
    config = dict(lambda_pd=1.0, lambda_s=1.0, lambda_orth=1.0,
                  lambda_cm=1.0, lambda_conf=1.0, gamma=1.0,
                  epsilon=0.05, risk_amplify=1.5)
    config.update(overrides)
    return CLSULoss(**config)


def fit_line(offset=None):
    offset = torch.tensor([10.0, -3.0]) if offset is None else offset
    points = offset + torch.tensor([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    subspace = RiskSubspace(rank_k=1, device="cpu", min_fit_samples=3)
    subspace.fit(points, torch.ones(3))
    return subspace


def test_projection_and_energy_use_the_weighted_centroid():
    subspace = fit_line()
    point = torch.tensor([[12.0, 1.0]])
    torch.testing.assert_close(subspace.project_risk(point), torch.tensor([[2.0, 0.0]]))
    torch.testing.assert_close(subspace.project_ortho(point), torch.tensor([[0.0, 4.0]]))
    torch.testing.assert_close(subspace.risk_energy(point), torch.tensor([4.0 / (20.0 + 1e-8)]))


def test_weighted_svd_fits_paper_equations_and_detaches_estimate():
    points = torch.tensor([[0.0, 2.0], [2.0, 0.0], [5.0, 3.0], [7.0, 1.0]], requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    subspace = RiskSubspace(1, device="cpu", min_fit_samples=3)
    subspace.fit(points, weights)
    w = weights.detach() / weights.detach().sum()
    centroid = (w[:, None] * points.detach()).sum(0)
    _, _, vt = torch.linalg.svd(w.sqrt()[:, None] * (points.detach() - centroid), full_matrices=False)
    torch.testing.assert_close(subspace.centroid, centroid)
    torch.testing.assert_close(subspace.basis @ subspace.basis.T, vt[:1].T @ vt[:1])
    assert not subspace.centroid.requires_grad
    assert not subspace.basis.requires_grad


def test_pooled_geometry_is_invariant_to_token_variance_and_global_translation():
    subspace = fit_line()
    tokens = torch.tensor([[[10.0, 10.0], [14.0, -8.0]]])
    pooled = tokens.mean(1)
    torch.testing.assert_close(subspace.project_risk(tokens), subspace.project_risk(pooled))
    torch.testing.assert_close(subspace.risk_energy(tokens), subspace.risk_energy(pooled))
    shift = torch.tensor([20.0, 30.0])
    shifted = fit_line(torch.tensor([10.0, -3.0]) + shift)
    torch.testing.assert_close(shifted.risk_energy(tokens + shift), subspace.risk_energy(tokens))


def test_suppression_is_squared_l2_of_pooled_centered_feature():
    tokens = torch.tensor([[[10.0, 10.0], [14.0, -8.0]]], requires_grad=True)
    loss, _ = make_loss().directional_suppress_loss(tokens, fit_line())
    torch.testing.assert_close(loss, torch.tensor(4.0))
    loss.backward()
    torch.testing.assert_close(tokens.grad, torch.tensor([[[2.0, 0.0], [2.0, 0.0]]]))


def test_orthogonal_loss_pools_before_projection_and_shares_centroid():
    current = torch.tensor([[[14.0, -2.0], [10.0, 4.0]]], requires_grad=True)
    reference = torch.tensor([[[11.0, -3.0], [11.0, -3.0]]], requires_grad=True)
    subspace = fit_line()
    loss, _ = make_loss().orthogonal_preserve_loss(current, reference, subspace)
    torch.testing.assert_close(loss, torch.tensor(16.0))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[[0.0, 4.0], [0.0, 4.0]]]))
    assert reference.grad is None
    torch.testing.assert_close(subspace.project_vector(torch.tensor([[2.0, 4.0]])), torch.tensor([[2.0, 0.0]]))


@pytest.mark.parametrize("case", ["too_few", "zero_weights", "negative_weights", "nonfinite_acts", "nonfinite_weights", "zero_variance", "rank_deficient", "weights_shape"])
def test_invalid_subspace_fit_raises_without_marking_fitted(case):
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    weights = torch.ones(3)
    rank = 1
    if case == "too_few":
        points, weights = points[:2], weights[:2]
    elif case == "zero_weights":
        weights.zero_()
    elif case == "negative_weights":
        weights[0] = -1.0
    elif case == "nonfinite_acts":
        points[0, 0] = float("nan")
    elif case == "nonfinite_weights":
        weights[0] = float("inf")
    elif case == "zero_variance":
        points.zero_()
    elif case == "rank_deficient":
        rank = 2
    elif case == "weights_shape":
        weights = weights[:, None]
    subspace = RiskSubspace(rank, device="cpu", min_fit_samples=3)
    with pytest.raises((ValueError, FloatingPointError)):
        subspace.fit(points, weights)
    assert not subspace.is_fitted
    assert subspace.basis is None


def test_unfitted_subspace_does_not_silently_return_zero():
    with pytest.raises(RuntimeError, match="fit"):
        RiskSubspace(1, device="cpu").project_risk(torch.ones(1, 2))


def test_failed_refit_preserves_previous_valid_geometry():
    subspace = fit_line()
    previous = subspace.project_risk(torch.tensor([[12.0, 0.0]]))
    with pytest.raises(ValueError):
        subspace.fit(torch.ones(3, 2), torch.zeros(3))
    torch.testing.assert_close(subspace.project_risk(torch.tensor([[12.0, 0.0]])), previous)


def test_subspace_checkpoint_is_tensor_only_and_restores_centered_geometry(tmp_path):
    original = fit_line()
    path = tmp_path / "subspace.pt"
    original.save(path)
    payload = torch.load(path, weights_only=True)
    assert isinstance(payload["eigenvalues"], torch.Tensor)
    restored = RiskSubspace(2, device="cpu", min_fit_samples=20)
    restored.load(path)
    assert restored.rank_k == 1
    assert restored.min_fit_samples == 3
    point = torch.tensor([[12.0, 1.0]])
    torch.testing.assert_close(restored.risk_energy(point), original.risk_energy(point))


@pytest.mark.parametrize("invalid", ["nonorthonormal", "centroid_shape", "nonfinite"])
def test_subspace_load_rejects_invalid_geometry_transactionally(tmp_path, invalid):
    subspace = fit_line()
    old_basis, old_centroid = subspace.basis.clone(), subspace.centroid.clone()
    basis, centroid = old_basis.clone(), old_centroid.clone()
    if invalid == "nonorthonormal":
        basis *= 2
    elif invalid == "centroid_shape":
        centroid = torch.zeros(3)
    else:
        basis[0, 0] = float("nan")
    path = tmp_path / "invalid.pt"
    torch.save(dict(format_version=1, basis=basis, centroid=centroid,
                    eigenvalues=torch.ones(1), explained_var=[1.0], rank_k=1,
                    min_fit_samples=3, is_fitted=True), path)
    with pytest.raises((ValueError, FloatingPointError)):
        subspace.load(path)
    torch.testing.assert_close(subspace.basis, old_basis)
    torch.testing.assert_close(subspace.centroid, old_centroid)


def test_response_risk_weight_remains_nonuniform_for_microbatch_one():
    criterion = make_loss(gamma=2.0, risk_amplify=1.5)
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    labels, mask = torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3)
    forget, retain = torch.tensor([True]), torch.tensor([False])
    low, _ = criterion.po_risk_loss(logits, labels, mask, torch.tensor([0.2]), forget, retain)
    high, _ = criterion.po_risk_loss(logits, labels, mask, torch.tensor([0.8]), forget, retain)
    torch.testing.assert_close(low, torch.log(torch.tensor(2.0)) * 1.5 * 0.2 ** 2)
    torch.testing.assert_close(high, 16 * low)


def test_response_weight_leaves_retain_cross_entropy_unweighted_and_risk_differentiable():
    criterion = make_loss()
    logits = torch.zeros(2, 3, 2, requires_grad=True)
    scores = torch.tensor([0.2, 0.8], requires_grad=True)
    forget, retain = torch.tensor([True, False]), torch.tensor([False, True])
    loss, _ = criterion.po_risk_loss(logits, torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3), scores, forget, retain)
    ce = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(loss, ce * (1.5 * 0.2 + 1.0) / 2)
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([1.5 * ce / 2, 0.0]))


def test_confidence_penalty_is_sum_of_forget_and_retain_expectations():
    scores = torch.tensor([0.2, 0.6, 0.1], requires_grad=True)
    loss, _ = make_loss().confidence_reg_loss(scores, torch.tensor([True, True, False]), torch.tensor([False, False, True]))
    torch.testing.assert_close(loss, torch.tensor(0.7))
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([-0.5, -0.5, 1.0]))


def test_response_ce_preserves_large_finite_logits_and_response_mask():
    logits = torch.tensor([[[100.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[-100, 1, -100]])
    actual = make_loss()._per_sample_ce(logits, labels, torch.ones(1, 3))
    expected = F.cross_entropy(logits[:, 0], torch.tensor([1]), reduction="none")
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("component", ["ce", "risk", "suppression", "orthogonal", "cosine", "confidence", "total"])
def test_nonfinite_losses_raise_instead_of_becoming_zero(component):
    criterion, bad = make_loss(), torch.tensor([[float("nan"), 1.0]])
    with pytest.raises(FloatingPointError):
        if component == "ce":
            criterion._per_sample_ce(bad.view(1, 1, 2).expand(1, 2, 2), torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2))
        elif component == "risk":
            criterion.risk_weight(bad[0])
        elif component == "suppression":
            criterion.directional_suppress_loss(bad, fit_line())
        elif component == "orthogonal":
            criterion.orthogonal_preserve_loss(bad, torch.zeros(1, 2), fit_line())
        elif component == "cosine":
            criterion.cm_loss(bad, torch.ones(1, 2))
        elif component == "confidence":
            criterion.confidence_reg_loss(bad[0], torch.tensor([True, False]), torch.tensor([False, True]))
        else:
            criterion._safe(torch.tensor(float("inf")), "total", {})

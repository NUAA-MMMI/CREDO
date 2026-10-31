"""Runtime-only OCOR/DM-UOR outcome token-weight construction.

The module consumes repaired-CF tensors that already exist on the driver.  It
does not recompute targets, teachers, frontiers, rewards, or advantages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

try:
    from .ocor_review_protocol import ReviewItem, intervention_metrics, select_dm_uor
except ImportError:  # direct contract-test execution from this directory
    from ocor_review_protocol import ReviewItem, intervention_metrics, select_dm_uor


ARMS = ("broad_uor", "dm_uor", "ocor", "polarity_protected")


@dataclass(frozen=True)
class RuntimeResult:
    weights: torch.Tensor
    eligible: torch.Tensor
    generic_candidate: torch.Tensor
    dm_selected: torch.Tensor
    diagnostics: dict[str, float | int | None]


def _sequence_advantage(advantages: torch.Tensor, actor: torch.Tensor) -> torch.Tensor:
    denominator = actor.sum(-1).clamp_min(1.0)
    return (advantages.to(torch.float32) * actor).sum(-1) / denominator


def build_runtime_weights(
    *,
    actor_mask: torch.Tensor,
    realization_mask: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    evidence_mask: torch.Tensor | None = None,
    cf_reasoning_valid: torch.Tensor,
    cf_reasoning_opportunity: torch.Tensor,
    advantages: torch.Tensor,
    uids: Sequence[object],
    qlabels: Sequence[str],
    qtypes: Sequence[str],
    arm: str,
    rho: float = 0.04,
    min_realization_fraction: float = 0.05,
    dm_seed: int = 0,
    tolerance: float = 1.0e-6,
) -> RuntimeResult:
    """Return per-token outcome weights with exact per-trajectory mass conservation."""

    if arm not in ARMS:
        raise ValueError(f"invalid OCOR arm: {arm}")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0,1)")
    if not (0.0 < min_realization_fraction <= 1.0):
        raise ValueError("min_realization_fraction must be in (0,1]")
    actor = torch.as_tensor(actor_mask, dtype=torch.float32).detach().cpu()
    realization = torch.as_tensor(realization_mask, dtype=torch.float32).detach().cpu()
    query = None if query_mask is None else torch.as_tensor(
        query_mask, dtype=torch.float32
    ).detach().cpu()
    evidence = None if evidence_mask is None else torch.as_tensor(
        evidence_mask, dtype=torch.float32
    ).detach().cpu()
    advantage = torch.as_tensor(advantages, dtype=torch.float32).detach().cpu()
    valid = torch.as_tensor(cf_reasoning_valid, dtype=torch.float32).detach().cpu().reshape(-1)
    opportunity = torch.as_tensor(
        cf_reasoning_opportunity, dtype=torch.float32
    ).detach().cpu().reshape(-1)
    batch_size = actor.shape[0]
    if actor.ndim != 2 or realization.shape != actor.shape or advantage.shape != actor.shape:
        raise ValueError("actor, realization, and advantage tensors must have identical [B,T] shape")
    if arm == "polarity_protected":
        if query is None or evidence is None:
            raise ValueError("polarity_protected requires query and evidence masks")
        if query.shape != actor.shape or evidence.shape != actor.shape:
            raise ValueError("query and evidence tensors must match actor [B,T] shape")
    if valid.numel() != batch_size or opportunity.numel() != batch_size:
        raise ValueError("CF validity/opportunity must have one value per trajectory")
    if not (len(uids) == len(qlabels) == len(qtypes) == batch_size):
        raise ValueError("metadata length must equal batch size")
    for name, tensor in (
        ("actor", actor), ("realization", realization), ("advantage", advantage),
        ("valid", valid), ("opportunity", opportunity),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
    actor = (actor > 0).to(torch.float32)
    realization = (realization > 0).to(torch.float32)
    if (realization * (1.0 - actor)).sum().item() != 0:
        raise ValueError("realization mask leaks outside actor mask")
    if arm == "polarity_protected":
        assert query is not None and evidence is not None
        if not torch.isfinite(query).all() or not torch.isfinite(evidence).all():
            raise ValueError("query or evidence contains NaN or Inf")
        query = (query > 0).to(torch.float32)
        evidence = (evidence > 0).to(torch.float32)
        union_sum = query + evidence + realization
        if (union_sum > 1.0).any():
            raise ValueError("query, evidence, and realization masks overlap")
        if (union_sum * (1.0 - actor)).sum().item() != 0:
            raise ValueError("query or evidence mask leaks outside actor mask")

    n = actor.sum(-1)
    m = realization.sum(-1)
    fraction = m / n.clamp_min(1.0)
    generic = (valid > 0) & (n > 0) & (m > 0) & (
        fraction + tolerance >= min_realization_fraction
    )
    ocor = generic & (opportunity > 0)
    sequence_adv = _sequence_advantage(advantage, actor)

    rows = [
        ReviewItem(
            key=f"{type(uids[index]).__qualname__}:{uids[index]!r}|{index}",
            uid=str(uids[index]),
            qlabel=str(qlabels[index]),
            qtype=str(qtypes[index]),
            actor_tokens=int(n[index].item()),
            realization_fraction=float(fraction[index].item()),
            outcome_advantage=float(sequence_adv[index].item()),
            generic_candidate=bool(generic[index].item()),
            ocor_eligible=bool(ocor[index].item()),
        )
        for index in range(batch_size)
    ]
    dm_keys, dm_diagnostics = select_dm_uor(rows, seed=dm_seed, rho=rho)
    dm = torch.tensor([row.key in dm_keys for row in rows], dtype=torch.bool)
    selected = (
        ocor
        if arm in ("ocor", "polarity_protected")
        else generic
        if arm == "broad_uor"
        else dm
    )

    weights = actor.clone()
    active = selected & generic
    positive_protected = torch.zeros(batch_size, dtype=torch.bool)
    negative_ocor = torch.zeros(batch_size, dtype=torch.bool)
    donor_weight = torch.ones(batch_size, dtype=torch.float32)
    if arm == "polarity_protected":
        assert query is not None and evidence is not None
        other = torch.clamp(actor - query - evidence - realization, 0.0, 1.0)
        donor_tokens = other.sum(-1)
        nonzero_advantage = sequence_adv.abs() > tolerance
        positive_candidate = active & (sequence_adv > tolerance)
        negative_ocor = active & (sequence_adv < -tolerance)
        net_transfer = rho * (n - m)
        donor_weight = torch.where(
            donor_tokens > 0,
            1.0 - net_transfer / donor_tokens.clamp_min(1.0),
            torch.ones_like(donor_tokens),
        )
        donor_safe = (donor_tokens > 0) & (donor_weight + tolerance >= 0.50)
        positive_protected = positive_candidate & donor_safe
        # Zero advantages and unsafe positive donors are exact no-ops.
        active = negative_ocor | positive_protected
    if active.any():
        coefficient = (rho * n[active] / m[active]).unsqueeze(-1)
        weights[active] = (1.0 - rho) * actor[active] + coefficient * realization[active]
    if positive_protected.any():
        assert query is not None and evidence is not None
        other = torch.clamp(actor - query - evidence - realization, 0.0, 1.0)
        realization_weight = (
            1.0 - rho + rho * n[positive_protected] / m[positive_protected]
        ).unsqueeze(-1)
        weights[positive_protected] = (
            query[positive_protected]
            + evidence[positive_protected]
            + donor_weight[positive_protected].unsqueeze(-1) * other[positive_protected]
            + realization_weight * realization[positive_protected]
        )
    mass_error = (weights.sum(-1) - n).abs()
    mass_relative_error = mass_error / n.clamp_min(1.0)
    if mass_error.max().item() > tolerance * max(float(n.max().item()), 1.0):
        raise AssertionError("OCOR per-trajectory token mass invariant failed")
    if (weights < 0).any() or (weights * (1.0 - actor)).sum().item() != 0:
        raise AssertionError("OCOR policy weight is negative or leaks outside actor mask")
    theoretical_max = 1.0 - rho + rho / min_realization_fraction
    if weights.max().item() > theoretical_max + tolerance:
        raise AssertionError(
            "OCOR policy weight exceeds the preregistered realization-fraction bound"
        )

    selected_rows = [
        ReviewItem(**{**row.__dict__, "ocor_eligible": bool(active[index].item())})
        for index, row in enumerate(rows)
    ]
    metrics = intervention_metrics(selected_rows, rho=rho)
    metrics.update(
        {
            "arm_id": float(ARMS.index(arm)),
            "raw_ocor_eligible_rate": float(ocor.float().mean().item()),
            "generic_candidate_rate": float(generic.float().mean().item()),
            "selected_rate": float(active.float().mean().item()),
            "positive_protected_rate": float(positive_protected.float().mean().item()),
            "negative_ocor_rate": float(negative_ocor.float().mean().item()),
            "positive_donor_weight_min": float(
                donor_weight[positive_protected].min().item()
            ) if positive_protected.any() else 1.0,
            "realization_fraction_p05": float(torch.quantile(fraction[generic], 0.05).item())
            if generic.any() else 0.0,
            "realization_fraction_p50": float(torch.quantile(fraction[generic], 0.50).item())
            if generic.any() else 0.0,
            "realization_fraction_p95": float(torch.quantile(fraction[generic], 0.95).item())
            if generic.any() else 0.0,
            "token_weight_max": float(weights.max().item()),
            "token_weight_theoretical_max": float(theoretical_max),
            "mass_error_max": float(mass_error.max().item()),
            "mass_relative_error_max": float(mass_relative_error.max().item()),
            **{f"dm_{key}": value for key, value in dm_diagnostics.items()},
        }
    )
    return RuntimeResult(weights, active, generic, dm, metrics)

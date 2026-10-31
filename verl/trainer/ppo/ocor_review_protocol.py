"""OCOR-CF review metrics and deterministic dose-matched UOR control.

This module is deliberately independent from the training runtime.  It does
not score targets, change CF teachers, or update model parameters.  It only
audits already-computed OCOR eligibility and builds an opportunity-blind
control with matched terminal reallocation dose.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ReviewItem:
    key: str
    uid: str
    qlabel: str
    qtype: str
    actor_tokens: int
    realization_fraction: float
    outcome_advantage: float
    generic_candidate: bool
    ocor_eligible: bool


def _sign(value: float, eps: float = 1.0e-8) -> int:
    return 1 if value > eps else -1 if value < -eps else 0


def _p_bucket(value: float) -> str:
    if value < 0.10:
        return "p05_10"
    if value < 0.20:
        return "p10_20"
    if value < 0.40:
        return "p20_40"
    return "p40_100"


def _n_bucket(value: int) -> str:
    if value <= 512:
        return "n_le512"
    if value <= 1024:
        return "n_513_1024"
    if value <= 2048:
        return "n_1025_2048"
    if value <= 4096:
        return "n_2049_4096"
    return "n_gt4096"


def _adv_bucket(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude < 1.0e-8:
        return "a_zero"
    if magnitude < 0.75:
        return "a_lt075"
    if magnitude < 1.25:
        return "a_075_125"
    if magnitude < 2.0:
        return "a_125_200"
    return "a_ge200"


def stable_rank(key: str, seed: int) -> int:
    payload = f"ocor-dm-uor-v1|{seed}|{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def intervention_metrics(
    items: Iterable[ReviewItem], *, rho: float = 0.04, eps: float = 1.0e-8
) -> dict[str, float | int | None]:
    """Return raw, effective, and advantage-weighted intervention diagnostics."""

    rows = list(items)
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0,1)")
    if any(row.actor_tokens < 0 for row in rows):
        raise ValueError("actor token counts must be non-negative")

    total_tokens = sum(row.actor_tokens for row in rows)
    nonzero_tokens = sum(row.actor_tokens for row in rows if abs(row.outcome_advantage) > eps)
    advantage_mass = sum(row.actor_tokens * abs(row.outcome_advantage) for row in rows)
    geometric_moved = sum(rho * row.actor_tokens for row in rows if row.ocor_eligible)
    effective_moved = sum(
        rho * row.actor_tokens
        for row in rows
        if row.ocor_eligible and abs(row.outcome_advantage) > eps
    )
    weighted_moved = sum(
        rho * row.actor_tokens * abs(row.outcome_advantage)
        for row in rows
        if row.ocor_eligible
    )
    z_eff = effective_moved
    wrong = sum(
        rho * row.actor_tokens
        for row in rows
        if row.ocor_eligible and row.outcome_advantage < -eps
    )
    correct = sum(
        rho * row.actor_tokens
        for row in rows
        if row.ocor_eligible and row.outcome_advantage > eps
    )
    wrong_adv = sum(
        rho * row.actor_tokens * abs(row.outcome_advantage)
        for row in rows
        if row.ocor_eligible and row.outcome_advantage < -eps
    )
    correct_adv = sum(
        rho * row.actor_tokens * abs(row.outcome_advantage)
        for row in rows
        if row.ocor_eligible and row.outcome_advantage > eps
    )

    def ratio(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator > 0 else None

    coefficient_sum = None
    if z_eff > 0:
        coefficient_sum = float((wrong + correct) / z_eff)
        if not math.isclose(coefficient_sum, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise AssertionError("wrong/correct coefficient shares do not conserve effective dose")

    return {
        "items": len(rows),
        "eligible": sum(row.ocor_eligible for row in rows),
        "generic_candidates": sum(row.generic_candidate for row in rows),
        "nonzero_advantage": sum(abs(row.outcome_advantage) > eps for row in rows),
        "intervention_mass_geom": ratio(geometric_moved, total_tokens),
        "intervention_mass_effective": ratio(effective_moved, nonzero_tokens),
        "intervention_mass_advantage_weighted": ratio(weighted_moved, advantage_mass),
        "wrong_blame_coefficient_share": ratio(wrong, z_eff),
        "correct_credit_coefficient_share": ratio(correct, z_eff),
        "wrong_blame_advantage_share": ratio(wrong_adv, weighted_moved),
        "correct_credit_advantage_share": ratio(correct_adv, weighted_moved),
        "coefficient_share_sum": coefficient_sum,
        "geometric_moved_mass": geometric_moved,
        "effective_moved_mass": effective_moved,
        "advantage_weighted_moved_mass": weighted_moved,
    }


def _stratum(item: ReviewItem) -> tuple[str, int, str, str, str]:
    return (
        item.qlabel,
        _sign(item.outcome_advantage),
        _p_bucket(item.realization_fraction),
        _n_bucket(item.actor_tokens),
        _adv_bucket(item.outcome_advantage),
    )


def _choose_prefix(candidates: list[ReviewItem], target_tokens: int, seed: int) -> list[ReviewItem]:
    """Choose a stable prefix whose actor-token mass is closest to the target."""

    if target_tokens <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda row: (stable_rank(row.key, seed), row.key))
    best_count = 0
    best_error = target_tokens
    running = 0
    for count, row in enumerate(ordered, 1):
        running += row.actor_tokens
        error = abs(running - target_tokens)
        if error < best_error:
            best_error = error
            best_count = count
        elif running > target_tokens and error > best_error:
            break
    return ordered[:best_count]


def select_dm_uor(
    items: Iterable[ReviewItem], *, seed: int = 0, rho: float = 0.04
) -> tuple[set[str], dict[str, float | int | None]]:
    """Build a deterministic opportunity-blind, approximately dose-matched control.

    OCOR eligibility is used only to define aggregate target dose per stratum.
    Candidate ranking never reads the individual opportunity/eligibility bit.
    All generic candidates, including chance overlap with OCOR, remain eligible
    for selection; overlap is measured explicitly.
    """

    rows = list(items)
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("ReviewItem keys must be unique")
    targets: dict[tuple[str, int, str, str, str], int] = defaultdict(int)
    pools: dict[tuple[str, int, str, str, str], list[ReviewItem]] = defaultdict(list)
    for row in rows:
        group = _stratum(row)
        if row.ocor_eligible:
            targets[group] += row.actor_tokens
        if row.generic_candidate:
            pools[group].append(row)

    selected: list[ReviewItem] = []
    for group in sorted(set(targets) | set(pools)):
        selected.extend(_choose_prefix(pools.get(group, []), targets.get(group, 0), seed))
    selected_keys = {row.key for row in selected}

    ocor_geom = sum(rho * row.actor_tokens for row in rows if row.ocor_eligible)
    dm_geom = sum(rho * row.actor_tokens for row in selected)
    ocor_adv = sum(
        rho * row.actor_tokens * abs(row.outcome_advantage)
        for row in rows
        if row.ocor_eligible
    )
    dm_adv = sum(rho * row.actor_tokens * abs(row.outcome_advantage) for row in selected)
    overlap = sum(row.ocor_eligible for row in selected)

    def relative_error(actual: float, target: float) -> float | None:
        return float(abs(actual - target) / target) if target > 0 else None

    diagnostics: dict[str, float | int | None] = {
        "ocor_count": sum(row.ocor_eligible for row in rows),
        "dm_count": len(selected),
        "ocor_geometric_dose": ocor_geom,
        "dm_geometric_dose": dm_geom,
        "geometric_dose_relative_error": relative_error(dm_geom, ocor_geom),
        "ocor_advantage_dose": ocor_adv,
        "dm_advantage_dose": dm_adv,
        "advantage_dose_relative_error": relative_error(dm_adv, ocor_adv),
        "opportunity_overlap_rate": float(overlap / len(selected)) if selected else None,
        "strata_with_target": sum(value > 0 for value in targets.values()),
        "strata_without_candidates": sum(targets[group] > 0 and not pools.get(group) for group in targets),
    }
    return selected_keys, diagnostics


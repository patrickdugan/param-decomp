"""Powered cell base for the constrained-choice lane (roadmap item 2).

A *cell* is a gate-compatible failure family at `top_target` granularity
(`{top}->{target}:letter`, gate fires on top action + margin, runner wildcarded)
that is eval-powered (>= MIN_TARGET_PER_FAMILY failing samples). For each cell we
train an ordinary low-rank adapter over the cached choice head and classify it as
*regression-positive* when it lifts the target family but damages the non-target
complement -- the dirty-adapter condition the extractor lane is meant to clean.

Item 1 showed eval power is only reachable at coarse granularity on a single
suite, so cells are pooled across scored suites (ARC-Challenge + ARC-Easy + ...).
Recurrence is an eval bootstrap of the powered sample set, not a re-decode: the
constrained-decode scorer is deterministic, so "is this cell representative or
lucky" is answered by resampling the eval rows, not by rescoring.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from param_decomp.experiments.trm_basis_constrained_edits.common import ChoiceSample, GateSpec
from param_decomp.experiments.trm_basis_constrained_edits.eval_power import MIN_TARGET_PER_FAMILY
from param_decomp.experiments.trm_basis_constrained_edits.head_to_head import (
    EvalContext,
    FailureFamily,
    evaluate_candidate,
)
from patch_optimizer.patch_runner import train_low_rank_adapter

GATE_MARGIN = 1.0
DEFAULT_RANK = 4
DEFAULT_STEPS = 30
DEFAULT_LR = 0.15
DEFAULT_BOOTSTRAPS = 200


def top_target_families(
    pooled: list[ChoiceSample],
    *,
    min_target: int = MIN_TARGET_PER_FAMILY,
    label_family: str = "letter",
) -> list[FailureFamily]:
    """Powered `top_target` failure families (runner wildcarded), >= min_target."""
    grouped: dict[tuple[str, str], list[ChoiceSample]] = defaultdict(list)
    for sample in pooled:
        if sample.baseline_hit or sample.label_family != label_family:
            continue
        grouped[(sample.top_action(), sample.candidate_actions[sample.target_position])].append(
            sample
        )
    families: list[FailureFamily] = []
    for (top, target), members in grouped.items():
        if len(members) < min_target:
            continue
        families.append(
            FailureFamily(
                family_key=f"{top}->{target}:{label_family}",
                top_action=top,
                runner_up_action="any",
                target_action=target,
                label_family=label_family,
                trajectory_ids=frozenset(sample.trajectory_id for sample in members),
                failure_count=len(members),
            )
        )
    families.sort(key=lambda fam: (-fam.failure_count, fam.family_key))
    return families


def _gate_for(family: FailureFamily) -> GateSpec:
    return GateSpec(
        gate_id=f"cell:{family.family_key}",
        top_action=family.top_action,
        runner_up_action="any",
        max_margin=GATE_MARGIN,
    )


def _is_regression_positive(target_delta: float, non_target_delta: float, damage_count: int) -> bool:
    return target_delta > 0 and (non_target_delta < 0 or damage_count > 0)


@dataclass(frozen=True)
class Cell:
    cell_id: str
    family_key: str
    top_action: str
    target_action: str
    n_target: int
    n_nontarget: int
    raw_target_delta: float
    raw_non_target_delta: float
    raw_damage_count: int
    regression_positive: bool
    bootstrap_recurrence: float


def _bootstrap_recurrence(
    family: FailureFamily,
    gate: GateSpec,
    pooled: list[ChoiceSample],
    edit_matrix: np.ndarray,
    *,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> float:
    pooled_arr = np.array(pooled, dtype=object)
    n = len(pooled_arr)
    positives = 0
    evaluated = 0
    for _ in range(n_bootstraps):
        resampled = list(pooled_arr[rng.integers(0, n, size=n)])
        if not any(sample.trajectory_id in family.trajectory_ids for sample in resampled):
            continue
        context = EvalContext.build(family, gate, {"boot": resampled})
        metrics = evaluate_candidate(context, edit_matrix)
        evaluated += 1
        positives += int(
            _is_regression_positive(
                metrics.target_delta, metrics.non_target_delta, metrics.damage_count
            )
        )
    return round(positives / evaluated, 6) if evaluated else 0.0


def build_cell(
    family: FailureFamily,
    pooled: list[ChoiceSample],
    *,
    rank: int,
    steps: int,
    lr: float,
    seed: int,
    n_bootstraps: int,
) -> Cell:
    gate = _gate_for(family)
    context = EvalContext.build(family, gate, {"pooled": pooled})
    adapter, _ = train_low_rank_adapter(
        context,
        rank=rank,
        steps=steps,
        lr=lr,
        seed=seed,
        replay_weight=0.0,
        init="random",
        basis=None,
    )
    edit_matrix = adapter.matrix()
    metrics = evaluate_candidate(context, edit_matrix)
    n_nontarget = context.total_samples() - family.failure_count
    recurrence = _bootstrap_recurrence(
        family,
        gate,
        pooled,
        edit_matrix,
        n_bootstraps=n_bootstraps,
        rng=np.random.default_rng(seed + 7919),
    )
    return Cell(
        cell_id=family.family_key,
        family_key=family.family_key,
        top_action=family.top_action,
        target_action=family.target_action,
        n_target=family.failure_count,
        n_nontarget=n_nontarget,
        raw_target_delta=metrics.target_delta,
        raw_non_target_delta=metrics.non_target_delta,
        raw_damage_count=metrics.damage_count,
        regression_positive=_is_regression_positive(
            metrics.target_delta, metrics.non_target_delta, metrics.damage_count
        ),
        bootstrap_recurrence=recurrence,
    )


@dataclass(frozen=True)
class CellBaseReport:
    n_pooled_samples: int
    n_powered_families: int
    n_cells: int
    n_regression_positive: int
    n_recurrent_regression_positive: float
    cells: tuple[Cell, ...]


def build_cell_base(
    pooled: list[ChoiceSample],
    *,
    min_target: int = MIN_TARGET_PER_FAMILY,
    rank: int = DEFAULT_RANK,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    seed: int = 20260612,
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    recurrence_threshold: float = 0.5,
) -> CellBaseReport:
    families = top_target_families(pooled, min_target=min_target)
    cells = [
        build_cell(
            family,
            pooled,
            rank=rank,
            steps=steps,
            lr=lr,
            seed=seed + index,
            n_bootstraps=n_bootstraps,
        )
        for index, family in enumerate(families)
    ]
    regression_positive = [cell for cell in cells if cell.regression_positive]
    recurrent = [
        cell for cell in regression_positive if cell.bootstrap_recurrence >= recurrence_threshold
    ]
    return CellBaseReport(
        n_pooled_samples=len(pooled),
        n_powered_families=len(families),
        n_cells=len(cells),
        n_regression_positive=len(regression_positive),
        n_recurrent_regression_positive=len(recurrent),
        cells=tuple(cells),
    )


def load_pooled_suites(score_files: dict[str, Path]) -> list[ChoiceSample]:
    """Pool scored suites; dedupe by suite-qualified trajectory id."""
    from param_decomp.experiments.trm_basis_constrained_edits.common import (
        load_seed_samples,
        pooled_samples,
    )

    return pooled_samples(load_seed_samples(score_files))

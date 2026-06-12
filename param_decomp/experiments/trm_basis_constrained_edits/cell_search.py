"""Best-of-N search and extractor comparison over powered cells (items 3 & 4).

Each cell's raw LoRA edit is a CHOICE_DIM x CHOICE_DIM matrix; rank atoms cap at
CHOICE_DIM, too few for a meaningful search, so we decompose the edit into its
matrix-entry coordinates (rank-1 `E_ij` atoms scaled by the entry value). A clean
subpatch is a sparse subset of entries that retains the target lift but drops the
regression-causing entries.

Item 3 (best-of-N): for each sparsity stratum (subset L0) we report whether a
clean subpatch exists, the search success rate at N in {4, 16, 64}, and a
norm-matched random-entry control success rate.

Item 4 (extractor vs best-of-N): the greedy entry extractor is compared against
best-of-N at its own eval budget; the per-cell regression difference feeds a
single pooled sign test that either rescues or retires the extractor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from param_decomp.experiments.trm_basis_constrained_edits.cell_base import (
    DEFAULT_LR,
    DEFAULT_RANK,
    DEFAULT_STEPS,
    top_target_families,
    train_cell_edit,
)
from param_decomp.experiments.trm_basis_constrained_edits.common import ChoiceSample
from param_decomp.experiments.trm_basis_constrained_edits.eval_power import MIN_TARGET_PER_FAMILY
from param_decomp.experiments.trm_basis_constrained_edits.head_to_head import (
    CandidateMetrics,
    EvalContext,
    FailureFamily,
    evaluate_candidate,
)

N_GRID = (4, 16, 64)
SUPPORT_STRATA = (1, 2, 4, 8)
DEFAULT_REPEATS = 64
DEFAULT_EXACT_CAP = 3000
RETENTION_FLOOR = 0.5
REG_TOL = 1e-9


def _regression(metrics: CandidateMetrics) -> float:
    return float(metrics.damage_count) + abs(min(0.0, float(metrics.non_target_delta)))


def entry_coords(matrix: np.ndarray, *, tol: float = 1e-12) -> list[tuple[int, int]]:
    return [
        (int(i), int(j))
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
        if abs(float(matrix[i, j])) > tol
    ]


def _compose(matrix: np.ndarray, coords: list[tuple[int, int]], indices: list[int]) -> np.ndarray:
    out = np.zeros_like(matrix)
    for index in indices:
        i, j = coords[index]
        out[i, j] = matrix[i, j]
    return out


def _norm_match(sub: np.ndarray, target_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(sub))
    if norm <= 1e-12 or target_norm <= 0:
        return sub
    return sub * (target_norm / norm)


def _is_clean(metrics: CandidateMetrics, raw_target: float) -> bool:
    target_delta = float(metrics.target_delta)
    if raw_target <= 0:
        return False
    retention = target_delta / raw_target
    return target_delta > 0 and retention >= RETENTION_FLOOR and _regression(metrics) <= REG_TOL


@dataclass(frozen=True)
class SupportSearch:
    support: int
    clean_exists: bool
    existence_checked: int
    success_rate_by_n: dict[int, float]
    control_success_rate_by_n: dict[int, float]


@dataclass(frozen=True)
class ExtractorResult:
    selected_support: int
    regression: float
    target_retention: float
    eval_budget: int


@dataclass(frozen=True)
class CellSearch:
    family_key: str
    raw_target_delta: float
    raw_regression: float
    n_entry_atoms: int
    supports: tuple[SupportSearch, ...]
    extractor: ExtractorResult
    matched_best_of_n_regression: float
    extractor_minus_best_of_n: float


def _sample_subset_matrix(
    matrix: np.ndarray,
    coords: list[tuple[int, int]],
    *,
    support: int,
    raw_norm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    support = min(support, len(coords))
    indices = [int(i) for i in rng.choice(len(coords), size=support, replace=False)]
    return _norm_match(_compose(matrix, coords, indices), raw_norm)


def _best_of_n_success(
    context: EvalContext,
    matrix: np.ndarray,
    coords: list[tuple[int, int]],
    raw_target: float,
    raw_norm: float,
    *,
    support: int,
    n: int,
    rng: np.random.Generator,
) -> bool:
    for _ in range(n):
        edit = _sample_subset_matrix(matrix, coords, support=support, raw_norm=raw_norm, rng=rng)
        if _is_clean(evaluate_candidate(context, edit), raw_target):
            return True
    return False


def _success_rate(
    context: EvalContext,
    matrix: np.ndarray,
    coords: list[tuple[int, int]],
    raw_target: float,
    raw_norm: float,
    *,
    support: int,
    n: int,
    repeats: int,
    rng: np.random.Generator,
) -> float:
    hits = sum(
        _best_of_n_success(
            context, matrix, coords, raw_target, raw_norm, support=support, n=n, rng=rng
        )
        for _ in range(repeats)
    )
    return round(hits / repeats, 6)


def _control_success_rate(
    context: EvalContext,
    raw_target: float,
    raw_norm: float,
    shape: tuple[int, int],
    *,
    support: int,
    n: int,
    repeats: int,
    rng: np.random.Generator,
) -> float:
    n_entries = shape[0] * shape[1]
    flat_coords = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    hits = 0
    for _ in range(repeats):
        found = False
        for _ in range(n):
            values = rng.normal(size=n_entries)
            indices = [int(i) for i in rng.choice(n_entries, size=min(support, n_entries), replace=False)]
            edit = np.zeros(shape)
            for index in indices:
                i, j = flat_coords[index]
                edit[i, j] = values[index]
            if _is_clean(evaluate_candidate(context, _norm_match(edit, raw_norm)), raw_target):
                found = True
                break
        hits += int(found)
    return round(hits / repeats, 6)


def _clean_exists(
    context: EvalContext,
    matrix: np.ndarray,
    coords: list[tuple[int, int]],
    raw_target: float,
    raw_norm: float,
    *,
    support: int,
    rng: np.random.Generator,
    exact_cap: int,
    ref_budget: int,
) -> tuple[bool, int]:
    support = min(support, len(coords))
    n_subsets = math.comb(len(coords), support)
    if n_subsets <= exact_cap:
        for combo in combinations(range(len(coords)), support):
            edit = _norm_match(_compose(matrix, coords, list(combo)), raw_norm)
            if _is_clean(evaluate_candidate(context, edit), raw_target):
                return True, n_subsets
        return False, n_subsets
    for _ in range(ref_budget):
        edit = _sample_subset_matrix(matrix, coords, support=support, raw_norm=raw_norm, rng=rng)
        if _is_clean(evaluate_candidate(context, edit), raw_target):
            return True, ref_budget
    return False, ref_budget


def greedy_extractor(
    context: EvalContext,
    matrix: np.ndarray,
    coords: list[tuple[int, int]],
    raw_target: float,
    raw_norm: float,
    *,
    max_support: int,
) -> ExtractorResult:
    selected: list[int] = []
    remaining = list(range(len(coords)))
    best_reg = math.inf
    best_metrics = evaluate_candidate(context, np.zeros_like(matrix))
    eval_budget = 0
    while remaining and len(selected) < max_support:
        round_best: tuple[tuple[float, float], int, CandidateMetrics, float] | None = None
        for index in remaining:
            edit = _norm_match(_compose(matrix, coords, selected + [index]), raw_norm)
            metrics = evaluate_candidate(context, edit)
            eval_budget += 1
            reg = _regression(metrics)
            retention = float(metrics.target_delta) / raw_target if raw_target > 0 else 0.0
            key = (reg, -float(metrics.target_delta))
            if retention >= RETENTION_FLOOR and (round_best is None or key < round_best[0]):
                round_best = (key, index, metrics, reg)
        if round_best is None:
            break
        _, chosen_index, chosen_metrics, chosen_reg = round_best
        selected.append(chosen_index)
        remaining.remove(chosen_index)
        if chosen_reg < best_reg:
            best_reg = chosen_reg
            best_metrics = chosen_metrics
    retention = float(best_metrics.target_delta) / raw_target if raw_target > 0 else 0.0
    return ExtractorResult(
        selected_support=len(selected),
        regression=round(best_reg if math.isfinite(best_reg) else _regression(best_metrics), 6),
        target_retention=round(retention, 6),
        eval_budget=eval_budget,
    )


def _best_of_budget_regression(
    context: EvalContext,
    matrix: np.ndarray,
    coords: list[tuple[int, int]],
    raw_target: float,
    raw_norm: float,
    *,
    budget: int,
    rng: np.random.Generator,
) -> float:
    best = math.inf
    for _ in range(max(1, budget)):
        support = int(rng.integers(1, len(coords) + 1))
        edit = _sample_subset_matrix(matrix, coords, support=support, raw_norm=raw_norm, rng=rng)
        metrics = evaluate_candidate(context, edit)
        retention = float(metrics.target_delta) / raw_target if raw_target > 0 else 0.0
        if retention >= RETENTION_FLOOR:
            best = min(best, _regression(metrics))
    return round(best, 6) if math.isfinite(best) else math.inf


def search_cell(
    family: FailureFamily,
    pooled: list[ChoiceSample],
    *,
    rank: int = DEFAULT_RANK,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    seed: int,
    repeats: int = DEFAULT_REPEATS,
    exact_cap: int = DEFAULT_EXACT_CAP,
    ref_budget: int = 512,
) -> CellSearch:
    context, matrix, raw_metrics = train_cell_edit(
        family, pooled, rank=rank, steps=steps, lr=lr, seed=seed
    )
    raw_target = float(raw_metrics.target_delta)
    raw_norm = float(np.linalg.norm(matrix))
    coords = entry_coords(matrix)
    rng = np.random.default_rng(seed + 104729)

    supports: list[SupportSearch] = []
    for support in SUPPORT_STRATA:
        if support > len(coords):
            continue
        clean_exists, checked = _clean_exists(
            context, matrix, coords, raw_target, raw_norm,
            support=support, rng=rng, exact_cap=exact_cap, ref_budget=ref_budget,
        )
        success = {
            n: _success_rate(
                context, matrix, coords, raw_target, raw_norm,
                support=support, n=n, repeats=repeats, rng=rng,
            )
            for n in N_GRID
        }
        control = {
            n: _control_success_rate(
                context, raw_target, raw_norm, matrix.shape,
                support=support, n=n, repeats=repeats, rng=rng,
            )
            for n in N_GRID
        }
        supports.append(
            SupportSearch(
                support=support,
                clean_exists=clean_exists,
                existence_checked=checked,
                success_rate_by_n=success,
                control_success_rate_by_n=control,
            )
        )

    extractor = greedy_extractor(
        context, matrix, coords, raw_target, raw_norm, max_support=len(coords)
    )
    matched = _best_of_budget_regression(
        context, matrix, coords, raw_target, raw_norm, budget=extractor.eval_budget, rng=rng
    )
    diff = extractor.regression - matched if math.isfinite(matched) else math.inf
    return CellSearch(
        family_key=family.family_key,
        raw_target_delta=round(raw_target, 6),
        raw_regression=round(_regression(raw_metrics), 6),
        n_entry_atoms=len(coords),
        supports=tuple(supports),
        extractor=extractor,
        matched_best_of_n_regression=matched,
        extractor_minus_best_of_n=round(diff, 6) if math.isfinite(diff) else math.inf,
    )


def _two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return round(min(1.0, 2.0 * tail), 6)


@dataclass(frozen=True)
class PooledExtractorTest:
    n_cells: int
    extractor_better: int
    best_of_n_better: int
    ties: int
    median_extractor_minus_best_of_n: float
    sign_test_p: float
    verdict: str


def pooled_extractor_test(cells: list[CellSearch]) -> PooledExtractorTest:
    diffs = [c.extractor_minus_best_of_n for c in cells if math.isfinite(c.extractor_minus_best_of_n)]
    wins = sum(1 for d in diffs if d < 0)
    losses = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    p = _two_sided_sign_p(wins, losses)
    median = round(float(np.median(diffs)), 6) if diffs else math.inf
    if wins > losses and p < 0.05:
        verdict = "EXTRACTOR_RESCUED"
    elif losses > wins and p < 0.05:
        verdict = "EXTRACTOR_RETIRED_BEST_OF_N_WINS"
    else:
        verdict = "EXTRACTOR_NO_ADVANTAGE"
    return PooledExtractorTest(
        n_cells=len(cells),
        extractor_better=wins,
        best_of_n_better=losses,
        ties=ties,
        median_extractor_minus_best_of_n=median,
        sign_test_p=p,
        verdict=verdict,
    )


def build_search_base(
    pooled: list[ChoiceSample],
    *,
    min_target: int = MIN_TARGET_PER_FAMILY,
    rank: int = DEFAULT_RANK,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    seed: int = 20260612,
    repeats: int = DEFAULT_REPEATS,
) -> tuple[tuple[CellSearch, ...], PooledExtractorTest]:
    families = top_target_families(pooled, min_target=min_target)
    cells = [
        search_cell(family, pooled, rank=rank, steps=steps, lr=lr, seed=seed + index, repeats=repeats)
        for index, family in enumerate(families)
    ]
    return tuple(cells), pooled_extractor_test(cells)

"""Unit tests for best-of-N search and extractor comparison (items 3 & 4)."""

from __future__ import annotations

import math

import numpy as np

from param_decomp.experiments.trm_basis_constrained_edits.cell_base import top_target_families
from param_decomp.experiments.trm_basis_constrained_edits.cell_search import (
    CellSearch,
    ExtractorResult,
    _compose,
    _norm_match,
    _two_sided_sign_p,
    entry_coords,
    pooled_extractor_test,
    search_cell,
)
from param_decomp.experiments.trm_basis_constrained_edits.common import LETTER_LABELS, ChoiceSample


def _sample(top: str, target: str, runner: str, *, hit: bool, tid: str) -> ChoiceSample:
    pos = {label: index for index, label in enumerate(LETTER_LABELS)}
    logprobs = np.full(4, -5.0)
    logprobs[pos[top]] = 0.0
    logprobs[pos[runner]] = -1.0
    target_pos = pos[top] if hit else pos[target]
    return ChoiceSample(
        seed_label="s",
        trajectory_id=tid,
        env_id="arc_challenge",
        candidate_actions=LETTER_LABELS,
        logprobs=logprobs,
        target_position=target_pos,
        label_family="letter",
    )


def test_entry_coords_and_compose_roundtrip():
    matrix = np.array([[0.0, 2.0], [3.0, 0.0]])
    coords = entry_coords(matrix)
    assert coords == [(0, 1), (1, 0)]
    composed = _compose(matrix, coords, [0, 1])
    assert np.allclose(composed, matrix)
    partial = _compose(matrix, coords, [0])
    assert partial[0, 1] == 2.0 and partial[1, 0] == 0.0


def test_norm_match_scales_to_target():
    sub = np.array([[3.0, 4.0], [0.0, 0.0]])
    matched = _norm_match(sub, 1.0)
    assert math.isclose(float(np.linalg.norm(matched)), 1.0)


def test_two_sided_sign_p_known_values():
    assert _two_sided_sign_p(0, 0) == 1.0
    assert math.isclose(_two_sided_sign_p(5, 0), 0.0625)
    assert _two_sided_sign_p(6, 0) < 0.05
    assert _two_sided_sign_p(3, 3) == 1.0


def _cell(diff: float) -> CellSearch:
    return CellSearch(
        family_key="x",
        raw_target_delta=0.5,
        raw_regression=2.0,
        n_entry_atoms=16,
        supports=(),
        extractor=ExtractorResult(1, 0.0, 1.0, 10),
        matched_best_of_n_regression=0.0,
        extractor_minus_best_of_n=diff,
    )


def test_pooled_extractor_test_verdicts():
    retired = pooled_extractor_test([_cell(1.0) for _ in range(6)])
    assert retired.best_of_n_better == 6 and retired.verdict == "EXTRACTOR_RETIRED_BEST_OF_N_WINS"
    rescued = pooled_extractor_test([_cell(-1.0) for _ in range(6)])
    assert rescued.extractor_better == 6 and rescued.verdict == "EXTRACTOR_RESCUED"
    noadv = pooled_extractor_test([_cell(0.0) for _ in range(6)])
    assert noadv.ties == 6 and noadv.verdict == "EXTRACTOR_NO_ADVANTAGE"


def test_search_cell_runs_end_to_end():
    samples = [_sample("B", "A", "C", hit=False, tid=f"f{i}") for i in range(60)]
    samples += [_sample("A", "A", "B", hit=True, tid=f"h{i}") for i in range(200)]
    family = top_target_families(samples, min_target=50)[0]
    result = search_cell(family, samples, seed=0, repeats=8, ref_budget=64)
    assert result.family_key == "B->A:letter"
    assert result.n_entry_atoms >= 1
    for support in result.supports:
        for n, rate in support.success_rate_by_n.items():
            assert n in (4, 16, 64)
            assert 0.0 <= rate <= 1.0
            assert 0.0 <= support.control_success_rate_by_n[n] <= 1.0
    assert result.extractor.eval_budget > 0

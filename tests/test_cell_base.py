"""Unit tests for the powered cell-base builder (roadmap item 2)."""

from __future__ import annotations

import numpy as np

from param_decomp.experiments.trm_basis_constrained_edits.cell_base import (
    _is_regression_positive,
    build_cell_base,
    top_target_families,
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


def test_top_target_families_wildcards_runner_and_pools_runners():
    samples = []
    for runner in ("C", "D"):
        for i in range(30):
            samples.append(_sample("B", "A", runner, hit=False, tid=f"{runner}{i}"))
    samples += [_sample("A", "A", "B", hit=True, tid=f"h{i}") for i in range(50)]
    families = top_target_families(samples, min_target=50)
    assert len(families) == 1
    fam = families[0]
    assert fam.family_key == "B->A:letter"
    assert fam.runner_up_action == "any"
    assert fam.failure_count == 60


def test_min_target_filters_underpowered_families():
    samples = [_sample("B", "A", "C", hit=False, tid=f"f{i}") for i in range(40)]
    samples += [_sample("A", "A", "B", hit=True, tid=f"h{i}") for i in range(100)]
    assert top_target_families(samples, min_target=50) == []
    assert len(top_target_families(samples, min_target=30)) == 1


def test_regression_positive_predicate():
    assert _is_regression_positive(0.1, -0.01, 0)
    assert _is_regression_positive(0.1, 0.0, 2)
    assert not _is_regression_positive(0.1, 0.0, 0)
    assert not _is_regression_positive(0.0, -0.01, 3)
    assert not _is_regression_positive(-0.1, -0.01, 2)


def test_build_cell_base_produces_powered_cells():
    rng = np.random.default_rng(0)
    samples = []
    for i in range(60):
        samples.append(_sample("B", "A", "C", hit=False, tid=f"ba{i}"))
    for i in range(60):
        samples.append(_sample("C", "D", "A", hit=False, tid=f"cd{i}"))
    for i in range(300):
        top = LETTER_LABELS[rng.integers(0, 4)]
        samples.append(_sample(top, top, "B", hit=True, tid=f"h{i}"))
    report = build_cell_base(samples, min_target=50, steps=10, n_bootstraps=20)
    assert report.n_powered_families == 2
    assert report.n_cells == 2
    assert report.n_pooled_samples == len(samples)
    for cell in report.cells:
        assert 0.0 <= cell.bootstrap_recurrence <= 1.0

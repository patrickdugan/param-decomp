"""Unit tests for the eval-power harness (roadmap item 1)."""

from __future__ import annotations

import numpy as np

from param_decomp.experiments.trm_basis_constrained_edits.common import LETTER_LABELS, ChoiceSample
from param_decomp.experiments.trm_basis_constrained_edits.eval_power import (
    MAX_RESOLVABLE_DELTA,
    family_key_for,
    measure_power,
    project_scale,
)


def _sample(top: str, target: str, runner: str, *, hit: bool, tid: str) -> ChoiceSample:
    """Build a letter ChoiceSample whose argmax/runner-up/target are controlled."""
    pos = {label: index for index, label in enumerate(LETTER_LABELS)}
    logprobs = np.full(4, -5.0)
    logprobs[pos[top]] = 0.0
    logprobs[pos[runner]] = -1.0
    target_pos = pos[target] if not hit else pos[top]
    return ChoiceSample(
        seed_label="s",
        trajectory_id=tid,
        env_id="arc_challenge",
        candidate_actions=LETTER_LABELS,
        logprobs=logprobs,
        target_position=target_pos,
        label_family="letter",
    )


def test_family_key_skips_hits_and_respects_granularity():
    hit = _sample("A", "A", "B", hit=True, tid="t0")
    miss = _sample("B", "A", "C", hit=False, tid="t1")
    assert family_key_for(hit, "top_target_runner") is None
    assert family_key_for(miss, "top_target_runner") == "B->A:C:letter"
    assert family_key_for(miss, "top_target") == "B->A:letter"
    assert family_key_for(miss, "target") == "A:letter"


def test_measure_power_target_count_and_nontarget_complement():
    samples = [_sample("B", "A", "C", hit=False, tid=f"f{i}") for i in range(60)]
    samples += [_sample("A", "A", "B", hit=True, tid=f"h{i}") for i in range(200)]
    report = measure_power(samples, granularity="top_target_runner")
    assert report.n_samples == 260
    assert report.n_families == 1
    fam = report.families[0]
    assert fam.n_target == 60
    assert fam.n_nontarget == 200
    assert fam.target_powered and fam.nontarget_powered and fam.powered
    assert fam.resolvable_target_delta <= MAX_RESOLVABLE_DELTA
    assert report.passes


def test_measure_power_underpowered_family_fails():
    samples = [_sample("B", "A", "C", hit=False, tid=f"f{i}") for i in range(10)]
    samples += [_sample("A", "A", "B", hit=True, tid=f"h{i}") for i in range(40)]
    report = measure_power(samples, granularity="top_target_runner")
    fam = report.families[0]
    assert fam.n_target == 10
    assert not fam.target_powered
    assert not fam.nontarget_powered  # only 40 non-target rows
    assert not report.passes


def test_coarsening_merges_families_into_one():
    samples = []
    for runner in ("A", "C", "D"):
        for i in range(20):
            samples.append(_sample("B", "A", runner, hit=False, tid=f"{runner}{i}"))
    samples += [_sample("A", "A", "B", hit=True, tid=f"h{i}") for i in range(200)]
    fine = measure_power(samples, granularity="top_target_runner")
    coarse = measure_power(samples, granularity="top_target")
    assert fine.n_families == 3
    assert coarse.n_families == 1
    assert coarse.families[0].n_target == 60


def test_project_scale_required_items_scales_inverse_with_share():
    # 10% fail rate, single failing family => required = min_target / (0.1 * 1.0).
    samples = [_sample("B", "A", "C", hit=False, tid=f"f{i}") for i in range(10)]
    samples += [_sample("A", "A", "B", hit=True, tid=f"h{i}") for i in range(90)]
    proj = project_scale(samples, granularity="top_target_runner", planned_items=256, min_target=50)
    assert proj.cached_fail_rate == 0.1
    assert proj.required_items_top_family == 500
    assert not proj.planned_powers_top_family
    assert proj.families[0].expected_target_at_planned == round(256 * 0.1, 3)

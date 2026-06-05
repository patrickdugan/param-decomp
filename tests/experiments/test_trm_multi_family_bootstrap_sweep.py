from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_multi_family_bootstrap_sweep import (
    failure_families,
    family_slice,
    run_sweep,
)


def _sample(trajectory_id: str, target: str, top: str, runner: str, *, gap: float = 0.2) -> dict[str, object]:
    return {
        "env_id": "arc",
        "trajectory_id": trajectory_id,
        "target_action": target,
        "scores": [{"action": top, "logprob": -1.0}, {"action": runner, "logprob": -1.0 - gap}],
    }


def _args(score_file: Path, out_dir: Path) -> Namespace:
    return Namespace(
        score_files=str(score_file),
        out_dir=out_dir,
        rounds=3,
        penalties="0.25",
        margins="0.25",
        touch_rate_max=1.0,
        touch_penalty=0.05,
        complexity_penalty=0.003,
        min_family_size=1,
        max_families=8,
        max_negative_per_family=8,
        max_prompt_tokens=1200,
    )


def test_failure_families_cluster_baseline_misses() -> None:
    samples = [
        _sample("miss1", "A", "B", "A"),
        _sample("miss2", "A", "B", "A"),
        _sample("hit", "C", "C", "D"),
    ]

    families = failure_families(samples, min_family_size=1, max_families=4)

    assert families[0]["family_key"].startswith("B->A:A")
    assert families[0]["failure_count"] == 2


def test_family_slice_includes_contrast_hits() -> None:
    samples = [
        _sample("miss", "A", "B", "A"),
        _sample("same_top_hit", "B", "B", "D"),
        _sample("filler_hit", "C", "C", "D"),
    ]
    family = failure_families(samples, min_family_size=1, max_families=4)[0]

    subset = family_slice(samples, family, max_negative=4)
    ids = {row["trajectory_id"] for row in subset}

    assert "miss" in ids
    assert "same_top_hit" in ids


def test_run_sweep_emits_gain_policy_rows(tmp_path: Path) -> None:
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            _sample("miss_b", "A", "B", "A"),
            _sample("miss_c", "A", "C", "A"),
            _sample("correct_b", "B", "B", "D"),
            _sample("correct_c", "C", "C", "D"),
            _sample("filler", "D", "D", "A", gap=1.0),
        ],
    )

    summary = run_sweep(_args(score_file, tmp_path / "out"))

    assert summary["status"] == "completed"
    assert summary["family_count"] >= 2
    assert summary["first_edit_family_count"] >= 1
    assert (tmp_path / "out" / "gain_policy_training_rows.jsonl").exists()

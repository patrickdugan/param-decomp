from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_edit_bootstrap_microcycle import (
    current_state_samples,
    run_microcycle,
    state_score,
)


def _args(score_file: Path, out_dir: Path, *, rounds: int = 3) -> Namespace:
    return Namespace(
        score_files=str(score_file),
        out_dir=out_dir,
        rounds=rounds,
        penalties="0.25",
        margins="0.25",
        touch_rate_max=1.0,
        touch_penalty=0.05,
        complexity_penalty=0.003,
        max_prompt_tokens=1200,
    )


def test_microcycle_accepts_stacked_conditional_edits(tmp_path: Path) -> None:
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "rescue_b",
                "target_action": "A",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
            },
            {
                "env_id": "arc",
                "trajectory_id": "rescue_c",
                "target_action": "A",
                "scores": [{"action": "C", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
            },
            {
                "env_id": "arc",
                "trajectory_id": "correct_b_wide",
                "target_action": "B",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
            },
            {
                "env_id": "arc",
                "trajectory_id": "correct_c_wide",
                "target_action": "C",
                "scores": [{"action": "C", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
            },
        ],
    )

    summary = run_microcycle(_args(score_file, tmp_path / "out"))

    assert summary["baseline_score"] == 0.5
    assert summary["final_score"] == 1.0
    assert summary["accepted_edit_count"] == 2
    assert summary["stacked_bootstrap_success"] is True
    assert (tmp_path / "out" / "bootstrap_candidate_scores.jsonl").exists()


def test_microcycle_stops_when_fixed_control_is_not_beaten(tmp_path: Path) -> None:
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "rescue_b",
                "target_action": "A",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
            },
            {
                "env_id": "arc",
                "trajectory_id": "already_a",
                "target_action": "A",
                "scores": [{"action": "A", "logprob": -1.0}, {"action": "D", "logprob": -2.0}],
            },
        ],
    )

    summary = run_microcycle(_args(score_file, tmp_path / "out"))

    assert summary["accepted_edit_count"] == 0
    assert summary["stop_reason"] == "fixed_control_not_beaten"
    assert summary["stacked_bootstrap_success"] is False


def test_current_state_samples_applies_edit_stack() -> None:
    samples = [
        {
            "env_id": "arc",
            "trajectory_id": "rescue_b",
            "target_action": "A",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
        }
    ]
    edits = [
        {
            "type": "conditional_top_margin_penalty",
            "top_action": "B",
            "runner_up_action": "A",
            "margin_bucket": "any",
            "max_margin": 0.25,
            "penalty": 0.25,
        }
    ]

    state = current_state_samples(samples, edits)

    assert state_score(samples) == 0.0
    assert state_score(state) == 1.0

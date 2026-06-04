from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_condition_cv import fold_result, run_cv


def write_score_file(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)


def test_fold_result_scores_train_best_on_heldout(tmp_path) -> None:
    train = tmp_path / "seed_train" / "choice_constrained_sample_scores.jsonl"
    heldout = tmp_path / "seed_holdout" / "choice_constrained_sample_scores.jsonl"
    rows = [
        {
            "env_id": "arc",
            "trajectory_id": "rescue",
            "target_action": "A",
            "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
        },
        {
            "env_id": "arc",
            "trajectory_id": "correct",
            "target_action": "B",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -3.0}],
        },
    ]
    write_score_file(train, rows)
    write_score_file(heldout, rows)

    row = fold_result(
        heldout_path=heldout,
        train_paths=[train],
        penalties=[1.0],
        margins=[1.0],
        touch_penalty=0.05,
        complexity_penalty=0.003,
    )

    assert row["heldout_accepted"] is True
    assert row["heldout_rescue_count"] == 1
    assert row["heldout_damage_count"] == 0


def test_fold_result_does_not_promote_when_control_beats_condition(tmp_path) -> None:
    train = tmp_path / "seed_train" / "choice_constrained_sample_scores.jsonl"
    heldout = tmp_path / "seed_holdout" / "choice_constrained_sample_scores.jsonl"
    write_score_file(
        train,
        [
            {
                "env_id": "arc",
                "trajectory_id": "train_rescue",
                "target_action": "A",
                "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
            }
        ],
    )
    write_score_file(
        heldout,
        [
            {
                "env_id": "arc",
                "trajectory_id": "control_rescue",
                "target_action": "A",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
            }
        ],
    )

    row = fold_result(
        heldout_path=heldout,
        train_paths=[train],
        penalties=[1.0],
        margins=[1.0],
        touch_penalty=0.05,
        complexity_penalty=0.003,
    )

    assert row["heldout_accepted"] is False
    assert row["beats_control"] is False
    assert row["best_control_candidate_id"] == "fixed:B:penalty_1_0"


def test_run_cv_writes_folds_and_summary(tmp_path) -> None:
    first = tmp_path / "seed1" / "choice_constrained_sample_scores.jsonl"
    second = tmp_path / "seed2" / "choice_constrained_sample_scores.jsonl"
    rows = [
        {
            "env_id": "arc",
            "trajectory_id": "rescue",
            "target_action": "A",
            "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
        },
        {
            "env_id": "arc",
            "trajectory_id": "correct",
            "target_action": "B",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -3.0}],
        },
    ]
    write_score_file(first, rows)
    write_score_file(second, rows)

    summary = run_cv(
        Namespace(
            score_files=f"{first};{second}",
            out_dir=tmp_path / "out",
            penalties="1",
            margins="1",
            touch_penalty=0.05,
            complexity_penalty=0.003,
            max_prompt_tokens=1200,
        )
    )

    assert summary["fold_count"] == 2
    assert summary["heldout_accepted_fold_count"] == 2
    assert (tmp_path / "out" / "condition_cv_folds.jsonl").exists()

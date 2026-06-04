from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_feature_search_packet import (
    classify_sample,
    policy_from_condition_id,
    run_packet,
)


CONDITION_ID = "cond_top_margin:top_D:runner_A:bucket_any:max_1_0:penalty_1_0"


def test_policy_from_condition_id_parses_controller_condition() -> None:
    policy = policy_from_condition_id(CONDITION_ID)

    assert policy["top_action"] == "D"
    assert policy["runner_up_action"] == "A"
    assert policy["max_margin"] == 1.0
    assert policy["penalty"] == 1.0


def test_classify_sample_marks_positive_rescue() -> None:
    policy = policy_from_condition_id(CONDITION_ID)
    row = classify_sample(
        {
            "env_id": "arc",
            "trajectory_id": "rescue",
            "target_action": "A",
            "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
        },
        policy,
    )

    assert row["label"] == "positive_rescue"
    assert row["condition_prediction"] == "A"


def test_classify_sample_marks_negative_correct_touched() -> None:
    policy = policy_from_condition_id(CONDITION_ID)
    row = classify_sample(
        {
            "env_id": "arc",
            "trajectory_id": "correct",
            "target_action": "D",
            "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
        },
        policy,
    )

    assert row["label"] == "negative_correct_touched"
    assert row["baseline_hit"] is True


def test_run_packet_writes_feature_search_contract(tmp_path) -> None:
    score_file = tmp_path / "seed" / "choice_constrained_sample_scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "rescue",
                "target_action": "A",
                "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
            },
            {
                "env_id": "arc",
                "trajectory_id": "correct",
                "target_action": "D",
                "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
            },
        ],
    )
    cv_run = tmp_path / "cv"
    cv_run.mkdir()
    (cv_run / "condition_cv_summary.json").write_text(
        '{"score_files": ["' + str(score_file).replace("\\", "\\\\") + '"]}\n',
        encoding="utf-8",
    )
    write_jsonl(
        cv_run / "condition_cv_folds.jsonl",
        [
            {
                "train_best_candidate_id": CONDITION_ID,
                "beats_control": True,
                "heldout_accepted": True,
                "heldout_reward": 0.1,
            }
        ],
    )

    summary = run_packet(
        Namespace(
            cv_run=cv_run,
            out_dir=tmp_path / "out",
            max_positive=4,
            max_negative=4,
            max_prompt_tokens=1200,
        )
    )

    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert (tmp_path / "out" / "activation_probe_contract.json").exists()

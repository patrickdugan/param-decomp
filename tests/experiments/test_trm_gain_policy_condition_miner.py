from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_condition_miner import (
    adjusted_prediction,
    candidate_policies,
    run_miner,
    score_policy,
)


def test_conditional_policy_adjusts_only_matching_top_action() -> None:
    policy = {
        "type": "conditional_top_margin_penalty",
        "top_action": "B",
        "runner_up_action": "any",
        "margin_bucket": "any",
        "max_margin": 0.25,
        "penalty": 0.25,
    }
    matching = {"target_action": "D", "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -1.2}]}
    nonmatching = {"target_action": "D", "scores": [{"action": "C", "logprob": -1.0}, {"action": "D", "logprob": -1.2}]}

    assert adjusted_prediction(matching, policy) == ("D", True)
    assert adjusted_prediction(nonmatching, policy) == ("C", False)


def test_efficiency_reward_prefers_lower_touch_tie() -> None:
    samples = [
        {
            "trajectory_id": "rescue",
            "target_action": "D",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
        },
        {
            "trajectory_id": "untouched",
            "target_action": "A",
            "scores": [{"action": "A", "logprob": -1.0}, {"action": "B", "logprob": -3.0}],
        },
    ]
    broad = score_policy(
        samples,
        {"type": "fixed_action_penalty", "action": "B", "penalty": 0.25},
        touch_penalty=0.05,
        complexity_penalty=0.003,
    )
    selective = score_policy(
        samples,
        {
            "type": "conditional_top_margin_penalty",
            "top_action": "B",
            "runner_up_action": "any",
            "margin_bucket": "any",
            "max_margin": 0.25,
            "penalty": 0.25,
        },
        touch_penalty=0.05,
        complexity_penalty=0.003,
    )

    assert selective["reward"] == broad["reward"]
    assert selective["touch_rate"] < broad["touch_rate"]
    assert selective["efficiency_reward"] > broad["efficiency_reward"]


def test_candidate_policies_include_top_and_runner_conditions() -> None:
    samples = [
        {
            "trajectory_id": "a",
            "target_action": "D",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
        }
    ]

    policies = candidate_policies(samples, penalties=[0.25], margins=[0.25])
    ids = {row.get("top_action", "fixed") + ":" + row.get("runner_up_action", "") for row in policies}

    assert "B:any" in ids
    assert "any:D" in ids


def test_run_miner_writes_ranked_scores(tmp_path) -> None:
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "rescue",
                "target_action": "D",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
            },
            {
                "env_id": "arc",
                "trajectory_id": "untouched",
                "target_action": "A",
                "scores": [{"action": "A", "logprob": -1.0}, {"action": "B", "logprob": -3.0}],
            },
        ],
    )

    summary = run_miner(
        Namespace(
            score_files=str(score_file),
            out_dir=tmp_path / "out",
            penalties="0.25",
            margins="0.25",
            touch_penalty=0.05,
            complexity_penalty=0.003,
            max_prompt_tokens=1200,
        )
    )

    assert summary["accepted_count"] > 0
    assert summary["best_touch_rate"] < 1.0
    assert (tmp_path / "out" / "condition_scores.jsonl").exists()

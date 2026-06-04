from __future__ import annotations

from param_decomp.experiments.trm_choice_rl_feedback_loop import dedupe_samples, score_policy


def test_dedupe_samples_keeps_first_env_trajectory_pair() -> None:
    samples = [
        {"env_id": "arc_challenge", "trajectory_id": "sample-1", "target_action": "A"},
        {"env_id": "arc_challenge", "trajectory_id": "sample-1", "target_action": "B"},
        {"env_id": "arc_easy", "trajectory_id": "sample-1", "target_action": "C"},
    ]

    deduped = dedupe_samples(samples)

    assert len(deduped) == 2
    assert deduped[0]["target_action"] == "A"
    assert deduped[1]["target_action"] == "C"


def test_fixed_action_policy_gets_immediate_positive_reward() -> None:
    samples = [
        {
            "trajectory_id": "a",
            "target_action": "A",
            "scores": [{"action": "A", "logprob": -1.0}, {"action": "B", "logprob": -5.0}],
        },
        {
            "trajectory_id": "b",
            "target_action": "A",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
        },
    ]
    policy = {"id": "fixed:B", "type": "fixed_action_penalty", "action": "B", "penalty": 1.0}

    scored = score_policy(samples, policy)

    assert scored["baseline_score"] == 0.5
    assert scored["policy_score"] == 1.0
    assert scored["delta"] == 0.5
    assert scored["rescue_count"] == 1
    assert scored["damage_count"] == 0
    assert scored["accepted"] is True


def test_top_margin_policy_only_touches_close_calls() -> None:
    samples = [
        {
            "trajectory_id": "close",
            "target_action": "A",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
        },
        {
            "trajectory_id": "wide",
            "target_action": "C",
            "scores": [{"action": "C", "logprob": -1.0}, {"action": "A", "logprob": -5.0}],
        },
    ]
    policy = {"id": "margin", "type": "top_margin_penalty", "max_margin": 0.5, "penalty": 0.5}

    scored = score_policy(samples, policy)

    assert scored["baseline_score"] == 0.5
    assert scored["policy_score"] == 1.0
    assert scored["touched_count"] == 1
    assert scored["accepted"] is True

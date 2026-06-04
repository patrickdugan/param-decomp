from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_hook_score import (
    policy_from_hook,
    run_score,
    score_candidate,
)


def test_policy_from_hook_maps_selective_top_penalty() -> None:
    policy = policy_from_hook(
        {
            "candidate_id": "hook",
            "hook_type": "candidate_logit_adjustment",
            "trigger": {"top_minus_runner_up_max_margin": 0.25},
            "action": {"penalize": "current_top_action", "penalty": 0.25},
        }
    )

    assert policy == {
        "id": "top_margin:max_0_25:penalty_0_25",
        "type": "top_margin_penalty",
        "max_margin": 0.25,
        "penalty": 0.25,
    }


def test_score_candidate_accepts_close_call_rescue() -> None:
    samples = [
        {
            "trajectory_id": "close",
            "target_action": "A",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
        },
        {
            "trajectory_id": "wide",
            "target_action": "C",
            "scores": [{"action": "C", "logprob": -1.0}, {"action": "A", "logprob": -3.0}],
        },
    ]
    row = score_candidate(
        samples,
        {
            "candidate_id": "logit_hook:top_margin",
            "edit_family": "selective_top_margin_suppression",
            "hook_type": "candidate_logit_adjustment",
            "trigger": {"top_minus_runner_up_max_margin": 0.25},
            "action": {"penalize": "current_top_action", "penalty": 0.25},
        },
        touch_rate_max=0.5,
    )

    assert row["delta"] == 0.5
    assert row["rescue_count"] == 1
    assert row["damage_count"] == 0
    assert row["accepted_hook"] is True


def test_run_score_keeps_promotion_false_when_control_wins(tmp_path) -> None:
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "logit_hook_candidate.json").write_text(
        """
{
  "candidate_id": "logit_hook:top_margin:max_0_25:penalty_0_25",
  "edit_family": "selective_top_margin_suppression",
  "hook_type": "candidate_logit_adjustment",
  "trigger": {"top_minus_runner_up_max_margin": 0.25},
  "action": {"penalize": "current_top_action", "penalty": 0.25}
}
""".strip(),
        encoding="utf-8",
    )
    write_jsonl(
        bridge / "matched_controls.jsonl",
        [
            {
                "candidate_id": "control_fixed_label:B:penalty_0_5",
                "edit_family": "broad_fixed_label_prior",
                "hook_type": "candidate_logit_adjustment",
                "trigger": {"scope": "all_choice_action_scores"},
                "action": {"penalize": "B", "penalty": 0.5},
            },
            {
                "candidate_id": "control_no_edit",
                "edit_family": "abstain",
                "hook_type": "none",
                "trigger": {"candidate_set_required": False},
                "action": {"mode": "no_edit"},
            },
        ],
    )
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "wide_label_bias",
                "target_action": "A",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.4}],
            }
        ],
    )

    summary = run_score(
        Namespace(
            bridge_dir=bridge,
            score_files=str(score_file),
            out_dir=tmp_path / "out",
            touch_rate_max=1.0,
            max_prompt_tokens=1200,
        )
    )

    assert summary["best_candidate_id"] == "control_fixed_label:B:penalty_0_5"
    assert summary["logit_hook_accepted"] is False
    assert summary["promotion_ready"] is False


def test_run_score_requires_hook_to_beat_controls(tmp_path) -> None:
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "logit_hook_candidate.json").write_text(
        """
{
  "candidate_id": "logit_hook:top_margin:max_0_25:penalty_0_25",
  "edit_family": "selective_top_margin_suppression",
  "hook_type": "candidate_logit_adjustment",
  "trigger": {"top_minus_runner_up_max_margin": 0.25},
  "action": {"penalize": "current_top_action", "penalty": 0.25}
}
""".strip(),
        encoding="utf-8",
    )
    write_jsonl(
        bridge / "matched_controls.jsonl",
        [
            {
                "candidate_id": "control_fixed_label:B:penalty_0_25",
                "edit_family": "broad_fixed_label_prior",
                "hook_type": "candidate_logit_adjustment",
                "trigger": {"scope": "all_choice_action_scores"},
                "action": {"penalize": "B", "penalty": 0.25},
            }
        ],
    )
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "shared_rescue",
                "target_action": "A",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
            }
        ],
    )

    summary = run_score(
        Namespace(
            bridge_dir=bridge,
            score_files=str(score_file),
            out_dir=tmp_path / "out",
            touch_rate_max=1.0,
            max_prompt_tokens=1200,
        )
    )

    assert summary["logit_hook_accepted"] is True
    assert summary["logit_hook_delta"] == summary["best_control_delta"]
    assert summary["promotion_ready"] is False

from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_rescue_family_analyzer import (
    aggregate_families,
    run_analysis,
    sample_analysis,
)


def hook_candidate() -> dict:
    return {
        "candidate_id": "logit_hook:top_margin:max_0_25:penalty_0_25",
        "edit_family": "selective_top_margin_suppression",
        "hook_type": "candidate_logit_adjustment",
        "trigger": {"top_minus_runner_up_max_margin": 0.25},
        "action": {"penalize": "current_top_action", "penalty": 0.25},
    }


def control_candidate() -> dict:
    return {
        "candidate_id": "control_fixed_label:B:penalty_0_25",
        "edit_family": "broad_fixed_label_prior",
        "hook_type": "candidate_logit_adjustment",
        "trigger": {"scope": "all_choice_action_scores"},
        "action": {"penalize": "B", "penalty": 0.25},
    }


def test_sample_analysis_marks_shared_hook_rescue() -> None:
    row = sample_analysis(
        {
            "env_id": "arc",
            "trajectory_id": "shared",
            "target_action": "A",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
        },
        [hook_candidate(), control_candidate()],
    )

    assert row["hook_rescue"] is True
    assert row["shared_hook_rescue"] is True
    assert row["isolated_hook_rescue"] is False
    assert row["family_key"] == "B->A:close_0_25"


def test_sample_analysis_marks_isolated_hook_rescue() -> None:
    row = sample_analysis(
        {
            "env_id": "arc",
            "trajectory_id": "isolated",
            "target_action": "D",
            "scores": [{"action": "C", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
        },
        [hook_candidate(), control_candidate()],
    )

    assert row["hook_rescue"] is True
    assert row["isolated_hook_rescue"] is True
    assert row["rescuing_controls"] == []


def test_aggregate_families_prefers_isolated_rescues() -> None:
    rows = [
        sample_analysis(
            {
                "env_id": "arc",
                "trajectory_id": "shared",
                "target_action": "A",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
            },
            [hook_candidate(), control_candidate()],
        ),
        sample_analysis(
            {
                "env_id": "arc",
                "trajectory_id": "isolated",
                "target_action": "D",
                "scores": [{"action": "C", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
            },
            [hook_candidate(), control_candidate()],
        ),
    ]

    families = aggregate_families(rows)

    assert families[0]["family_key"] == "C->D:close_0_25"
    assert families[0]["isolated_hook_rescue_count"] == 1


def test_run_analysis_writes_contract(tmp_path) -> None:
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
    write_jsonl(bridge / "matched_controls.jsonl", [control_candidate()])
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "isolated",
                "target_action": "D",
                "scores": [{"action": "C", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
            }
        ],
    )

    summary = run_analysis(
        Namespace(
            bridge_dir=bridge,
            score_files=str(score_file),
            out_dir=tmp_path / "out",
            max_prompt_tokens=1200,
        )
    )

    assert summary["isolated_hook_rescue_count"] == 1
    assert summary["promotion_ready"] is False
    assert (tmp_path / "out" / "next_edit_contract.json").exists()

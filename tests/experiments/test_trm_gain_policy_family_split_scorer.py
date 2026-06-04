from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_family_split_scorer import run_split_score, score_split, split_samples


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


def no_edit_candidate() -> dict:
    return {
        "candidate_id": "control_no_edit",
        "edit_family": "abstain",
        "hook_type": "none",
        "trigger": {"candidate_set_required": False},
        "action": {"mode": "no_edit"},
    }


def test_score_split_promotes_when_hook_beats_control() -> None:
    split = {
        "split_id": "C->D:close_0_25",
        "split_type": "family_key",
        "samples": [
            {
                "trajectory_id": "isolated",
                "target_action": "D",
                "scores": [{"action": "C", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
            },
            {
                "trajectory_id": "correct",
                "target_action": "A",
                "scores": [{"action": "A", "logprob": -1.0}, {"action": "B", "logprob": -2.0}],
            },
        ],
    }

    row = score_split(split, [hook_candidate(), control_candidate(), no_edit_candidate()], touch_rate_max=0.5)

    assert row["hook_beats_control"] is True
    assert row["promotion_ready"] is True
    assert row["hook_rescue_count"] == 1


def test_score_split_keeps_tie_as_non_promotion() -> None:
    split = {
        "split_id": "B->D:close_0_25",
        "split_type": "family_key",
        "samples": [
            {
                "trajectory_id": "shared",
                "target_action": "D",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
            },
            {
                "trajectory_id": "wide",
                "target_action": "A",
                "scores": [{"action": "A", "logprob": -1.0}, {"action": "B", "logprob": -3.0}],
            },
        ],
    }

    row = score_split(split, [hook_candidate(), control_candidate(), no_edit_candidate()], touch_rate_max=0.5)

    assert row["hook_beats_control"] is False
    assert row["hook_ties_control_with_lower_touch"] is True
    assert row["promotion_ready"] is False


def test_split_samples_adds_family_margin_and_heldout_splits() -> None:
    samples = [
        {
            "env_id": "arc",
            "source_score_file_name": "seed1",
            "trajectory_id": "a",
            "target_action": "D",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
        },
        {
            "env_id": "arc",
            "source_score_file_name": "seed1",
            "trajectory_id": "b",
            "target_action": "D",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "D", "logprob": -1.1}],
        },
    ]

    splits = split_samples(samples, min_split_samples=2)
    split_ids = {row["split_id"] for row in splits}

    assert "all" in split_ids
    assert "B->D:close_0_25" in split_ids
    assert "close_0_25" in split_ids
    assert "seed1" in split_ids


def test_run_split_score_writes_iteration_contract(tmp_path) -> None:
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "gain_policy_vpd_bridge_summary.json").write_text('{"status": "completed"}\n', encoding="utf-8")
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
    write_jsonl(bridge / "matched_controls.jsonl", [control_candidate(), no_edit_candidate()])
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "isolated",
                "target_action": "D",
                "scores": [{"action": "C", "logprob": -1.0}, {"action": "D", "logprob": -1.2}],
            },
            {
                "env_id": "arc",
                "trajectory_id": "correct",
                "target_action": "A",
                "scores": [{"action": "A", "logprob": -1.0}, {"action": "B", "logprob": -2.0}],
            },
        ],
    )

    summary = run_split_score(
        Namespace(
            bridge_dir=bridge,
            score_files=str(score_file),
            out_dir=tmp_path / "out",
            touch_rate_max=0.5,
            min_split_samples=1,
            max_prompt_tokens=1200,
        )
    )

    assert summary["promotion_ready_split_count"] >= 1
    assert (tmp_path / "out" / "next_iteration_contract.json").exists()

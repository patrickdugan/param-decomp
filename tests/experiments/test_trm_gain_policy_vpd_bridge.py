from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_decomp.experiments.trm_gain_policy_vpd_bridge import (
    build_logit_hook_candidate,
    build_matched_controls,
    build_vpd_search_plan,
    run_bridge,
)


def policy_state() -> dict:
    return {
        "policy_id": "vpd_edit_policy_arc_bootstrap_v1",
        "best_candidate": {
            "candidate_id": "gain:top_margin:max_0_25:penalty_0_25",
            "edit_family": "selective_top_margin_suppression",
            "fitness": 0.071161,
            "reward": 0.078387,
            "score_delta": 0.048387,
            "touch_rate": 0.16129,
            "source_feature_hint": "overselected_top_choice",
            "policy": {
                "id": "top_margin:max_0_25:penalty_0_25",
                "type": "top_margin_penalty",
                "max_margin": 0.25,
                "penalty": 0.25,
            },
        },
    }


def test_top_margin_policy_maps_to_logit_hook_candidate() -> None:
    candidate = build_logit_hook_candidate(policy_state(), "arc_challenge")

    assert candidate["edit_family"] == "selective_top_margin_suppression"
    assert candidate["trigger"]["top_minus_runner_up_max_margin"] == 0.25
    assert candidate["action"]["penalize"] == "current_top_action"
    assert candidate["claim_boundary"] == "logit-level route-rule analogue; not a VPD weight edit."


def test_matched_controls_include_broad_priors_and_no_edit() -> None:
    controls = build_matched_controls(policy_state(), "arc_challenge", ["B", "D"])

    assert [row["edit_family"] for row in controls] == [
        "broad_fixed_label_prior",
        "broad_fixed_label_prior",
        "abstain",
    ]
    assert all(row["matched_to"] == "logit_hook:top_margin:max_0_25:penalty_0_25" for row in controls)


def test_vpd_search_plan_is_not_claimable_before_feature_mapping() -> None:
    plan = build_vpd_search_plan(policy_state(), "arc_challenge")

    assert plan["claimable"] is False
    assert "random_feature_same_touch_rate" in plan["matched_controls_required"]
    assert plan["candidate_mechanisms"][0]["mechanism"] == "final_choice_logit_hook"


def test_run_bridge_writes_claim_safe_artifacts(tmp_path: Path) -> None:
    state_path = tmp_path / "policy_state.json"
    state_path.write_text(json.dumps(policy_state()), encoding="utf-8")
    args = argparse.Namespace(
        policy_state=state_path,
        out_dir=tmp_path / "out",
        env_id="arc_challenge",
        control_actions="B,D",
        max_prompt_tokens=1200,
    )

    summary = run_bridge(args)

    assert summary["status"] == "completed"
    assert summary["matched_control_count"] == 3
    assert (args.out_dir / "logit_hook_candidate.json").exists()
    assert (args.out_dir / "matched_controls.jsonl").exists()
    assert (args.out_dir / "vpd_search_plan.json").exists()

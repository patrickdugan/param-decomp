"""Bridge trained gain policies into VPD/logit-hook search candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl


DEFAULT_POLICY_STATE = Path(r"D:\Research_Engine\runs\trm_gain_function_memetic_arc_4seed_20260604\vpd_edit_policy_state.json")
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_vpd_bridge")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map a trained gain policy into claim-safe VPD/logit-hook candidates.")
    parser.add_argument("--policy-state", type=Path, default=DEFAULT_POLICY_STATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-id", default="arc_challenge")
    parser.add_argument("--control-actions", default="B,D")
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_actions(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def best_policy(policy_state: dict[str, Any]) -> dict[str, Any]:
    best = policy_state.get("best_candidate") or {}
    policy = best.get("policy") or {}
    if not policy:
        raise ValueError("Policy state does not contain best_candidate.policy")
    return policy


def build_logit_hook_candidate(policy_state: dict[str, Any], env_id: str) -> dict[str, Any]:
    policy = best_policy(policy_state)
    if policy.get("type") != "top_margin_penalty":
        raise ValueError(f"Unsupported best policy for logit bridge: {policy.get('type')}")
    return {
        "candidate_id": f"logit_hook:{policy['id']}",
        "source_policy_id": policy_state["policy_id"],
        "source_gain_candidate": (policy_state.get("best_candidate") or {}).get("candidate_id"),
        "env_id": env_id,
        "edit_family": "selective_top_margin_suppression",
        "hook_type": "candidate_logit_adjustment",
        "trigger": {
            "candidate_set_required": True,
            "top_minus_runner_up_max_margin": policy["max_margin"],
            "scope": "final_choice_action_scores",
        },
        "action": {
            "penalize": "current_top_action",
            "penalty": policy["penalty"],
        },
        "expected_behavior": "Suppress only close-call overselected top choices and preserve wider-margin predictions.",
        "claim_boundary": "logit-level route-rule analogue; not a VPD weight edit.",
        "claimable": True,
    }


def build_matched_controls(policy_state: dict[str, Any], env_id: str, actions: list[str]) -> list[dict[str, Any]]:
    policy = best_policy(policy_state)
    controls = []
    for action in actions:
        controls.append(
            {
                "candidate_id": f"control_fixed_label:{action}:penalty_{str(policy['penalty']).replace('.', '_')}",
                "source_policy_id": policy_state["policy_id"],
                "env_id": env_id,
                "edit_family": "broad_fixed_label_prior",
                "hook_type": "candidate_logit_adjustment",
                "trigger": {"candidate_set_required": True, "scope": "all_choice_action_scores"},
                "action": {"penalize": action, "penalty": policy["penalty"]},
                "matched_to": f"logit_hook:{policy['id']}",
                "claim_boundary": "matched broad-prior control for selective gain-function policy.",
                "claimable": True,
            }
        )
    controls.append(
        {
            "candidate_id": "control_no_edit",
            "source_policy_id": policy_state["policy_id"],
            "env_id": env_id,
            "edit_family": "abstain",
            "hook_type": "none",
            "trigger": {"candidate_set_required": False},
            "action": {"mode": "no_edit"},
            "matched_to": f"logit_hook:{policy['id']}",
            "claim_boundary": "null control for route-rule/logit-hook gain measurement.",
            "claimable": True,
        }
    )
    return controls


def build_vpd_search_plan(policy_state: dict[str, Any], env_id: str) -> dict[str, Any]:
    policy = best_policy(policy_state)
    return {
        "plan_id": f"vpd_search_from_{policy['id']}",
        "source_policy_id": policy_state["policy_id"],
        "env_id": env_id,
        "source_feature_hint": (policy_state.get("best_candidate") or {}).get("source_feature_hint"),
        "target_edit_family": "selective_top_margin_suppression",
        "candidate_mechanisms": [
            {
                "mechanism": "final_choice_logit_hook",
                "status": "ready",
                "maps_to": f"logit_hook:{policy['id']}",
            },
            {
                "mechanism": "decision-token_activation_suppression",
                "status": "candidate_generation_required",
                "selection_rule": "features active on close-call false top choices but inactive on wide-margin correct choices",
            },
            {
                "mechanism": "adapter_lora_row_vector_edit",
                "status": "candidate_generation_required",
                "selection_rule": "rows whose output raises current top choice on rescued close-call samples",
            },
        ],
        "matched_controls_required": [
            "broad_fixed_label_prior",
            "random_feature_same_touch_rate",
            "no_edit",
        ],
        "promotion_gate": [
            "fresh_seed_fitness > 0",
            "damage_count == 0 or damage_rate below matched controls",
            "touch_rate <= matched broad-prior controls",
            "holdout_delta >= 0",
        ],
        "claim_boundary": "Search plan only. No VPD feature or module hook has been causally validated yet.",
        "claimable": False,
    }


def compact_packet(summary: dict[str, Any], policy_state: dict[str, Any]) -> str:
    best = policy_state["best_candidate"]
    lines = [
        "TASK: Promote trained gain policy into VPD/logit-hook search.",
        "CURRENT STATE:",
        f"- policy_id: {policy_state['policy_id']}",
        f"- best_candidate: {best['candidate_id']}",
        f"- best_family: {best['edit_family']}",
        f"- fitness/reward/delta: {best['fitness']}/{best['reward']}/{best['score_delta']}",
        f"- touch_rate: {best['touch_rate']}",
        "BRIDGE OUTPUT:",
        f"- logit_hook_candidate: {summary['outputs']['logit_hook_candidate']}",
        f"- matched_controls: {summary['outputs']['matched_controls']}",
        f"- vpd_search_plan: {summary['outputs']['vpd_search_plan']}",
        "NEXT ACTION:",
        "- Score logit hook and matched controls on one fresh ARC card.",
        "- Generate VPD feature candidates only if logit hook remains positive.",
    ]
    return "\n".join(lines) + "\n"


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_state = read_json(args.policy_state)
    logit_hook = build_logit_hook_candidate(policy_state, args.env_id)
    controls = build_matched_controls(policy_state, args.env_id, parse_actions(args.control_actions))
    vpd_plan = build_vpd_search_plan(policy_state, args.env_id)
    outputs = {
        "summary": str(args.out_dir / "gain_policy_vpd_bridge_summary.json"),
        "logit_hook_candidate": str(args.out_dir / "logit_hook_candidate.json"),
        "matched_controls": str(args.out_dir / "matched_controls.jsonl"),
        "vpd_search_plan": str(args.out_dir / "vpd_search_plan.json"),
        "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
    }
    (args.out_dir / "logit_hook_candidate.json").write_text(json.dumps(logit_hook, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.out_dir / "matched_controls.jsonl", controls)
    (args.out_dir / "vpd_search_plan.json").write_text(json.dumps(vpd_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "policy_state": str(args.policy_state),
        "policy_id": policy_state["policy_id"],
        "env_id": args.env_id,
        "logit_hook_candidate_id": logit_hook["candidate_id"],
        "matched_control_count": len(controls),
        "claim_boundary": "Bridge artifacts only; VPD hook promotion requires fresh scoring and feature/module mapping.",
        "outputs": outputs,
    }
    packet = compact_packet(summary, policy_state)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "gain_policy_vpd_bridge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_bridge(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

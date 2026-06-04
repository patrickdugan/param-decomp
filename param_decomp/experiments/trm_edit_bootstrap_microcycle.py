"""Run a cached edit-bootstrap microcycle for VPD/TRM gain policies.

This is intentionally not a checkpoint editor. It tests whether accepted,
low-touch controller predicates can be stacked as stateful edits and then
searched again on the residual failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import (
    dedupe_samples,
    parse_floats,
    parse_paths,
    read_jsonl,
    write_jsonl,
)
from param_decomp.experiments.trm_gain_policy_condition_miner import (
    candidate_policies,
    policy_id,
    runner_up_action,
    top_action,
)
from param_decomp.experiments.trm_gain_policy_rescue_family_analyzer import margin_bucket, score_margin


DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604\choice_constrained_sample_scores.jsonl"
)
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_edit_bootstrap_microcycle_arc_20260604")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack accepted cached VPD/TRM edit predicates over residual failures.")
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--penalties", default="0.25,0.5,1")
    parser.add_argument("--margins", default="0.25,0.5,1")
    parser.add_argument("--touch-rate-max", type=float, default=0.25)
    parser.add_argument("--touch-penalty", type=float, default=0.05)
    parser.add_argument("--complexity-penalty", type=float, default=0.003)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def _score_key(row: dict[str, Any]) -> float:
    return float(row.get("adjusted_logprob", row.get("logprob", 0.0)))


def _with_adjusted_scores(sample: dict[str, Any], scores: list[dict[str, Any]]) -> dict[str, Any]:
    next_sample = deepcopy(sample)
    next_sample["scores"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"adjusted_score", "adjusted_logprob"}
        }
        | {"logprob": float(row.get("adjusted_logprob", row.get("logprob", 0.0)))}
        for row in scores
    ]
    next_sample["scores"].sort(key=lambda row: float(row["logprob"]), reverse=True)
    return next_sample


def current_state_samples(samples: list[dict[str, Any]], edit_stack: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = [deepcopy(sample) for sample in samples]
    for policy in edit_stack:
        state = [_with_adjusted_scores(sample, adjusted_scores(sample, policy)[0]) for sample in state]
    return state


def policy_active_on_state(sample: dict[str, Any], policy: dict[str, Any]) -> bool:
    if policy["type"] == "fixed_action_penalty":
        return any(str(row["action"]) == str(policy["action"]) for row in sample.get("scores", []))
    if policy["type"] != "conditional_top_margin_penalty":
        raise ValueError(f"Unsupported policy type: {policy['type']}")
    margin = score_margin(sample)
    if margin > float(policy["max_margin"]):
        return False
    if policy.get("top_action") not in {None, "any"} and top_action(sample) != policy["top_action"]:
        return False
    if policy.get("runner_up_action") not in {None, "any"} and runner_up_action(sample) != policy["runner_up_action"]:
        return False
    if policy.get("margin_bucket") not in {None, "any"} and margin_bucket(margin) != policy["margin_bucket"]:
        return False
    return True


def adjusted_scores(sample: dict[str, Any], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    scores = sorted(sample.get("scores", []), key=lambda row: float(row["logprob"]), reverse=True)
    if not scores:
        return [], False
    adjusted = []
    touched = False
    if policy["type"] == "fixed_action_penalty":
        for row in scores:
            active = str(row["action"]) == str(policy["action"])
            touched = touched or active
            adjusted.append({**row, "adjusted_logprob": float(row["logprob"]) - (float(policy["penalty"]) if active else 0.0)})
    elif policy["type"] == "conditional_top_margin_penalty":
        active = policy_active_on_state(sample, policy)
        current_top = top_action(sample)
        for row in scores:
            penalize = active and str(row["action"]) == current_top
            touched = touched or penalize
            adjusted.append({**row, "adjusted_logprob": float(row["logprob"]) - (float(policy["penalty"]) if penalize else 0.0)})
    else:
        raise ValueError(f"Unsupported policy type: {policy['type']}")
    adjusted.sort(key=_score_key, reverse=True)
    return adjusted, touched


def prediction(sample: dict[str, Any]) -> str:
    scores = sorted(sample.get("scores", []), key=lambda row: float(row["logprob"]), reverse=True)
    return str(scores[0]["action"]) if scores else ""


def state_score(samples: list[dict[str, Any]]) -> float:
    hits = sum(int(prediction(sample) == sample.get("target_action")) for sample in samples)
    return round(hits / max(1, len(samples)), 6)


def policy_complexity(policy: dict[str, Any]) -> int:
    if policy["type"] == "fixed_action_penalty":
        return 1
    return 1 + int(policy.get("top_action") not in {None, "any"}) + int(policy.get("runner_up_action") not in {None, "any"}) + int(policy.get("margin_bucket") not in {None, "any"})


def score_candidate(
    state_samples: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    round_index: int,
    candidate_family: str,
    touch_penalty: float,
    complexity_penalty: float,
) -> dict[str, Any]:
    current_hits = 0
    edited_hits = 0
    rescues = 0
    damages = 0
    touched = 0
    details = []
    for sample in state_samples:
        before = prediction(sample)
        before_hit = before == sample.get("target_action")
        rows, did_touch = adjusted_scores(sample, policy)
        after = str(rows[0]["action"]) if rows else ""
        after_hit = after == sample.get("target_action")
        current_hits += int(before_hit)
        edited_hits += int(after_hit)
        rescues += int(not before_hit and after_hit)
        damages += int(before_hit and not after_hit)
        touched += int(did_touch)
        details.append(
            {
                "env_id": sample.get("env_id", ""),
                "trajectory_id": sample.get("trajectory_id", ""),
                "target_action": sample.get("target_action"),
                "before_prediction": before,
                "after_prediction": after,
                "before_hit": before_hit,
                "after_hit": after_hit,
                "touched": did_touch,
                "top_margin": score_margin(sample),
                "margin_bucket": margin_bucket(score_margin(sample)),
            }
        )
    current_score = round(current_hits / max(1, len(state_samples)), 6)
    edited_score = round(edited_hits / max(1, len(state_samples)), 6)
    delta = round(edited_score - current_score, 6)
    reward = round(delta + 0.01 * (rescues - damages), 6)
    touch_rate = round(touched / max(1, len(state_samples)), 6)
    complexity = policy_complexity(policy)
    efficiency_reward = round(reward - touch_penalty * touch_rate - complexity_penalty * complexity, 6)
    policy_with_id = {**policy, "id": policy_id(policy)}
    return {
        "round_index": round_index,
        "candidate_id": policy_with_id["id"],
        "candidate_family": candidate_family,
        "policy": policy_with_id,
        "sample_count": len(state_samples),
        "current_score": current_score,
        "edited_score": edited_score,
        "delta": delta,
        "reward": reward,
        "efficiency_reward": efficiency_reward,
        "rescue_count": rescues,
        "damage_count": damages,
        "touched_count": touched,
        "touch_rate": touch_rate,
        "complexity": complexity,
        "details": details,
    }


def promotion_status(row: dict[str, Any], best_control: dict[str, Any] | None, touch_rate_max: float) -> tuple[bool, str]:
    if row["delta"] <= 0:
        return False, "no_positive_delta"
    if row["rescue_count"] <= row["damage_count"]:
        return False, "rescues_not_above_damages"
    if row["damage_count"] > 0:
        return False, "damage_detected"
    if row["touch_rate"] > touch_rate_max:
        return False, "touch_rate_above_cap"
    if best_control and row["reward"] <= best_control["reward"]:
        return False, "fixed_control_not_beaten"
    return True, "promoted"


def round_candidates(state_samples: list[dict[str, Any]], penalties: list[float], margins: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies = candidate_policies(state_samples, penalties, margins)
    edits = [
        policy
        for policy in policies
        if policy["type"] == "conditional_top_margin_penalty"
        and policy.get("top_action") not in {None, "any"}
        and policy.get("runner_up_action") not in {None, "any"}
    ]
    controls = [policy for policy in policies if policy["type"] == "fixed_action_penalty"]
    return edits, controls


def compact_packet(summary: dict[str, Any], rounds: list[dict[str, Any]]) -> str:
    last = rounds[-1] if rounds else {}
    lines = [
        "TASK: Decide whether cached VPD/TRM edits show ability bootstrap.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- samples: {summary['sample_count']}",
        f"- baseline_score: {summary['baseline_score']}",
        f"- final_score: {summary['final_score']}",
        f"- accepted_edit_count: {summary['accepted_edit_count']}",
        f"- stacked_bootstrap_success: {summary['stacked_bootstrap_success']}",
        "LATEST ROUND:",
        f"- index: {last.get('round_index')}",
        f"- selected: {last.get('selected_candidate_id')}",
        f"- promoted: {last.get('promoted')}",
        f"- stop_reason: {last.get('stop_reason')}",
        "PROMOTION CONTRACT:",
        "- conditional edit only; fixed-label policies are controls",
        "- require positive delta, zero damage, low touch, and reward above best fixed-label control",
        "- cached route-rule success is not a checkpoint or activation-local VPD edit claim",
    ]
    return "\n".join(lines) + "\n"


def run_microcycle(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    samples = dedupe_samples(raw_samples)
    penalties = parse_floats(args.penalties)
    margins = parse_floats(args.margins)
    baseline_score = state_score(samples)
    edit_stack: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    stop_reason = "round_limit"

    for round_index in range(1, args.rounds + 1):
        state_samples = current_state_samples(samples, edit_stack)
        edit_policies, control_policies = round_candidates(state_samples, penalties, margins)
        scored_edits = [
            score_candidate(
                state_samples,
                policy,
                round_index=round_index,
                candidate_family="conditional_edit",
                touch_penalty=args.touch_penalty,
                complexity_penalty=args.complexity_penalty,
            )
            for policy in edit_policies
        ]
        scored_controls = [
            score_candidate(
                state_samples,
                policy,
                round_index=round_index,
                candidate_family="fixed_label_control",
                touch_penalty=args.touch_penalty,
                complexity_penalty=args.complexity_penalty,
            )
            for policy in control_policies
        ]
        scored_edits.sort(key=lambda row: (row["reward"], row["delta"], row["efficiency_reward"], -row["touch_rate"]), reverse=True)
        scored_controls.sort(key=lambda row: (row["reward"], row["delta"], row["efficiency_reward"], -row["touch_rate"]), reverse=True)
        best_edit = scored_edits[0] if scored_edits else None
        best_control = scored_controls[0] if scored_controls else None
        candidate_rows.extend(scored_edits)
        candidate_rows.extend(scored_controls)
        if not best_edit:
            stop_reason = "no_candidate_edit"
            break
        promoted, reason = promotion_status(best_edit, best_control, args.touch_rate_max)
        round_row = {
            "round_index": round_index,
            "state_score": state_score(state_samples),
            "candidate_count": len(scored_edits),
            "control_count": len(scored_controls),
            "selected_candidate_id": best_edit["candidate_id"],
            "selected_delta": best_edit["delta"],
            "selected_reward": best_edit["reward"],
            "selected_touch_rate": best_edit["touch_rate"],
            "selected_rescues": best_edit["rescue_count"],
            "selected_damages": best_edit["damage_count"],
            "best_control_id": best_control["candidate_id"] if best_control else None,
            "best_control_delta": best_control["delta"] if best_control else 0.0,
            "best_control_reward": best_control["reward"] if best_control else 0.0,
            "best_control_touch_rate": best_control["touch_rate"] if best_control else 0.0,
            "promoted": promoted,
            "stop_reason": reason,
            "policy": best_edit["policy"],
        }
        round_rows.append(round_row)
        if not promoted:
            stop_reason = reason
            break
        edit_stack.append(best_edit["policy"])

    final_samples = current_state_samples(samples, edit_stack)
    final_score = state_score(final_samples)
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "score_files": [str(path) for path in score_files],
        "raw_sample_count": len(raw_samples),
        "sample_count": len(samples),
        "round_limit": args.rounds,
        "round_count": len(round_rows),
        "baseline_score": baseline_score,
        "final_score": final_score,
        "cumulative_delta": round(final_score - baseline_score, 6),
        "accepted_edit_count": len(edit_stack),
        "first_edit_success": bool(edit_stack),
        "stacked_bootstrap_success": bool(len(edit_stack) >= 2 and final_score > baseline_score),
        "stop_reason": stop_reason,
        "edit_stack": edit_stack,
        "claim_boundary": "Cached route-rule bootstrap only; not an activation-local VPD edit or checkpoint mutation.",
        "paper_interpretation": (
            "stacked cached edits improved the current state under fixed-label controls"
            if len(edit_stack) >= 2 and final_score > baseline_score
            else "bootstrap gate did not promote a stacked route-rule edit under fixed-label controls"
        ),
        "outputs": {
            "summary": str(args.out_dir / "bootstrap_summary.json"),
            "rounds": str(args.out_dir / "bootstrap_rounds.jsonl"),
            "candidate_scores": str(args.out_dir / "bootstrap_candidate_scores.jsonl"),
            "state": str(args.out_dir / "bootstrap_state.json"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    state = {
        "edit_stack": edit_stack,
        "final_samples": [
            {
                "env_id": sample.get("env_id", ""),
                "trajectory_id": sample.get("trajectory_id", ""),
                "target_action": sample.get("target_action"),
                "prediction": prediction(sample),
                "hit": prediction(sample) == sample.get("target_action"),
                "scores": sample.get("scores", []),
            }
            for sample in final_samples
        ],
    }
    write_jsonl(args.out_dir / "bootstrap_rounds.jsonl", round_rows)
    write_jsonl(args.out_dir / "bootstrap_candidate_scores.jsonl", candidate_rows)
    (args.out_dir / "bootstrap_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    packet = compact_packet(summary, round_rows)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_microcycle(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

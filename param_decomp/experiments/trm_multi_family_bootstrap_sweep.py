"""Sweep cached edit-bootstrap microcycles across failure families."""

from __future__ import annotations

import argparse
import json
import sys
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
from param_decomp.experiments.trm_edit_bootstrap_microcycle import (
    current_state_samples,
    prediction,
    promotion_status,
    round_candidates,
    score_candidate,
    state_score,
)
from param_decomp.experiments.trm_gain_policy_condition_miner import runner_up_action, top_action
from param_decomp.experiments.trm_gain_policy_rescue_family_analyzer import margin_bucket, score_margin


DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604\choice_constrained_sample_scores.jsonl"
)
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_multi_family_bootstrap_sweep_arc_20260604")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run edit-bootstrap microcycles over many cached failure families.")
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--penalties", default="0.25,0.5,1")
    parser.add_argument("--margins", default="0.25,0.5,1")
    parser.add_argument("--touch-rate-max", type=float, default=0.35)
    parser.add_argument("--touch-penalty", type=float, default=0.05)
    parser.add_argument("--complexity-penalty", type=float, default=0.003)
    parser.add_argument("--min-family-size", type=int, default=1)
    parser.add_argument("--max-families", type=int, default=24)
    parser.add_argument("--max-negative-per-family", type=int, default=24)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def sample_key(sample: dict[str, Any]) -> str:
    return f"{sample.get('env_id', '')}:{sample.get('trajectory_id', '')}"


def baseline_hit(sample: dict[str, Any]) -> bool:
    return prediction(sample) == sample.get("target_action")


def failure_family_key(sample: dict[str, Any]) -> str:
    return f"{top_action(sample)}->{sample.get('target_action')}:{runner_up_action(sample) or 'none'}:{margin_bucket(score_margin(sample))}"


def family_fields(family_key: str) -> dict[str, str]:
    transition, runner, bucket = family_key.split(":", 2)
    top, target = transition.split("->", 1)
    return {"top_action": top, "target_action": target, "runner_up_action": runner, "margin_bucket": bucket}


def failure_families(samples: list[dict[str, Any]], min_family_size: int, max_families: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        if baseline_hit(sample):
            continue
        grouped.setdefault(failure_family_key(sample), []).append(sample)
    rows = []
    for key, members in grouped.items():
        fields = family_fields(key)
        rows.append(
            {
                "family_key": key,
                "failure_count": len(members),
                "top_action": fields["top_action"],
                "target_action": fields["target_action"],
                "runner_up_action": fields["runner_up_action"],
                "margin_bucket": fields["margin_bucket"],
                "mean_margin": round(sum(score_margin(sample) for sample in members) / max(1, len(members)), 6),
                "trajectory_ids": [sample.get("trajectory_id", "") for sample in members],
            }
        )
    rows = [row for row in rows if row["failure_count"] >= min_family_size]
    rows.sort(key=lambda row: (row["failure_count"], -row["mean_margin"], row["family_key"]), reverse=True)
    return rows[:max_families]


def family_slice(samples: list[dict[str, Any]], family: dict[str, Any], max_negative: int) -> list[dict[str, Any]]:
    positives = [sample for sample in samples if not baseline_hit(sample) and failure_family_key(sample) == family["family_key"]]
    positive_keys = {sample_key(sample) for sample in positives}
    negatives = []
    for sample in samples:
        if sample_key(sample) in positive_keys:
            continue
        if not baseline_hit(sample):
            continue
        same_top = top_action(sample) == family["top_action"]
        same_target = sample.get("target_action") == family["target_action"]
        same_runner = runner_up_action(sample) == family["runner_up_action"]
        if same_top or same_target or same_runner:
            negatives.append(sample)
    if len(negatives) < max_negative:
        for sample in samples:
            if sample_key(sample) in positive_keys or sample in negatives or not baseline_hit(sample):
                continue
            negatives.append(sample)
            if len(negatives) >= max_negative:
                break
    negatives.sort(key=sample_key)
    return positives + negatives[:max_negative]


def run_family_cycle(
    family: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    rounds: int,
    penalties: list[float],
    margins: list[float],
    touch_rate_max: float,
    touch_penalty: float,
    complexity_penalty: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    edit_stack: list[dict[str, Any]] = []
    round_rows = []
    candidate_rows = []
    stop_reason = "round_limit"
    baseline_score = state_score(samples)
    for round_index in range(1, rounds + 1):
        state_samples = current_state_samples(samples, edit_stack)
        edits, controls = round_candidates(state_samples, penalties, margins)
        scored_edits = [
            score_candidate(
                state_samples,
                policy,
                round_index=round_index,
                candidate_family="conditional_edit",
                touch_penalty=touch_penalty,
                complexity_penalty=complexity_penalty,
            )
            for policy in edits
        ]
        scored_controls = [
            score_candidate(
                state_samples,
                policy,
                round_index=round_index,
                candidate_family="fixed_label_control",
                touch_penalty=touch_penalty,
                complexity_penalty=complexity_penalty,
            )
            for policy in controls
        ]
        scored_edits.sort(key=lambda row: (row["reward"], row["delta"], row["efficiency_reward"], -row["touch_rate"]), reverse=True)
        scored_controls.sort(key=lambda row: (row["reward"], row["delta"], row["efficiency_reward"], -row["touch_rate"]), reverse=True)
        best_edit = scored_edits[0] if scored_edits else None
        best_control = scored_controls[0] if scored_controls else None
        for row in scored_edits + scored_controls:
            candidate_rows.append({k: v for k, v in row.items() if k != "details"} | {"family_key": family["family_key"]})
        if not best_edit:
            stop_reason = "no_candidate_edit"
            break
        promoted, reason = promotion_status(best_edit, best_control, touch_rate_max)
        round_row = {
            "family_key": family["family_key"],
            "round_index": round_index,
            "state_score": state_score(state_samples),
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
    final_score = state_score(current_state_samples(samples, edit_stack))
    result = {
        **family,
        "sample_count": len(samples),
        "negative_count": max(0, len(samples) - family["failure_count"]),
        "baseline_score": baseline_score,
        "final_score": final_score,
        "cumulative_delta": round(final_score - baseline_score, 6),
        "accepted_edit_count": len(edit_stack),
        "first_edit_success": bool(edit_stack),
        "stacked_bootstrap_success": bool(len(edit_stack) >= 2 and final_score > baseline_score),
        "stop_reason": stop_reason,
        "edit_stack": edit_stack,
    }
    return result, round_rows, candidate_rows


def gain_policy_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "family_key": result["family_key"],
        "top_action": result["top_action"],
        "runner_up_action": result["runner_up_action"],
        "target_action": result["target_action"],
        "margin_bucket": result["margin_bucket"],
        "failure_count": result["failure_count"],
        "sample_count": result["sample_count"],
        "accepted_edit_count": result["accepted_edit_count"],
        "cumulative_delta": result["cumulative_delta"],
        "first_edit_success": result["first_edit_success"],
        "stacked_bootstrap_success": result["stacked_bootstrap_success"],
        "stop_reason": result["stop_reason"],
        "label": "stacked_gain" if result["stacked_bootstrap_success"] else ("first_edit_gain" if result["first_edit_success"] else "blocked"),
    }


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None) -> str:
    lines = [
        "TASK: Broaden VPD/TRM edit learning beyond one residual family.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- families_scored: {summary['family_count']}",
        f"- first_edit_family_count: {summary['first_edit_family_count']}",
        f"- stacked_bootstrap_family_count: {summary['stacked_bootstrap_family_count']}",
        f"- control_blocked_family_count: {summary['control_blocked_family_count']}",
        "BEST FAMILY:",
        f"- key: {best.get('family_key') if best else None}",
        f"- cumulative_delta: {best.get('cumulative_delta') if best else 0}",
        f"- accepted_edit_count: {best.get('accepted_edit_count') if best else 0}",
        "NEXT ACTION:",
        "- Train or rank a gain policy over family features, then hold out families before activation-local VPD mapping.",
    ]
    return "\n".join(lines) + "\n"


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    samples = dedupe_samples(raw_samples)
    families = failure_families(samples, args.min_family_size, args.max_families)
    family_results = []
    all_rounds = []
    all_candidates = []
    penalties = parse_floats(args.penalties)
    margins = parse_floats(args.margins)
    for family in families:
        subset = family_slice(samples, family, args.max_negative_per_family)
        result, rounds, candidates = run_family_cycle(
            family,
            subset,
            rounds=args.rounds,
            penalties=penalties,
            margins=margins,
            touch_rate_max=args.touch_rate_max,
            touch_penalty=args.touch_penalty,
            complexity_penalty=args.complexity_penalty,
        )
        family_results.append(result)
        all_rounds.extend(rounds)
        all_candidates.extend(candidates)
    family_results.sort(key=lambda row: (row["stacked_bootstrap_success"], row["accepted_edit_count"], row["cumulative_delta"], row["failure_count"]), reverse=True)
    best = family_results[0] if family_results else None
    policy_rows = [gain_policy_row(row) for row in family_results]
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "score_files": [str(path) for path in score_files],
        "raw_sample_count": len(raw_samples),
        "sample_count": len(samples),
        "family_count": len(family_results),
        "first_edit_family_count": sum(int(row["first_edit_success"]) for row in family_results),
        "stacked_bootstrap_family_count": sum(int(row["stacked_bootstrap_success"]) for row in family_results),
        "control_blocked_family_count": sum(int(row["stop_reason"] == "fixed_control_not_beaten") for row in family_results),
        "best_family_key": best["family_key"] if best else None,
        "best_cumulative_delta": best["cumulative_delta"] if best else 0.0,
        "best_accepted_edit_count": best["accepted_edit_count"] if best else 0,
        "claim_boundary": "Cached multi-family route-rule sweep only; not an activation-local VPD edit or checkpoint mutation.",
        "outputs": {
            "summary": str(args.out_dir / "multi_family_bootstrap_summary.json"),
            "family_scores": str(args.out_dir / "family_bootstrap_scores.jsonl"),
            "rounds": str(args.out_dir / "family_bootstrap_rounds.jsonl"),
            "candidate_scores": str(args.out_dir / "family_bootstrap_candidate_scores.jsonl"),
            "gain_policy_rows": str(args.out_dir / "gain_policy_training_rows.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    write_jsonl(args.out_dir / "family_bootstrap_scores.jsonl", family_results)
    write_jsonl(args.out_dir / "family_bootstrap_rounds.jsonl", all_rounds)
    write_jsonl(args.out_dir / "family_bootstrap_candidate_scores.jsonl", all_candidates)
    write_jsonl(args.out_dir / "gain_policy_training_rows.jsonl", policy_rows)
    packet = compact_packet(summary, best)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "multi_family_bootstrap_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_sweep(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

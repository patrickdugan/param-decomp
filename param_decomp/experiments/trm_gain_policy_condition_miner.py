"""Mine lower-touch conditional policies from gain-hook split evidence."""

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
    baseline_hit,
    dedupe_samples,
    parse_floats,
    parse_paths,
    read_jsonl,
    write_jsonl,
)
from param_decomp.experiments.trm_gain_policy_rescue_family_analyzer import margin_bucket, score_margin


DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604\choice_constrained_sample_scores.jsonl"
)
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_condition_miner")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine stricter conditional policies from cached candidate scores.")
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--penalties", default="0.25,0.5,1")
    parser.add_argument("--margins", default="0.25,0.5,1")
    parser.add_argument("--touch-penalty", type=float, default=0.05)
    parser.add_argument("--complexity-penalty", type=float, default=0.003)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def top_scores(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(sample.get("scores", []), key=lambda row: float(row["logprob"]), reverse=True)


def top_action(sample: dict[str, Any]) -> str:
    scores = top_scores(sample)
    return str(scores[0]["action"]) if scores else ""


def runner_up_action(sample: dict[str, Any]) -> str:
    scores = top_scores(sample)
    return str(scores[1]["action"]) if len(scores) > 1 else ""


def policy_id(policy: dict[str, Any]) -> str:
    if policy["type"] == "conditional_top_margin_penalty":
        parts = [
            "cond_top_margin",
            f"top_{policy.get('top_action', 'any')}",
            f"runner_{policy.get('runner_up_action', 'any')}",
            f"bucket_{policy.get('margin_bucket', 'any')}",
            f"max_{str(policy['max_margin']).replace('.', '_')}",
            f"penalty_{str(policy['penalty']).replace('.', '_')}",
        ]
        return ":".join(parts)
    if policy["type"] == "fixed_action_penalty":
        return f"fixed:{policy['action']}:penalty_{str(policy['penalty']).replace('.', '_')}"
    raise ValueError(f"Unsupported policy type: {policy['type']}")


def policy_active(sample: dict[str, Any], policy: dict[str, Any]) -> bool:
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


def adjusted_prediction(sample: dict[str, Any], policy: dict[str, Any]) -> tuple[str, bool]:
    scores = top_scores(sample)
    if not scores:
        return "", False
    adjusted = []
    touched = False
    if policy["type"] == "fixed_action_penalty":
        for row in scores:
            active = str(row["action"]) == str(policy["action"])
            touched = touched or active
            adjusted.append({**row, "adjusted_score": float(row["logprob"]) - (float(policy["penalty"]) if active else 0.0)})
    else:
        active = policy_active(sample, policy)
        current_top = top_action(sample)
        for row in scores:
            penalize = active and str(row["action"]) == current_top
            touched = touched or penalize
            adjusted.append({**row, "adjusted_score": float(row["logprob"]) - (float(policy["penalty"]) if penalize else 0.0)})
    adjusted.sort(key=lambda row: row["adjusted_score"], reverse=True)
    return str(adjusted[0]["action"]), touched


def score_policy(samples: list[dict[str, Any]], policy: dict[str, Any], *, touch_penalty: float, complexity_penalty: float) -> dict[str, Any]:
    baseline_hits = 0
    policy_hits = 0
    rescues = 0
    damages = 0
    touched = 0
    details = []
    for sample in samples:
        base_hit = baseline_hit(sample)
        prediction, did_touch = adjusted_prediction(sample, policy)
        hit = prediction == sample["target_action"]
        baseline_hits += int(base_hit)
        policy_hits += int(hit)
        rescues += int(not base_hit and hit)
        damages += int(base_hit and not hit)
        touched += int(did_touch)
        details.append(
            {
                "trajectory_id": sample["trajectory_id"],
                "target_action": sample["target_action"],
                "baseline_prediction": top_action(sample),
                "runner_up_prediction": runner_up_action(sample),
                "prediction": prediction,
                "baseline_hit": base_hit,
                "hit": hit,
                "touched": did_touch,
                "top_margin": score_margin(sample),
                "margin_bucket": margin_bucket(score_margin(sample)),
            }
        )
    baseline_score = baseline_hits / max(1, len(samples))
    policy_score = policy_hits / max(1, len(samples))
    delta = round(policy_score - baseline_score, 6)
    reward = round(delta + 0.01 * (rescues - damages), 6)
    touch_rate = round(touched / max(1, len(samples)), 6)
    complexity = 1 + int(policy.get("top_action") not in {None, "any"}) + int(policy.get("runner_up_action") not in {None, "any"}) + int(policy.get("margin_bucket") not in {None, "any"})
    efficiency_reward = round(reward - touch_penalty * touch_rate - complexity_penalty * complexity, 6)
    return {
        "candidate_id": policy_id(policy),
        "policy": {**policy, "id": policy_id(policy)},
        "sample_count": len(samples),
        "baseline_score": baseline_score,
        "policy_score": policy_score,
        "delta": delta,
        "reward": reward,
        "efficiency_reward": efficiency_reward,
        "rescue_count": rescues,
        "damage_count": damages,
        "touched_count": touched,
        "touch_rate": touch_rate,
        "complexity": complexity,
        "accepted": bool(delta > 0 and rescues > damages and damages == 0),
        "details": details,
    }


def candidate_policies(samples: list[dict[str, Any]], penalties: list[float], margins: list[float]) -> list[dict[str, Any]]:
    actions = sorted({top_action(sample) for sample in samples if top_action(sample)})
    runners = sorted({runner_up_action(sample) for sample in samples if runner_up_action(sample)})
    buckets = sorted({margin_bucket(score_margin(sample)) for sample in samples})
    policies = []
    for action in actions:
        for penalty in penalties:
            policies.append({"type": "fixed_action_penalty", "action": action, "penalty": penalty})
    for penalty in penalties:
        for max_margin in margins:
            policies.append(
                {
                    "type": "conditional_top_margin_penalty",
                    "top_action": "any",
                    "runner_up_action": "any",
                    "margin_bucket": "any",
                    "max_margin": max_margin,
                    "penalty": penalty,
                }
            )
            for action in actions:
                policies.append(
                    {
                        "type": "conditional_top_margin_penalty",
                        "top_action": action,
                        "runner_up_action": "any",
                        "margin_bucket": "any",
                        "max_margin": max_margin,
                        "penalty": penalty,
                    }
                )
            for runner in runners:
                policies.append(
                    {
                        "type": "conditional_top_margin_penalty",
                        "top_action": "any",
                        "runner_up_action": runner,
                        "margin_bucket": "any",
                        "max_margin": max_margin,
                        "penalty": penalty,
                    }
                )
            for bucket in buckets:
                policies.append(
                    {
                        "type": "conditional_top_margin_penalty",
                        "top_action": "any",
                        "runner_up_action": "any",
                        "margin_bucket": bucket,
                        "max_margin": max_margin,
                        "penalty": penalty,
                    }
                )
            for action in actions:
                for runner in runners:
                    policies.append(
                        {
                            "type": "conditional_top_margin_penalty",
                            "top_action": action,
                            "runner_up_action": runner,
                            "margin_bucket": "any",
                            "max_margin": max_margin,
                            "penalty": penalty,
                        }
                    )
    deduped = {policy_id(policy): policy for policy in policies}
    return list(deduped.values())


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None) -> str:
    lines = [
        "TASK: Continue continuous VPD-TRM edit learning by selecting a lower-touch predicate.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- samples: {summary['sample_count']}",
        f"- candidates_scored: {summary['candidate_count']}",
        f"- accepted_candidates: {summary['accepted_count']}",
        "BEST CONDITION:",
        f"- id: {best.get('candidate_id') if best else None}",
        f"- delta/reward/efficiency: {best.get('delta') if best else 0}/{best.get('reward') if best else 0}/{best.get('efficiency_reward') if best else 0}",
        f"- rescues/damages: {best.get('rescue_count') if best else 0}/{best.get('damage_count') if best else 0}",
        f"- touch_rate: {best.get('touch_rate') if best else 0}",
        "NEXT ACTION:",
        "- Map the best lower-touch condition to activation-local VPD feature candidates.",
        "- Keep broad-prior controls as the reward gate; do not claim a VPD edit until reward beats controls.",
    ]
    return "\n".join(lines) + "\n"


def run_miner(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    samples = dedupe_samples(raw_samples)
    policies = candidate_policies(samples, parse_floats(args.penalties), parse_floats(args.margins))
    rows = [score_policy(samples, policy, touch_penalty=args.touch_penalty, complexity_penalty=args.complexity_penalty) for policy in policies]
    rows.sort(key=lambda row: (row["accepted"], row["efficiency_reward"], row["reward"], -row["touch_rate"]), reverse=True)
    best = rows[0] if rows else None
    accepted = [row for row in rows if row["accepted"]]
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "score_files": [str(path) for path in score_files],
        "raw_sample_count": len(raw_samples),
        "sample_count": len(samples),
        "candidate_count": len(rows),
        "accepted_count": len(accepted),
        "best_candidate_id": best["candidate_id"] if best else None,
        "best_delta": best["delta"] if best else 0.0,
        "best_reward": best["reward"] if best else 0.0,
        "best_efficiency_reward": best["efficiency_reward"] if best else 0.0,
        "best_touch_rate": best["touch_rate"] if best else 0.0,
        "claim_boundary": "Cached condition mining only; selected predicate is a controller prior, not a VPD feature edit.",
        "outputs": {
            "summary": str(args.out_dir / "condition_miner_summary.json"),
            "condition_scores": str(args.out_dir / "condition_scores.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    write_jsonl(args.out_dir / "condition_scores.jsonl", rows)
    packet = compact_packet(summary, best)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "condition_miner_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_miner(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

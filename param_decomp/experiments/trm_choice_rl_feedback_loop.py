"""Tight cached RL-style feedback loop for choice-rule discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_choice_rl_feedback_loop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run immediate reward updates over cached choice score cards.")
    parser.add_argument("--score-files", required=True, help="Comma or semicolon-separated sample-score files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--penalties", default="0.5,1,2,4")
    parser.add_argument("--margins", default="0.25,0.5,1,2")
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def parse_paths(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in raw.replace(",", ";").split(";") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dedupe_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for sample in samples:
        key = f"{sample.get('env_id', '')}:{sample.get('trajectory_id', '')}"
        if key not in deduped:
            deduped[key] = sample
    return list(deduped.values())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def baseline_hit(sample: dict[str, Any]) -> bool:
    return bool(sample.get("scores")) and sample["scores"][0]["action"] == sample["target_action"]


def adjusted_prediction(sample: dict[str, Any], policy: dict[str, Any]) -> tuple[str, bool]:
    scores = sorted(sample["scores"], key=lambda row: float(row["logprob"]), reverse=True)
    if not scores:
        return "", False
    adjusted = []
    touched = False
    if policy["type"] == "fixed_action_penalty":
        for row in scores:
            active = row["action"] == policy["action"]
            touched = touched or active
            adjusted.append({**row, "adjusted_score": float(row["logprob"]) - (policy["penalty"] if active else 0.0)})
    elif policy["type"] == "top_margin_penalty":
        margin = float("inf")
        if len(scores) > 1:
            margin = float(scores[0]["logprob"]) - float(scores[1]["logprob"])
        active = margin <= policy["max_margin"]
        top_action = scores[0]["action"]
        for row in scores:
            penalize = active and row["action"] == top_action
            touched = touched or penalize
            adjusted.append({**row, "adjusted_score": float(row["logprob"]) - (policy["penalty"] if penalize else 0.0)})
    else:
        raise ValueError(f"Unsupported policy type: {policy['type']}")
    adjusted.sort(key=lambda row: row["adjusted_score"], reverse=True)
    return adjusted[0]["action"], touched


def score_policy(samples: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
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
        touched += int(did_touch)
        rescues += int(not base_hit and hit)
        damages += int(base_hit and not hit)
        details.append(
            {
                "trajectory_id": sample["trajectory_id"],
                "target_action": sample["target_action"],
                "baseline_prediction": sample["scores"][0]["action"] if sample.get("scores") else "",
                "prediction": prediction,
                "baseline_hit": base_hit,
                "hit": hit,
                "touched": did_touch,
            }
        )
    baseline_score = baseline_hits / max(1, len(samples))
    policy_score = policy_hits / max(1, len(samples))
    delta = round(policy_score - baseline_score, 6)
    reward = round(delta + 0.01 * (rescues - damages), 6)
    return {
        "policy": policy,
        "sample_count": len(samples),
        "baseline_score": baseline_score,
        "policy_score": policy_score,
        "delta": delta,
        "reward": reward,
        "rescue_count": rescues,
        "damage_count": damages,
        "touched_count": touched,
        "accepted": delta > 0 and rescues > damages,
        "details": details,
    }


def candidate_policies(samples: list[dict[str, Any]], penalties: list[float], margins: list[float]) -> list[dict[str, Any]]:
    actions = sorted({row["action"] for sample in samples for row in sample.get("scores", [])})
    policies = []
    for action in actions:
        for penalty in penalties:
            policies.append(
                {
                    "id": f"fixed:{action}:penalty_{str(penalty).replace('.', '_')}",
                    "type": "fixed_action_penalty",
                    "action": action,
                    "penalty": penalty,
                }
            )
    for margin in margins:
        for penalty in penalties:
            policies.append(
                {
                    "id": f"top_margin:max_{str(margin).replace('.', '_')}:penalty_{str(penalty).replace('.', '_')}",
                    "type": "top_margin_penalty",
                    "max_margin": margin,
                    "penalty": penalty,
                }
            )
    return policies


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None, score_files: list[Path]) -> str:
    best_policy = best["policy"] if best else {}
    replay_hint = None
    if best:
        replay_hint = next((row for row in best["details"] if not row["baseline_hit"] and row["hit"]), None)
    lines = [
        "TASK: Choose next constrained-choice TRM rule probe.",
        "CURRENT STATE:",
        f"- envs: {', '.join(summary['env_ids'])}",
        f"- score_files: {len(score_files)} cached cards",
        f"- samples: {summary['sample_count']}",
        f"- candidates_scored: {summary['candidate_count']}",
        f"- accepted_candidates: {summary['accepted_count']}",
        "BEST CANDIDATE:",
        f"- id: {best_policy.get('id')}",
        f"- type: {best_policy.get('type')}",
        f"- reward: {best.get('reward') if best else 0}",
        f"- delta: {best.get('delta') if best else 0}",
        f"- rescues/damages: {best.get('rescue_count') if best else 0}/{best.get('damage_count') if best else 0}",
        "REWARD CONTRACT:",
        "- immediate_reward = score_delta + 0.01 * (rescues - damages)",
        "- accept only if delta > 0 and rescues > damages",
        "REPLAY HINT:",
        "- none" if not replay_hint else (
            f"- {replay_hint['trajectory_id']}: baseline {replay_hint['baseline_prediction']} -> "
            f"policy {replay_hint['prediction']} matched target {replay_hint['target_action']}"
        ),
        "NEXT ACTION:",
        "- If best is accepted, schedule one live score card on a fresh seed.",
        "- If fixed action varies by seed, prefer top-margin family over fixed anti-label rule.",
        "MCP RESOURCES:",
        f"- score_history: {summary['outputs']['reward_history']}",
        f"- self_model: {summary['outputs']['policy_state']}",
        f"- replay_candidates: {summary['outputs']['replay_candidates']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    samples = dedupe_samples(raw_samples)
    policies = candidate_policies(samples, parse_floats(args.penalties), parse_floats(args.margins))
    rewards = [score_policy(samples, policy) for policy in policies]
    rewards.sort(key=lambda row: (row["accepted"], row["reward"], row["delta"]), reverse=True)
    best = rewards[0] if rewards else None
    accepted = [row for row in rewards if row["accepted"]]
    replay_candidates = []
    if best:
        for row in best["details"]:
            if not row["baseline_hit"] and row["hit"]:
                replay_candidates.append(row)
    summary = {
        "status": "completed",
        "score_files": [str(path) for path in score_files],
        "env_ids": sorted({sample.get("env_id", "") for sample in samples}),
        "raw_sample_count": len(raw_samples),
        "sample_count": len(samples),
        "deduped_sample_count": len(samples),
        "candidate_count": len(rewards),
        "accepted_count": len(accepted),
        "best_policy": best["policy"] if best else None,
        "best_reward": best["reward"] if best else 0.0,
        "best_delta": best["delta"] if best else 0.0,
        "outputs": {
            "summary": str(args.out_dir / "choice_rl_feedback_summary.json"),
            "reward_history": str(args.out_dir / "reward_history.jsonl"),
            "policy_state": str(args.out_dir / "rl_policy_state.json"),
            "replay_candidates": str(args.out_dir / "replay_candidates.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    policy_state = {
        "recent_pass_rate": best["policy_score"] if best else 0.0,
        "baseline_pass_rate": best["baseline_score"] if best else 0.0,
        "accepted_policy_count": len(accepted),
        "recurring_failure_modes": ["overselected_top_choice", "label_bias"],
        "best_policy": best["policy"] if best else None,
        "prompt_budget_target_tokens": args.max_prompt_tokens,
    }
    write_jsonl(args.out_dir / "reward_history.jsonl", rewards)
    write_jsonl(args.out_dir / "replay_candidates.jsonl", replay_candidates)
    (args.out_dir / "rl_policy_state.json").write_text(
        json.dumps(policy_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    packet = compact_packet(summary, best, score_files)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "choice_rl_feedback_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

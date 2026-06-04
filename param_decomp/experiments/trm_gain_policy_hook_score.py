"""Score bridged gain policies as logit-hook candidates against controls."""

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
    parse_paths,
    read_jsonl,
    score_policy,
    write_jsonl,
)


DEFAULT_BRIDGE_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_vpd_bridge_arc_20260604")
DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604\choice_constrained_sample_scores.jsonl"
)
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_hook_score")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a bridged gain-policy logit hook and matched controls.")
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--touch-rate-max", type=float, default=0.25)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_from_hook(candidate: dict[str, Any]) -> dict[str, Any]:
    action = candidate.get("action") or {}
    trigger = candidate.get("trigger") or {}
    if candidate.get("hook_type") == "none":
        return {"id": "no_edit", "type": "fixed_action_penalty", "action": "__never__", "penalty": 0.0}
    if action.get("penalize") == "current_top_action":
        margin = float(trigger["top_minus_runner_up_max_margin"])
        penalty = float(action["penalty"])
        return {
            "id": f"top_margin:max_{str(margin).replace('.', '_')}:penalty_{str(penalty).replace('.', '_')}",
            "type": "top_margin_penalty",
            "max_margin": margin,
            "penalty": penalty,
        }
    if isinstance(action.get("penalize"), str):
        penalty = float(action["penalty"])
        label = str(action["penalize"])
        return {
            "id": f"fixed:{label}:penalty_{str(penalty).replace('.', '_')}",
            "type": "fixed_action_penalty",
            "action": label,
            "penalty": penalty,
        }
    raise ValueError(f"Unsupported hook candidate: {candidate.get('candidate_id')}")


def score_candidate(samples: list[dict[str, Any]], candidate: dict[str, Any], *, touch_rate_max: float) -> dict[str, Any]:
    scored = score_policy(samples, policy_from_hook(candidate))
    touch_rate = round(scored["touched_count"] / max(1, scored["sample_count"]), 6)
    accepted = bool(scored["accepted"] and scored["damage_count"] == 0 and touch_rate <= touch_rate_max)
    return {
        "candidate_id": candidate["candidate_id"],
        "edit_family": candidate["edit_family"],
        "hook_type": candidate["hook_type"],
        "policy": scored["policy"],
        "baseline_score": scored["baseline_score"],
        "hook_score": scored["policy_score"],
        "delta": scored["delta"],
        "reward": scored["reward"],
        "rescue_count": scored["rescue_count"],
        "damage_count": scored["damage_count"],
        "touched_count": scored["touched_count"],
        "sample_count": scored["sample_count"],
        "touch_rate": touch_rate,
        "accepted_hook": accepted,
        "failed_gates": [
            name
            for name, passed in {
                "positive_delta": scored["delta"] > 0.0,
                "rescues_exceed_damages": scored["rescue_count"] > scored["damage_count"],
                "zero_damage": scored["damage_count"] == 0,
                "touch_rate": touch_rate <= touch_rate_max,
            }.items()
            if not passed
        ],
        "claim_boundary": candidate.get("claim_boundary", "logit-hook score; not a VPD weight edit."),
        "details": scored["details"],
    }


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None) -> str:
    replay_hint = None
    if best:
        replay_hint = next((row for row in best["details"] if not row["baseline_hit"] and row["hit"]), None)
    lines = [
        "TASK: Decide whether to promote a gain-policy hook toward VPD feature search.",
        "CURRENT STATE:",
        f"- bridge_run: {summary['bridge_dir']}",
        f"- samples: {summary['sample_count']} from {summary['score_file_count']} cached score files",
        f"- candidates_scored: {summary['candidate_count']}",
        "BEST HOOK:",
        f"- id: {best.get('candidate_id') if best else None}",
        f"- family: {best.get('edit_family') if best else None}",
        f"- delta/reward: {best.get('delta') if best else 0}/{best.get('reward') if best else 0}",
        f"- rescues/damages: {best.get('rescue_count') if best else 0}/{best.get('damage_count') if best else 0}",
        f"- touch_rate: {best.get('touch_rate') if best else 0}",
        f"- accepted_hook: {best.get('accepted_hook') if best else False}",
        f"- promotion_ready: {summary['promotion_ready']}",
        "REPLAY HINT:",
        "- none" if not replay_hint else (
            f"- {replay_hint['trajectory_id']}: baseline {replay_hint['baseline_prediction']} -> "
            f"hook {replay_hint['prediction']} matched target {replay_hint['target_action']}"
        ),
        "NEXT ACTION:",
        "- If accepted_hook is true and beats controls, generate a small VPD feature candidate set.",
        "- If controls beat it, treat this as a route-rule prior rather than a feature-local effect.",
    ]
    return "\n".join(lines) + "\n"


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logit_hook = read_json(args.bridge_dir / "logit_hook_candidate.json")
    controls = read_jsonl(args.bridge_dir / "matched_controls.jsonl")
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    samples = dedupe_samples(raw_samples)
    rows = [score_candidate(samples, row, touch_rate_max=args.touch_rate_max) for row in [logit_hook, *controls]]
    rows.sort(key=lambda row: (row["accepted_hook"], row["reward"], row["delta"], -row["touch_rate"]), reverse=True)
    best = rows[0] if rows else None
    hook_row = next((row for row in rows if row["candidate_id"] == logit_hook["candidate_id"]), None)
    control_best = next((row for row in rows if row["candidate_id"] != logit_hook["candidate_id"]), None)
    hook_beats_controls = bool(
        hook_row
        and hook_row["accepted_hook"]
        and (
            control_best is None
            or (
                float(hook_row["reward"]) > float(control_best["reward"])
                and float(hook_row["delta"]) >= float(control_best["delta"])
            )
        )
    )
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "bridge_dir": str(args.bridge_dir),
        "score_files": [str(path) for path in score_files],
        "score_file_count": len(score_files),
        "raw_sample_count": len(raw_samples),
        "sample_count": len(samples),
        "candidate_count": len(rows),
        "touch_rate_max": args.touch_rate_max,
        "best_candidate_id": best["candidate_id"] if best else None,
        "best_delta": best["delta"] if best else 0.0,
        "best_reward": best["reward"] if best else 0.0,
        "logit_hook_delta": hook_row["delta"] if hook_row else None,
        "logit_hook_accepted": hook_row["accepted_hook"] if hook_row else False,
        "best_control_candidate_id": control_best["candidate_id"] if control_best else None,
        "best_control_delta": control_best["delta"] if control_best else None,
        "promotion_ready": hook_beats_controls,
        "claim_boundary": "Cached logit-hook scoring only; VPD feature/module hook promotion still requires causal feature mapping.",
        "outputs": {
            "summary": str(args.out_dir / "gain_policy_hook_score_summary.json"),
            "scores": str(args.out_dir / "hook_scores.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    write_jsonl(args.out_dir / "hook_scores.jsonl", rows)
    packet = compact_packet(summary, best)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "gain_policy_hook_score_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_score(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

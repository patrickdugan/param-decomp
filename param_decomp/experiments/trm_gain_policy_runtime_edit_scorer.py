"""Score activation-gated gain-policy runtime edit trials against controls."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import dedupe_samples, parse_paths, read_jsonl, write_jsonl
from param_decomp.experiments.trm_gain_policy_condition_miner import (
    runner_up_action,
    score_margin,
    top_action,
)
from param_decomp.experiments.trm_gain_policy_feature_search_packet import policy_from_condition_id


DEFAULT_FEATURE_MAP_RUN = Path(r"D:\Research_Engine\runs\trm_gain_policy_activation_feature_map_arc_20260604")
DEFAULT_ACTIVATION_STATS = Path(r"D:\Research_Engine\runs\trm_gain_policy_activation_capture_arc_20260604\activation_stats.jsonl")
DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604\choice_constrained_sample_scores.jsonl"
)
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_runtime_edit_score_arc_20260604")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score activation-gated gain-policy runtime edit trials.")
    parser.add_argument("--feature-map-run", type=Path, default=DEFAULT_FEATURE_MAP_RUN)
    parser.add_argument("--activation-stats", type=Path, default=DEFAULT_ACTIVATION_STATS)
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--controller-touch-rate", type=float, default=0.064516)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_id(sample: dict[str, Any]) -> str:
    return f"{sample.get('env_id', '')}:{sample.get('trajectory_id', '')}"


def top_scores(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(sample.get("scores", []), key=lambda row: float(row["logprob"]), reverse=True)


def baseline_prediction(sample: dict[str, Any]) -> str:
    scores = top_scores(sample)
    return str(scores[0]["action"]) if scores else ""


def adjusted_prediction(sample: dict[str, Any], penalize_action: str | None, penalty: float) -> str:
    scores = top_scores(sample)
    adjusted = []
    for row in scores:
        active = penalize_action is not None and str(row["action"]) == penalize_action
        adjusted.append({**row, "adjusted_score": float(row["logprob"]) - (penalty if active else 0.0)})
    adjusted.sort(key=lambda row: row["adjusted_score"], reverse=True)
    return str(adjusted[0]["action"]) if adjusted else ""


def activation_by_module(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_module: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(path):
        module_path = str(row["module_path"])
        sid = str(row["sample_id"])
        by_module.setdefault(module_path, {})[sid] = row
    return by_module


def activation_rule(module_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    positives = [float(row["activation_value"]) for row in module_rows.values() if str(row.get("label")) == "positive_rescue"]
    negatives = [float(row["activation_value"]) for row in module_rows.values() if str(row.get("label", "")).startswith("negative_")]
    if not positives or not negatives:
        return {"positive_mean": 0.0, "negative_mean": 0.0, "threshold": 0.0, "direction": "unrankable"}
    positive_mean = sum(positives) / len(positives)
    negative_mean = sum(negatives) / len(negatives)
    return {
        "positive_mean": round(positive_mean, 6),
        "negative_mean": round(negative_mean, 6),
        "threshold": round((positive_mean + negative_mean) / 2.0, 6),
        "direction": "positive_high" if positive_mean >= negative_mean else "positive_low",
    }


def activation_active(row: dict[str, Any] | None, rule: dict[str, Any]) -> bool:
    if row is None or rule["direction"] == "unrankable":
        return False
    value = float(row["activation_value"])
    threshold = float(rule["threshold"])
    return value >= threshold if rule["direction"] == "positive_high" else value <= threshold


def condition_active(sample: dict[str, Any], policy: dict[str, Any]) -> bool:
    if top_action(sample) != policy["top_action"]:
        return False
    if runner_up_action(sample) != policy["runner_up_action"]:
        return False
    return score_margin(sample) <= float(policy["max_margin"])


def score_rows(
    samples: list[dict[str, Any]],
    *,
    candidate_id: str,
    family: str,
    active_sample_ids: set[str],
    penalty: float,
    penalize_mode: str,
    controller_touch_rate: float,
) -> dict[str, Any]:
    baseline_hits = 0
    edited_hits = 0
    rescues = 0
    damages = 0
    touched = 0
    details = []
    for sample in samples:
        sid = sample_id(sample)
        base_pred = baseline_prediction(sample)
        base_hit = base_pred == str(sample["target_action"])
        active = sid in active_sample_ids
        penalized_action = None
        if active:
            penalized_action = base_pred if penalize_mode == "current_top" else penalize_mode
        pred = adjusted_prediction(sample, penalized_action, penalty)
        hit = pred == str(sample["target_action"])
        baseline_hits += int(base_hit)
        edited_hits += int(hit)
        rescues += int(not base_hit and hit)
        damages += int(base_hit and not hit)
        touched += int(active)
        details.append(
            {
                "sample_id": sid,
                "target_action": sample["target_action"],
                "baseline_prediction": base_pred,
                "prediction": pred,
                "baseline_hit": base_hit,
                "hit": hit,
                "touched": active,
            }
        )
    baseline_score = baseline_hits / max(1, len(samples))
    edited_score = edited_hits / max(1, len(samples))
    delta = round(edited_score - baseline_score, 6)
    reward = round(delta + 0.01 * (rescues - damages), 6)
    touch_rate = round(touched / max(1, len(samples)), 6)
    return {
        "candidate_id": candidate_id,
        "edit_family": family,
        "baseline_score": baseline_score,
        "edited_score": edited_score,
        "delta": delta,
        "reward": reward,
        "rescue_count": rescues,
        "damage_count": damages,
        "touched_count": touched,
        "sample_count": len(samples),
        "touch_rate": touch_rate,
        "accepted_runtime_candidate": bool(delta > 0 and rescues > damages and damages == 0 and touch_rate <= controller_touch_rate),
        "details": details,
    }


def fixed_control(samples: list[dict[str, Any]], action: str, penalty: float, controller_touch_rate: float) -> dict[str, Any]:
    active = {sample_id(sample) for sample in samples if any(str(row["action"]) == action for row in sample.get("scores", []))}
    return score_rows(
        samples,
        candidate_id=f"control_fixed_label:{action}:penalty_{str(penalty).replace('.', '_')}",
        family="fixed_label_control",
        active_sample_ids=active,
        penalty=penalty,
        penalize_mode=action,
        controller_touch_rate=controller_touch_rate,
    )


def no_edit_control(samples: list[dict[str, Any]], controller_touch_rate: float) -> dict[str, Any]:
    return score_rows(
        samples,
        candidate_id="control_no_edit",
        family="no_edit",
        active_sample_ids=set(),
        penalty=0.0,
        penalize_mode="current_top",
        controller_touch_rate=controller_touch_rate,
    )


def matched_touch_control(samples: list[dict[str, Any]], touch_count: int, penalty: float, controller_touch_rate: float) -> dict[str, Any]:
    ordered = sorted(sample_id(sample) for sample in samples)
    active = set(reversed(ordered[: max(0, touch_count)]))
    return score_rows(
        samples,
        candidate_id=f"control_matched_touch:current_top:penalty_{str(penalty).replace('.', '_')}",
        family="matched_touch_control",
        active_sample_ids=active,
        penalty=penalty,
        penalize_mode="current_top",
        controller_touch_rate=controller_touch_rate,
    )


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None) -> str:
    lines = [
        "TASK: Decide whether ranked activation-map trials promote to runtime-edit evidence.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- samples: {summary['sample_count']}",
        f"- trial_count: {summary['trial_count']}",
        f"- control_count: {summary['control_count']}",
        f"- promotion_ready: {summary['promotion_ready']}",
        "BEST TRIAL:",
        f"- id: {best.get('candidate_id') if best else None}",
        f"- delta/reward: {best.get('delta') if best else 0}/{best.get('reward') if best else 0}",
        f"- rescues/damages: {best.get('rescue_count') if best else 0}/{best.get('damage_count') if best else 0}",
        f"- touch_rate: {best.get('touch_rate') if best else 0}",
        "NEXT ACTION:",
        "- If promotion_ready, run the same edit on fresh ARC cards.",
        "- If not, report activation-local runtime edit as a boundary result.",
    ]
    return "\n".join(lines) + "\n"


def run_runtime_score(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_summary = read_json(args.feature_map_run / "activation_feature_map_summary.json")
    feature_map = read_json(args.feature_map_run / "activation_feature_map.json")
    trials = read_jsonl(args.feature_map_run / "activation_edit_trials.jsonl")
    policy = policy_from_condition_id(str(feature_summary["condition_id"]))
    penalty = float(policy["penalty"])
    activation = activation_by_module(args.activation_stats)
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    all_samples = dedupe_samples(raw_samples)
    stat_sample_ids = {sid for module_rows in activation.values() for sid in module_rows}
    samples = [sample for sample in all_samples if sample_id(sample) in stat_sample_ids]
    if not samples:
        raise RuntimeError("No score samples overlap with activation stats.")
    samples_by_id = {sample_id(sample): sample for sample in samples}

    trial_rows = []
    for trial in trials:
        module_path = str(trial["module_path"])
        module_rows = activation.get(module_path, {})
        rule = activation_rule(module_rows)
        active = {
            sid
            for sid, row in module_rows.items()
            if sid in samples_by_id
            if activation_active(row, rule)
            and condition_active(samples_by_id[sid], policy)
        }
        scored = score_rows(
            samples,
            candidate_id=str(trial["trial_id"]),
            family="activation_gated_runtime_proxy",
            active_sample_ids=active,
            penalty=penalty,
            penalize_mode="current_top",
            controller_touch_rate=args.controller_touch_rate,
        )
        trial_rows.append({**scored, "module_path": module_path, "activation_rule": rule, "claim_boundary": "Activation-gated cached runtime proxy; no checkpoint mutation."})

    best_trial = max(trial_rows, key=lambda row: (row["accepted_runtime_candidate"], row["reward"], row["delta"], -row["touch_rate"]), default=None)
    controls = [
        fixed_control(samples, policy["top_action"], penalty, args.controller_touch_rate),
        fixed_control(samples, policy["runner_up_action"], penalty, args.controller_touch_rate),
        no_edit_control(samples, args.controller_touch_rate),
        matched_touch_control(samples, int(best_trial["touched_count"] if best_trial else 0), penalty, args.controller_touch_rate),
    ]
    best_control = max(controls, key=lambda row: (row["reward"], row["delta"], -row["touch_rate"]), default=None)
    promotion_ready = bool(
        best_trial
        and best_trial["accepted_runtime_candidate"]
        and (
            best_control is None
            or (
                float(best_trial["reward"]) > float(best_control["reward"])
                and float(best_trial["delta"]) >= float(best_control["delta"])
            )
        )
    )
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "feature_map_run": str(args.feature_map_run),
        "activation_stats": str(args.activation_stats),
        "score_files": [str(path) for path in score_files],
        "condition_id": feature_summary["condition_id"],
        "feature_map_entry_count": len(feature_map.get("entries") or []),
        "raw_sample_count": len(raw_samples),
        "sample_count": len(samples),
        "trial_count": len(trial_rows),
        "control_count": len(controls),
        "controller_touch_rate": args.controller_touch_rate,
        "best_trial_id": best_trial["candidate_id"] if best_trial else None,
        "best_trial_delta": best_trial["delta"] if best_trial else 0.0,
        "best_trial_reward": best_trial["reward"] if best_trial else 0.0,
        "best_trial_touch_rate": best_trial["touch_rate"] if best_trial else 0.0,
        "best_control_id": best_control["candidate_id"] if best_control else None,
        "best_control_delta": best_control["delta"] if best_control else 0.0,
        "best_control_reward": best_control["reward"] if best_control else 0.0,
        "promotion_ready": promotion_ready,
        "claim_boundary": "Scores activation-gated cached runtime proxies; not a VPD checkpoint or live eval edit claim.",
        "outputs": {
            "summary": str(args.out_dir / "runtime_edit_summary.json"),
            "runtime_edit_scores": str(args.out_dir / "runtime_edit_scores.jsonl"),
            "matched_controls": str(args.out_dir / "matched_controls.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    write_jsonl(args.out_dir / "runtime_edit_scores.jsonl", trial_rows)
    write_jsonl(args.out_dir / "matched_controls.jsonl", controls)
    packet = compact_packet(summary, best_trial)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "runtime_edit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_runtime_score(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

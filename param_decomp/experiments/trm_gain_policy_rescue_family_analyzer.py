"""Analyze rescue families for gain-policy hooks and matched controls."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import (
    adjusted_prediction,
    baseline_hit,
    dedupe_samples,
    parse_paths,
    read_jsonl,
    write_jsonl,
)
from param_decomp.experiments.trm_gain_policy_hook_score import policy_from_hook, read_json


DEFAULT_BRIDGE_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_vpd_bridge_arc_20260604")
DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604\choice_constrained_sample_scores.jsonl"
)
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_rescue_families")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster hook rescues against matched controls.")
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def score_margin(sample: dict[str, Any]) -> float:
    scores = sorted(sample.get("scores", []), key=lambda row: float(row["logprob"]), reverse=True)
    if len(scores) < 2:
        return float("inf")
    return round(float(scores[0]["logprob"]) - float(scores[1]["logprob"]), 6)


def margin_bucket(margin: float) -> str:
    if margin <= 0.25:
        return "close_0_25"
    if margin <= 0.5:
        return "close_0_5"
    if margin <= 1.0:
        return "medium_1_0"
    return "wide"


def candidate_rows(bridge_dir: Path) -> list[dict[str, Any]]:
    return [read_json(bridge_dir / "logit_hook_candidate.json"), *read_jsonl(bridge_dir / "matched_controls.jsonl")]


def candidate_outcome(sample: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    prediction, touched = adjusted_prediction(sample, policy_from_hook(candidate))
    base_hit = baseline_hit(sample)
    hit = prediction == sample["target_action"]
    return {
        "candidate_id": candidate["candidate_id"],
        "edit_family": candidate["edit_family"],
        "prediction": prediction,
        "hit": hit,
        "touched": touched,
        "rescue": bool(not base_hit and hit),
        "damage": bool(base_hit and not hit),
    }


def sample_analysis(sample: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scores = sorted(sample.get("scores", []), key=lambda row: float(row["logprob"]), reverse=True)
    baseline_prediction = scores[0]["action"] if scores else ""
    margin = score_margin(sample)
    outcomes = [candidate_outcome(sample, candidate) for candidate in candidates]
    hook = outcomes[0]
    controls = outcomes[1:]
    rescuing_controls = [row["candidate_id"] for row in controls if row["rescue"]]
    damaging_controls = [row["candidate_id"] for row in controls if row["damage"]]
    return {
        "trajectory_id": sample["trajectory_id"],
        "env_id": sample.get("env_id", ""),
        "target_action": sample["target_action"],
        "baseline_prediction": baseline_prediction,
        "baseline_hit": baseline_hit(sample),
        "top_margin": margin,
        "margin_bucket": margin_bucket(margin),
        "family_key": f"{baseline_prediction}->{sample['target_action']}:{margin_bucket(margin)}",
        "hook_prediction": hook["prediction"],
        "hook_touched": hook["touched"],
        "hook_rescue": hook["rescue"],
        "hook_damage": hook["damage"],
        "control_rescue_count": len(rescuing_controls),
        "control_damage_count": len(damaging_controls),
        "rescuing_controls": rescuing_controls,
        "damaging_controls": damaging_controls,
        "isolated_hook_rescue": bool(hook["rescue"] and not rescuing_controls),
        "shared_hook_rescue": bool(hook["rescue"] and rescuing_controls),
        "outcomes": outcomes,
    }


def aggregate_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["family_key"]].append(row)
    aggregates = []
    for key, members in groups.items():
        hook_rescues = [row for row in members if row["hook_rescue"]]
        isolated = [row for row in members if row["isolated_hook_rescue"]]
        shared = [row for row in members if row["shared_hook_rescue"]]
        hook_damages = [row for row in members if row["hook_damage"]]
        control_rescue_total = sum(int(row["control_rescue_count"] > 0) for row in members)
        aggregates.append(
            {
                "family_key": key,
                "sample_count": len(members),
                "baseline_prediction": members[0]["baseline_prediction"],
                "target_action": members[0]["target_action"],
                "margin_bucket": members[0]["margin_bucket"],
                "hook_rescue_count": len(hook_rescues),
                "isolated_hook_rescue_count": len(isolated),
                "shared_hook_rescue_count": len(shared),
                "hook_damage_count": len(hook_damages),
                "control_rescue_sample_count": control_rescue_total,
                "specificity_score": round((len(isolated) + 0.25 * len(shared) - len(hook_damages)) / max(1, len(members)), 6),
                "example_trajectory_ids": [row["trajectory_id"] for row in members[:5]],
            }
        )
    aggregates.sort(
        key=lambda row: (
            row["isolated_hook_rescue_count"],
            row["hook_rescue_count"],
            row["specificity_score"],
            -row["hook_damage_count"],
        ),
        reverse=True,
    )
    return aggregates


def next_edit_contract(summary: dict[str, Any], families: list[dict[str, Any]]) -> dict[str, Any]:
    best = families[0] if families else {}
    return {
        "task": "Generate a narrower VPD/TRM edit from hook rescue-family evidence.",
        "objective": "Beat broad-prior controls by rescuing close-margin failures with lower touch and no damage.",
        "current_gate": {
            "hook_positive": summary["hook_rescue_count"] > summary["hook_damage_count"],
            "promotion_blocked_by_control_tie": True,
            "required_next_condition": "isolated_hook_rescue_count > 0 or hook delta exceeds best broad-prior control on held-out family split",
        },
        "preferred_family": best.get("family_key"),
        "feature_search_hint": {
            "positive_set": "samples with hook_rescue=true and margin_bucket close_0_25",
            "negative_set": "same baseline label where broad controls rescue or already-correct close calls",
            "candidate_feature": "activation/module row active on close-call wrong-top-choice states, inactive on broad fixed-label prior states",
        },
        "claim_boundary": "Family analysis only; no VPD feature has been identified or edited yet.",
    }


def compact_packet(summary: dict[str, Any], contract: dict[str, Any]) -> str:
    lines = [
        "TASK: Continue VPD-TRM edit learning from hook rescue-family evidence.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- samples: {summary['sample_count']}",
        f"- hook rescues/damages: {summary['hook_rescue_count']}/{summary['hook_damage_count']}",
        f"- isolated/shared hook rescues: {summary['isolated_hook_rescue_count']}/{summary['shared_hook_rescue_count']}",
        f"- promotion_ready: {summary['promotion_ready']}",
        "NEXT EDIT CONTRACT:",
        f"- preferred_family: {contract['preferred_family']}",
        f"- positive_set: {contract['feature_search_hint']['positive_set']}",
        f"- negative_set: {contract['feature_search_hint']['negative_set']}",
        f"- required_condition: {contract['current_gate']['required_next_condition']}",
    ]
    return "\n".join(lines) + "\n"


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_rows(args.bridge_dir)
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    samples = dedupe_samples(raw_samples)
    rows = [sample_analysis(sample, candidates) for sample in samples]
    families = aggregate_families(rows)
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "bridge_dir": str(args.bridge_dir),
        "score_files": [str(path) for path in score_files],
        "raw_sample_count": len(raw_samples),
        "sample_count": len(samples),
        "family_count": len(families),
        "hook_rescue_count": sum(int(row["hook_rescue"]) for row in rows),
        "hook_damage_count": sum(int(row["hook_damage"]) for row in rows),
        "isolated_hook_rescue_count": sum(int(row["isolated_hook_rescue"]) for row in rows),
        "shared_hook_rescue_count": sum(int(row["shared_hook_rescue"]) for row in rows),
        "promotion_ready": False,
        "claim_boundary": "Rescue-family analysis only; next edit still needs VPD feature/module mapping.",
        "outputs": {
            "summary": str(args.out_dir / "rescue_family_summary.json"),
            "sample_families": str(args.out_dir / "sample_families.jsonl"),
            "family_aggregates": str(args.out_dir / "family_aggregates.jsonl"),
            "next_edit_contract": str(args.out_dir / "next_edit_contract.json"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    contract = next_edit_contract(summary, families)
    packet = compact_packet(summary, contract)
    write_jsonl(args.out_dir / "sample_families.jsonl", rows)
    write_jsonl(args.out_dir / "family_aggregates.jsonl", families)
    (args.out_dir / "next_edit_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "rescue_family_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_analysis(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

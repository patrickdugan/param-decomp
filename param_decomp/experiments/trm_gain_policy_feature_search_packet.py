"""Build feature-search packets from validated conditional gain policies."""

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
    parse_paths,
    read_jsonl,
    write_jsonl,
)
from param_decomp.experiments.trm_gain_policy_condition_miner import (
    adjusted_prediction,
    policy_active,
    runner_up_action,
    score_margin,
    top_action,
)


DEFAULT_CV_RUN = Path(r"D:\Research_Engine\runs\trm_gain_policy_condition_cv_arc_4seed_20260604")
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_feature_search_packet")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact VPD feature-search packets from condition CV artifacts.")
    parser.add_argument("--cv-run", type=Path, default=DEFAULT_CV_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-positive", type=int, default=16)
    parser.add_argument("--max-negative", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_from_condition_id(condition_id: str) -> dict[str, Any]:
    parts = condition_id.split(":")
    if len(parts) != 6 or parts[0] != "cond_top_margin":
        raise ValueError(f"Unsupported condition id: {condition_id}")
    values = {}
    for part in parts[1:]:
        key, value = part.split("_", 1)
        values[key] = value
    return {
        "id": condition_id,
        "type": "conditional_top_margin_penalty",
        "top_action": values["top"],
        "runner_up_action": values["runner"],
        "margin_bucket": values["bucket"],
        "max_margin": float(values["max"].replace("_", ".")),
        "penalty": float(values["penalty"].replace("_", ".")),
    }


def fold_rows(cv_run: Path) -> list[dict[str, Any]]:
    return read_jsonl(cv_run / "condition_cv_folds.jsonl")


def best_condition_id(folds: list[dict[str, Any]]) -> str:
    ranked = sorted(
        folds,
        key=lambda row: (
            row.get("beats_control", False),
            row.get("ties_control_lower_touch", False),
            row.get("heldout_accepted", False),
            float(row.get("heldout_reward") or 0.0),
        ),
        reverse=True,
    )
    if not ranked or not ranked[0].get("train_best_candidate_id"):
        raise ValueError("CV run does not contain a train_best_candidate_id")
    return str(ranked[0]["train_best_candidate_id"])


def load_samples(paths: list[Path]) -> list[dict[str, Any]]:
    samples = []
    seen = set()
    for path in paths:
        for sample in read_jsonl(path):
            key = f"{sample.get('env_id', '')}:{sample.get('trajectory_id', '')}"
            if key in seen:
                continue
            seen.add(key)
            samples.append({**sample, "source_score_file": str(path), "source_score_file_name": path.parent.name})
    return samples


def classify_sample(sample: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    base_hit = baseline_hit(sample)
    prediction, touched = adjusted_prediction(sample, policy)
    hit = prediction == sample["target_action"]
    active = policy_active(sample, policy)
    top = top_action(sample)
    runner = runner_up_action(sample)
    margin = score_margin(sample)
    label = "background"
    if active and touched and not base_hit and hit:
        label = "positive_rescue"
    elif active and touched and base_hit:
        label = "negative_correct_touched"
    elif active and touched and not hit:
        label = "negative_uncorrected_active"
    elif top == policy["top_action"] and runner == policy["runner_up_action"] and not active:
        label = "negative_same_pair_inactive"
    return {
        "sample_id": f"{sample.get('env_id', '')}:{sample.get('trajectory_id', '')}",
        "trajectory_id": sample.get("trajectory_id"),
        "env_id": sample.get("env_id", ""),
        "source_score_file_name": sample.get("source_score_file_name"),
        "label": label,
        "target_action": sample["target_action"],
        "baseline_prediction": top,
        "runner_up_prediction": runner,
        "condition_prediction": prediction,
        "baseline_hit": base_hit,
        "condition_hit": hit,
        "condition_active": active,
        "condition_touched": touched,
        "top_margin": margin,
    }


def select_contrast_rows(rows: list[dict[str, Any]], max_positive: int, max_negative: int) -> list[dict[str, Any]]:
    positives = [row for row in rows if row["label"] == "positive_rescue"][:max_positive]
    negatives = [
        row
        for row in rows
        if row["label"] in {"negative_correct_touched", "negative_uncorrected_active", "negative_same_pair_inactive"}
    ][:max_negative]
    return [*positives, *negatives]


def activation_probe_contract(condition_id: str, policy: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Find activation-local VPD features that reproduce the validated conditional gain policy.",
        "condition_id": condition_id,
        "controller_policy": policy,
        "positive_set": "positive_rescue rows: baseline miss corrected by condition",
        "negative_sets": [
            "negative_correct_touched: condition would touch an already-correct case",
            "negative_uncorrected_active: condition is active but does not rescue",
            "negative_same_pair_inactive: same top/runner pair outside the active margin",
        ],
        "candidate_feature_rule": "Rank modules/components by positive-vs-negative activation contrast, then test runtime edits that suppress wrong top-action D only on positive-like states.",
        "promotion_gate": [
            "feature-local edit reproduces held-out condition reward",
            "feature-local edit beats fixed-label controls on at least one held-out score file",
            "damage_count == 0",
            "touch_rate <= controller predicate touch_rate",
        ],
        "mcp_resources": {
            "contrast_sets": summary["outputs"]["contrast_sets"],
            "feature_search_manifest": summary["outputs"]["feature_search_manifest"],
            "prompt_packet": summary["outputs"]["prompt_packet"],
        },
        "claim_boundary": "Feature-search packet only; no activation feature has been scored or edited yet.",
    }


def compact_packet(summary: dict[str, Any], condition_id: str, contract: dict[str, Any]) -> str:
    lines = [
        "TASK: Map validated conditional gain policy to VPD feature-search candidates.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- condition: {condition_id}",
        f"- positive_count: {summary['positive_count']}",
        f"- negative_count: {summary['negative_count']}",
        f"- prompt_budget: {summary['prompt_packet_est_tokens']} est tokens",
        "FEATURE SEARCH CONTRACT:",
        f"- positive_set: {contract['positive_set']}",
        f"- candidate_rule: {contract['candidate_feature_rule']}",
        f"- promotion_gate: {contract['promotion_gate'][0]}; {contract['promotion_gate'][1]}",
        "NEXT ACTION:",
        "- Run activation/component contrast ranking over this packet.",
        "- Keep fixed-label controls as the reward gate.",
    ]
    return "\n".join(lines) + "\n"


def run_packet(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv_summary = read_json(args.cv_run / "condition_cv_summary.json")
    folds = fold_rows(args.cv_run)
    condition_id = best_condition_id(folds)
    policy = policy_from_condition_id(condition_id)
    score_files = parse_paths(";".join(cv_summary["score_files"]))
    rows = [classify_sample(sample, policy) for sample in load_samples(score_files)]
    contrast_rows = select_contrast_rows(rows, args.max_positive, args.max_negative)
    positive_count = sum(int(row["label"] == "positive_rescue") for row in rows)
    negative_count = sum(
        int(row["label"] in {"negative_correct_touched", "negative_uncorrected_active", "negative_same_pair_inactive"})
        for row in rows
    )
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "cv_run": str(args.cv_run),
        "condition_id": condition_id,
        "sample_count": len(rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "contrast_row_count": len(contrast_rows),
        "claim_boundary": "Packet generator only; activation-local VPD feature scoring remains pending.",
        "outputs": {
            "summary": str(args.out_dir / "feature_search_packet_summary.json"),
            "contrast_sets": str(args.out_dir / "contrast_sets.jsonl"),
            "all_sample_labels": str(args.out_dir / "all_sample_labels.jsonl"),
            "feature_search_manifest": str(args.out_dir / "feature_search_manifest.json"),
            "activation_probe_contract": str(args.out_dir / "activation_probe_contract.json"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    contract = activation_probe_contract(condition_id, policy, summary)
    packet = compact_packet({**summary, "prompt_packet_est_tokens": 0}, condition_id, contract)
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    packet = compact_packet(summary, condition_id, contract)
    manifest = {
        "condition_id": condition_id,
        "controller_policy": policy,
        "cv_summary": cv_summary,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "contrast_labels": sorted({row["label"] for row in contrast_rows}),
        "claim_boundary": summary["claim_boundary"],
    }
    write_jsonl(args.out_dir / "all_sample_labels.jsonl", rows)
    write_jsonl(args.out_dir / "contrast_sets.jsonl", contrast_rows)
    (args.out_dir / "feature_search_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "activation_probe_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    (args.out_dir / "feature_search_packet_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_packet(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score gain-policy hooks against controls across family and held-out splits."""

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
    dedupe_samples,
    parse_paths,
    read_jsonl,
    write_jsonl,
)
from param_decomp.experiments.trm_gain_policy_hook_score import read_json, score_candidate
from param_decomp.experiments.trm_gain_policy_rescue_family_analyzer import candidate_rows, margin_bucket, score_margin


DEFAULT_BRIDGE_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_vpd_bridge_arc_20260604")
DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604\choice_constrained_sample_scores.jsonl"
)
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_family_split_score")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score hook-vs-control deltas over family and held-out slices.")
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--touch-rate-max", type=float, default=0.25)
    parser.add_argument("--min-split-samples", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def baseline_prediction(sample: dict[str, Any]) -> str:
    scores = sorted(sample.get("scores", []), key=lambda row: float(row["logprob"]), reverse=True)
    return str(scores[0]["action"]) if scores else ""


def family_key(sample: dict[str, Any]) -> str:
    return f"{baseline_prediction(sample)}->{sample['target_action']}:{margin_bucket(score_margin(sample))}"


def load_samples_with_source(score_files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in score_files:
        for sample in read_jsonl(path):
            rows.append({**sample, "source_score_file": str(path), "source_score_file_name": path.parent.name})
    return rows


def split_samples(samples: list[dict[str, Any]], min_split_samples: int) -> list[dict[str, Any]]:
    splits = [{"split_id": "all", "split_type": "all", "samples": dedupe_samples(samples)}]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[("family_key", family_key(sample))].append(sample)
        groups[("margin_bucket", margin_bucket(score_margin(sample)))].append(sample)
        groups[("heldout_score_file", str(sample["source_score_file_name"]))].append(sample)
    for (split_type, split_id), members in sorted(groups.items()):
        deduped = dedupe_samples(members)
        if len(deduped) >= min_split_samples:
            splits.append({"split_id": split_id, "split_type": split_type, "samples": deduped})
    return splits


def score_split(split: dict[str, Any], candidates: list[dict[str, Any]], touch_rate_max: float) -> dict[str, Any]:
    samples = split["samples"]
    scored = [score_candidate(samples, candidate, touch_rate_max=touch_rate_max) for candidate in candidates]
    hook = scored[0]
    controls = scored[1:]
    best_control = max(controls, key=lambda row: (row["reward"], row["delta"], -row["touch_rate"])) if controls else None
    hook_beats_control = bool(
        hook["accepted_hook"]
        and best_control
        and float(hook["reward"]) > float(best_control["reward"])
        and float(hook["delta"]) >= float(best_control["delta"])
    )
    hook_ties_control_with_lower_touch = bool(
        hook["accepted_hook"]
        and best_control
        and float(hook["reward"]) == float(best_control["reward"])
        and float(hook["delta"]) == float(best_control["delta"])
        and float(hook["touch_rate"]) < float(best_control["touch_rate"])
    )
    return {
        "split_id": split["split_id"],
        "split_type": split["split_type"],
        "sample_count": len(samples),
        "hook_candidate_id": hook["candidate_id"],
        "hook_delta": hook["delta"],
        "hook_reward": hook["reward"],
        "hook_rescue_count": hook["rescue_count"],
        "hook_damage_count": hook["damage_count"],
        "hook_touch_rate": hook["touch_rate"],
        "hook_accepted": hook["accepted_hook"],
        "best_control_candidate_id": best_control["candidate_id"] if best_control else None,
        "best_control_delta": best_control["delta"] if best_control else None,
        "best_control_reward": best_control["reward"] if best_control else None,
        "best_control_touch_rate": best_control["touch_rate"] if best_control else None,
        "hook_beats_control": hook_beats_control,
        "hook_ties_control_with_lower_touch": hook_ties_control_with_lower_touch,
        "promotion_ready": hook_beats_control,
        "control_gap": round(float(hook["reward"]) - float(best_control["reward"]), 6) if best_control else None,
        "claim_boundary": "Split-level cached scoring; no VPD feature or permanent edit claim.",
    }


def next_iteration_contract(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    beat_rows = [row for row in rows if row["hook_beats_control"]]
    tie_rows = [row for row in rows if row["hook_ties_control_with_lower_touch"]]
    best = (beat_rows or tie_rows or rows)[0] if rows else {}
    return {
        "task": "Select the next VPD/TRM edit-learning iteration from split scores.",
        "result": "promotion_ready" if beat_rows else ("touch_specificity_tie" if tie_rows else "no_isolated_split_gain"),
        "preferred_split": {
            "split_type": best.get("split_type"),
            "split_id": best.get("split_id"),
            "sample_count": best.get("sample_count"),
        },
        "next_action": (
            "generate activation-local feature candidates for the winning split"
            if beat_rows
            else "search for a more specific condition than answer-label B before feature promotion"
        ),
        "required_gate": "hook reward must exceed best matched control reward on a held-out or family split",
        "claim_boundary": "Controller iteration contract only; VPD edit remains unclaimed.",
    }


def compact_packet(summary: dict[str, Any], contract: dict[str, Any]) -> str:
    split = contract["preferred_split"]
    lines = [
        "TASK: Continue continuous VPD-TRM edit learning from split scores.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- splits_scored: {summary['split_count']}",
        f"- promotion_ready_split_count: {summary['promotion_ready_split_count']}",
        f"- touch_specificity_tie_count: {summary['touch_specificity_tie_count']}",
        "NEXT ITERATION:",
        f"- result: {contract['result']}",
        f"- preferred_split: {split.get('split_type')}:{split.get('split_id')} ({split.get('sample_count')} samples)",
        f"- next_action: {contract['next_action']}",
        f"- required_gate: {contract['required_gate']}",
    ]
    return "\n".join(lines) + "\n"


def run_split_score(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _bridge_summary = read_json(args.bridge_dir / "gain_policy_vpd_bridge_summary.json")
    candidates = candidate_rows(args.bridge_dir)
    score_files = parse_paths(args.score_files)
    raw_samples = load_samples_with_source(score_files)
    splits = split_samples(raw_samples, args.min_split_samples)
    rows = [score_split(split, candidates, args.touch_rate_max) for split in splits]
    rows.sort(
        key=lambda row: (
            row["hook_beats_control"],
            row["hook_ties_control_with_lower_touch"],
            row["control_gap"] if row["control_gap"] is not None else -999.0,
            row["hook_reward"],
            -row["hook_touch_rate"],
        ),
        reverse=True,
    )
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "bridge_dir": str(args.bridge_dir),
        "score_files": [str(path) for path in score_files],
        "raw_sample_count": len(raw_samples),
        "deduped_sample_count": len(dedupe_samples(raw_samples)),
        "split_count": len(rows),
        "promotion_ready_split_count": sum(int(row["hook_beats_control"]) for row in rows),
        "touch_specificity_tie_count": sum(int(row["hook_ties_control_with_lower_touch"]) for row in rows),
        "best_split_id": rows[0]["split_id"] if rows else None,
        "best_split_type": rows[0]["split_type"] if rows else None,
        "claim_boundary": "Cached split scoring only; use as edit-policy state for next VPD feature search.",
        "outputs": {
            "summary": str(args.out_dir / "family_split_score_summary.json"),
            "split_scores": str(args.out_dir / "split_scores.jsonl"),
            "next_iteration_contract": str(args.out_dir / "next_iteration_contract.json"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    contract = next_iteration_contract(summary, rows)
    packet = compact_packet(summary, contract)
    write_jsonl(args.out_dir / "split_scores.jsonl", rows)
    (args.out_dir / "next_iteration_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "family_split_score_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_split_score(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

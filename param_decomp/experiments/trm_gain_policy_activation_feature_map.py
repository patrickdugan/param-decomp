"""Build activation feature-map candidates from validated gain-policy packets."""

from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_FEATURE_MAP_SPEC = importlib.util.spec_from_file_location(
    "param_decomp_editing_feature_map_light",
    Path(__file__).resolve().parents[1] / "editing" / "feature_map.py",
)
if _FEATURE_MAP_SPEC is None or _FEATURE_MAP_SPEC.loader is None:
    raise ImportError("Could not load lightweight feature_map module")
_FEATURE_MAP_MODULE = importlib.util.module_from_spec(_FEATURE_MAP_SPEC)
sys.modules[_FEATURE_MAP_SPEC.name] = _FEATURE_MAP_MODULE
_FEATURE_MAP_SPEC.loader.exec_module(_FEATURE_MAP_MODULE)
FeatureMapEntry = _FEATURE_MAP_MODULE.FeatureMapEntry
FeatureMapPlan = _FEATURE_MAP_MODULE.FeatureMapPlan
save_feature_map = _FEATURE_MAP_MODULE.save_feature_map
from param_decomp.experiments.trm_choice_rl_feedback_loop import read_jsonl, write_jsonl
from param_decomp.experiments.trm_gain_policy_activation_contrast_ranker import label_sets, read_activation_stats, rank_activation_contrast


DEFAULT_CONTRAST_RUN = Path(r"D:\Research_Engine\runs\trm_gain_policy_activation_contrast_arc_20260604")
DEFAULT_PROBE_RUN = Path(r"D:\Research_Engine\runs\trm_gain_policy_activation_contrast_arc_20260604")
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_activation_feature_map")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build activation feature-map candidates from probe requests.")
    parser.add_argument("--contrast-run", type=Path, default=DEFAULT_CONTRAST_RUN)
    parser.add_argument("--probe-run", type=Path, default=DEFAULT_PROBE_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--activation-stats", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def module_probe_rows(contrast_run: Path) -> list[dict[str, Any]]:
    return read_jsonl(contrast_run / "module_probe_requests.jsonl")


def probe_feature_name(module_path: str, condition_id: str) -> str:
    module_tail = module_path.rsplit(".", 1)[-1]
    layer = "unknown"
    for part in module_path.split("."):
        if part.isdigit():
            layer = part
            break
    return f"gain_policy::{condition_id}::layer_{layer}::{module_tail}"


def build_probe_map(probe_rows: list[dict[str, Any]], condition_id: str) -> FeatureMapPlan:
    entries = []
    for row in probe_rows:
        module_path = str(row["module_path"])
        entries.append(
            FeatureMapEntry(
                key=module_path,
                feature_name=probe_feature_name(module_path, condition_id),
                scale=1.0,
                source="activation_probe_request",
                confidence=0.0,
                notes=f"probe_request={row['request_id']}; claim_boundary=probe_only",
            )
        )
    return FeatureMapPlan(
        name=f"gain-policy-activation-probe::{condition_id}",
        description="Probe-only activation map for a validated conditional TRM policy.",
        entries=entries,
    )


def build_ranked_map(ranking_rows: list[dict[str, Any]], condition_id: str) -> FeatureMapPlan:
    entries = []
    for row in ranking_rows:
        scale = 1.25 if float(row.get("positive_minus_negative") or 0.0) > 0.0 else 0.8
        confidence = max(0.0, min(1.0, 0.5 + abs(float(row.get("positive_minus_negative") or 0.0))))
        entries.append(
            FeatureMapEntry(
                key=str(row["module_path"]),
                feature_name=f"gain_policy::{condition_id}::{row['module_path'].rsplit('.', 1)[-1]}",
                scale=scale,
                source="activation_contrast_ranking",
                confidence=confidence,
                notes=(
                    f"positive_mean={row['positive_mean_activation']}; "
                    f"negative_mean={row['negative_mean_activation']}; "
                    f"contrast={row['positive_minus_negative']}"
                ),
            )
        )
    return FeatureMapPlan(
        name=f"gain-policy-activation-feature-map::{condition_id}",
        description="Ranked activation contrast map for a validated conditional TRM policy.",
        entries=entries,
    )


def contrast_summary(contrast_rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives, negatives = label_sets(contrast_rows)
    return {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "contrast_row_count": len(contrast_rows),
        "positive_sample_ids": sorted(positives),
        "negative_sample_ids": sorted(negatives),
    }


def build_edit_trials(plan: FeatureMapPlan, *, source: str) -> list[dict[str, Any]]:
    trials = []
    for index, entry in enumerate(plan.entries):
        trials.append(
            {
                "trial_id": f"activation_edit_trial:{index:04d}:{entry.key}",
                "module_path": entry.key,
                "feature_name": entry.feature_name,
                "source": source,
                "scale": entry.scale,
                "confidence": entry.confidence,
                "runtime_only": True,
                "acceptance_rule": [
                    "feature-local edit reproduces held-out condition reward",
                    "feature-local edit beats fixed-label controls on at least one held-out score file",
                    "damage_count == 0",
                    "touch_rate <= controller predicate touch_rate",
                ],
                "claim_boundary": "Trial request only; no runtime edit has been run.",
            }
        )
    return trials


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None) -> str:
    lines = [
        "TASK: Turn activation contrast evidence into a feature-map plan.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- condition: {summary['condition_id']}",
        f"- status: {summary['status']}",
        f"- probe_run: {summary['probe_run']}",
        f"- probe_rows: {summary['probe_row_count']}",
        f"- activation_stats_present: {summary['activation_stats_present']}",
        f"- feature_map_entries: {summary['feature_map_entry_count']}",
        "BEST MODULE:",
        f"- module_path: {best.get('key') if best else None}",
        f"- scale: {best.get('scale') if best else None}",
        f"- confidence: {best.get('confidence') if best else None}",
        "NEXT ACTION:",
        "- If ranked, feed the top entries into runtime component edit testing.",
        "- If probe-only, capture activations using the probe requests first.",
    ]
    return "\n".join(lines) + "\n"


def run_feature_map(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contrast_manifest = read_json(args.contrast_run / "feature_search_manifest.json")
    probe_rows = module_probe_rows(args.probe_run)
    contrast_rows = read_jsonl(args.contrast_run / "contrast_sets.jsonl")
    condition_id = str(contrast_manifest["condition_id"])
    activation_stats_present = args.activation_stats is not None
    ranking_rows: list[dict[str, Any]] = []
    if activation_stats_present:
        stats = read_activation_stats(args.activation_stats)
        positive_ids, negative_ids = label_sets(contrast_rows)
        ranking_rows = rank_activation_contrast(stats, positive_ids=positive_ids, negative_ids=negative_ids)
    plan = build_ranked_map(ranking_rows[: args.top_k], condition_id) if ranking_rows else build_probe_map(probe_rows[: args.top_k], condition_id)
    trials = build_edit_trials(plan, source="activation_contrast_ranking" if ranking_rows else "activation_probe_request")
    summary = {
        "status": "ranked" if ranking_rows else "probe_only",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "contrast_run": str(args.contrast_run),
        "probe_run": str(args.probe_run),
        "condition_id": condition_id,
        "probe_row_count": len(probe_rows),
        "activation_stats_present": activation_stats_present,
        "feature_map_entry_count": len(plan.entries),
        "edit_trial_count": len(trials),
        "claim_boundary": "Feature-map bridge only; runtime VPD edit still untested.",
        "outputs": {
            "summary": str(args.out_dir / "activation_feature_map_summary.json"),
            "feature_map": str(args.out_dir / "activation_feature_map.json"),
            "edit_trials": str(args.out_dir / "activation_edit_trials.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    packet = compact_packet(summary, plan.entries[0].__dict__ if plan.entries else None)
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    packet = compact_packet(summary, plan.entries[0].__dict__ if plan.entries else None)
    save_feature_map(plan, args.out_dir / "activation_feature_map.json")
    write_jsonl(args.out_dir / "activation_edit_trials.jsonl", trials)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    (args.out_dir / "activation_feature_map_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_feature_map(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

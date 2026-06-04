"""Rank or request activation contrasts for validated gain-policy packets."""

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

from param_decomp.experiments.trm_choice_rl_feedback_loop import read_jsonl, write_jsonl


DEFAULT_PACKET_RUN = Path(r"D:\Research_Engine\runs\trm_gain_policy_feature_search_packet_arc_20260604")
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_activation_contrast")
DEFAULT_MODULE_PATHS = (
    "base_model.model.model.language_model.layers.23.self_attn.o_proj,"
    "base_model.model.model.language_model.layers.23.mlp.down_proj,"
    "base_model.model.model.language_model.layers.19.self_attn.o_proj,"
    "base_model.model.model.language_model.layers.19.mlp.down_proj,"
    "base_model.model.model.language_model.layers.15.self_attn.o_proj,"
    "base_model.model.model.language_model.layers.15.mlp.down_proj,"
    "base_model.model.model.language_model.layers.11.self_attn.o_proj,"
    "base_model.model.model.language_model.layers.11.mlp.down_proj"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank activation contrast candidates or emit module probe requests.")
    parser.add_argument("--packet-run", type=Path, default=DEFAULT_PACKET_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--activation-stats", type=Path, default=None)
    parser.add_argument("--module-paths", default=DEFAULT_MODULE_PATHS)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_module_paths(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def label_sets(contrast_rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    positive = {str(row["sample_id"]) for row in contrast_rows if row.get("label") == "positive_rescue"}
    negative = {str(row["sample_id"]) for row in contrast_rows if str(row.get("label", "")).startswith("negative_")}
    return positive, negative


def read_activation_stats(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    normalized = []
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        module_path = str(row.get("module_path") or "")
        value = row.get("activation_value", row.get("mean_abs_activation", row.get("value")))
        if sample_id and module_path and value is not None:
            normalized.append({"sample_id": sample_id, "module_path": module_path, "activation_value": float(value)})
    return normalized


def rank_activation_contrast(
    activation_rows: list[dict[str, Any]],
    *,
    positive_ids: set[str],
    negative_ids: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in activation_rows:
        grouped[str(row["module_path"])].append(row)
    rankings = []
    for module_path, rows in grouped.items():
        pos_values = [float(row["activation_value"]) for row in rows if row["sample_id"] in positive_ids]
        neg_values = [float(row["activation_value"]) for row in rows if row["sample_id"] in negative_ids]
        if not pos_values or not neg_values:
            continue
        pos_mean = sum(pos_values) / len(pos_values)
        neg_mean = sum(neg_values) / len(neg_values)
        contrast = pos_mean - neg_mean
        rankings.append(
            {
                "module_path": module_path,
                "positive_mean_activation": round(pos_mean, 6),
                "negative_mean_activation": round(neg_mean, 6),
                "positive_minus_negative": round(contrast, 6),
                "abs_contrast": round(abs(contrast), 6),
                "positive_count": len(pos_values),
                "negative_count": len(neg_values),
                "suggested_edit_direction": "suppress_when_positive_high" if contrast > 0 else "amplify_or_check_inhibitory_feature",
                "claim_boundary": "Activation contrast ranking only; no runtime edit has been scored.",
            }
        )
    rankings.sort(key=lambda row: (row["abs_contrast"], row["positive_count"], row["negative_count"]), reverse=True)
    return rankings


def build_probe_requests(module_paths: list[str], contrast_rows: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    positive_ids, negative_ids = label_sets(contrast_rows)
    rows = []
    for index, module_path in enumerate(module_paths):
        rows.append(
            {
                "request_id": f"activation_probe:{index:04d}",
                "module_path": module_path,
                "condition_id": contract["condition_id"],
                "positive_sample_ids": sorted(positive_ids),
                "negative_sample_ids": sorted(negative_ids),
                "metric": "mean_abs_activation",
                "scoring_contract": {
                    "rank_by": "abs(mean_positive_activation - mean_negative_activation)",
                    "then_test": "runtime module/component edit against fixed-label controls",
                },
                "claim_boundary": "Probe request only; requires model activation capture.",
            }
        )
    return rows


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None) -> str:
    lines = [
        "TASK: Rank activation-local candidates for validated D-over-A gain policy.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- status: {summary['status']}",
        f"- condition: {summary['condition_id']}",
        f"- positive/negative samples: {summary['positive_count']}/{summary['negative_count']}",
        f"- ranked_modules: {summary['ranked_module_count']}",
        f"- probe_requests: {summary['probe_request_count']}",
        "BEST MODULE:",
        f"- module_path: {best.get('module_path') if best else None}",
        f"- abs_contrast: {best.get('abs_contrast') if best else None}",
        "NEXT ACTION:",
        "- Capture real activations if ranking is empty.",
        "- If ranking exists, score top runtime edits against held-out fixed-label controls.",
    ]
    return "\n".join(lines) + "\n"


def run_ranker(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.packet_run / "feature_search_manifest.json")
    contract = read_json(args.packet_run / "activation_probe_contract.json")
    contrast_rows = read_jsonl(args.packet_run / "contrast_sets.jsonl")
    positive_ids, negative_ids = label_sets(contrast_rows)
    probe_requests = build_probe_requests(parse_module_paths(args.module_paths), contrast_rows, contract)
    rankings: list[dict[str, Any]] = []
    status = "probe_requests_ready"
    if args.activation_stats is not None:
        rankings = rank_activation_contrast(
            read_activation_stats(args.activation_stats),
            positive_ids=positive_ids,
            negative_ids=negative_ids,
        )
        status = "ranked" if rankings else "no_rankable_activation_stats"
    top_rankings = rankings[: max(0, args.top_k)]
    summary = {
        "status": status,
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "packet_run": str(args.packet_run),
        "condition_id": str(manifest["condition_id"]),
        "positive_count": len(positive_ids),
        "negative_count": len(negative_ids),
        "ranked_module_count": len(rankings),
        "probe_request_count": len(probe_requests),
        "claim_boundary": "Activation contrast/probe artifact only; no VPD runtime edit has been tested.",
        "outputs": {
            "summary": str(args.out_dir / "activation_contrast_summary.json"),
            "activation_contrast_ranking": str(args.out_dir / "activation_contrast_ranking.jsonl"),
            "module_probe_requests": str(args.out_dir / "module_probe_requests.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    write_jsonl(args.out_dir / "activation_contrast_ranking.jsonl", top_rankings)
    write_jsonl(args.out_dir / "module_probe_requests.jsonl", probe_requests)
    packet = compact_packet(summary, top_rankings[0] if top_rankings else None)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "activation_contrast_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_ranker(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

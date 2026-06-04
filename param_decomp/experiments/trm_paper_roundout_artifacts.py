"""Build deterministic paper roundout artifacts from completed VPD-TRM runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


RUN_ROOT = Path(r"D:\Research_Engine\runs")
DEFAULT_OUT_DIR = RUN_ROOT / "vpd_trm_paper_roundout_20260604"


PAPER_RUNS = [
    {
        "run_id": "metta_organelle_arena_paper_20260601T024147Z",
        "summary": "arena_summary.json",
        "tier": "claim_support",
        "role": "strict organelle transfer replication",
    },
    {
        "run_id": "trm_feedback_loop_full_sweep_filtered_20260601T163437Z",
        "summary": "outer_loop_summary.json",
        "tier": "claim_support",
        "role": "filtered feedback-loop policy comparison",
    },
    {
        "run_id": "trm_eval_aligned_edits_intellect3_logic_20260602",
        "summary": "eval_aligned_summary.json",
        "tier": "boundary_result",
        "role": "strict Intellect-3 eval-alignment gate",
    },
    {
        "run_id": "trm_eval_hill_climb_intellect3_logic_full_20260602",
        "summary": "hill_climb_summary.json",
        "tier": "boundary_result",
        "role": "non-targeted eval hill-climb cliff",
    },
    {
        "run_id": "trm_gain_policy_condition_cv_arc_4seed_20260604",
        "summary": "condition_cv_summary.json",
        "tier": "claim_support",
        "role": "held-out ARC controller-predicate validation",
    },
    {
        "run_id": "trm_gain_policy_activation_capture_arc_20260604",
        "summary": "activation_capture_summary.json",
        "tier": "open_positive_track",
        "role": "activation capture for ARC gain-policy bridge",
    },
    {
        "run_id": "trm_gain_policy_activation_feature_map_arc_20260604",
        "summary": "activation_feature_map_summary.json",
        "tier": "open_positive_track",
        "role": "ranked activation feature-map handoff",
    },
    {
        "run_id": "trm_feedback_loop_paper_20260601T115019Z",
        "summary": "outer_loop_summary.json",
        "tier": "excluded",
        "role": "superseded scoring artifact with replay-gain overwrite bug",
    },
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final paper manifest, tables, and compact figures.")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def manifest_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for spec in PAPER_RUNS:
        path = run_root / spec["run_id"]
        summary_path = path / spec["summary"]
        rows.append(
            {
                "run_id": spec["run_id"],
                "path": str(path),
                "summary_path": str(summary_path),
                "exists": path.exists(),
                "summary_exists": summary_path.exists(),
                "tier": spec["tier"],
                "role": spec["role"],
                "include_for_main_claims": spec["tier"] == "claim_support",
            }
        )
    return rows


def metric_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    arena = read_json(run_root / "metta_organelle_arena_paper_20260601T024147Z" / "arena_summary.json")
    if arena:
        for metric in ("accepted_count", "stable_graft_count", "graft_record_count", "organelle_count"):
            rows.append({"result_family": "organelle_transfer", "metric": metric, "value": arena.get(metric), "claim_tier": "claim_support"})

    feedback = read_json(run_root / "trm_feedback_loop_full_sweep_filtered_20260601T163437Z" / "outer_loop_summary.json")
    if feedback:
        for row in feedback.get("filtered_policy_comparison") or []:
            rows.append(
                {
                    "result_family": "filtered_feedback_loop",
                    "metric": f"{row['policy']}:filtered_metric_gain",
                    "value": row.get("filtered_metric_gain"),
                    "claim_tier": "claim_support",
                }
            )
            rows.append(
                {
                    "result_family": "filtered_feedback_loop",
                    "metric": f"{row['policy']}:filtered_accept_count",
                    "value": row.get("filtered_accept_count"),
                    "claim_tier": "claim_support",
                }
            )

    eval_aligned = read_json(run_root / "trm_eval_aligned_edits_intellect3_logic_20260602" / "eval_aligned_summary.json")
    if eval_aligned:
        for metric in ("candidate_count", "cluster_count", "result_count", "accepted_eval_aligned_count"):
            rows.append({"result_family": "eval_alignment_boundary", "metric": metric, "value": eval_aligned.get(metric), "claim_tier": "boundary_result"})

    cv = read_json(run_root / "trm_gain_policy_condition_cv_arc_4seed_20260604" / "condition_cv_summary.json")
    if cv:
        for metric in ("fold_count", "heldout_accepted_fold_count", "beats_control_fold_count", "ties_control_lower_touch_fold_count", "best_heldout_delta"):
            rows.append({"result_family": "arc_gain_policy", "metric": metric, "value": cv.get(metric), "claim_tier": "claim_support"})

    activation = read_json(run_root / "trm_gain_policy_activation_feature_map_arc_20260604" / "activation_feature_map_summary.json")
    if activation:
        for metric in ("status", "feature_map_entry_count", "activation_stats_present", "prompt_packet_est_tokens"):
            rows.append({"result_family": "activation_runtime_track", "metric": metric, "value": activation.get(metric), "claim_tier": "open_positive_track"})

    return rows


def bar_svg(rows: list[dict[str, Any]]) -> str:
    selected = [
        ("VPD grafts", float(next((row["value"] for row in rows if row["metric"] == "accepted_count"), 0) or 0)),
        ("Eval accepts", float(next((row["value"] for row in rows if row["metric"] == "accepted_eval_aligned_count"), 0) or 0)),
        ("ARC folds", float(next((row["value"] for row in rows if row["metric"] == "heldout_accepted_fold_count"), 0) or 0)),
    ]
    width, height = 720, 260
    left, top, plot_w, plot_h = 72, 36, 560, 150
    max_value = max([value for _, value in selected] + [1.0])
    bars = []
    colors = ["#356f66", "#8d3f3f", "#4c5f9d"]
    for index, ((label, value), color) in enumerate(zip(selected, colors, strict=True)):
        bar_w = 120
        x = left + index * 180
        h = plot_h * value / max_value
        y = top + plot_h - h
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{color}"/>')
        bars.append(f'<text x="{x}" y="{top + plot_h + 24}" font-family="Arial" font-size="13" fill="#222">{label}</text>')
        bars.append(f'<text x="{x}" y="{max(18, y - 8):.2f}" font-family="Arial" font-size="13" fill="#222">{value:g}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        f'<text x="{left}" y="24" font-family="Arial" font-size="17" fill="#1d2327">VPD-TRM paper roundout summary</text>'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>'
        f'{"".join(bars)}'
        '</svg>'
    )


def compact_packet(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "TASK: Freeze paper-ready VPD-TRM experiment evidence while preserving positive-track open work.",
            "CURRENT STATE:",
            f"- run: {summary['run_dir']}",
            f"- manifest_rows: {summary['manifest_row_count']}",
            f"- metric_rows: {summary['metric_row_count']}",
            f"- included_main_claim_runs: {summary['included_main_claim_run_count']}",
            "CLAIM POSTURE:",
            "- Main paper: proxy control motifs and filtered feedback-loop evidence.",
            "- Boundary: Intellect-3 eval-aligned accepts remain zero.",
            "- Open track: activation-local ARC runtime edit scoring.",
        ]
    ) + "\n"


def run_roundout(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_rows(args.run_root)
    metrics = metric_rows(args.run_root)
    write_csv(
        args.out_dir / "paper_experiment_manifest.csv",
        manifest,
        ["run_id", "path", "summary_path", "exists", "summary_exists", "tier", "role", "include_for_main_claims"],
    )
    write_csv(args.out_dir / "paper_metric_table.csv", metrics, ["result_family", "metric", "value", "claim_tier"])
    (args.out_dir / "paper_experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figures = args.out_dir / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "roundout_summary.svg").write_text(bar_svg(metrics), encoding="utf-8")
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "run_root": str(args.run_root),
        "manifest_row_count": len(manifest),
        "metric_row_count": len(metrics),
        "included_main_claim_run_count": sum(int(row["include_for_main_claims"]) for row in manifest),
        "excluded_run_count": sum(int(row["tier"] == "excluded") for row in manifest),
        "claim_boundary": "Paper artifact builder only; does not rerun experiments.",
        "outputs": {
            "summary": str(args.out_dir / "paper_roundout_summary.json"),
            "manifest_csv": str(args.out_dir / "paper_experiment_manifest.csv"),
            "manifest_json": str(args.out_dir / "paper_experiment_manifest.json"),
            "metric_table": str(args.out_dir / "paper_metric_table.csv"),
            "summary_figure": str(figures / "roundout_summary.svg"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    packet = compact_packet(summary)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "paper_roundout_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_roundout(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

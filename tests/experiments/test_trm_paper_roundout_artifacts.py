from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from param_decomp.experiments.trm_paper_roundout_artifacts import run_roundout


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_roundout_builds_manifest_and_metric_table(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_json(
        root / "metta_organelle_arena_paper_20260601T024147Z" / "arena_summary.json",
        {"accepted_count": 2, "stable_graft_count": 4, "graft_record_count": 20, "organelle_count": 3},
    )
    _write_json(
        root / "trm_feedback_loop_full_sweep_filtered_20260601T163437Z" / "outer_loop_summary.json",
        {"filtered_policy_comparison": [{"policy": "vpd_greedy", "filtered_metric_gain": 0.2, "filtered_accept_count": 5}]},
    )
    _write_json(
        root / "trm_eval_aligned_edits_intellect3_logic_20260602" / "eval_aligned_summary.json",
        {"candidate_count": 10, "cluster_count": 2, "result_count": 4, "accepted_eval_aligned_count": 0},
    )

    summary = run_roundout(Namespace(run_root=root, out_dir=tmp_path / "out"))

    assert summary["status"] == "completed"
    assert summary["manifest_row_count"] > 0
    assert summary["metric_row_count"] > 0
    assert (tmp_path / "out" / "paper_experiment_manifest.csv").exists()
    assert (tmp_path / "out" / "paper_metric_table.csv").exists()
    assert (tmp_path / "out" / "figures" / "roundout_summary.svg").exists()


def test_roundout_marks_superseded_artifact_excluded(tmp_path: Path) -> None:
    summary = run_roundout(Namespace(run_root=tmp_path / "runs", out_dir=tmp_path / "out"))

    manifest = json.loads((tmp_path / "out" / "paper_experiment_manifest.json").read_text(encoding="utf-8"))
    excluded = [row for row in manifest if row["tier"] == "excluded"]

    assert summary["excluded_run_count"] == len(excluded)
    assert excluded
    assert excluded[0]["include_for_main_claims"] is False

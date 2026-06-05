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
        {
            "policy_comparison": [{"policy": "vpd_greedy", "total_metric_gain": 0.3, "accepted_metric_gain": 0.2, "accepted_count": 5}],
            "filtered_policy_comparison": [{"policy": "vpd_greedy", "filtered_metric_gain": 0.2, "filtered_accept_count": 5}],
        },
    )
    _write_json(
        root / "trm_eval_aligned_edits_intellect3_logic_20260602" / "eval_aligned_summary.json",
        {"candidate_count": 10, "cluster_count": 2, "result_count": 4, "accepted_eval_aligned_count": 0},
    )
    _write_json(
        root / "trm_edit_bootstrap_microcycle_arc_20260604" / "bootstrap_summary.json",
        {"accepted_edit_count": 1, "baseline_score": 0.7, "final_score": 0.77, "cumulative_delta": 0.07, "stacked_bootstrap_success": False},
    )
    _write_json(
        root / "trm_multi_family_bootstrap_sweep_arc_20260604" / "multi_family_bootstrap_summary.json",
        {"family_count": 3, "first_edit_family_count": 1, "stacked_bootstrap_family_count": 0, "control_blocked_family_count": 2, "best_cumulative_delta": 0.1},
    )

    summary = run_roundout(Namespace(run_root=root, out_dir=tmp_path / "out"))

    assert summary["status"] == "completed"
    assert summary["manifest_row_count"] > 0
    assert summary["metric_row_count"] > 0
    assert (tmp_path / "out" / "paper_experiment_manifest.csv").exists()
    assert (tmp_path / "out" / "paper_metric_table.csv").exists()
    assert (tmp_path / "out" / "paper_policy_comparison.csv").exists()
    assert (tmp_path / "out" / "paper_claim_ledger.csv").exists()
    assert (tmp_path / "out" / "figures" / "roundout_summary.svg").exists()
    assert (tmp_path / "out" / "figures" / "filtered_feedback_policy_gain.svg").exists()
    assert (tmp_path / "out" / "figures" / "eval_alignment_collapse.svg").exists()


def test_roundout_marks_superseded_artifact_excluded(tmp_path: Path) -> None:
    summary = run_roundout(Namespace(run_root=tmp_path / "runs", out_dir=tmp_path / "out"))

    manifest = json.loads((tmp_path / "out" / "paper_experiment_manifest.json").read_text(encoding="utf-8"))
    excluded = [row for row in manifest if row["tier"] == "excluded"]

    assert summary["excluded_run_count"] == len(excluded)
    assert excluded
    assert excluded[0]["include_for_main_claims"] is False


def test_claim_ledger_marks_runtime_edit_as_open_track(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_json(
        root / "trm_gain_policy_runtime_edit_score_arc_20260604" / "runtime_edit_summary.json",
        {"promotion_ready": False, "best_trial_delta": 0.1, "best_control_delta": 0.2},
    )

    run_roundout(Namespace(run_root=root, out_dir=tmp_path / "out"))
    ledger = (tmp_path / "out" / "paper_claim_ledger.csv").read_text(encoding="utf-8")

    assert "not_supported_open_track" in ledger
    assert "promotion_ready=False" in ledger


def test_claim_ledger_marks_bootstrap_boundary(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_json(
        root / "trm_edit_bootstrap_microcycle_arc_20260604" / "bootstrap_summary.json",
        {"accepted_edit_count": 1, "cumulative_delta": 0.064517, "stacked_bootstrap_success": False, "stop_reason": "fixed_control_not_beaten"},
    )

    run_roundout(Namespace(run_root=root, out_dir=tmp_path / "out"))
    ledger = (tmp_path / "out" / "paper_claim_ledger.csv").read_text(encoding="utf-8")

    assert "Stateful cached edit search bootstraps ability" in ledger
    assert "not_supported_boundary" in ledger
    assert "fixed_control_not_beaten" in ledger


def test_claim_ledger_marks_multi_family_methodology(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_json(
        root / "trm_multi_family_bootstrap_sweep_arc_20260604" / "multi_family_bootstrap_summary.json",
        {"family_count": 15, "first_edit_family_count": 3, "stacked_bootstrap_family_count": 0, "best_family_key": "D->A:A:medium_1_0"},
    )

    run_roundout(Namespace(run_root=root, out_dir=tmp_path / "out"))
    ledger = (tmp_path / "out" / "paper_claim_ledger.csv").read_text(encoding="utf-8")

    assert "Broadening the loop across failure families" in ledger
    assert "supported_methodology" in ledger
    assert "D->A:A:medium_1_0" in ledger

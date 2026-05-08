from __future__ import annotations

import json
from pathlib import Path

from param_decomp.experiments.metta_amalgam.data import GATE_FAMILIES, build_default_rows, load_rows, write_jsonl
from param_decomp.experiments.metta_amalgam.run import main


def test_default_amalgam_rows_cover_all_gates_and_splits() -> None:
    rows = load_rows()

    assert {row.gate_family for row in rows} == set(GATE_FAMILIES)
    assert {row.split for row in rows} >= {"train", "validation", "holdout_seen", "holdout_unseen", "backdoor_probe"}
    assert len(rows) == 53


def test_amalgam_rejects_invalid_gate_decision(tmp_path: Path) -> None:
    rows = build_default_rows()
    rows[0]["expected_decision"] = "commit"
    trace = tmp_path / "bad.jsonl"
    write_jsonl(trace, rows)

    try:
        load_rows(trace)
    except ValueError as exc:
        assert "invalid for" in str(exc)
    else:
        raise AssertionError("expected invalid gate decision to fail")


def test_amalgam_smoke_run_writes_component_registry(tmp_path: Path) -> None:
    out_dir = tmp_path / "amalgam"
    code = main(
        [
            "--out-dir",
            str(out_dir),
            "--backend",
            "python",
            "--epochs",
            "45",
            "--hidden-dim",
            "16",
            "--recursive-steps",
            "2",
        ]
    )

    assert code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    registry = json.loads((out_dir / "component_registry.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["row_summary"]["gate_counts"] == {
        "commit_veto": 21,
        "repair_or_abstain": 16,
        "route_to_tool": 16,
    }
    assert len(registry) == 3
    assert (out_dir / "task_graph.json").exists()

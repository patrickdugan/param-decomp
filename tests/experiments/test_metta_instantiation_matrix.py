from __future__ import annotations

import json
from pathlib import Path

from param_decomp.experiments.metta_instantiation_matrix.data import (
    GATE_DECISIONS,
    METTA_VARIANTS,
    build_rows,
    write_jsonl,
)
from param_decomp.experiments.metta_instantiation_matrix.run import main


def _write_logic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "intellect_3_logic.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    metadata = {
        "grid": [["T", "X"], ["X", "X"]],
        "solution": [["T", "C"], ["X", "X"]],
        "row_constraints": [1, 0],
        "col_constraints": [0, 1],
    }
    source_row = {
        "env_id": "intellect_3_logic",
        "trajectory_id": "intellect_3_logic_fixture_0",
        "state_prompt": "Place one camp next to the tree.",
        "action": json.dumps({"game_data_str": json.dumps({"metadata": metadata})}),
        "reward": 1.0,
        "success": True,
    }
    write_jsonl(source, [source_row])
    write_jsonl(
        predictions,
        [
            {
                "row_id": "intellect_3_logic_fixture_0",
                "arm": "logic_skill_trm",
                "task": "intellect_3_logic",
                "final": {
                    "action": repr([["T", "C"], ["X", "X"]]),
                    "token_total": 128,
                    "latency_seconds": 1.0,
                    "visible_output_emitted": True,
                },
            },
            {
                "row_id": "intellect_3_logic_fixture_0",
                "arm": "generic_skill",
                "task": "intellect_3_logic",
                "final": {
                    "action": repr([["T", "X"], ["X", "X"]]),
                    "token_total": 128,
                    "latency_seconds": 1.0,
                    "visible_output_emitted": True,
                },
            },
        ],
    )
    return source, predictions


def test_intellect3_adapter_expands_all_metta_variants_without_target_leakage(tmp_path: Path) -> None:
    source, predictions = _write_logic_fixture(tmp_path)

    rows = build_rows(source_path=source, predictions_path=predictions, limit=1, include_storyworld=False)

    assert {row.metta_variant for row in rows} == set(METTA_VARIANTS)
    assert {row.gate_family for row in rows} >= {"signature_route", "candidate_verify", "repair_step", "format_commit"}
    candidate_rows = [row for row in rows if row.gate_family == "candidate_verify"]
    assert candidate_rows
    for row in candidate_rows:
        assert "exact_match" not in row.row["evidence"]
        assert "cell_accuracy" not in row.row["evidence"]
        assert "exact_match" in row.row["score_signals"]


def test_matrix_rows_have_valid_gate_decisions(tmp_path: Path) -> None:
    source, predictions = _write_logic_fixture(tmp_path)

    rows = build_rows(source_path=source, predictions_path=predictions, limit=1, include_storyworld=True, storyworld_limit=6)

    for row in rows:
        assert row.expected_decision in GATE_DECISIONS[row.gate_family]


def test_metta_instantiation_matrix_smoke_run_writes_outputs(tmp_path: Path) -> None:
    source, predictions = _write_logic_fixture(tmp_path)
    out_dir = tmp_path / "matrix"

    code = main(
        [
            "--source",
            str(source),
            "--predictions",
            str(predictions),
            "--out-dir",
            str(out_dir),
            "--backend",
            "python",
            "--epochs",
            "8",
            "--hidden-dim",
            "8",
            "--recursive-steps",
            "1",
            "--limit",
            "1",
            "--no-storyworld",
        ]
    )

    assert code == 0
    summary = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    registry = json.loads((out_dir / "component_registry.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["experiment"] == "metta_instantiation_matrix_v1"
    assert summary["row_summary"]["variant_counts"] == {variant: 7 for variant in METTA_VARIANTS}
    assert registry
    assert (out_dir / "rows.normalized.jsonl").exists()
    assert (out_dir / "ablation_summary.csv").exists()
    assert (out_dir / "report.md").exists()

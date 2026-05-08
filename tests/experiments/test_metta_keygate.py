from __future__ import annotations

import json
from pathlib import Path

import pytest

from param_decomp.experiments.metta_keygate.data import encode_rows, load_trace_rows
from param_decomp.experiments.metta_keygate.run import main


def _row(step_id: str, input_kind: str, expected: str, split: str) -> dict:
    evidence = {
        "schema_valid": input_kind == "exact_positive_repair",
        "repair_gain": {
            "exact_positive_repair": "complete",
            "partial_positive_repair": "partial",
            "no_gain_repair": "none",
            "hard_negative": "partial",
        }[input_kind],
        "missing_fields": [] if input_kind == "exact_positive_repair" else ["field"],
        "verifier_disagreement": input_kind == "hard_negative",
    }
    return {
        "episode_id": "test",
        "step_id": step_id,
        "scenario": "post_repair_commit_veto",
        "gate_family": "commit_veto",
        "input_kind": input_kind,
        "risk_tags": ["clean_repair"] if expected == "commit" else ["schema_invalid"],
        "prompt": f"Verifier sees {input_kind} and must decide whether to commit, veto, or repair.",
        "evidence": evidence,
        "expected_decision": expected,
        "actual_decision": None,
        "label": "test",
        "split": split,
        "holdout_group": "test",
    }


def _write_trace(path: Path) -> None:
    rows = [
        _row("s0001", "exact_positive_repair", "commit", "train"),
        _row("s0002", "partial_positive_repair", "veto", "train"),
        _row("s0003", "no_gain_repair", "repair", "train"),
        _row("s0004", "hard_negative", "veto", "train"),
        _row("s0005", "exact_positive_repair", "commit", "validation"),
        _row("s0006", "partial_positive_repair", "veto", "validation"),
        _row("s0007", "no_gain_repair", "repair", "holdout_seen"),
        _row("s0008", "hard_negative", "veto", "backdoor_probe"),
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_load_trace_rejects_duplicate_step_id(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    row = _row("s0001", "exact_positive_repair", "commit", "train")
    trace.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate step_id"):
        load_trace_rows(trace)


def test_feature_encoder_is_deterministic(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    rows = load_trace_rows(trace)

    features_a, labels_a, names_a = encode_rows(rows)
    features_b, labels_b, names_b = encode_rows(rows)

    assert names_a == names_b
    assert labels_a.tolist() == labels_b.tolist()
    assert features_a.tolist() == features_b.tolist()


def test_metta_keygate_smoke_run_writes_outputs(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    out_dir = tmp_path / "out"
    _write_trace(trace)

    exit_code = main(
        [
            "--trace",
            str(trace),
            "--out-dir",
            str(out_dir),
            "--backend",
            "python",
            "--seed",
            "7",
            "--epochs",
            "30",
            "--hidden-dim",
            "12",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "component_scores.json").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "manifest.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["trace_summary"]["row_count"] == 8
    assert summary["component_scoring"]["component_count"] == 6

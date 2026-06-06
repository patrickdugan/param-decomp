from __future__ import annotations

import json
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_tinylora_live_trainer import run_trainer


def test_live_trainer_dry_run_emits_events_and_results(tmp_path: Path) -> None:
    candidate = {
        "candidate_id": "task:candidate:001",
        "organism_id": "tiny_lora:g0:1",
        "target_module": "layer.1",
        "source_family_key": "D->A:A:medium_1_0",
        "adapter_config": {"rank": 1, "scale": 1.0},
        "proxy_score": {"delta": 0.1, "control_margin": 0.05, "fitness": 0.2},
    }
    control = {**candidate, "candidate_id": "task:random_control:001", "organism_id": "random_tinylora:001"}
    write_jsonl(tmp_path / "candidates.jsonl", [candidate])
    write_jsonl(tmp_path / "controls.jsonl", [control])
    manifest = {
        "training_task_id": "task",
        "caps": {"ram_mb": 1024, "cpu_pct": 25, "io_mb_s": 20},
        "paths": {
            "candidates": str(tmp_path / "candidates.jsonl"),
            "random_controls": str(tmp_path / "controls.jsonl"),
            "plan": str(tmp_path / "plan.json"),
            "comparison_table": str(tmp_path / "comparison.csv"),
            "event_schema": str(tmp_path / "schema.json"),
        },
        "default_output_dir": str(tmp_path / "out"),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "dry_run", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "completed"
    assert summary["accepted_count"] == 0
    events = (tmp_path / "out" / "tinylora_training_events.jsonl").read_text(encoding="utf-8")
    assert "candidate_start" in events
    assert "dry_run_cleanup_passed" in events
    results = [json.loads(line) for line in (tmp_path / "out" / "tinylora_training_candidate_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert results[0]["decision_reason"] == "dry_run_only_no_adapter_trained"


def _write_manifest(tmp_path: Path) -> Path:
    candidate = {
        "candidate_id": "task:candidate:001",
        "organism_id": "tiny_lora:g0:1",
        "target_module": "layer.1",
        "source_family_key": "D->A:A:medium_1_0",
        "adapter_config": {"rank": 1, "scale": 1.0},
        "proxy_score": {"delta": 0.1, "control_margin": 0.05, "fitness": 0.2},
    }
    control = {**candidate, "candidate_id": "task:random_control:001", "organism_id": "random_tinylora:001"}
    write_jsonl(tmp_path / "candidates.jsonl", [candidate])
    write_jsonl(tmp_path / "controls.jsonl", [control])
    manifest = {
        "training_task_id": "task",
        "caps": {"ram_mb": 1024, "cpu_pct": 25, "io_mb_s": 20},
        "paths": {
            "candidates": str(tmp_path / "candidates.jsonl"),
            "random_controls": str(tmp_path / "controls.jsonl"),
            "plan": str(tmp_path / "plan.json"),
            "comparison_table": str(tmp_path / "comparison.csv"),
            "event_schema": str(tmp_path / "schema.json"),
        },
        "default_output_dir": str(tmp_path / "out"),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_live_trainer_train_one_blocks_outside_wrapper(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["candidate_count"] == 1
    assert summary["block_reason"] == "blocked_not_inside_generated_jobobject_wrapper"
    events = (tmp_path / "out" / "tinylora_training_events.jsonl").read_text(encoding="utf-8")
    assert "train_one_blocked" in events
    results = [json.loads(line) for line in (tmp_path / "out" / "tinylora_training_candidate_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert results[0]["decision_reason"] == "blocked_not_inside_generated_jobobject_wrapper"


def test_live_trainer_train_one_inside_wrapper_blocks_for_missing_backend(tmp_path: Path, monkeypatch: object) -> None:
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("TINYLORA_JOB_OBJECT", "1")

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "blocked_missing_adapter_training_backend"
    results = [json.loads(line) for line in (tmp_path / "out" / "tinylora_training_candidate_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert results[0]["decision_reason"] == "blocked_missing_adapter_training_backend"

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
    control["proxy_score"] = {"delta": 0.0, "control_margin": 0.0, "fitness": 0.0}
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

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "dry_run", "backend": "none", "max_candidates": 0, "candidate_id": None})())

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
    control["proxy_score"] = {"delta": 0.0, "control_margin": 0.0, "fitness": 0.0}
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
    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "none", "max_candidates": 0, "candidate_id": None})())

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

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "none", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "blocked_missing_adapter_training_backend"
    results = [json.loads(line) for line in (tmp_path / "out" / "tinylora_training_candidate_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert results[0]["decision_reason"] == "blocked_missing_adapter_training_backend"


def test_live_trainer_scorecard_rehearsal_accepts_inside_wrapper(tmp_path: Path, monkeypatch: object) -> None:
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("TINYLORA_JOB_OBJECT", "1")

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "scorecard_rehearsal", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "completed"
    assert summary["backend"] == "scorecard_rehearsal"
    assert summary["accepted_count"] == 1
    assert "not a trained model-edit gain" in summary["claim_boundary"]
    results = [json.loads(line) for line in (tmp_path / "out" / "tinylora_training_candidate_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert results[0]["accepted"] is True
    assert results[0]["decision_reason"] == "scorecard_rehearsal_accept"
    assert results[0]["live_target_score"] == 0.2


def test_live_trainer_peft_preflight_blocks_without_model_load_opt_in(tmp_path: Path, monkeypatch: object) -> None:
    manifest_path = _write_manifest(tmp_path)
    monkeypatch.setenv("TINYLORA_JOB_OBJECT", "1")

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["backend"] == "peft_train_one"
    assert summary["block_reason"] == "blocked_model_load_not_enabled"
    request_path = Path(summary["outputs"]["backend_request"])
    assert request_path.exists()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["candidate_id"] == "task:candidate:001"
    assert request["required_runtime"]["model_load_marker"] == "TINYLORA_ENABLE_MODEL_LOAD=1"


def test_live_trainer_peft_preflight_records_fully_specified_backend_gap(tmp_path: Path, monkeypatch: object) -> None:
    manifest_path = _write_manifest(tmp_path)
    model_path = tmp_path / "model"
    eval_spec = tmp_path / "eval.json"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    eval_spec.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TINYLORA_JOB_OBJECT", "1")
    monkeypatch.setenv("TINYLORA_ENABLE_MODEL_LOAD", "1")
    monkeypatch.setenv("TINYLORA_MODEL_PATH", str(model_path))
    monkeypatch.setenv("TINYLORA_EVAL_SPEC", str(eval_spec))

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "blocked_peft_training_body_not_enabled_after_preflight"
    request = json.loads(Path(summary["outputs"]["backend_request"]).read_text(encoding="utf-8"))
    assert request["model_path"] == str(model_path)
    assert request["eval_spec"] == str(eval_spec)
    assert summary["backend_probe"]["model_size_mb"] >= 1


def test_live_trainer_peft_preflight_blocks_missing_model_path(tmp_path: Path, monkeypatch: object) -> None:
    manifest_path = _write_manifest(tmp_path)
    eval_spec = tmp_path / "eval.json"
    eval_spec.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TINYLORA_JOB_OBJECT", "1")
    monkeypatch.setenv("TINYLORA_ENABLE_MODEL_LOAD", "1")
    monkeypatch.setenv("TINYLORA_MODEL_PATH", str(tmp_path / "missing-model"))
    monkeypatch.setenv("TINYLORA_EVAL_SPEC", str(eval_spec))

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "blocked_model_path_not_found"

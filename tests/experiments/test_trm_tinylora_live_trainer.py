from __future__ import annotations

import json
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
import param_decomp.experiments.trm_tinylora_live_trainer as live_trainer
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
    assert summary["backend_probe"]["safe_model_mb"] == 768


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


def test_live_trainer_peft_preflight_blocks_when_model_exceeds_fractional_cap(tmp_path: Path, monkeypatch: object) -> None:
    manifest_path = _write_manifest(tmp_path)
    model_path = tmp_path / "model"
    eval_spec = tmp_path / "eval.json"
    model_path.mkdir()
    (model_path / "weights.bin").write_bytes(b"0" * 2 * 1024 * 1024)
    eval_spec.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TINYLORA_JOB_OBJECT", "1")
    monkeypatch.setenv("TINYLORA_ENABLE_MODEL_LOAD", "1")
    monkeypatch.setenv("TINYLORA_MODEL_PATH", str(model_path))
    monkeypatch.setenv("TINYLORA_EVAL_SPEC", str(eval_spec))
    monkeypatch.setenv("TINYLORA_MODEL_SIZE_SAFETY_FRACTION", "0.0001")

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "blocked_model_size_exceeds_safe_cap"
    assert summary["backend_probe"]["safe_model_mb"] == 1


def test_live_trainer_peft_preflight_blocks_missing_target_module(tmp_path: Path, monkeypatch: object) -> None:
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
    monkeypatch.setenv("TINYLORA_VALIDATE_TARGET_MODULES", "1")
    monkeypatch.setattr(
        live_trainer,
        "inspect_target_modules",
        lambda _model_path, target_modules: {
            "all_found": False,
            "requested": target_modules,
            "found": [],
            "missing": target_modules,
            "suggestions": ["model.L_module.layers.0.self_attn.o_proj"],
        },
    )

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "blocked_target_module_not_found"
    assert summary["backend_probe"]["target_probe"]["suggestions"] == ["model.L_module.layers.0.self_attn.o_proj"]


def test_static_hrm_target_probe_suggests_native_modules(tmp_path: Path) -> None:
    model_path = tmp_path / "hrm"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "hrm_text", "num_hidden_layers": 2}),
        encoding="utf-8",
    )

    probe = live_trainer.inspect_target_modules(
        model_path,
        ["base_model.model.model.language_model.layers.19.self_attn.o_proj"],
    )

    assert probe["all_found"] is False
    assert probe["source"] == "static_hrm_text_config"
    assert "model.L_module.layers.0.self_attn.o_proj" in probe["suggestions"]


def test_transformers_interval_compat_installs_missing_helper(monkeypatch: object) -> None:
    import transformers.utils.type_validators as type_validators

    original = getattr(type_validators, "interval", None)
    if hasattr(type_validators, "interval"):
        monkeypatch.delattr(type_validators, "interval")

    live_trainer.ensure_transformers_interval_compat()

    assert type_validators.interval(min=0.0, max=1.0)(default=0.02) == 0.02
    if original is not None:
        monkeypatch.setattr(type_validators, "interval", original)


def test_live_trainer_peft_preflight_remaps_target_when_enabled(tmp_path: Path, monkeypatch: object) -> None:
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
    monkeypatch.setenv("TINYLORA_VALIDATE_TARGET_MODULES", "1")
    monkeypatch.setenv("TINYLORA_TARGET_REMAP", "first_suffix")
    monkeypatch.setattr(
        live_trainer,
        "inspect_target_modules",
        lambda _model_path, target_modules: {
            "all_found": False,
            "requested": target_modules,
            "found": [],
            "missing": target_modules,
            "suggestions": ["model.L_module.layers.0.self_attn.o_proj"],
        },
    )

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "blocked_target_remap_ready_adapter_smoke_not_enabled"
    request = json.loads(Path(summary["outputs"]["backend_request"]).read_text(encoding="utf-8"))
    assert request["target_module"] == "model.L_module.layers.0.self_attn.o_proj"


def test_live_trainer_peft_adapter_smoke_completed_path(tmp_path: Path, monkeypatch: object) -> None:
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
    monkeypatch.setenv("TINYLORA_ENABLE_ADAPTER_SMOKE", "1")
    monkeypatch.setattr(
        live_trainer,
        "run_peft_adapter_smoke",
        lambda out_dir, _model_path, _manifest, candidate, controls: {
            "status": "completed",
            "reason": "adapter_smoke_completed",
            "adapter_dir": str(out_dir / "adapter_smoke" / candidate["candidate_id"].replace(":", "_")),
            "trainable_parameters": [2, 10],
            "random_control_count": len(controls),
        },
    )

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "completed"
    assert summary["block_reason"] is None
    assert summary["backend_probe"]["reason"] == "adapter_smoke_completed"
    assert "Adapter smoke only" in summary["claim_boundary"]
    results = [json.loads(line) for line in (tmp_path / "out" / "tinylora_training_candidate_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert results[0]["decision_reason"] == "adapter_smoke_completed"


def test_live_trainer_peft_remap_can_reach_adapter_smoke(tmp_path: Path, monkeypatch: object) -> None:
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
    monkeypatch.setenv("TINYLORA_VALIDATE_TARGET_MODULES", "1")
    monkeypatch.setenv("TINYLORA_TARGET_REMAP", "first_suffix")
    monkeypatch.setenv("TINYLORA_ENABLE_ADAPTER_SMOKE", "1")
    monkeypatch.setattr(
        live_trainer,
        "inspect_target_modules",
        lambda _model_path, target_modules: {
            "all_found": False,
            "requested": target_modules,
            "found": [],
            "missing": target_modules,
            "suggestions": ["model.L_module.layers.0.self_attn.o_proj"],
        },
    )
    monkeypatch.setattr(
        live_trainer,
        "run_peft_adapter_smoke",
        lambda out_dir, _model_path, _manifest, candidate, controls: {
            "status": "completed",
            "reason": "adapter_smoke_completed",
            "adapter_dir": str(out_dir / "adapter_smoke" / candidate["candidate_id"].replace(":", "_")),
            "trainable_parameters": [2, 10],
            "random_control_count": len(controls),
            "target_module": candidate["target_module"],
        },
    )

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "completed"
    assert summary["backend_probe"]["target_module"] == "model.L_module.layers.0.self_attn.o_proj"


def test_live_trainer_peft_one_batch_completed_path(tmp_path: Path, monkeypatch: object) -> None:
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
    monkeypatch.setenv("TINYLORA_ENABLE_ADAPTER_SMOKE", "1")
    monkeypatch.setenv("TINYLORA_ENABLE_ONE_BATCH_TRAIN", "1")
    monkeypatch.setattr(
        live_trainer,
        "run_peft_adapter_smoke",
        lambda out_dir, _model_path, _manifest, candidate, controls: {
            "status": "completed",
            "reason": "one_batch_train_completed",
            "adapter_dir": str(out_dir / "adapter_one_batch" / candidate["candidate_id"].replace(":", "_")),
            "before_loss": 2.0,
            "after_loss": 1.9,
            "loss_delta": -0.1,
            "train_loss": 2.0,
            "optimizer": "sgd",
            "random_control_count": len(controls),
        },
    )

    summary = run_trainer(type("Args", (), {"manifest": manifest_path, "out_dir": None, "mode": "train_one", "backend": "peft_train_one", "max_candidates": 0, "candidate_id": None})())

    assert summary["status"] == "completed"
    assert summary["backend_probe"]["reason"] == "one_batch_train_completed"
    assert summary["backend_probe"]["loss_delta"] == -0.1
    assert summary["backend_probe"]["optimizer"] == "sgd"
    assert "One-batch adapter update" in summary["claim_boundary"]
    results = [json.loads(line) for line in (tmp_path / "out" / "tinylora_training_candidate_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert results[0]["decision_reason"] == "one_batch_train_completed"

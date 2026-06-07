from __future__ import annotations

import json
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_tinylora_overnight_search import (
    build_trial_grid,
    wait_for_ram_gate,
    materialize_trial_wrapper,
    prepare_trial_manifest,
    score_trial,
)


def _source_manifest(tmp_path: Path) -> Path:
    candidate = {
        "candidate_id": "task:candidate:001",
        "organism_id": "tiny_lora:g0:1",
        "target_module": "old.module",
        "source_family_key": "D->A:A:medium_1_0",
        "adapter_config": {
            "adapter_seed": 11,
            "rank": 1,
            "scale": 1.0,
            "target_modules": ["old.module"],
        },
        "proxy_score": {"delta": 0.1, "control_margin": 0.05, "fitness": 0.2},
    }
    control = {**candidate, "candidate_id": "task:random_control:001", "organism_id": "random_tinylora:001"}
    write_jsonl(tmp_path / "candidates.jsonl", [candidate])
    write_jsonl(tmp_path / "controls.jsonl", [control])
    manifest = {
        "training_task_id": "task",
        "caps": {"ram_mb": 4096, "cpu_pct": 50, "io_mb_s": 50},
        "paths": {
            "candidates": str(tmp_path / "candidates.jsonl"),
            "random_controls": str(tmp_path / "controls.jsonl"),
            "plan": str(tmp_path / "plan.json"),
            "comparison_table": str(tmp_path / "comparison.csv"),
            "event_schema": str(tmp_path / "schema.json"),
        },
        "default_output_dir": str(tmp_path / "source_out"),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_build_trial_grid_limits_cartesian_product() -> None:
    trials = build_trial_grid(["a", "b"], [0.1, 0.2, 0.3], max_seq_len=8, optimizer="sgd", max_trials=4)

    assert [trial.trial_id for trial in trials] == ["trial_001", "trial_002", "trial_003", "trial_004"]
    assert trials[0].target_module == "a"
    assert trials[3].target_module == "b"
    assert trials[3].learning_rate == 0.1


def test_prepare_trial_manifest_retargets_candidate_without_mutating_source(tmp_path: Path) -> None:
    manifest_path = _source_manifest(tmp_path)
    trial = build_trial_grid(["model.L_module.layers.0.self_attn.o_proj"], [0.0001], max_seq_len=8, optimizer="sgd", max_trials=1)[0]

    trial_manifest = prepare_trial_manifest(manifest_path, tmp_path / "search", trial)

    manifest = json.loads(trial_manifest.read_text(encoding="utf-8"))
    candidate_path = Path(manifest["paths"]["candidates"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8").splitlines()[0])
    source = json.loads((tmp_path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert manifest["default_output_dir"].endswith("trial_001\\execution") or manifest["default_output_dir"].endswith("trial_001/execution")
    assert candidate["target_module"] == "model.L_module.layers.0.self_attn.o_proj"
    assert candidate["adapter_config"]["target_modules"] == ["model.L_module.layers.0.self_attn.o_proj"]
    assert candidate["adapter_config"]["search_trial"]["learning_rate"] == 0.0001
    assert source["target_module"] == "old.module"


def test_score_trial_accepts_negative_finite_loss_delta() -> None:
    scored = score_trial(
        {
            "status": "completed",
            "backend_probe": {
                "reason": "one_batch_train_completed",
                "loss_delta": -0.25,
            },
        }
    )

    assert scored["accepted"] is True
    assert scored["score"] == 0.25
    assert scored["reason"] == "loss_improved"


def test_score_trial_rejects_nonfinite_or_blocked() -> None:
    scored = score_trial({"status": "blocked", "block_reason": "blocked_adapter_smoke_exception", "backend_probe": {}})

    assert scored["accepted"] is False
    assert scored["score"] < -999
    assert scored["reason"] == "blocked_adapter_smoke_exception"


def test_materialize_trial_wrapper_rewrites_job_memory_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.ps1"
    source.write_text("$MemoryLimitBytes = 4096MB\nWrite-Host 'ok'\n", encoding="utf-8")

    trial_dir = tmp_path / "trial_001"
    wrapper = materialize_trial_wrapper(source, trial_dir, 3072)

    assert wrapper.exists()
    text = wrapper.read_text(encoding="utf-8")
    assert "$MemoryLimitBytes = 3072MB" in text
    assert "$MemoryLimitBytes = 4096MB" not in text


def test_wait_for_ram_gate_requires_ram_pagefile_and_gpu(monkeypatch: object) -> None:
    values = {
        "ram": [5000, 7000],
        "pagefile": [3000, 5000],
        "gpu": [1024, 4096],
    }

    monkeypatch.setattr(
        "param_decomp.experiments.trm_tinylora_overnight_search.available_ram_mb",
        lambda: values["ram"].pop(0),
    )
    monkeypatch.setattr(
        "param_decomp.experiments.trm_tinylora_overnight_search.available_pagefile_mb",
        lambda: values["pagefile"].pop(0),
    )
    monkeypatch.setattr(
        "param_decomp.experiments.trm_tinylora_overnight_search.gpu_free_mb",
        lambda: values["gpu"].pop(0),
    )
    monkeypatch.setattr(
        "param_decomp.experiments.trm_tinylora_overnight_search.time.sleep",
        lambda _seconds: None,
    )

    ok, samples = wait_for_ram_gate(6144, 4096, 2048, 60, 1)

    assert ok is True
    assert samples[0]["available_ram_mb"] == 5000
    assert samples[0]["available_pagefile_mb"] == 3000
    assert samples[0]["gpu_free_mb"] == 1024

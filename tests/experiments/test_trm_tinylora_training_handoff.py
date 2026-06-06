from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_tinylora_training_handoff import run_handoff


def _accepted_row(organism_id: str, fitness: float) -> dict[str, object]:
    return {
        "organism_id": organism_id,
        "parent_ids": [],
        "source_family_key": "D->A:A:medium_1_0",
        "target_module": "layer.1.o_proj",
        "rank": 1,
        "alpha": 2.0,
        "scale": 1.0,
        "adapter_seed": 7,
        "trigger_top_action": "D",
        "trigger_runner_up_action": "A",
        "trigger_target_action": "A",
        "trigger_margin_bucket": "medium_1_0",
        "max_margin": 1.0,
        "score": {"fitness": fitness, "delta": 0.1, "control_margin": 0.05, "rescue_count": 2, "damage_count": 0, "touch_rate": 0.1},
    }


def test_handoff_emits_candidates_and_wrapper(tmp_path: Path) -> None:
    swarm = tmp_path / "swarm"
    swarm.mkdir()
    write_jsonl(swarm / "accepted_tinylora_organisms.jsonl", [_accepted_row("a", 0.1), _accepted_row("b", 0.2)])

    summary = run_handoff(
        Namespace(
            swarm_dir=swarm,
            out_dir=tmp_path / "out",
            top_n=1,
            random_control_count=1,
            ram_mb=2048,
            cpu_pct=50,
            io_mb_s=50,
            checkpoint_interval="generation_or_120s",
            training_task_id="test-task",
            max_prompt_tokens=1200,
        )
    )

    assert summary["status"] == "completed"
    assert summary["candidate_count"] == 1
    assert (tmp_path / "out" / "run_tinylora_jobobject.ps1").exists()
    wrapper = (tmp_path / "out" / "run_tinylora_jobobject.ps1").read_text(encoding="utf-8")
    assert '$env:TINYLORA_JOB_OBJECT = "1"' in wrapper
    assert '"--mode", "train_one"' in wrapper
    assert '"--backend", $Backend' in wrapper
    candidates = [json.loads(line) for line in (tmp_path / "out" / "tinylora_training_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert candidates[0]["organism_id"] == "b"
    assert candidates[0]["adapter_config"]["rank"] == 1
    assert candidates[0]["acceptance_gate"]["must_beat_random_tinylora_control"] is True
    assert (tmp_path / "out" / "tinylora_random_controls.jsonl").exists()
    assert (tmp_path / "out" / "tinylora_training_comparison.csv").exists()


def test_handoff_records_abort_as_valid_status(tmp_path: Path) -> None:
    swarm = tmp_path / "swarm"
    swarm.mkdir()
    write_jsonl(swarm / "accepted_tinylora_organisms.jsonl", [_accepted_row("a", 0.1)])

    run_handoff(
        Namespace(
            swarm_dir=swarm,
            out_dir=tmp_path / "out",
            top_n=1,
            random_control_count=1,
            ram_mb=1024,
            cpu_pct=25,
            io_mb_s=20,
            checkpoint_interval="generation_or_120s",
            training_task_id="test-task",
            max_prompt_tokens=1200,
        )
    )

    schema = json.loads((tmp_path / "out" / "tinylora_training_event_schema.json").read_text(encoding="utf-8"))
    plan = json.loads((tmp_path / "out" / "tinylora_training_plan.json").read_text(encoding="utf-8"))
    assert schema["abort_is_valid_status"] is True
    assert plan["caps"]["ram_mb"] == 1024

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_tinylora_auto_research_loop import run_auto_research_loop


def _sample(trajectory_id: str, target: str, top: str, runner: str, *, gap: float = 0.2) -> dict[str, object]:
    return {
        "env_id": "arc",
        "trajectory_id": trajectory_id,
        "target_action": target,
        "scores": [{"action": top, "logprob": -1.0}, {"action": runner, "logprob": -1.0 - gap}],
    }


def test_auto_research_loop_emits_agent_manageable_state(tmp_path: Path) -> None:
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            _sample("miss_b", "A", "B", "A"),
            _sample("correct_b", "B", "B", "D"),
            _sample("filler_a", "A", "A", "D"),
            _sample("filler_c", "C", "C", "D"),
        ],
    )

    summary = run_auto_research_loop(
        Namespace(
            score_files=str(score_file),
            out_dir=tmp_path / "loop",
            cycles=1,
            generations=2,
            population_size=8,
            elite_count=2,
            modules="m0,m1",
            ranks="1,2",
            scales="0.25",
            margins="0.25",
            top_n=1,
            random_control_count=1,
            trainer_mode="dry_run",
            ram_mb=1024,
            cpu_pct=25,
            io_mb_s=20,
            checkpoint_interval="generation_or_120s",
            seed=1,
            max_families=4,
            touch_rate_max=1.0,
            touch_penalty=0.05,
            complexity_penalty=0.003,
            max_prompt_tokens=1200,
        )
    )

    assert summary["status"] == "completed"
    assert summary["cycle_count"] == 1
    assert summary["research_state"] == "ready_for_train_one_backend"
    assert Path(summary["outputs"]["agent_packet"]).exists()
    state = json.loads((tmp_path / "loop" / "tinylora_auto_research_state.json").read_text(encoding="utf-8"))
    assert state["next_agent_action"] == "implement_or_run_train_one_backend_under_generated_jobobject_wrapper"
    cycles = [json.loads(line) for line in (tmp_path / "loop" / "tinylora_auto_research_cycles.jsonl").read_text(encoding="utf-8").splitlines()]
    assert cycles[0]["accepted_proxy_organisms"] >= 1
    assert cycles[0]["accepted_live_edits"] == 0
    assert cycles[0]["trainer_status"] == "completed"

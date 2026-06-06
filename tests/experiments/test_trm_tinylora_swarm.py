from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_tinylora_swarm import (
    TinyLoraOrganism,
    organism_policy,
    run_swarm,
    seed_organisms,
)


def _sample(trajectory_id: str, target: str, top: str, runner: str, *, gap: float = 0.2) -> dict[str, object]:
    return {
        "env_id": "arc",
        "trajectory_id": trajectory_id,
        "target_action": target,
        "scores": [{"action": top, "logprob": -1.0}, {"action": runner, "logprob": -1.0 - gap}],
    }


def _args(score_file: Path, out_dir: Path) -> Namespace:
    return Namespace(
        score_files=str(score_file),
        out_dir=out_dir,
        generations=2,
        population_size=8,
        elite_count=2,
        modules="m0,m1",
        ranks="1,2",
        scales="0.25",
        margins="0.25",
        touch_rate_max=1.0,
        touch_penalty=0.05,
        complexity_penalty=0.003,
        seed=1,
        max_families=4,
        max_prompt_tokens=1200,
    )


def test_organism_policy_is_conditional_trigger() -> None:
    organism = TinyLoraOrganism(
        organism_id="o",
        generation=0,
        parent_ids=(),
        target_module="m0",
        rank=1,
        alpha=2.0,
        scale=0.25,
        adapter_seed=1,
        trigger_top_action="B",
        trigger_runner_up_action="A",
        trigger_target_action="A",
        trigger_margin_bucket="close_0_25",
        max_margin=0.25,
        mutation_ops=("seed",),
        source_family_key="B->A:A:close_0_25",
    )

    policy = organism_policy(organism)

    assert policy["type"] == "conditional_top_margin_penalty"
    assert policy["top_action"] == "B"
    assert policy["runner_up_action"] == "A"
    assert policy["penalty"] == 0.25


def test_seed_organisms_carries_family_and_adapter_fields() -> None:
    organisms = seed_organisms(
        [{"family_key": "B->A:A:close_0_25", "top_action": "B", "runner_up_action": "A", "target_action": "A", "margin_bucket": "close_0_25"}],
        modules=["m0"],
        ranks=[1],
        scales=[0.25],
        margins=[0.25],
        population_size=1,
        rng=__import__("random").Random(1),
    )

    assert organisms[0].source_family_key == "B->A:A:close_0_25"
    assert organisms[0].rank == 1
    assert organisms[0].alpha == 2.0
    assert organisms[0].target_module == "m0"


def test_run_swarm_emits_accepted_proxy_organism(tmp_path: Path) -> None:
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

    summary = run_swarm(_args(score_file, tmp_path / "out"))

    assert summary["status"] == "completed"
    assert summary["organism_count"] > 0
    assert summary["accepted_organism_count"] >= 1
    assert "Cached tinyLoRA swarm proxy" in summary["claim_boundary"]
    assert (tmp_path / "out" / "tinylora_organisms.jsonl").exists()
    assert (tmp_path / "out" / "tinylora_swarm_events.jsonl").exists()

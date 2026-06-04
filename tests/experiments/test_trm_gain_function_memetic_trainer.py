from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_function_memetic_trainer import (
    gain_candidate_from_score,
    matched_control_penalty,
    policy_state_from_elites,
    raw_gain_score,
    train_gain_policy,
)


WEIGHTS = {
    "score_delta": 1.0,
    "rescue_bonus": 0.01,
    "damage_penalty": 0.02,
    "touch_rate_penalty": 0.02,
    "complexity_penalty": 0.002,
    "matched_control_penalty": 0.05,
}


def scored(policy: dict, *, reward: float, delta: float, rescues: int, damages: int, touched: int, samples: int = 10) -> dict:
    return {
        "policy": policy,
        "reward": reward,
        "delta": delta,
        "rescue_count": rescues,
        "damage_count": damages,
        "touched_count": touched,
        "touch_rate": touched / samples,
        "sample_count": samples,
        "accepted": delta > 0 and rescues > damages,
    }


def test_raw_gain_score_penalizes_damage_when_delta_ties() -> None:
    clean = scored(
        {"id": "top_margin:max_0_25:penalty_0_25", "type": "top_margin_penalty"},
        reward=0.08,
        delta=0.05,
        rescues=3,
        damages=0,
        touched=2,
    )
    damaged = scored(
        {"id": "top_margin:max_0_5:penalty_0_5", "type": "top_margin_penalty"},
        reward=0.08,
        delta=0.05,
        rescues=5,
        damages=3,
        touched=2,
    )

    assert raw_gain_score(clean, WEIGHTS) > raw_gain_score(damaged, WEIGHTS)


def test_matched_control_penalty_blocks_broad_fixed_label_tie() -> None:
    broad = scored(
        {"id": "fixed:D:penalty_1_0", "type": "fixed_action_penalty", "action": "D", "penalty": 1.0},
        reward=0.08,
        delta=0.05,
        rescues=4,
        damages=1,
        touched=10,
    )
    selective = scored(
        {"id": "top_margin:max_0_25:penalty_0_25", "type": "top_margin_penalty", "max_margin": 0.25, "penalty": 0.25},
        reward=0.08,
        delta=0.05,
        rescues=3,
        damages=0,
        touched=2,
    )

    assert matched_control_penalty(broad, [broad, selective], WEIGHTS["matched_control_penalty"]) == 0.05
    assert matched_control_penalty(selective, [broad, selective], WEIGHTS["matched_control_penalty"]) == 0.0


def test_gain_candidate_maps_top_margin_to_vpd_edit_family() -> None:
    row = scored(
        {"id": "top_margin:max_0_25:penalty_0_25", "type": "top_margin_penalty", "max_margin": 0.25, "penalty": 0.25},
        reward=0.08,
        delta=0.05,
        rescues=3,
        damages=0,
        touched=2,
    )

    candidate = gain_candidate_from_score(row, env_ids=["arc_challenge"], weights=WEIGHTS, peers=[row])

    assert candidate["edit_family"] == "selective_top_margin_suppression"
    assert candidate["source_feature_hint"] == "overselected_top_choice"
    assert candidate["promotion_status"] == "promote"


def test_policy_state_prefers_selective_and_downranks_broad_prior() -> None:
    elites = [
        {
            "candidate_id": "gain:top",
            "edit_family": "selective_top_margin_suppression",
            "fitness": 0.07,
            "matched_control_penalty": 0.0,
        },
        {
            "candidate_id": "gain:fixed",
            "edit_family": "broad_fixed_label_prior",
            "fitness": 0.02,
            "matched_control_penalty": 0.05,
        },
    ]

    state = policy_state_from_elites(elites, weights=WEIGHTS, ram_cap_mb=8192, max_prompt_tokens=1200)

    assert "selective_top_margin_suppression" in state["preferred_families"]
    assert "broad_fixed_label_prior" in state["downranked_families"]
    assert state["trained_object"] == "vpd_edit_policy"


def test_train_gain_policy_writes_controller_artifacts(tmp_path: Path) -> None:
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc_challenge",
                "trajectory_id": "a",
                "target_action": "A",
                "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.1}],
            },
            {
                "env_id": "arc_challenge",
                "trajectory_id": "b",
                "target_action": "C",
                "scores": [{"action": "C", "logprob": -1.0}, {"action": "B", "logprob": -4.0}],
            },
        ],
    )
    args = argparse.Namespace(
        score_files=str(score_file),
        out_dir=tmp_path / "out",
        penalties="0.25,0.5",
        margins="0.25,0.5",
        generations=2,
        elite_count=4,
        population_size=8,
        ram_cap_mb=8192,
        checkpoint_interval_generations=1,
        checkpoint_interval_seconds=600,
        max_prompt_tokens=1200,
        score_delta_weight=1.0,
        rescue_bonus=0.01,
        damage_penalty=0.02,
        touch_rate_penalty=0.02,
        complexity_penalty=0.002,
        matched_control_penalty=0.05,
    )

    summary = train_gain_policy(args)
    state = json.loads((args.out_dir / "vpd_edit_policy_state.json").read_text(encoding="utf-8"))

    assert summary["status"] == "completed"
    assert (args.out_dir / "gain_function_population.jsonl").exists()
    assert (args.out_dir / "training_events.jsonl").exists()
    assert (args.out_dir / "checkpoints" / "generation_0000.json").exists()
    assert state["policy_id"] == "vpd_edit_policy_arc_bootstrap_v1"

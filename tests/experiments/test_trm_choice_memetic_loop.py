from __future__ import annotations

from param_decomp.experiments.trm_choice_memetic_loop import (
    evaluate_policy,
    fitness_from_score,
    local_refine,
    run_memetic_search,
)


def test_fitness_prefers_selective_policy_when_reward_ties() -> None:
    broad = {
        "policy": {"type": "fixed_action_penalty"},
        "reward": 0.08,
        "touched_count": 50,
        "sample_count": 50,
    }
    selective = {
        "policy": {"type": "top_margin_penalty"},
        "reward": 0.08,
        "touched_count": 5,
        "sample_count": 50,
    }

    assert fitness_from_score(selective, touch_penalty=0.02, complexity_penalty=0.002) > fitness_from_score(
        broad, touch_penalty=0.02, complexity_penalty=0.002
    )


def test_local_refine_selects_best_top_margin_variant() -> None:
    samples = [
        {
            "env_id": "arc",
            "trajectory_id": "close",
            "target_action": "A",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
        },
        {
            "env_id": "arc",
            "trajectory_id": "wide",
            "target_action": "C",
            "scores": [{"action": "C", "logprob": -1.0}, {"action": "A", "logprob": -5.0}],
        },
    ]

    refined = local_refine(
        samples,
        {"type": "top_margin_penalty", "max_margin": 0.25, "penalty": 0.25},
        penalties=[0.25, 0.5],
        margins=[0.1, 0.25],
        touch_penalty=0.02,
        complexity_penalty=0.002,
    )

    assert refined["accepted"] is True
    assert refined["policy"]["type"] == "top_margin_penalty"
    assert refined["policy_score"] == 1.0


def test_memetic_search_promotes_selective_top_margin_elite() -> None:
    samples = [
        {
            "env_id": "arc",
            "trajectory_id": "a",
            "target_action": "A",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
        },
        {
            "env_id": "arc",
            "trajectory_id": "b",
            "target_action": "C",
            "scores": [{"action": "C", "logprob": -1.0}, {"action": "A", "logprob": -4.0}],
        },
        {
            "env_id": "arc",
            "trajectory_id": "c",
            "target_action": "D",
            "scores": [{"action": "D", "logprob": -1.0}, {"action": "B", "logprob": -4.0}],
        },
    ]

    _history, elites = run_memetic_search(
        samples,
        penalties=[0.5],
        margins=[0.25],
        generations=2,
        elite_count=2,
        population_size=8,
        touch_penalty=0.02,
        complexity_penalty=0.002,
    )

    assert elites[0]["policy"]["type"] == "top_margin_penalty"
    assert elites[0]["accepted"] is True
    assert elites[0]["touch_rate"] < 1.0


def test_memetic_initial_generation_is_not_population_truncated() -> None:
    samples = [
        {
            "env_id": "arc",
            "trajectory_id": "a",
            "target_action": "A",
            "scores": [
                {"action": "Z9", "logprob": -1.0},
                {"action": "A", "logprob": -1.1},
                {"action": "B", "logprob": -3.0},
            ],
        },
        {
            "env_id": "arc",
            "trajectory_id": "b",
            "target_action": "C",
            "scores": [{"action": "C", "logprob": -1.0}, {"action": "B", "logprob": -3.0}],
        },
    ]

    history, elites = run_memetic_search(
        samples,
        penalties=[0.5],
        margins=[0.25],
        generations=1,
        elite_count=2,
        population_size=1,
        touch_penalty=0.02,
        complexity_penalty=0.002,
    )

    assert any(row["policy"]["type"] == "top_margin_penalty" for row in history)
    assert elites


def test_evaluate_policy_records_touch_rate_and_fitness() -> None:
    samples = [
        {
            "trajectory_id": "a",
            "target_action": "A",
            "scores": [{"action": "B", "logprob": -1.0}, {"action": "A", "logprob": -1.2}],
        }
    ]

    scored = evaluate_policy(
        samples,
        {"type": "top_margin_penalty", "max_margin": 0.25, "penalty": 0.5},
        touch_penalty=0.02,
        complexity_penalty=0.002,
    )

    assert scored["touch_rate"] == 1.0
    assert "fitness" in scored

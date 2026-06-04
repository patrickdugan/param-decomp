"""Memetic cached search over constrained-choice TRM route rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import (
    candidate_policies,
    dedupe_samples,
    parse_floats,
    parse_paths,
    read_jsonl,
    score_policy,
    write_jsonl,
)


DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_choice_memetic_loop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic memetic search over cached choice score cards.")
    parser.add_argument("--score-files", required=True, help="Comma or semicolon-separated sample-score files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--penalties", default="0.25,0.5,1,2,4")
    parser.add_argument("--margins", default="0.25,0.5,1,2")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--touch-penalty", type=float, default=0.02)
    parser.add_argument("--complexity-penalty", type=float, default=0.002)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def policy_id(policy: dict[str, Any]) -> str:
    if policy["type"] == "fixed_action_penalty":
        return f"fixed:{policy['action']}:penalty_{str(policy['penalty']).replace('.', '_')}"
    if policy["type"] == "top_margin_penalty":
        margin = str(policy["max_margin"]).replace(".", "_")
        penalty = str(policy["penalty"]).replace(".", "_")
        return f"top_margin:max_{margin}:penalty_{penalty}"
    raise ValueError(f"Unsupported policy type: {policy['type']}")


def normalized_policy(policy: dict[str, Any]) -> dict[str, Any]:
    row = dict(policy)
    row["id"] = policy_id(row)
    return row


def fitness_from_score(scored: dict[str, Any], touch_penalty: float, complexity_penalty: float) -> float:
    touch_rate = scored["touched_count"] / max(1, scored["sample_count"])
    policy_complexity = 2 if scored["policy"]["type"] == "top_margin_penalty" else 1
    return round(scored["reward"] - touch_penalty * touch_rate - complexity_penalty * policy_complexity, 6)


def evaluate_policy(
    samples: list[dict[str, Any]],
    policy: dict[str, Any],
    touch_penalty: float,
    complexity_penalty: float,
) -> dict[str, Any]:
    scored = score_policy(samples, normalized_policy(policy))
    scored["touch_rate"] = round(scored["touched_count"] / max(1, scored["sample_count"]), 6)
    scored["fitness"] = fitness_from_score(scored, touch_penalty, complexity_penalty)
    return scored


def local_refine(
    samples: list[dict[str, Any]],
    policy: dict[str, Any],
    penalties: list[float],
    margins: list[float],
    touch_penalty: float,
    complexity_penalty: float,
) -> dict[str, Any]:
    variants = []
    if policy["type"] == "fixed_action_penalty":
        variants = [
            {"type": "fixed_action_penalty", "action": policy["action"], "penalty": penalty}
            for penalty in penalties
        ]
    elif policy["type"] == "top_margin_penalty":
        variants = [
            {"type": "top_margin_penalty", "max_margin": margin, "penalty": penalty}
            for margin in margins
            for penalty in penalties
        ]
    else:
        raise ValueError(f"Unsupported policy type: {policy['type']}")
    scored = [evaluate_policy(samples, variant, touch_penalty, complexity_penalty) for variant in variants]
    scored.sort(key=lambda row: (row["accepted"], row["fitness"], row["reward"], row["delta"]), reverse=True)
    return scored[0]


def mutate_policy(policy: dict[str, Any], penalties: list[float], margins: list[float]) -> list[dict[str, Any]]:
    mutated = []
    if policy["type"] == "fixed_action_penalty":
        current = policy["penalty"]
        if current in penalties:
            index = penalties.index(current)
            for next_index in {max(0, index - 1), min(len(penalties) - 1, index + 1)}:
                mutated.append(
                    {"type": "fixed_action_penalty", "action": policy["action"], "penalty": penalties[next_index]}
                )
    elif policy["type"] == "top_margin_penalty":
        penalty = policy["penalty"]
        margin = policy["max_margin"]
        if penalty in penalties:
            index = penalties.index(penalty)
            for next_index in {max(0, index - 1), min(len(penalties) - 1, index + 1)}:
                mutated.append({"type": "top_margin_penalty", "max_margin": margin, "penalty": penalties[next_index]})
        if margin in margins:
            index = margins.index(margin)
            for next_index in {max(0, index - 1), min(len(margins) - 1, index + 1)}:
                mutated.append({"type": "top_margin_penalty", "max_margin": margins[next_index], "penalty": penalty})
    return [normalized_policy(row) for row in mutated]


def crossover_policy(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    if left["type"] != right["type"]:
        return None
    if left["type"] == "fixed_action_penalty":
        if left["action"] != right["action"]:
            return None
        return normalized_policy(
            {"type": "fixed_action_penalty", "action": left["action"], "penalty": min(left["penalty"], right["penalty"])}
        )
    if left["type"] == "top_margin_penalty":
        return normalized_policy(
            {
                "type": "top_margin_penalty",
                "max_margin": min(left["max_margin"], right["max_margin"]),
                "penalty": min(left["penalty"], right["penalty"]),
            }
        )
    return None


def next_population(
    elites: list[dict[str, Any]],
    penalties: list[float],
    margins: list[float],
    population_size: int,
) -> list[dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for elite in elites:
        policy = normalized_policy(elite["policy"])
        policies[policy["id"]] = policy
        for mutated in mutate_policy(policy, penalties, margins):
            policies[mutated["id"]] = mutated
    for left_index, left in enumerate(elites):
        for right in elites[left_index + 1 :]:
            crossed = crossover_policy(left["policy"], right["policy"])
            if crossed is not None:
                policies[crossed["id"]] = crossed
    ordered = sorted(policies.values(), key=lambda row: row["id"])
    return ordered[:population_size]


def run_memetic_search(
    samples: list[dict[str, Any]],
    penalties: list[float],
    margins: list[float],
    generations: int,
    elite_count: int,
    population_size: int,
    touch_penalty: float,
    complexity_penalty: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    population = [normalized_policy(row) for row in candidate_policies(samples, penalties, margins)]
    history = []
    final_elites = []
    for generation in range(generations):
        refined_by_id: dict[str, dict[str, Any]] = {}
        for policy in population:
            refined = local_refine(samples, policy, penalties, margins, touch_penalty, complexity_penalty)
            refined_by_id[refined["policy"]["id"]] = refined
        ranked = sorted(
            refined_by_id.values(),
            key=lambda row: (row["accepted"], row["fitness"], row["reward"], row["delta"]),
            reverse=True,
        )
        final_elites = ranked[:elite_count]
        for rank, row in enumerate(ranked, start=1):
            history.append(
                {
                    "generation": generation,
                    "rank": rank,
                    "policy": row["policy"],
                    "accepted": row["accepted"],
                    "baseline_score": row["baseline_score"],
                    "policy_score": row["policy_score"],
                    "delta": row["delta"],
                    "reward": row["reward"],
                    "fitness": row["fitness"],
                    "touch_rate": row["touch_rate"],
                    "rescue_count": row["rescue_count"],
                    "damage_count": row["damage_count"],
                    "touched_count": row["touched_count"],
                }
            )
        population = next_population(final_elites, penalties, margins, population_size)
    return history, final_elites


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None) -> str:
    policy = best["policy"] if best else {}
    replay = None
    if best:
        replay = next((row for row in best["details"] if not row["baseline_hit"] and row["hit"]), None)
    lines = [
        "TASK: Evolve next constrained-choice TRM route rule.",
        "CURRENT STATE:",
        f"- envs: {', '.join(summary['env_ids'])}",
        f"- samples: {summary['sample_count']} deduped from {summary['raw_sample_count']} raw",
        f"- generations: {summary['generations']}",
        f"- elite_count: {summary['elite_count']}",
        "FITNESS:",
        "- fitness = reward - touch_penalty*touch_rate - complexity_penalty*policy_complexity",
        f"- touch_penalty: {summary['touch_penalty']}",
        "ELITE:",
        f"- id: {policy.get('id')}",
        f"- type: {policy.get('type')}",
        f"- reward/fitness: {best.get('reward') if best else 0}/{best.get('fitness') if best else 0}",
        f"- delta: {best.get('delta') if best else 0}",
        f"- touch_rate: {best.get('touch_rate') if best else 0}",
        f"- rescues/damages: {best.get('rescue_count') if best else 0}/{best.get('damage_count') if best else 0}",
        "REPLAY HINT:",
        "- none" if not replay else (
            f"- {replay['trajectory_id']}: baseline {replay['baseline_prediction']} -> "
            f"policy {replay['prediction']} matched target {replay['target_action']}"
        ),
        "NEXT ACTION:",
        "- Prefer selective elites over broad fixed-label priors when reward ties.",
        "- Schedule a fresh score card if elite fitness remains positive.",
        "MCP RESOURCES:",
        f"- population_history: {summary['outputs']['population_history']}",
        f"- elite_candidates: {summary['outputs']['elite_candidates']}",
        f"- self_model: {summary['outputs']['memetic_state']}",
        f"- replay_candidates: {summary['outputs']['replay_candidates']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    samples = dedupe_samples(raw_samples)
    penalties = parse_floats(args.penalties)
    margins = parse_floats(args.margins)
    history, elites = run_memetic_search(
        samples,
        penalties,
        margins,
        args.generations,
        args.elite_count,
        args.population_size,
        args.touch_penalty,
        args.complexity_penalty,
    )
    best = elites[0] if elites else None
    replay_candidates = []
    if best:
        replay_candidates = [row for row in best["details"] if not row["baseline_hit"] and row["hit"]]
    summary = {
        "status": "completed",
        "score_files": [str(path) for path in score_files],
        "env_ids": sorted({sample.get("env_id", "") for sample in samples}),
        "raw_sample_count": len(raw_samples),
        "sample_count": len(samples),
        "deduped_sample_count": len(samples),
        "generations": args.generations,
        "elite_count": len(elites),
        "population_history_count": len(history),
        "touch_penalty": args.touch_penalty,
        "complexity_penalty": args.complexity_penalty,
        "best_policy": best["policy"] if best else None,
        "best_reward": best["reward"] if best else 0.0,
        "best_fitness": best["fitness"] if best else 0.0,
        "best_delta": best["delta"] if best else 0.0,
        "outputs": {
            "summary": str(args.out_dir / "choice_memetic_summary.json"),
            "population_history": str(args.out_dir / "population_history.jsonl"),
            "elite_candidates": str(args.out_dir / "elite_candidates.jsonl"),
            "memetic_state": str(args.out_dir / "memetic_state.json"),
            "replay_candidates": str(args.out_dir / "replay_candidates.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    state = {
        "best_policy": best["policy"] if best else None,
        "recent_pass_rate": best["policy_score"] if best else 0.0,
        "baseline_pass_rate": best["baseline_score"] if best else 0.0,
        "best_reward": best["reward"] if best else 0.0,
        "best_fitness": best["fitness"] if best else 0.0,
        "recurring_failure_modes": ["overselected_top_choice", "broad_label_prior"],
        "selection_rule": "prefer positive reward, then lower touch_rate, then compact policy",
        "prompt_budget_target_tokens": args.max_prompt_tokens,
    }
    write_jsonl(args.out_dir / "population_history.jsonl", history)
    write_jsonl(args.out_dir / "elite_candidates.jsonl", elites)
    write_jsonl(args.out_dir / "replay_candidates.jsonl", replay_candidates)
    (args.out_dir / "memetic_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    packet = compact_packet(summary, best)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "choice_memetic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

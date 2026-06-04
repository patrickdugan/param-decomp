"""Controlled memetic trainer for VPD edit-policy gain functions."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_memetic_loop import (
    evaluate_policy,
    next_population,
    normalized_policy,
)
from param_decomp.experiments.trm_choice_rl_feedback_loop import (
    candidate_policies,
    dedupe_samples,
    parse_floats,
    parse_paths,
    read_jsonl,
    write_jsonl,
)


DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_function_memetic_trainer")
DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact VPD edit-policy state from memetic route-rule evidence.")
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--penalties", default="0.25,0.5,1,2,4")
    parser.add_argument("--margins", default="0.25,0.5,1,2")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--ram-cap-mb", type=int, default=8192)
    parser.add_argument("--checkpoint-interval-generations", type=int, default=1)
    parser.add_argument("--checkpoint-interval-seconds", type=int, default=600)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    parser.add_argument("--score-delta-weight", type=float, default=1.0)
    parser.add_argument("--rescue-bonus", type=float, default=0.01)
    parser.add_argument("--damage-penalty", type=float, default=0.02)
    parser.add_argument("--touch-rate-penalty", type=float, default=0.02)
    parser.add_argument("--complexity-penalty", type=float, default=0.002)
    parser.add_argument("--matched-control-penalty", type=float, default=0.05)
    return parser.parse_args()


def policy_complexity(policy: dict[str, Any]) -> int:
    if policy["type"] == "top_margin_penalty":
        return 2
    return 1


def edit_family_for(policy: dict[str, Any]) -> str:
    if policy["type"] == "top_margin_penalty":
        return "selective_top_margin_suppression"
    if policy["type"] == "fixed_action_penalty":
        return "broad_fixed_label_prior"
    return "unknown"


def feature_hint_for(policy: dict[str, Any]) -> str:
    if policy["type"] == "top_margin_penalty":
        return "overselected_top_choice"
    if policy["type"] == "fixed_action_penalty":
        return "label_bias"
    return "unknown"


def gate_for(policy: dict[str, Any], env_ids: list[str]) -> str:
    env = env_ids[0] if env_ids else "choice_env"
    if policy["type"] == "top_margin_penalty":
        return f"{env}:choice_margin"
    return f"{env}:answer_label"


def raw_gain_score(scored: dict[str, Any], weights: dict[str, float]) -> float:
    sample_count = max(1, int(scored["sample_count"]))
    touch_rate = float(scored["touched_count"]) / sample_count
    return round(
        weights["score_delta"] * float(scored["delta"])
        + weights["rescue_bonus"] * float(scored["rescue_count"])
        - weights["damage_penalty"] * float(scored["damage_count"])
        - weights["touch_rate_penalty"] * touch_rate
        - weights["complexity_penalty"] * policy_complexity(scored["policy"]),
        6,
    )


def matched_control_penalty(scored: dict[str, Any], peers: list[dict[str, Any]], weight: float) -> float:
    policy = scored["policy"]
    if policy["type"] != "fixed_action_penalty":
        return 0.0
    score_reward = float(scored["reward"])
    score_touch = float(scored["touch_rate"])
    for peer in peers:
        if peer["policy"]["type"] != "top_margin_penalty":
            continue
        if float(peer["reward"]) >= score_reward and float(peer["touch_rate"]) < score_touch:
            return weight
    return 0.0


def gain_candidate_from_score(
    scored: dict[str, Any],
    *,
    env_ids: list[str],
    weights: dict[str, float],
    peers: list[dict[str, Any]],
) -> dict[str, Any]:
    penalty = matched_control_penalty(scored, peers, weights["matched_control_penalty"])
    fitness = round(raw_gain_score(scored, weights) - penalty, 6)
    policy = scored["policy"]
    return {
        "candidate_id": f"gain:{policy['id']}",
        "policy": policy,
        "edit_family": edit_family_for(policy),
        "gate": gate_for(policy, env_ids),
        "parameters": {key: value for key, value in policy.items() if key not in {"id", "type"}},
        "source_feature_hint": feature_hint_for(policy),
        "score_delta": scored["delta"],
        "reward": scored["reward"],
        "fitness": fitness,
        "rescue_count": scored["rescue_count"],
        "damage_count": scored["damage_count"],
        "touch_rate": scored["touch_rate"],
        "matched_control_penalty": penalty,
        "accepted": bool(scored["accepted"] and fitness > 0.0),
        "promotion_status": "promote" if scored["accepted"] and fitness > 0.0 and penalty == 0.0 else "hold",
        "claim_boundary": "controller_policy_training_from_route_rule_evidence",
    }


def rank_scored(scored_rows: list[dict[str, Any]], weights: dict[str, float], env_ids: list[str]) -> list[dict[str, Any]]:
    enriched = []
    for scored in scored_rows:
        scored = dict(scored)
        scored["touch_rate"] = round(float(scored["touched_count"]) / max(1, int(scored["sample_count"])), 6)
        enriched.append(scored)
    candidates = [gain_candidate_from_score(row, env_ids=env_ids, weights=weights, peers=enriched) for row in enriched]
    candidates.sort(key=lambda row: (row["accepted"], row["fitness"], row["reward"], -row["touch_rate"]), reverse=True)
    return candidates


def policy_state_from_elites(
    elites: list[dict[str, Any]],
    *,
    weights: dict[str, float],
    ram_cap_mb: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    family_scores: dict[str, list[float]] = {}
    for elite in elites:
        family_scores.setdefault(str(elite["edit_family"]), []).append(float(elite["fitness"]))
    preferred = [
        family
        for family, scores in sorted(family_scores.items(), key=lambda item: max(item[1]), reverse=True)
        if max(scores) > 0.0 and family != "broad_fixed_label_prior"
    ]
    downranked = []
    if any(elite["edit_family"] == "broad_fixed_label_prior" and elite["matched_control_penalty"] > 0.0 for elite in elites):
        downranked.append("broad_fixed_label_prior")
    if "selective_top_margin_suppression" not in preferred and any(
        elite["edit_family"] == "selective_top_margin_suppression" and elite["fitness"] > 0.0 for elite in elites
    ):
        preferred.append("selective_top_margin_suppression")
    if "selective_top_margin_suppression" in preferred and "broad_fixed_label_prior" not in downranked:
        downranked.append("broad_fixed_label_prior")
    return {
        "policy_id": "vpd_edit_policy_arc_bootstrap_v1",
        "trained_object": "vpd_edit_policy",
        "preferred_families": preferred,
        "downranked_families": downranked,
        "reward_terms": weights,
        "promotion_rule": "promote only if positive fitness on fresh seed and lower/equal touch than matched controls",
        "resource_profile": {
            "ram_cap_mb": ram_cap_mb,
            "checkpoint_interval_generations": 1,
            "abort_is_valid_output": True,
        },
        "best_candidate": elites[0] if elites else None,
        "prompt_budget_target_tokens": max_prompt_tokens,
    }


def memory_mb() -> float:
    current, _peak = tracemalloc.get_traced_memory()
    return round(current / (1024 * 1024), 3)


def peak_memory_mb() -> float:
    _current, peak = tracemalloc.get_traced_memory()
    return round(peak / (1024 * 1024), 3)


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_packet(summary: dict[str, Any], policy_state: dict[str, Any]) -> str:
    best = policy_state.get("best_candidate") or {}
    lines = [
        "TASK: Choose next VPD edit-policy scoring action.",
        "CURRENT STATE:",
        f"- policy_id: {policy_state['policy_id']}",
        f"- samples: {summary['deduped_sample_count']} deduped from {summary['raw_sample_count']} raw",
        f"- generations: {summary['generations_completed']}",
        f"- best_candidate: {best.get('candidate_id')}",
        f"- best_family: {best.get('edit_family')}",
        f"- fitness/reward/delta: {best.get('fitness')}/{best.get('reward')}/{best.get('score_delta')}",
        f"- touch_rate: {best.get('touch_rate')}",
        f"- rescues/damages: {best.get('rescue_count')}/{best.get('damage_count')}",
        "POLICY:",
        f"- preferred: {', '.join(policy_state['preferred_families']) or 'none'}",
        f"- downranked: {', '.join(policy_state['downranked_families']) or 'none'}",
        "NEXT ACTION:",
        "- Run one fresh ARC Challenge score card for the best selective family.",
        "- Promote to VPD-hook search only after fresh-seed fitness remains positive.",
        "MCP RESOURCES:",
        f"- population: {summary['outputs']['population']}",
        f"- fitness_history: {summary['outputs']['fitness_history']}",
        f"- self_model: {summary['outputs']['policy_state']}",
        f"- training_events: {summary['outputs']['training_events']}",
    ]
    return "\n".join(lines) + "\n"


def train_gain_policy(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    weights = {
        "score_delta": args.score_delta_weight,
        "rescue_bonus": args.rescue_bonus,
        "damage_penalty": args.damage_penalty,
        "touch_rate_penalty": args.touch_rate_penalty,
        "complexity_penalty": args.complexity_penalty,
        "matched_control_penalty": args.matched_control_penalty,
    }
    tracemalloc.start()
    start = time.time()
    last_checkpoint = start
    events = [
        {
            "ts": utc_now(),
            "event": "start",
            "training_task_id": "trm_gain_function_memetic_arc_bootstrap",
            "caps": {"ram_mb": args.ram_cap_mb, "checkpoint_interval_seconds": args.checkpoint_interval_seconds},
        }
    ]
    score_files = parse_paths(args.score_files)
    raw_samples = [sample for path in score_files for sample in read_jsonl(path)]
    samples = dedupe_samples(raw_samples)
    env_ids = sorted({str(sample.get("env_id") or "") for sample in samples})
    penalties = parse_floats(args.penalties)
    margins = parse_floats(args.margins)
    population = [normalized_policy(row) for row in candidate_policies(samples, penalties, margins)]
    fitness_history = []
    all_candidates = []
    final_elites: list[dict[str, Any]] = []
    status = "completed"
    abort_reason = None

    for generation in range(args.generations):
        scored_rows = [
            evaluate_policy(samples, policy, args.touch_rate_penalty, args.complexity_penalty)
            for policy in population
        ]
        ranked = rank_scored(scored_rows, weights, env_ids)
        final_elites = ranked[: args.elite_count]
        for rank, row in enumerate(ranked, start=1):
            record = {"generation": generation, "rank": rank, **row}
            fitness_history.append(record)
            all_candidates.append(record)
        events.append(
            {
                "ts": utc_now(),
                "event": "generation",
                "generation": generation,
                "population_size": len(population),
                "best_candidate": final_elites[0]["candidate_id"] if final_elites else None,
                "best_fitness": final_elites[0]["fitness"] if final_elites else 0.0,
                "peak_ram_mb": peak_memory_mb(),
            }
        )
        checkpoint_due = (
            (generation + 1) % max(1, args.checkpoint_interval_generations) == 0
            or time.time() - last_checkpoint >= args.checkpoint_interval_seconds
        )
        if checkpoint_due:
            write_checkpoint(
                checkpoint_dir / f"generation_{generation:04d}.json",
                {
                    "generation": generation,
                    "population": population,
                    "elite_candidates": final_elites,
                    "weights": weights,
                    "peak_ram_mb": peak_memory_mb(),
                },
            )
            events.append({"ts": utc_now(), "event": "checkpoint", "generation": generation, "path": str(checkpoint_dir / f"generation_{generation:04d}.json")})
            last_checkpoint = time.time()
        if peak_memory_mb() > args.ram_cap_mb:
            status = "aborted"
            abort_reason = "ram_cap_exceeded"
            events.append({"ts": utc_now(), "event": "abort", "reason": abort_reason, "peak_ram_mb": peak_memory_mb()})
            break
        elite_as_scored = [
            {
                "policy": elite["policy"],
                "accepted": elite["accepted"],
                "fitness": elite["fitness"],
                "reward": elite["reward"],
                "delta": elite["score_delta"],
            }
            for elite in final_elites
        ]
        population = next_population(elite_as_scored, penalties, margins, args.population_size)

    policy_state = policy_state_from_elites(
        final_elites,
        weights=weights,
        ram_cap_mb=args.ram_cap_mb,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    write_jsonl(out_dir / "gain_function_population.jsonl", all_candidates)
    write_jsonl(out_dir / "gain_function_fitness.jsonl", fitness_history)
    write_jsonl(out_dir / "training_events.jsonl", events)
    (out_dir / "vpd_edit_policy_state.json").write_text(json.dumps(policy_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": status,
        "abort_reason": abort_reason,
        "training_task_id": "trm_gain_function_memetic_arc_bootstrap",
        "score_files": [str(path) for path in score_files],
        "raw_sample_count": len(raw_samples),
        "deduped_sample_count": len(samples),
        "generations_completed": len({row["generation"] for row in fitness_history}),
        "elite_count": len(final_elites),
        "best_candidate": final_elites[0] if final_elites else None,
        "ram_cap_mb": args.ram_cap_mb,
        "peak_ram_mb": peak_memory_mb(),
        "duration_seconds": round(time.time() - start, 3),
        "outputs": {
            "summary": str(out_dir / "gain_function_training_summary.json"),
            "population": str(out_dir / "gain_function_population.jsonl"),
            "fitness_history": str(out_dir / "gain_function_fitness.jsonl"),
            "policy_state": str(out_dir / "vpd_edit_policy_state.json"),
            "training_events": str(out_dir / "training_events.jsonl"),
            "prompt_packet": str(out_dir / "prompt_packet.txt"),
            "checkpoints": str(checkpoint_dir),
        },
    }
    packet = compact_packet(summary, policy_state)
    (out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (out_dir / "gain_function_training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    events.append({"ts": utc_now(), "event": status, "peak_ram_mb": peak_memory_mb(), "duration_seconds": summary["duration_seconds"]})
    write_jsonl(out_dir / "training_events.jsonl", events)
    tracemalloc.stop()
    return summary


def main() -> int:
    summary = train_gain_policy(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

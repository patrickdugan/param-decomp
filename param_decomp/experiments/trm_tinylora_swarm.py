"""Cached tinyLoRA-swarm scaffold for VPD/TRM edit search.

This formalizes the edit organism and memetic loop without loading model
weights. Fitness is evaluated through cached constrained-choice score cards,
so this run is a planning/screening stage before hard-capped LoRA training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import (
    dedupe_samples,
    parse_floats,
    parse_paths,
    read_jsonl,
    write_jsonl,
)
from param_decomp.experiments.trm_edit_bootstrap_microcycle import (
    promotion_status,
    score_candidate,
)
from param_decomp.experiments.trm_gain_policy_condition_miner import policy_id
from param_decomp.experiments.trm_multi_family_bootstrap_sweep import (
    DEFAULT_SCORE_FILES,
    failure_families,
)


DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_tinylora_swarm_arc_20260606")
DEFAULT_MODULES = (
    "base_model.model.model.language_model.layers.23.mlp.down_proj,"
    "base_model.model.model.language_model.layers.19.self_attn.o_proj,"
    "base_model.model.model.language_model.layers.15.mlp.down_proj,"
    "base_model.model.model.language_model.layers.11.self_attn.o_proj"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cached tinyLoRA-swarm edit search over TRM score cards.")
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument("--ranks", default="1,2,4")
    parser.add_argument("--scales", default="0.25,0.5,1")
    parser.add_argument("--margins", default="0.25,0.5,1")
    parser.add_argument("--touch-rate-max", type=float, default=0.35)
    parser.add_argument("--touch-penalty", type=float, default=0.05)
    parser.add_argument("--complexity-penalty", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--max-families", type=int, default=12)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def parse_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


@dataclass(frozen=True)
class TinyLoraOrganism:
    organism_id: str
    generation: int
    parent_ids: tuple[str, ...]
    target_module: str
    rank: int
    alpha: float
    scale: float
    adapter_seed: int
    trigger_top_action: str
    trigger_runner_up_action: str
    trigger_target_action: str
    trigger_margin_bucket: str
    max_margin: float
    mutation_ops: tuple[str, ...]
    source_family_key: str


def organism_policy(organism: TinyLoraOrganism) -> dict[str, Any]:
    return {
        "type": "conditional_top_margin_penalty",
        "top_action": organism.trigger_top_action,
        "runner_up_action": organism.trigger_runner_up_action,
        "margin_bucket": "any",
        "max_margin": organism.max_margin,
        "penalty": organism.scale,
    }


def fixed_control(policy: dict[str, Any]) -> dict[str, Any]:
    return {"type": "fixed_action_penalty", "action": policy["top_action"], "penalty": policy["penalty"]}


def organism_record(organism: TinyLoraOrganism, score: dict[str, Any] | None = None) -> dict[str, Any]:
    row = asdict(organism)
    row["parent_ids"] = list(organism.parent_ids)
    row["mutation_ops"] = list(organism.mutation_ops)
    row["policy"] = {**organism_policy(organism), "id": policy_id(organism_policy(organism))}
    if score:
        row["score"] = score
    return row


def seed_organisms(
    families: list[dict[str, Any]],
    modules: list[str],
    ranks: list[int],
    scales: list[float],
    margins: list[float],
    population_size: int,
    rng: random.Random,
) -> list[TinyLoraOrganism]:
    organisms = []
    index = 0
    for family in families:
        for module in modules:
            for rank in ranks:
                scale = scales[index % len(scales)]
                max_margin = margins[index % len(margins)]
                organisms.append(
                    TinyLoraOrganism(
                        organism_id=f"tiny_lora:g0:{index:04d}",
                        generation=0,
                        parent_ids=(),
                        target_module=module,
                        rank=rank,
                        alpha=float(rank) * 2.0,
                        scale=scale,
                        adapter_seed=rng.randrange(1_000_000),
                        trigger_top_action=family["top_action"],
                        trigger_runner_up_action=family["runner_up_action"],
                        trigger_target_action=family["target_action"],
                        trigger_margin_bucket=family["margin_bucket"],
                        max_margin=max_margin,
                        mutation_ops=("seed_from_failure_family",),
                        source_family_key=family["family_key"],
                    )
                )
                index += 1
                if len(organisms) >= population_size:
                    return organisms
    return organisms


def mutate_organism(
    parent: TinyLoraOrganism,
    generation: int,
    child_index: int,
    modules: list[str],
    ranks: list[int],
    scales: list[float],
    margins: list[float],
    rng: random.Random,
) -> TinyLoraOrganism:
    op = child_index % 4
    module = parent.target_module
    rank = parent.rank
    scale = parent.scale
    max_margin = parent.max_margin
    ops: list[str] = []
    if op == 0:
        module = modules[(modules.index(module) + 1) % len(modules)] if module in modules else rng.choice(modules)
        ops.append("module_shift")
    elif op == 1:
        rank = ranks[(ranks.index(rank) + 1) % len(ranks)] if rank in ranks else rng.choice(ranks)
        ops.append("rank_shift")
    elif op == 2:
        scale = scales[(scales.index(scale) + 1) % len(scales)] if scale in scales else rng.choice(scales)
        ops.append("scale_shift")
    else:
        max_margin = margins[(margins.index(max_margin) + 1) % len(margins)] if max_margin in margins else rng.choice(margins)
        ops.append("trigger_margin_shift")
    return TinyLoraOrganism(
        organism_id=f"tiny_lora:g{generation}:{child_index:04d}",
        generation=generation,
        parent_ids=(parent.organism_id,),
        target_module=module,
        rank=rank,
        alpha=float(rank) * 2.0,
        scale=scale,
        adapter_seed=rng.randrange(1_000_000),
        trigger_top_action=parent.trigger_top_action,
        trigger_runner_up_action=parent.trigger_runner_up_action,
        trigger_target_action=parent.trigger_target_action,
        trigger_margin_bucket=parent.trigger_margin_bucket,
        max_margin=max_margin,
        mutation_ops=tuple(ops),
        source_family_key=parent.source_family_key,
    )


def score_organism(
    samples: list[dict[str, Any]],
    organism: TinyLoraOrganism,
    *,
    generation: int,
    touch_rate_max: float,
    touch_penalty: float,
    complexity_penalty: float,
) -> dict[str, Any]:
    policy = organism_policy(organism)
    edit = score_candidate(
        samples,
        policy,
        round_index=generation,
        candidate_family="tinylora_proxy_organism",
        touch_penalty=touch_penalty,
        complexity_penalty=complexity_penalty,
    )
    control = score_candidate(
        samples,
        fixed_control(policy),
        round_index=generation,
        candidate_family="fixed_label_control",
        touch_penalty=touch_penalty,
        complexity_penalty=complexity_penalty,
    )
    promoted, reason = promotion_status(edit, control, touch_rate_max)
    control_margin = round(edit["reward"] - control["reward"], 6)
    sparsity_bonus = round(0.001 / max(1, organism.rank), 6)
    fitness = round(control_margin + 0.02 * edit["rescue_count"] - 0.02 * edit["damage_count"] - 0.01 * edit["touch_rate"] + sparsity_bonus, 6)
    return {
        "organism_id": organism.organism_id,
        "generation": generation,
        "policy_id": edit["candidate_id"],
        "current_score": edit["current_score"],
        "edited_score": edit["edited_score"],
        "delta": edit["delta"],
        "reward": edit["reward"],
        "control_reward": control["reward"],
        "control_delta": control["delta"],
        "control_margin": control_margin,
        "fitness": fitness,
        "rescue_count": edit["rescue_count"],
        "damage_count": edit["damage_count"],
        "touch_rate": edit["touch_rate"],
        "rank": organism.rank,
        "scale": organism.scale,
        "target_module": organism.target_module,
        "promoted": promoted,
        "promotion_reason": reason,
        "claim_boundary": "Cached tinyLoRA proxy only; no model weights were loaded or trained.",
    }


def next_generation(
    elites: list[TinyLoraOrganism],
    generation: int,
    population_size: int,
    modules: list[str],
    ranks: list[int],
    scales: list[float],
    margins: list[float],
    rng: random.Random,
) -> list[TinyLoraOrganism]:
    children = []
    index = 0
    while len(children) < population_size and elites:
        parent = elites[index % len(elites)]
        children.append(mutate_organism(parent, generation, index, modules, ranks, scales, margins, rng))
        index += 1
    return children


def compact_packet(summary: dict[str, Any], best: dict[str, Any] | None) -> str:
    lines = [
        "TASK: Promote VPD/TRM edit search to tinyLoRA swarm organisms.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- generations: {summary['generation_count']}",
        f"- organisms_scored: {summary['organism_count']}",
        f"- accepted_organisms: {summary['accepted_organism_count']}",
        "BEST ORGANISM:",
        f"- id: {best.get('organism_id') if best else None}",
        f"- module: {best.get('target_module') if best else None}",
        f"- rank/scale: {best.get('rank') if best else None}/{best.get('scale') if best else None}",
        f"- delta/control_margin/fitness: {best.get('delta') if best else 0}/{best.get('control_margin') if best else 0}/{best.get('fitness') if best else 0}",
        "SAFETY:",
        "- cached proxy only; real tinyLoRA training must run under hard caps, checkpointing, and PID-owned cleanup.",
    ]
    return "\n".join(lines) + "\n"


def run_swarm(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    score_files = parse_paths(args.score_files)
    samples = dedupe_samples([sample for path in score_files for sample in read_jsonl(path)])
    families = failure_families(samples, min_family_size=1, max_families=args.max_families)
    modules = parse_strings(args.modules)
    ranks = [int(value) for value in parse_floats(args.ranks)]
    scales = parse_floats(args.scales)
    margins = parse_floats(args.margins)
    population = seed_organisms(families, modules, ranks, scales, margins, args.population_size, rng)
    organism_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    best_score: dict[str, Any] | None = None
    best_organism: TinyLoraOrganism | None = None
    for generation in range(args.generations):
        scored_generation: list[tuple[TinyLoraOrganism, dict[str, Any]]] = []
        for organism in population:
            score = score_organism(
                samples,
                organism,
                generation=generation,
                touch_rate_max=args.touch_rate_max,
                touch_penalty=args.touch_penalty,
                complexity_penalty=args.complexity_penalty,
            )
            scored_generation.append((organism, score))
            organism_rows.append(organism_record(organism, score))
            score_rows.append(score)
            if score["promoted"]:
                accepted.append(organism_record(organism, score))
            if best_score is None or score["fitness"] > best_score["fitness"]:
                best_score = score
                best_organism = organism
        scored_generation.sort(key=lambda pair: (pair[1]["promoted"], pair[1]["fitness"], pair[1]["control_margin"], pair[1]["delta"]), reverse=True)
        elites = [pair[0] for pair in scored_generation[: max(1, args.elite_count)]]
        event_rows.append(
            {
                "ts": utc_now(),
                "event": "generation_complete",
                "generation": generation,
                "population_size": len(population),
                "elite_ids": [organism.organism_id for organism in elites],
                "best_organism_id": scored_generation[0][0].organism_id if scored_generation else None,
                "best_fitness": scored_generation[0][1]["fitness"] if scored_generation else 0.0,
                "accepted_count": sum(int(pair[1]["promoted"]) for pair in scored_generation),
            }
        )
        if generation < args.generations - 1:
            population = next_generation(elites, generation + 1, args.population_size, modules, ranks, scales, margins, rng)
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "score_files": [str(path) for path in score_files],
        "sample_count": len(samples),
        "family_count": len(families),
        "generation_count": args.generations,
        "organism_count": len(score_rows),
        "accepted_organism_count": len(accepted),
        "best_organism_id": best_score["organism_id"] if best_score else None,
        "best_fitness": best_score["fitness"] if best_score else 0.0,
        "best_delta": best_score["delta"] if best_score else 0.0,
        "best_control_margin": best_score["control_margin"] if best_score else 0.0,
        "best_target_module": best_score["target_module"] if best_score else None,
        "claim_boundary": "Cached tinyLoRA swarm proxy only; no adapter weights were trained or merged.",
        "training_safety_contract": {
            "real_training_required_caps": {"ram_mb": 2048, "cpu_pct": 50, "io_mb_s": 50},
            "checkpoint_interval": "every generation or 120 seconds",
            "chunk_strategy": "score organisms in bounded batches; never materialize full activation matrices",
            "cleanup_required": "PID-owned process cleanup plus CUDA/object release before any real LoRA run",
        },
        "outputs": {
            "summary": str(args.out_dir / "tinylora_swarm_summary.json"),
            "organisms": str(args.out_dir / "tinylora_organisms.jsonl"),
            "scores": str(args.out_dir / "tinylora_swarm_scores.jsonl"),
            "accepted": str(args.out_dir / "accepted_tinylora_organisms.jsonl"),
            "events": str(args.out_dir / "tinylora_swarm_events.jsonl"),
            "state": str(args.out_dir / "tinylora_swarm_state.json"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    state = {
        "best_organism": organism_record(best_organism, best_score) if best_organism and best_score else None,
        "families": families,
        "modules": modules,
        "ranks": ranks,
        "scales": scales,
        "margins": margins,
        "accepted_count": len(accepted),
    }
    write_jsonl(args.out_dir / "tinylora_organisms.jsonl", organism_rows)
    write_jsonl(args.out_dir / "tinylora_swarm_scores.jsonl", score_rows)
    write_jsonl(args.out_dir / "accepted_tinylora_organisms.jsonl", accepted)
    write_jsonl(args.out_dir / "tinylora_swarm_events.jsonl", event_rows)
    (args.out_dir / "tinylora_swarm_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    packet = compact_packet(summary, best_score)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "tinylora_swarm_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_swarm(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

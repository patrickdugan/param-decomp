"""Agent-manageable tinyLoRA auto-research loop for VPD/TRM hill climbing.

This orchestrates bounded research cycles over the existing cached swarm,
training handoff, and trainer bridge. The default trainer mode is dry-run, so
the loop can be tested without loading model weights. A real adapter backend
can replace the trainer bridge later while preserving the same state files.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_tinylora_live_trainer import run_trainer
from param_decomp.experiments.trm_tinylora_swarm import DEFAULT_MODULES, DEFAULT_OUT_DIR as DEFAULT_SWARM_OUT_DIR
from param_decomp.experiments.trm_tinylora_swarm import DEFAULT_SCORE_FILES, run_swarm
from param_decomp.experiments.trm_tinylora_training_handoff import run_handoff


DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_tinylora_auto_research_arc_20260606")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded tinyLoRA auto-research hill-climb loop.")
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument("--ranks", default="1,2,4")
    parser.add_argument("--scales", default="0.25,0.5,1")
    parser.add_argument("--margins", default="0.25,0.5,1")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--random-control-count", type=int, default=8)
    parser.add_argument("--trainer-mode", choices=["dry_run", "train_one"], default="dry_run")
    parser.add_argument("--trainer-backend", choices=["none", "scorecard_rehearsal"], default="none")
    parser.add_argument("--ram-mb", type=int, default=2048)
    parser.add_argument("--cpu-pct", type=int, default=50)
    parser.add_argument("--io-mb-s", type=int, default=50)
    parser.add_argument("--checkpoint-interval", default="generation_or_120s")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--max-families", type=int, default=12)
    parser.add_argument("--touch-rate-max", type=float, default=0.35)
    parser.add_argument("--touch-penalty", type=float, default=0.05)
    parser.add_argument("--complexity-penalty", type=float, default=0.003)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compact_packet(summary: dict[str, Any], cycles: list[dict[str, Any]]) -> str:
    best = summary.get("best_cycle") or {}
    last = cycles[-1] if cycles else {}
    status = "real_training_needed" if summary.get("accepted_live_edit_count", 0) == 0 else "continue_hill_climb"
    lines = [
        "TASK: Continue VPD/TRM tinyLoRA hill climbing under hard caps.",
        "STATE:",
        f"- auto_loop: {summary['run_dir']}",
        f"- cycles_completed: {summary['cycle_count']}",
        f"- best_proxy_fitness: {summary.get('best_proxy_fitness', 0.0)}",
        f"- accepted_live_edits: {summary.get('accepted_live_edit_count', 0)}",
        f"- accepted_rehearsal_edits: {summary.get('accepted_rehearsal_edit_count', 0)}",
        f"- status: {status}",
        f"- block_reason: {last.get('trainer_block_reason')}",
        "BEST CYCLE:",
        f"- cycle_id: {best.get('cycle_id')}",
        f"- swarm_dir: {best.get('swarm_dir')}",
        f"- handoff_manifest: {best.get('handoff_manifest')}",
        "NEXT ACTION:",
        "- run the generated trainer wrapper for one top candidate when a real adapter backend is available",
        "- compare live target score against random tinyLoRA controls before accepting any edit",
        "- if blocked_missing_adapter_training_backend persists, implement train_one backend rather than claiming model gain",
        "LAST CYCLE:",
        f"- trainer_summary: {last.get('trainer_summary_path')}",
    ]
    return "\n".join(lines) + "\n"


def cycle_record(cycle_id: str, swarm_summary: dict[str, Any], handoff_summary: dict[str, Any], trainer_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": utc_now(),
        "cycle_id": cycle_id,
        "swarm_dir": swarm_summary["run_dir"],
        "handoff_dir": handoff_summary["run_dir"],
        "handoff_manifest": handoff_summary["outputs"]["manifest"],
        "trainer_dir": trainer_summary["run_dir"],
        "trainer_summary_path": trainer_summary["outputs"]["summary"],
        "best_proxy_fitness": swarm_summary.get("best_fitness", 0.0),
        "best_proxy_delta": swarm_summary.get("best_delta", 0.0),
        "best_proxy_control_margin": swarm_summary.get("best_control_margin", 0.0),
        "accepted_proxy_organisms": swarm_summary.get("accepted_organism_count", 0),
        "candidate_count": handoff_summary.get("candidate_count", 0),
        "random_control_count": trainer_summary.get("random_control_count", 0),
        "accepted_live_edits": trainer_summary.get("accepted_count", 0) if trainer_summary.get("backend") != "scorecard_rehearsal" else 0,
        "accepted_rehearsal_edits": trainer_summary.get("accepted_count", 0) if trainer_summary.get("backend") == "scorecard_rehearsal" else 0,
        "trainer_status": trainer_summary.get("status"),
        "trainer_backend": trainer_summary.get("backend"),
        "trainer_block_reason": trainer_summary.get("block_reason"),
        "claim_boundary": trainer_summary.get("claim_boundary", ""),
    }


def run_cycle(args: argparse.Namespace, cycle_index: int) -> dict[str, Any]:
    cycle_id = f"cycle_{cycle_index:03d}"
    cycle_dir = args.out_dir / cycle_id
    swarm_dir = cycle_dir / "swarm"
    handoff_dir = cycle_dir / "handoff"
    trainer_dir = cycle_dir / "trainer"
    swarm_summary = run_swarm(
        Namespace(
            score_files=args.score_files,
            out_dir=swarm_dir,
            generations=args.generations,
            population_size=args.population_size,
            elite_count=args.elite_count,
            modules=args.modules,
            ranks=args.ranks,
            scales=args.scales,
            margins=args.margins,
            touch_rate_max=args.touch_rate_max,
            touch_penalty=args.touch_penalty,
            complexity_penalty=args.complexity_penalty,
            seed=args.seed + cycle_index,
            max_families=args.max_families,
            max_prompt_tokens=args.max_prompt_tokens,
        )
    )
    handoff_summary = run_handoff(
        Namespace(
            swarm_dir=swarm_dir,
            out_dir=handoff_dir,
            top_n=args.top_n,
            random_control_count=args.random_control_count,
            ram_mb=args.ram_mb,
            cpu_pct=args.cpu_pct,
            io_mb_s=args.io_mb_s,
            checkpoint_interval=args.checkpoint_interval,
            training_task_id=f"tinylora-auto-research-{cycle_id}",
            max_prompt_tokens=args.max_prompt_tokens,
        )
    )
    trainer_summary = run_trainer(
        Namespace(
            manifest=Path(handoff_summary["outputs"]["manifest"]),
            out_dir=trainer_dir,
            mode=args.trainer_mode,
            backend=args.trainer_backend,
            max_candidates=0,
            candidate_id=None,
        )
    )
    return cycle_record(cycle_id, swarm_summary, handoff_summary, trainer_summary)


def run_auto_research_loop(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = [
        {
            "ts": utc_now(),
            "event": "start",
            "cycles_requested": args.cycles,
            "trainer_mode": args.trainer_mode,
            "trainer_backend": args.trainer_backend,
            "caps": {"ram_mb": args.ram_mb, "cpu_pct": args.cpu_pct, "io_mb_s": args.io_mb_s},
        }
    ]
    cycles: list[dict[str, Any]] = []
    best_cycle: dict[str, Any] | None = None
    for index in range(1, args.cycles + 1):
        events.append({"ts": utc_now(), "event": "cycle_start", "cycle_index": index})
        record = run_cycle(args, index)
        cycles.append(record)
        if best_cycle is None or record["best_proxy_fitness"] > best_cycle["best_proxy_fitness"]:
            best_cycle = record
        events.append(
            {
                "ts": utc_now(),
                "event": "cycle_complete",
                "cycle_id": record["cycle_id"],
                "best_proxy_fitness": record["best_proxy_fitness"],
                "accepted_live_edits": record["accepted_live_edits"],
            }
        )
    accepted_rehearsal = sum(int(record["accepted_rehearsal_edits"]) for record in cycles)
    accepted_live = 0 if args.trainer_backend == "scorecard_rehearsal" else sum(int(record["accepted_live_edits"]) for record in cycles)
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "cycle_count": len(cycles),
        "trainer_mode": args.trainer_mode,
        "trainer_backend": args.trainer_backend,
        "best_cycle": best_cycle,
        "best_proxy_fitness": best_cycle["best_proxy_fitness"] if best_cycle else 0.0,
        "accepted_live_edit_count": accepted_live,
        "accepted_rehearsal_edit_count": accepted_rehearsal,
        "research_state": "scorecard_rehearsal_signal_observed" if accepted_rehearsal else ("ready_for_train_one_backend" if accepted_live == 0 else "live_hill_climb_signal_observed"),
        "claim_boundary": "Auto-research manager loop only; blocked or dry-run trainer modes do not prove model-weight gains.",
        "outputs": {
            "summary": str(args.out_dir / "tinylora_auto_research_summary.json"),
            "events": str(args.out_dir / "tinylora_auto_research_events.jsonl"),
            "cycles": str(args.out_dir / "tinylora_auto_research_cycles.jsonl"),
            "state": str(args.out_dir / "tinylora_auto_research_state.json"),
            "agent_packet": str(args.out_dir / "next_agent_packet.txt"),
        },
    }
    state = {
        "cycles": cycles,
        "best_cycle": best_cycle,
        "next_agent_action": "implement_or_run_train_one_backend_under_generated_jobobject_wrapper",
        "acceptance_rule": "accept only if live target score beats fixed-label and random tinyLoRA controls with guardrail preservation",
        "caps": {"ram_mb": args.ram_mb, "cpu_pct": args.cpu_pct, "io_mb_s": args.io_mb_s},
    }
    events.append({"ts": utc_now(), "event": "summary", "status": summary["status"], "accepted_live_edit_count": accepted_live, "accepted_rehearsal_edit_count": accepted_rehearsal})
    packet = compact_packet(summary, cycles)
    summary["agent_packet_est_tokens"] = len(packet) // 4
    write_jsonl(args.out_dir / "tinylora_auto_research_events.jsonl", events)
    write_jsonl(args.out_dir / "tinylora_auto_research_cycles.jsonl", cycles)
    (args.out_dir / "tinylora_auto_research_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "next_agent_packet.txt").write_text(packet, encoding="utf-8")
    (args.out_dir / "tinylora_auto_research_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_auto_research_loop(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Dry-run/live harness shell for hard-capped tinyLoRA training.

The default mode is dry-run. It consumes a tinyLoRA training manifest, validates
candidates and controls, and emits the same event/result files a real capped
trainer must emit. It does not load model weights or train adapters.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import read_jsonl, write_jsonl


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tinyLoRA trainer bridge from a handoff manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["dry_run", "train_one"], default="dry_run")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means all candidates.")
    parser.add_argument("--candidate-id", default=None, help="Optional candidate id for train_one mode.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event(ts_event: str, **kwargs: Any) -> dict[str, Any]:
    return {"ts": utc_now(), "event": ts_event, **kwargs}


def candidate_result(candidate: dict[str, Any], kind: str, *, reason: str, claim_boundary: str) -> dict[str, Any]:
    proxy = candidate.get("proxy_score") or {}
    accepted = False
    return {
        "candidate_id": candidate["candidate_id"],
        "organism_id": candidate["organism_id"],
        "kind": kind,
        "target_module": candidate["target_module"],
        "source_family_key": candidate["source_family_key"],
        "rank": candidate["adapter_config"]["rank"],
        "scale": candidate["adapter_config"]["scale"],
        "proxy_delta": proxy.get("delta", 0.0),
        "proxy_control_margin": proxy.get("control_margin", 0.0),
        "proxy_fitness": proxy.get("fitness", 0.0),
        "live_target_score": None,
        "live_guardrail_score": None,
        "live_control_score": None,
        "accepted": accepted,
        "decision_reason": reason,
        "claim_boundary": claim_boundary,
    }


def selected_candidates(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidate_id = getattr(args, "candidate_id", None)
    if candidate_id:
        candidates = [candidate for candidate in candidates if candidate["candidate_id"] == candidate_id]
    if args.mode == "train_one":
        return candidates[:1]
    if args.max_candidates > 0:
        return candidates[: args.max_candidates]
    return candidates


def selected_controls(controls: list[dict[str, Any]], args: argparse.Namespace, count: int) -> list[dict[str, Any]]:
    if args.mode == "train_one":
        return controls[: max(1, count)]
    if args.max_candidates > 0:
        return controls[: args.max_candidates]
    return controls


def train_one_block_reason() -> str | None:
    if os.environ.get("TINYLORA_JOB_OBJECT") != "1":
        return "blocked_not_inside_generated_jobobject_wrapper"
    return "blocked_missing_adapter_training_backend"


def run_trainer(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.manifest)
    out_dir = args.out_dir or Path(manifest["default_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    candidates = selected_candidates(read_jsonl(paths["candidates"]), args)
    controls = selected_controls(read_jsonl(paths["random_controls"]), args, len(candidates))
    train_one_reason = train_one_block_reason() if args.mode == "train_one" else None
    if args.mode == "train_one" and not candidates:
        train_one_reason = "blocked_candidate_not_found"
    events: list[dict[str, Any]] = [
        event(
            "start",
            training_task_id=manifest["training_task_id"],
            mode=args.mode,
            caps=manifest["caps"],
            candidate_count=len(candidates),
            random_control_count=len(controls),
        )
    ]
    results: list[dict[str, Any]] = []
    if train_one_reason:
        events.append(event("train_one_blocked", reason=train_one_reason, inside_jobobject=os.environ.get("TINYLORA_JOB_OBJECT") == "1"))
    for index, candidate in enumerate(candidates, start=1):
        events.append(event("candidate_start", candidate_id=candidate["candidate_id"], index=index, kind="vpd_seeded"))
        events.append(event("checkpoint", candidate_id=candidate["candidate_id"], step=0, path=str(out_dir / "checkpoints" / candidate["candidate_id"].replace(":", "_")), ram_mb=0, cpu_pct=0))
        reason = train_one_reason or "dry_run_only_no_adapter_trained"
        claim = (
            "Train-one request blocked before model load; no adapter weights were trained, merged, or scored."
            if train_one_reason
            else "Dry-run validation only; no model weights were loaded, trained, or merged."
        )
        result = candidate_result(candidate, "vpd_seeded", reason=reason, claim_boundary=claim)
        results.append(result)
        events.append(event("candidate_score", candidate_id=candidate["candidate_id"], target_score=None, guardrail_score=None, control_score=None, proxy_delta=result["proxy_delta"]))
        events.append(event("candidate_accept_or_reject", candidate_id=candidate["candidate_id"], accepted=False, reason=result["decision_reason"]))
    for index, control in enumerate(controls, start=1):
        events.append(event("candidate_start", candidate_id=control["candidate_id"], index=index, kind="random_control"))
        result = candidate_result(
            control,
            "random_control",
            reason=train_one_reason or "dry_run_only_no_adapter_trained",
            claim_boundary="Control adapter was not trained or scored.",
        )
        results.append(result)
        events.append(event("candidate_accept_or_reject", candidate_id=control["candidate_id"], accepted=False, reason=result["decision_reason"]))
    gc.collect()
    cleanup_status = "blocked_cleanup_passed" if train_one_reason else "dry_run_cleanup_passed"
    events.append(event("cleanup", owned_pids_stopped=[], cuda_cleanup=False, ram_after_mb=None, status=cleanup_status))
    status = "blocked" if train_one_reason else "completed"
    summary_name = "tinylora_training_train_one_summary.json" if args.mode == "train_one" else "tinylora_training_dry_run_summary.json"
    summary = {
        "status": status,
        "mode": args.mode,
        "generated_at_utc": utc_now(),
        "manifest": str(args.manifest),
        "run_dir": str(out_dir),
        "training_task_id": manifest["training_task_id"],
        "candidate_count": len(candidates),
        "random_control_count": len(controls),
        "accepted_count": 0,
        "block_reason": train_one_reason,
        "claim_boundary": "Train-one bridge is blocked until a capped adapter backend is implemented." if train_one_reason else "Dry-run trainer only; no tinyLoRA adapter training was executed.",
        "outputs": {
            "summary": str(out_dir / summary_name),
            "events": str(out_dir / "tinylora_training_events.jsonl"),
            "candidate_results": str(out_dir / "tinylora_training_candidate_results.jsonl"),
        },
    }
    events.append(event("summary", status=summary["status"], accepted_count=0, candidate_count=len(candidates), block_reason=train_one_reason))
    write_jsonl(out_dir / "tinylora_training_events.jsonl", events)
    write_jsonl(out_dir / "tinylora_training_candidate_results.jsonl", results)
    (out_dir / summary_name).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_trainer(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

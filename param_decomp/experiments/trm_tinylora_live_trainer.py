"""Dry-run/live harness shell for hard-capped tinyLoRA training.

The default mode is dry-run. It consumes a tinyLoRA training manifest, validates
candidates and controls, and emits the same event/result files a real capped
trainer must emit. It does not load model weights or train adapters.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
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
    parser.add_argument("--backend", choices=["none", "scorecard_rehearsal", "peft_train_one"], default="none")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means all candidates.")
    parser.add_argument("--candidate-id", default=None, help="Optional candidate id for train_one mode.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event(ts_event: str, **kwargs: Any) -> dict[str, Any]:
    return {"ts": utc_now(), "event": ts_event, **kwargs}


def candidate_result(
    candidate: dict[str, Any],
    kind: str,
    *,
    reason: str,
    claim_boundary: str,
    accepted: bool = False,
    live_target_score: float | None = None,
    live_guardrail_score: float | None = None,
    live_control_score: float | None = None,
) -> dict[str, Any]:
    proxy = candidate.get("proxy_score") or {}
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
        "live_target_score": live_target_score,
        "live_guardrail_score": live_guardrail_score,
        "live_control_score": live_control_score,
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


def peft_train_one_block_reason() -> str | None:
    if os.environ.get("TINYLORA_JOB_OBJECT") != "1":
        return "blocked_not_inside_generated_jobobject_wrapper"
    if os.environ.get("TINYLORA_ENABLE_MODEL_LOAD") != "1":
        return "blocked_model_load_not_enabled"
    model_path = os.environ.get("TINYLORA_MODEL_PATH")
    eval_spec = os.environ.get("TINYLORA_EVAL_SPEC")
    if not model_path:
        return "blocked_missing_model_path"
    if not Path(model_path).exists():
        return "blocked_model_path_not_found"
    if not eval_spec:
        return "blocked_missing_eval_spec"
    if not Path(eval_spec).exists():
        return "blocked_eval_spec_not_found"
    missing = [name for name in ("torch", "transformers", "peft", "accelerate") if importlib.util.find_spec(name) is None]
    if missing:
        return "blocked_missing_python_packages:" + ",".join(missing)
    return None


def directory_size_mb(path: Path) -> int:
    if path.is_file():
        return max(1, path.stat().st_size // (1024 * 1024))
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return max(1, total // (1024 * 1024))


def peft_train_one_smoke(out_dir: Path, manifest: dict[str, Any], candidate: dict[str, Any], controls: list[dict[str, Any]]) -> dict[str, Any]:
    model_path = Path(os.environ["TINYLORA_MODEL_PATH"])
    eval_spec = Path(os.environ["TINYLORA_EVAL_SPEC"])
    cap_mb = int((manifest.get("caps") or {}).get("ram_mb", 2048))
    model_size_mb = directory_size_mb(model_path)
    request_path = write_peft_train_one_request(out_dir, manifest, candidate, controls)
    if model_size_mb > max(1, cap_mb // 2):
        return {
            "status": "blocked",
            "reason": "blocked_model_size_exceeds_safe_cap",
            "request_path": str(request_path),
            "model_size_mb": model_size_mb,
            "cap_mb": cap_mb,
            "live_target_score": None,
            "live_guardrail_score": None,
            "live_control_score": None,
        }
    return {
        "status": "blocked",
        "reason": "blocked_peft_training_body_not_enabled_after_preflight",
        "request_path": str(request_path),
        "model_size_mb": model_size_mb,
        "cap_mb": cap_mb,
        "eval_spec": str(eval_spec),
        "live_target_score": None,
        "live_guardrail_score": None,
        "live_control_score": None,
    }


def write_peft_train_one_request(out_dir: Path, manifest: dict[str, Any], candidate: dict[str, Any], controls: list[dict[str, Any]]) -> Path:
    request = {
        "training_task_id": manifest["training_task_id"],
        "candidate_id": candidate["candidate_id"],
        "organism_id": candidate["organism_id"],
        "model_path": os.environ.get("TINYLORA_MODEL_PATH"),
        "eval_spec": os.environ.get("TINYLORA_EVAL_SPEC"),
        "caps": manifest["caps"],
        "checkpoint_interval": manifest.get("checkpoint_interval"),
        "adapter_config": candidate["adapter_config"],
        "target_module": candidate["target_module"],
        "trigger": candidate.get("trigger"),
        "acceptance_gate": candidate.get("acceptance_gate"),
        "random_control_ids": [control["candidate_id"] for control in controls],
        "required_runtime": {
            "wrapper_marker": "TINYLORA_JOB_OBJECT=1",
            "model_load_marker": "TINYLORA_ENABLE_MODEL_LOAD=1",
            "model_path_env": "TINYLORA_MODEL_PATH",
            "eval_spec_env": "TINYLORA_EVAL_SPEC",
        },
        "claim_boundary": "PEFT train-one request artifact only; this run did not load model weights.",
    }
    path = out_dir / "peft_train_one_request.json"
    path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def scorecard_rehearsal(candidate: dict[str, Any], controls: list[dict[str, Any]]) -> dict[str, Any]:
    proxy = candidate.get("proxy_score") or {}
    best_control = max((float((control.get("proxy_score") or {}).get("fitness", 0.0)) for control in controls), default=0.0)
    target_score = float(proxy.get("fitness", 0.0))
    guardrail_score = 1.0 - float(proxy.get("touch_rate", 0.0))
    accepted = (
        float(proxy.get("delta", 0.0)) > 0.0
        and float(proxy.get("control_margin", 0.0)) > 0.0
        and target_score > best_control
        and guardrail_score >= 0.65
    )
    return {
        "accepted": accepted,
        "reason": "scorecard_rehearsal_accept" if accepted else "scorecard_rehearsal_reject",
        "live_target_score": round(target_score, 6),
        "live_guardrail_score": round(guardrail_score, 6),
        "live_control_score": round(best_control, 6),
    }


def run_trainer(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.manifest)
    out_dir = args.out_dir or Path(manifest["default_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    candidates = selected_candidates(read_jsonl(paths["candidates"]), args)
    controls = selected_controls(read_jsonl(paths["random_controls"]), args, len(candidates))
    train_one_reason = train_one_block_reason() if args.mode == "train_one" else None
    backend = getattr(args, "backend", "none")
    if args.mode == "train_one" and backend == "scorecard_rehearsal" and os.environ.get("TINYLORA_JOB_OBJECT") == "1":
        train_one_reason = None
    if args.mode == "train_one" and backend == "peft_train_one":
        train_one_reason = peft_train_one_block_reason()
    if args.mode == "train_one" and not candidates:
        train_one_reason = "blocked_candidate_not_found"
    events: list[dict[str, Any]] = [
        event(
            "start",
            training_task_id=manifest["training_task_id"],
            mode=args.mode,
            backend=backend,
            caps=manifest["caps"],
            candidate_count=len(candidates),
            random_control_count=len(controls),
        )
    ]
    results: list[dict[str, Any]] = []
    if train_one_reason:
        events.append(event("train_one_blocked", reason=train_one_reason, inside_jobobject=os.environ.get("TINYLORA_JOB_OBJECT") == "1"))
    backend_request_path: Path | None = None
    backend_probe: dict[str, Any] | None = None
    for index, candidate in enumerate(candidates, start=1):
        events.append(event("candidate_start", candidate_id=candidate["candidate_id"], index=index, kind="vpd_seeded"))
        events.append(event("checkpoint", candidate_id=candidate["candidate_id"], step=0, path=str(out_dir / "checkpoints" / candidate["candidate_id"].replace(":", "_")), ram_mb=0, cpu_pct=0))
        reason = train_one_reason or "dry_run_only_no_adapter_trained"
        rehearsal = None
        if args.mode == "train_one" and backend == "scorecard_rehearsal" and not train_one_reason:
            rehearsal = scorecard_rehearsal(candidate, controls)
            reason = rehearsal["reason"]
        if args.mode == "train_one" and backend == "peft_train_one":
            if train_one_reason:
                backend_request_path = write_peft_train_one_request(out_dir, manifest, candidate, controls)
                reason = train_one_reason
            else:
                backend_probe = peft_train_one_smoke(out_dir, manifest, candidate, controls)
                backend_request_path = Path(backend_probe["request_path"])
                train_one_reason = backend_probe["reason"]
                reason = train_one_reason
            events.append(event("backend_preflight", backend=backend, candidate_id=candidate["candidate_id"], request_path=str(backend_request_path), block_reason=train_one_reason, probe=backend_probe))
        claim = (
            "Train-one request blocked before model load; no adapter weights were trained, merged, or scored."
            if train_one_reason
            else (
                "Scorecard rehearsal only; no adapter weights were loaded, trained, merged, or live-scored."
                if rehearsal
                else (
                    "PEFT train-one preflight only; no model weights were loaded, trained, merged, or scored."
                    if backend == "peft_train_one"
                    else "Dry-run validation only; no model weights were loaded, trained, or merged."
                )
            )
        )
        result = candidate_result(
            candidate,
            "vpd_seeded",
            reason=reason,
            claim_boundary=claim,
            accepted=bool(rehearsal and rehearsal["accepted"]),
            live_target_score=rehearsal["live_target_score"] if rehearsal else None,
            live_guardrail_score=rehearsal["live_guardrail_score"] if rehearsal else None,
            live_control_score=rehearsal["live_control_score"] if rehearsal else None,
        )
        results.append(result)
        events.append(event("candidate_score", candidate_id=candidate["candidate_id"], target_score=result["live_target_score"], guardrail_score=result["live_guardrail_score"], control_score=result["live_control_score"], proxy_delta=result["proxy_delta"]))
        events.append(event("candidate_accept_or_reject", candidate_id=candidate["candidate_id"], accepted=result["accepted"], reason=result["decision_reason"]))
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
    accepted_count = sum(1 for result in results if result["kind"] == "vpd_seeded" and result["accepted"])
    cleanup_status = "blocked_cleanup_passed" if train_one_reason else ("scorecard_rehearsal_cleanup_passed" if backend == "scorecard_rehearsal" else "dry_run_cleanup_passed")
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
        "accepted_count": accepted_count,
        "block_reason": train_one_reason,
        "backend": backend,
        "backend_probe": backend_probe,
        "claim_boundary": (
            "Train-one bridge is blocked until a capped adapter backend is implemented."
            if train_one_reason
            else (
                "Scorecard rehearsal only; accepted_count is not a trained model-edit gain."
                if backend == "scorecard_rehearsal"
                else "Dry-run trainer only; no tinyLoRA adapter training was executed."
            )
        ),
        "outputs": {
            "summary": str(out_dir / summary_name),
            "events": str(out_dir / "tinylora_training_events.jsonl"),
            "candidate_results": str(out_dir / "tinylora_training_candidate_results.jsonl"),
            "backend_request": str(backend_request_path) if backend_request_path else None,
        },
    }
    events.append(event("summary", status=summary["status"], accepted_count=accepted_count, candidate_count=len(candidates), block_reason=train_one_reason, backend=backend))
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

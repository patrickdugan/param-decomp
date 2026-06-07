"""Sequential tinyLoRA local-maxima search under the capped trainer wrapper.

The controller does not load model weights. Each trial writes a trial-specific
manifest and invokes the generated Windows Job Object wrapper, so the risky
model work stays under the hard cap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import read_jsonl, write_jsonl


DEFAULT_MANIFEST = Path(r"D:\Research_Engine\runs\trm_tinylora_peft_4gb_probe_20260606\cycle_001\handoff\tinylora_training_manifest.json")
DEFAULT_WRAPPER = Path(r"D:\Research_Engine\runs\trm_tinylora_peft_4gb_probe_20260606\cycle_001\handoff\run_tinylora_jobobject.ps1")
DEFAULT_TRAINING_SCRIPT = Path(r"D:\projects\VDP\scripts\run_trm_tinylora_live_trainer.py")
DEFAULT_MODEL_PATH = Path(r"D:\Research_Engine\models\HRM-Text-1B")
DEFAULT_EVAL_SPEC = Path(r"D:\Research_Engine\runs\trm_tinylora_peft_4gb_probe_20260606\cycle_001\handoff\dry_run_execution\peft_train_one_request.json")
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_tinylora_overnight_search_20260606")
DEFAULT_TARGETS = [
    "model.L_module.layers.0.self_attn.o_proj",
    "model.L_module.layers.1.self_attn.o_proj",
    "model.H_module.layers.0.self_attn.o_proj",
    "model.H_module.layers.1.self_attn.o_proj",
]
DEFAULT_LEARNING_RATES = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001]
DEFAULT_JOB_MEMORY_MB = 3072
DEFAULT_MIN_AVAILABLE_RAM_MB = 6144
DEFAULT_RAM_RESERVE_MB = 1024
DEFAULT_COOLDOWN_SECONDS = 20
DEFAULT_RAM_POLL_SECONDS = 30
DEFAULT_RAM_WAIT_SECONDS = 600


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    target_module: str
    learning_rate: float
    max_seq_len: int
    optimizer: str


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential tinyLoRA local-maxima overnight search.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--training-script", type=Path, default=DEFAULT_TRAINING_SCRIPT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--eval-spec", type=Path, default=DEFAULT_EVAL_SPEC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--learning-rates", default=",".join(str(value) for value in DEFAULT_LEARNING_RATES))
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--job-memory-mb", type=int, default=DEFAULT_JOB_MEMORY_MB)
    parser.add_argument("--min-available-ram-mb", type=int, default=DEFAULT_MIN_AVAILABLE_RAM_MB)
    parser.add_argument("--ram-reserve-mb", type=int, default=DEFAULT_RAM_RESERVE_MB)
    parser.add_argument("--cooldown-seconds", type=int, default=DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument("--ram-poll-seconds", type=int, default=DEFAULT_RAM_POLL_SECONDS)
    parser.add_argument("--ram-wait-seconds", type=int, default=DEFAULT_RAM_WAIT_SECONDS)
    parser.add_argument("--resume", action="store_true", help="Resume from an existing out-dir by skipping completed trial ids.")
    return parser.parse_args()


def build_trial_grid(targets: list[str], learning_rates: list[float], *, max_seq_len: int, optimizer: str, max_trials: int) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for target in targets:
        for learning_rate in learning_rates:
            specs.append(
                TrialSpec(
                    trial_id=f"trial_{len(specs) + 1:03d}",
                    target_module=target,
                    learning_rate=learning_rate,
                    max_seq_len=max_seq_len,
                    optimizer=optimizer,
                )
            )
            if max_trials > 0 and len(specs) >= max_trials:
                return specs
    return specs


def _replace_target(candidate: dict[str, Any], trial: TrialSpec) -> dict[str, Any]:
    updated = json.loads(json.dumps(candidate))
    updated["candidate_id"] = f"{candidate['candidate_id']}:{trial.trial_id}"
    updated["target_module"] = trial.target_module
    adapter = updated.setdefault("adapter_config", {})
    adapter["target_modules"] = [trial.target_module]
    adapter["search_trial"] = {
        "trial_id": trial.trial_id,
        "learning_rate": trial.learning_rate,
        "max_seq_len": trial.max_seq_len,
        "optimizer": trial.optimizer,
    }
    return updated


def available_ram_mb() -> int:
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys // (1024 * 1024))
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:
        return 0


def wait_for_ram_gate(min_available_ram_mb: int, ram_wait_seconds: int, ram_poll_seconds: int) -> tuple[bool, list[dict[str, Any]]]:
    waited = 0
    samples: list[dict[str, Any]] = []
    while True:
        current = available_ram_mb()
        samples.append({"ts": utc_now(), "available_ram_mb": current, "waited_seconds": waited})
        if current >= min_available_ram_mb:
            return True, samples
        if waited >= ram_wait_seconds:
            return False, samples
        time.sleep(max(1, ram_poll_seconds))
        waited += max(1, ram_poll_seconds)


def materialize_trial_wrapper(source_wrapper: Path, trial_dir: Path, job_memory_mb: int) -> Path:
    wrapper_text = source_wrapper.read_text(encoding="utf-8")
    marker = "$MemoryLimitBytes = "
    replacement = f"$MemoryLimitBytes = {job_memory_mb}MB"
    if marker not in wrapper_text:
        raise ValueError(f"wrapper {source_wrapper} does not contain the expected memory limit assignment")
    wrapper_text = wrapper_text.replace("$MemoryLimitBytes = 4096MB", replacement, 1)
    wrapper_text = wrapper_text.replace("$MemoryLimitBytes = 3072MB", replacement, 1)
    wrapper_text = wrapper_text.replace("$MemoryLimitBytes = 2048MB", replacement, 1)
    wrapper_dir = trial_dir / "handoff"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / "run_tinylora_jobobject.ps1"
    wrapper_path.write_text(wrapper_text, encoding="utf-8")
    return wrapper_path


def prepare_trial_manifest(source_manifest_path: Path, out_dir: Path, trial: TrialSpec) -> Path:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_paths = {key: Path(value) for key, value in source_manifest["paths"].items()}
    candidates = read_jsonl(source_paths["candidates"])
    controls = read_jsonl(source_paths["random_controls"])
    if not candidates:
        raise ValueError("source manifest has no candidates")

    trial_dir = out_dir / trial.trial_id
    handoff_dir = trial_dir / "handoff"
    execution_dir = trial_dir / "execution"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    execution_dir.mkdir(parents=True, exist_ok=True)

    trial_candidates = [_replace_target(candidates[0], trial)]
    trial_controls = [_replace_target(control, trial) for control in controls[: max(1, len(trial_candidates))]]
    candidates_path = handoff_dir / "tinylora_training_candidates.jsonl"
    controls_path = handoff_dir / "tinylora_random_controls.jsonl"
    write_jsonl(candidates_path, trial_candidates)
    write_jsonl(controls_path, trial_controls)

    manifest = json.loads(json.dumps(source_manifest))
    manifest["training_task_id"] = f"{source_manifest['training_task_id']}-{trial.trial_id}"
    manifest["default_output_dir"] = str(execution_dir)
    manifest["paths"]["candidates"] = str(candidates_path)
    manifest["paths"]["random_controls"] = str(controls_path)
    manifest["claim_boundary"] = "Trial manifest only; model work must run inside the generated hard-cap wrapper."
    manifest_path = handoff_dir / "tinylora_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (handoff_dir / "trial_spec.json").write_text(json.dumps(trial.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def wrapper_command(args: argparse.Namespace, trial_manifest: Path, trial_wrapper: Path) -> list[str]:
    return [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(trial_wrapper),
        "-PythonExe",
        str(args.python_exe),
        "-TrainingScript",
        str(args.training_script),
        "-ManifestPath",
        str(trial_manifest),
        "-Backend",
        "peft_train_one",
    ]


def trial_env(args: argparse.Namespace, trial: TrialSpec) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TINYLORA_ENABLE_MODEL_LOAD": "1",
            "TINYLORA_MODEL_PATH": str(args.model_path),
            "TINYLORA_EVAL_SPEC": str(args.eval_spec),
            "TINYLORA_VALIDATE_TARGET_MODULES": "1",
            "TINYLORA_ENABLE_ADAPTER_SMOKE": "1",
            "TINYLORA_ENABLE_ONE_BATCH_TRAIN": "1",
            "TINYLORA_MAX_SEQ_LEN": str(trial.max_seq_len),
            "TINYLORA_LEARNING_RATE": str(trial.learning_rate),
            "TINYLORA_OPTIMIZER": trial.optimizer,
        }
    )
    env.pop("TINYLORA_TARGET_REMAP", None)
    return env


def score_trial(summary: dict[str, Any] | None, returncode: int | None = None) -> dict[str, Any]:
    if not summary:
        return {"score": -1000000.0, "accepted": False, "reason": "missing_summary", "loss_delta": None}
    probe = summary.get("backend_probe") or {}
    loss_delta = probe.get("loss_delta")
    finite_delta = isinstance(loss_delta, int | float) and math.isfinite(float(loss_delta))
    completed = summary.get("status") == "completed" and probe.get("reason") == "one_batch_train_completed"
    if completed and finite_delta:
        score = -float(loss_delta)
        return {
            "score": score,
            "accepted": float(loss_delta) < 0.0,
            "reason": "loss_improved" if float(loss_delta) < 0.0 else "loss_not_improved",
            "loss_delta": float(loss_delta),
        }
    reason = summary.get("block_reason") or probe.get("reason") or f"returncode_{returncode}"
    return {"score": -1000000.0, "accepted": False, "reason": reason, "loss_delta": loss_delta if finite_delta else None}


def read_summary(trial_manifest: Path) -> dict[str, Any] | None:
    manifest = json.loads(trial_manifest.read_text(encoding="utf-8"))
    summary_path = Path(manifest["default_output_dir"]) / "tinylora_training_train_one_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_leaderboard(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "trial_id",
        "accepted",
        "score",
        "loss_delta",
        "target_module",
        "learning_rate",
        "max_seq_len",
        "optimizer",
        "status",
        "reason",
        "adapter_dir",
        "summary_path",
    ]
    ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    with (out_dir / "tinylora_overnight_leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(ranked, start=1):
            writer.writerow({**{field: row.get(field) for field in fields}, "rank": index})


def read_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def next_agent_packet(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    best = ranked[0] if ranked else {}
    lines = [
        "TASK: Continue capped tinyLoRA steering search from the overnight local-maxima run.",
        f"run: {out_dir}",
        f"status: {summary['status']}",
        f"trials_completed: {summary['trials_completed']}",
        f"accepted_loss_improvements: {summary['accepted_count']}",
        f"best_trial: {best.get('trial_id')}",
        f"best_target: {best.get('target_module')}",
        f"best_learning_rate: {best.get('learning_rate')}",
        f"best_loss_delta: {best.get('loss_delta')}",
        "claim_boundary: one-batch self-loss hill climb only; benchmark acceptance still needs live eval scoring and random-control comparison.",
    ]
    (out_dir / "next_agent_packet.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


Runner = Callable[[list[str], dict[str, str], Path, Path, int], subprocess.CompletedProcess[str]]


def default_runner(command: list[str], env: dict[str, str], stdout_path: Path, stderr_path: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        return subprocess.run(
            command,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )


def run_search(args: argparse.Namespace, runner: Runner = default_runner) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials = build_trial_grid(
        parse_csv_list(args.targets),
        parse_float_list(args.learning_rates),
        max_seq_len=args.max_seq_len,
        optimizer=args.optimizer,
        max_trials=args.max_trials,
    )
    events: list[dict[str, Any]] = [{"ts": utc_now(), "event": "start", "trial_count": len(trials), "out_dir": str(args.out_dir)}]
    rows: list[dict[str, Any]] = []
    if args.resume:
        rows = read_existing_jsonl(args.out_dir / "tinylora_overnight_trials.jsonl")
        events = read_existing_jsonl(args.out_dir / "tinylora_overnight_events.jsonl") or events
    completed_trials = {row["trial_id"] for row in rows if row.get("status") in {"completed", "aborted"}}
    abort_reason: str | None = None
    fatal_exception: str | None = None
    try:
        for trial in trials:
            if trial.trial_id in completed_trials:
                continue
            ok, ram_samples = wait_for_ram_gate(args.min_available_ram_mb, args.ram_wait_seconds, args.ram_poll_seconds)
            if not ok:
                abort_reason = "ram_gate_timeout"
                events.append(
                    {
                        "ts": utc_now(),
                        "event": "ram_gate_blocked",
                        "trial_id": trial.trial_id,
                        "min_available_ram_mb": args.min_available_ram_mb,
                        "reserve_mb": args.ram_reserve_mb,
                        "samples": ram_samples,
                    }
                )
                rows.append(
                    {
                        "trial_id": trial.trial_id,
                        "target_module": trial.target_module,
                        "learning_rate": trial.learning_rate,
                        "max_seq_len": trial.max_seq_len,
                        "optimizer": trial.optimizer,
                        "returncode": None,
                        "timed_out": False,
                        "status": "aborted",
                        "reason": abort_reason,
                        "score": -1000000.0,
                        "accepted": False,
                        "loss_delta": None,
                        "before_loss": None,
                        "after_loss": None,
                        "adapter_dir": None,
                        "summary_path": None,
                        "available_ram_mb": ram_samples[-1]["available_ram_mb"] if ram_samples else None,
                    }
                )
                break
            trial_manifest = prepare_trial_manifest(args.manifest, args.out_dir, trial)
            trial_dir = args.out_dir / trial.trial_id
            trial_wrapper = materialize_trial_wrapper(args.wrapper, trial_dir, args.job_memory_mb)
            command = wrapper_command(args, trial_manifest, trial_wrapper)
            events.append({"ts": utc_now(), "event": "trial_start", **trial.__dict__, "manifest": str(trial_manifest)})
            returncode: int | None = None
            timed_out = False
            try:
                completed = runner(command, trial_env(args, trial), trial_dir / "stdout.log", trial_dir / "stderr.log", args.timeout_seconds)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            summary = read_summary(trial_manifest)
            score = score_trial(summary, returncode)
            probe = (summary or {}).get("backend_probe") or {}
            manifest = json.loads(trial_manifest.read_text(encoding="utf-8"))
            available_after = available_ram_mb()
            row = {
                "trial_id": trial.trial_id,
                "target_module": trial.target_module,
                "learning_rate": trial.learning_rate,
                "max_seq_len": trial.max_seq_len,
                "optimizer": trial.optimizer,
                "returncode": returncode,
                "timed_out": timed_out,
                "status": (summary or {}).get("status", "missing_summary"),
                "reason": score["reason"],
                "score": score["score"],
                "accepted": score["accepted"],
                "loss_delta": score["loss_delta"],
                "before_loss": probe.get("before_loss"),
                "after_loss": probe.get("after_loss"),
                "adapter_dir": probe.get("adapter_dir"),
                "summary_path": str(Path(manifest["default_output_dir"]) / "tinylora_training_train_one_summary.json"),
                "available_ram_mb": available_after,
            }
            rows = [existing for existing in rows if existing.get("trial_id") != trial.trial_id]
            rows.append(row)
            events.append({"ts": utc_now(), "event": "trial_complete", **row})
            write_jsonl(args.out_dir / "tinylora_overnight_trials.jsonl", rows)
            write_jsonl(args.out_dir / "tinylora_overnight_events.jsonl", events)
            write_leaderboard(args.out_dir, rows)
            if args.cooldown_seconds > 0 and trial != trials[-1]:
                events.append(
                    {
                        "ts": utc_now(),
                        "event": "cooldown",
                        "trial_id": trial.trial_id,
                        "cooldown_seconds": args.cooldown_seconds,
                        "available_ram_mb": available_after,
                    }
                )
                write_jsonl(args.out_dir / "tinylora_overnight_events.jsonl", events)
                time.sleep(args.cooldown_seconds)
    except Exception as exc:  # pragma: no cover - resilience path.
        fatal_exception = f"{type(exc).__name__}: {exc}"
        events.append({"ts": utc_now(), "event": "fatal_exception", "exception": fatal_exception})
    summary = {
        "status": "aborted" if abort_reason or fatal_exception else "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "trials_planned": len(trials),
        "trials_completed": len(rows),
        "accepted_count": sum(1 for row in rows if row["accepted"]),
        "abort_reason": abort_reason,
        "fatal_exception": fatal_exception,
        "job_memory_mb": args.job_memory_mb,
        "min_available_ram_mb": args.min_available_ram_mb,
        "ram_reserve_mb": args.ram_reserve_mb,
        "claim_boundary": "One-batch self-loss local-maxima search only; not benchmark evidence until live eval and random-control scoring are added.",
        "outputs": {
            "summary": str(args.out_dir / "tinylora_overnight_summary.json"),
            "events": str(args.out_dir / "tinylora_overnight_events.jsonl"),
            "trials": str(args.out_dir / "tinylora_overnight_trials.jsonl"),
            "leaderboard": str(args.out_dir / "tinylora_overnight_leaderboard.csv"),
            "next_agent_packet": str(args.out_dir / "next_agent_packet.txt"),
        },
    }
    events.append({"ts": utc_now(), "event": "summary", **summary})
    write_jsonl(args.out_dir / "tinylora_overnight_events.jsonl", events)
    write_jsonl(args.out_dir / "tinylora_overnight_trials.jsonl", rows)
    (args.out_dir / "tinylora_overnight_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    next_agent_packet(args.out_dir, rows, summary)
    return summary


def main() -> int:
    summary = run_search(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

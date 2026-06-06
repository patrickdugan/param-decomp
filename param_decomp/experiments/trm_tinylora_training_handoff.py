"""Prepare hard-capped real-training handoff artifacts for tinyLoRA organisms.

This does not train adapters. It converts accepted cached tinyLoRA proxy
organisms into a capped training manifest and wrapper plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import read_jsonl, write_jsonl


DEFAULT_SWARM_DIR = Path(r"D:\Research_Engine\runs\trm_tinylora_swarm_arc_20260606")
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_tinylora_training_handoff_arc_20260606")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hard-capped tinyLoRA training handoff artifacts.")
    parser.add_argument("--swarm-dir", type=Path, default=DEFAULT_SWARM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--ram-mb", type=int, default=2048)
    parser.add_argument("--cpu-pct", type=int, default=50)
    parser.add_argument("--io-mb-s", type=int, default=50)
    parser.add_argument("--checkpoint-interval", default="generation_or_120s")
    parser.add_argument("--training-task-id", default="tinylora-swarm-arc-d-over-a-v1")
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def score_of(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("score") or {}


def ranked_accepted(swarm_dir: Path, top_n: int) -> list[dict[str, Any]]:
    rows = read_jsonl(swarm_dir / "accepted_tinylora_organisms.jsonl")
    rows.sort(
        key=lambda row: (
            score_of(row).get("fitness", 0.0),
            score_of(row).get("control_margin", 0.0),
            score_of(row).get("delta", 0.0),
            -int(row.get("rank", 999)),
        ),
        reverse=True,
    )
    return rows[:top_n]


def candidate_row(row: dict[str, Any], rank_index: int, task_id: str) -> dict[str, Any]:
    score = score_of(row)
    target_module = row["target_module"]
    organism_id = row["organism_id"]
    return {
        "candidate_id": f"{task_id}:candidate:{rank_index:03d}",
        "organism_id": organism_id,
        "parent_ids": row.get("parent_ids", []),
        "source_family_key": row["source_family_key"],
        "target_module": target_module,
        "adapter_config": {
            "target_modules": [target_module],
            "rank": int(row["rank"]),
            "alpha": float(row["alpha"]),
            "scale": float(row["scale"]),
            "adapter_seed": int(row["adapter_seed"]),
            "init": "zero_b_low_rank_a_gaussian",
            "merge_policy": "runtime_only_until_guardrails_pass",
        },
        "trigger": {
            "top_action": row["trigger_top_action"],
            "runner_up_action": row["trigger_runner_up_action"],
            "target_action": row["trigger_target_action"],
            "margin_bucket": row["trigger_margin_bucket"],
            "max_margin": float(row["max_margin"]),
        },
        "proxy_score": {
            "delta": score.get("delta", 0.0),
            "control_margin": score.get("control_margin", 0.0),
            "fitness": score.get("fitness", 0.0),
            "rescue_count": score.get("rescue_count", 0),
            "damage_count": score.get("damage_count", 0),
            "touch_rate": score.get("touch_rate", 0.0),
        },
        "acceptance_gate": {
            "must_improve_target_family": True,
            "must_preserve_guardrails": True,
            "must_beat_fixed_label_control": True,
            "must_beat_random_tinylora_control": True,
            "max_touch_rate": 0.35,
            "abort_on_damage": True,
        },
        "claim_boundary": "Training candidate only; no adapter has been trained, merged, or accepted by live score.",
    }


def event_schema() -> dict[str, Any]:
    return {
        "event_log": "tinylora_training_events.jsonl",
        "required_events": [
            "start",
            "candidate_start",
            "checkpoint",
            "candidate_score",
            "candidate_accept_or_reject",
            "cleanup",
            "summary",
        ],
        "event_fields": {
            "start": ["ts", "event", "training_task_id", "caps", "candidate_count"],
            "checkpoint": ["ts", "event", "candidate_id", "step", "path", "ram_mb", "cpu_pct"],
            "candidate_score": ["ts", "event", "candidate_id", "target_score", "guardrail_score", "control_score"],
            "cleanup": ["ts", "event", "owned_pids_stopped", "cuda_cleanup", "ram_after_mb", "status"],
        },
        "abort_is_valid_status": True,
    }


def wrapper_script(caps: dict[str, Any]) -> str:
    memory_mb = int(caps["ram_mb"])
    cpu_permille = int(caps["cpu_pct"]) * 100
    return f"""param(
  [Parameter(Mandatory=$true)][string]$PythonExe,
  [Parameter(Mandatory=$true)][string]$TrainingScript,
  [Parameter(Mandatory=$true)][string]$ManifestPath
)

$ErrorActionPreference = "Stop"
$MemoryLimitBytes = {memory_mb}MB
$CpuRatePermille = {cpu_permille}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class JobObject {{
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
  public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);
  [DllImport("kernel32.dll")]
  public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
  [DllImport("kernel32.dll")]
  public static extern bool SetInformationJobObject(IntPtr hJob, int infoType, IntPtr lpJobObjectInfo, uint cbJobObjectInfoLength);
  public const int JobObjectExtendedLimitInformation = 9;
  public const int JobObjectCpuRateControlInformation = 15;
  public const uint JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100;
  public const uint JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1;
  public const uint JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4;
  [StructLayout(LayoutKind.Sequential)]
  public struct IO_COUNTERS {{ public ulong ReadOperationCount; public ulong WriteOperationCount; public ulong OtherOperationCount; public ulong ReadTransferCount; public ulong WriteTransferCount; public ulong OtherTransferCount; }}
  [StructLayout(LayoutKind.Sequential)]
  public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {{ public long PerProcessUserTimeLimit; public long PerJobUserTimeLimit; public uint LimitFlags; public UIntPtr MinimumWorkingSetSize; public UIntPtr MaximumWorkingSetSize; public uint ActiveProcessLimit; public long Affinity; public uint PriorityClass; public uint SchedulingClass; }}
  [StructLayout(LayoutKind.Sequential)]
  public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {{ public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation; public IO_COUNTERS IoInfo; public UIntPtr ProcessMemoryLimit; public UIntPtr JobMemoryLimit; public UIntPtr PeakProcessMemoryUsed; public UIntPtr PeakJobMemoryUsed; }}
  [StructLayout(LayoutKind.Sequential)]
  public struct JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {{ public uint ControlFlags; public uint CpuRate; }}
}}
'@

$job = [JobObject]::CreateJobObject([IntPtr]::Zero, "tinylora-swarm-training")

$limit = New-Object JobObject+JOBOBJECT_EXTENDED_LIMIT_INFORMATION
$limit.BasicLimitInformation.LimitFlags = [JobObject]::JOB_OBJECT_LIMIT_PROCESS_MEMORY
$limit.ProcessMemoryLimit = [UIntPtr]$MemoryLimitBytes
$size = [System.Runtime.InteropServices.Marshal]::SizeOf($limit)
$ptr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[System.Runtime.InteropServices.Marshal]::StructureToPtr($limit, $ptr, $false)
[JobObject]::SetInformationJobObject($job, [JobObject]::JobObjectExtendedLimitInformation, $ptr, $size) | Out-Null
[System.Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)

$cpu = New-Object JobObject+JOBOBJECT_CPU_RATE_CONTROL_INFORMATION
$cpu.ControlFlags = [JobObject]::JOB_OBJECT_CPU_RATE_CONTROL_ENABLE -bor [JobObject]::JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
$cpu.CpuRate = $CpuRatePermille
$size2 = [System.Runtime.InteropServices.Marshal]::SizeOf($cpu)
$ptr2 = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size2)
[System.Runtime.InteropServices.Marshal]::StructureToPtr($cpu, $ptr2, $false)
[JobObject]::SetInformationJobObject($job, [JobObject]::JobObjectCpuRateControlInformation, $ptr2, $size2) | Out-Null
[System.Runtime.InteropServices.Marshal]::FreeHGlobal($ptr2)

$args = @($TrainingScript, "--manifest", $ManifestPath)
$proc = Start-Process -FilePath $PythonExe -ArgumentList $args -PassThru -WindowStyle Hidden
[JobObject]::AssignProcessToJobObject($job, $proc.Handle) | Out-Null
$proc.WaitForExit()
exit $proc.ExitCode
"""


def compact_packet(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "TASK: Train top tinyLoRA organisms only under hard caps.",
            "CURRENT STATE:",
            f"- run: {summary['run_dir']}",
            f"- candidates: {summary['candidate_count']}",
            f"- source_swarm: {summary['source_swarm_dir']}",
            f"- caps: {summary['caps']}",
            "CONTRACT:",
            "- no unmanaged training; use the generated Job Object wrapper",
            "- checkpoint each candidate and treat abort as a valid outcome",
            "- accept only if live score beats fixed-label and random tinyLoRA controls",
        ]
    ) + "\n"


def run_handoff(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    accepted = ranked_accepted(args.swarm_dir, args.top_n)
    candidates = [candidate_row(row, index, args.training_task_id) for index, row in enumerate(accepted, start=1)]
    caps = {"ram_mb": args.ram_mb, "cpu_pct": args.cpu_pct, "io_mb_s": args.io_mb_s}
    plan = {
        "training_task_id": args.training_task_id,
        "status": "handoff_ready",
        "source_swarm_dir": str(args.swarm_dir),
        "candidate_count": len(candidates),
        "caps": caps,
        "checkpoint_interval": args.checkpoint_interval,
        "chunk_strategy": "one candidate at a time; batch score cards; never materialize full activation matrices",
        "control_policy": "fixed-label and random tinyLoRA controls required before promotion",
        "abort_semantics": "abort is a valid logged outcome, not a failure of the harness",
        "cleanup": "release model/tokenizer/adapter/datasets, run gc, clear CUDA if loaded, stop only owned PIDs",
    }
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "source_swarm_dir": str(args.swarm_dir),
        "candidate_count": len(candidates),
        "caps": caps,
        "claim_boundary": "Handoff only; no tinyLoRA adapter training was executed.",
        "outputs": {
            "summary": str(args.out_dir / "tinylora_training_handoff_summary.json"),
            "plan": str(args.out_dir / "tinylora_training_plan.json"),
            "candidates": str(args.out_dir / "tinylora_training_candidates.jsonl"),
            "event_schema": str(args.out_dir / "tinylora_training_event_schema.json"),
            "wrapper": str(args.out_dir / "run_tinylora_jobobject.ps1"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    write_jsonl(args.out_dir / "tinylora_training_candidates.jsonl", candidates)
    (args.out_dir / "tinylora_training_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "tinylora_training_event_schema.json").write_text(json.dumps(event_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "run_tinylora_jobobject.ps1").write_text(wrapper_script(caps), encoding="utf-8")
    packet = compact_packet(summary)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "tinylora_training_handoff_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_handoff(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

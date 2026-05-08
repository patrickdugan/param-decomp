from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from param_decomp.experiments.metta_instantiation_matrix import run as matrix_run
from param_decomp.experiments.metta_instantiation_matrix.data import (
    DEFAULT_INTELLECT3_PREDICTIONS,
    DEFAULT_INTELLECT3_SOURCE,
    DEFAULT_STORYWORLD_BUILD_ROOT,
    DEFAULT_STORYWORLD_PLAY_ROOT,
    build_prediction_gate_rows,
    build_signature_route_row,
    build_storyworld_build_rows,
    build_storyworld_play_rows,
    candidate_metrics,
    expand_metta_variants,
    load_logic_instances,
    load_prediction_groups,
    parse_grid,
    prediction_grid,
    write_jsonl,
)
from param_decomp.settings import REPO_ROOT

DEFAULT_BASE_URL = "http://snacksack-ms-7d32.tail3156cd.ts.net:8083/v1"
DEFAULT_COMPONENT_REGISTRY = REPO_ROOT / "logs" / "metta_instantiation_matrix_smoke" / "component_registry.json"
ARM_IDS = (
    "27b_direct",
    "27b_skill_prompt",
    "27b_metta_rules",
    "27b_metta_trm_frozen",
    "27b_metta_trm_adaptive",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.loads(handle.read())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_path(raw: str | None, default: Path) -> Path:
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def endpoint_json(base_url: str, suffix: str, *, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(f"{normalize_base_url(base_url)}/{suffix.lstrip('/')}")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_model_id(base_url: str, *, timeout: int) -> str:
    payload = endpoint_json(base_url, "models", timeout=timeout)
    data = payload.get("data") or payload.get("models") or []
    if not data:
        raise RuntimeError(f"no models returned by {base_url}/models")
    first = data[0]
    if isinstance(first, dict):
        return str(first.get("id") or first.get("model") or first.get("name") or "").strip()
    return str(first).strip()


def completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{normalize_base_url(base_url)}/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"completion HTTP {exc.code}: {message[:500]}") from exc
    elapsed = time.monotonic() - started
    choices = raw.get("choices") if isinstance(raw, dict) else None
    text = ""
    if choices and isinstance(choices, list):
        first = choices[0]
        if isinstance(first, dict):
            text = str(first.get("text") or first.get("message", {}).get("content") or "")
    return {
        "text": text,
        "latency_seconds": elapsed,
        "usage": raw.get("usage", {}) if isinstance(raw, dict) else {},
        "raw_response": raw,
    }


def extract_grid_from_text(text: str) -> list[list[str]] | None:
    direct = parse_grid(text)
    if direct is not None:
        return direct
    candidates = bracket_candidates(text)
    for candidate in reversed(candidates):
        grid = parse_grid(candidate)
        if grid is not None:
            return grid
    return None


def bracket_candidates(text: str) -> list[str]:
    out: list[str] = []
    starts: list[int] = []
    for index, char in enumerate(text):
        if char == "[":
            starts.append(index)
        elif char == "]" and starts:
            start = starts.pop()
            if not starts:
                out.append(text[start : index + 1])
    return out


def load_gate_cards(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if not isinstance(payload, list):
        return []
    return [
        row
        for row in payload
        if isinstance(row, dict) and row.get("accepted_for_growth") and row.get("gate_family") and row.get("component_id")
    ]


def render_gate_cards(cards: list[dict[str, Any]], limit: int = 8) -> str:
    if not cards:
        return "signature_route route_trm; candidate_verify veto visible defects; repair_step repair visible defects."
    lines = []
    for card in cards[:limit]:
        lines.append(f"{card.get('gate_family')}:{card.get('component_id')}")
    return "; ".join(lines)


def prompt_for_arm(
    *,
    arm: str,
    state_prompt: str,
    gate_cards: list[dict[str, Any]],
    previous_text: str | None = None,
    verifier_feedback: dict[str, Any] | None = None,
) -> str:
    output_contract = (
        "Return only the completed grid as a Python 2D list using only 'T', 'C', and 'X'. "
        "No markdown fences. No prose explanation. Do not write analysis. The first visible answer character must be '['."
    )
    if arm == "27b_direct":
        return f"{state_prompt}\n\n{output_contract}\n"
    if arm == "27b_skill_prompt":
        return (
            "You are operating as Hermes/Intellect-3-Logic-v1.\n"
            "Contract: parse the grid, satisfy row/column tent counts, preserve trees, place each C orthogonally "
            "adjacent to at least one T, prevent C adjacency including diagonals, then commit only the final grid.\n\n"
            f"{state_prompt}\n\n{output_contract}\n"
        )
    if arm == "27b_metta_rules":
        return (
            "Use this MeTTa-style control plane internally before answering:\n"
            "(: signature_route GateFamily)\n"
            "(: candidate_verify GateFamily)\n"
            "(: repair_step GateFamily)\n"
            "(= (commit? candidate) (and row_signature_ok col_signature_ok camp_adjacency_ok no_touching_camps))\n"
            "(= (repair? candidate) (and shape_ok visible_signature_defect))\n"
            "Do not print the MeTTa. Use it to structure the solution.\n\n"
            f"{state_prompt}\n\n{output_contract}\n"
        )
    if arm == "27b_metta_trm_frozen":
        return (
            f"{state_prompt}\n\n"
            f"SILENT_CONTROL_HINT={render_gate_cards(gate_cards, limit=4)}\n"
            "Use the hint only to decide whether to repair before committing. Do not print or discuss the hint.\n\n"
            f"{output_contract}\n"
        )
    if arm == "27b_metta_trm_adaptive":
        return (
            "Repair your previous candidate using only verifier-visible defects. Do not assume a hidden target grid.\n"
            "Use MeTTa/TRM gates: candidate_verify -> repair_step -> format_commit.\n\n"
            f"PREVIOUS_CANDIDATE\n{previous_text or ''}\n\n"
            f"VERIFIER_VISIBLE_FEEDBACK\n{json.dumps(verifier_feedback or {}, ensure_ascii=True, sort_keys=True)}\n\n"
            f"{state_prompt}\n\n{output_contract}\n"
        )
    raise ValueError(f"unsupported arm: {arm}")


def public_verifier_feedback(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_valid": bool(metrics.get("format_valid")),
        "shape_valid": bool(metrics.get("shape_valid")),
        "row_signature_l1": float(metrics.get("row_signature_l1") or 0.0),
        "col_signature_l1": float(metrics.get("col_signature_l1") or 0.0),
        "c_count_delta": float(metrics.get("c_count_delta") or 0.0),
        "camp_adjacent_ok": bool(metrics.get("camp_adjacent_ok")),
        "camps_non_touching_ok": bool(metrics.get("camps_non_touching_ok")),
        "tree_count_match": bool(metrics.get("tree_count_match")),
        "public_valid": bool(metrics.get("public_valid")),
    }


def completion_record(
    *,
    instance_id: str,
    arm: str,
    model: str,
    prompt: str,
    response: dict[str, Any],
    grid: list[list[str]] | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    text = str(response.get("text") or "")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    token_total = int(usage.get("total_tokens") or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0) or 0)
    return {
        "row_id": instance_id,
        "arm": arm,
        "task": "intellect_3_logic",
        "model_name": model,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt),
        "final": {
            "action": repr(grid) if grid is not None else text,
            "raw_action": repr(grid) if grid is not None else "",
            "raw_text": text,
            "exact_match": bool(metrics.get("exact_match")),
            "grid_cell_accuracy": float(metrics.get("cell_accuracy") or 0.0),
            "public_valid": bool(metrics.get("public_valid")),
            "latency_seconds": round(float(response.get("latency_seconds") or 0.0), 3),
            "usage": usage,
            "token_total": token_total,
            "output_status": "parsed_grid" if grid is not None else "unparsed",
            "visible_output_emitted": bool(text.strip()),
            "error": str(response.get("raw_response", {}).get("error", "")) if isinstance(response.get("raw_response"), dict) else "",
        },
    }


def select_logic_instances(source_path: Path, predictions_path: Path | None, limit: int) -> list[Any]:
    pool_limit = max(limit * 8, limit + 32)
    instances = load_logic_instances(source_path, pool_limit)
    if not instances:
        return []
    groups = load_prediction_groups(predictions_path, set(instances)) if predictions_path and predictions_path.exists() else {}
    scored = []
    for index, instance_id in enumerate(sorted(instances), start=1):
        instance = instances[instance_id]
        records = groups.get(instance_id, [])
        best_cell = 0.0
        any_exact = False
        trm_best = 0.0
        plain_best = 0.0
        for record in records:
            metrics = candidate_metrics(prediction_grid(record), instance)
            cell = float(metrics["cell_accuracy"])
            best_cell = max(best_cell, cell)
            any_exact = any_exact or bool(metrics["exact_match"])
            if record.get("arm") == "logic_skill_trm":
                trm_best = max(trm_best, cell)
            else:
                plain_best = max(plain_best, cell)
        scored.append(
            {
                "index": index,
                "instance": instance,
                "best_cell": best_cell,
                "any_exact": any_exact,
                "trm_best": trm_best,
                "plain_best": plain_best,
                "ambiguous": abs(trm_best - plain_best) <= 0.01,
                "near_miss": best_cell >= 0.82 and not any_exact,
            }
        )
    selected: list[Any] = []
    seen: set[str] = set()

    def add(rows: list[dict[str, Any]], count: int) -> None:
        for row in rows:
            instance = row["instance"]
            if len(selected) >= limit or count <= 0:
                break
            if instance.instance_id in seen:
                continue
            selected.append(instance)
            seen.add(instance.instance_id)
            count -= 1

    near = [row for row in scored if row["near_miss"]]
    near.sort(key=lambda row: (-row["best_cell"], row["index"]))
    ambiguous = [row for row in scored if row["ambiguous"]]
    ambiguous.sort(key=lambda row: (abs(row["trm_best"] - row["plain_best"]), row["index"]))
    heldout = list(reversed(scored))
    add(near, max(1, limit // 2))
    add(ambiguous, max(1, limit // 4))
    add(heldout, limit - len(selected))
    return selected[:limit]


def run_logic_collection(
    *,
    instances: list[Any],
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    gate_cards: list[dict[str, Any]],
    arms: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_rows: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []
    records_by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _index, instance in enumerate(instances, start=1):
        frozen_raw_text: str | None = None
        frozen_feedback: dict[str, Any] | None = None
        for arm in arms:
            prompt = prompt_for_arm(
                arm=arm,
                state_prompt=instance.prompt,
                gate_cards=gate_cards,
                previous_text=frozen_raw_text,
                verifier_feedback=frozen_feedback,
            )
            call_started = time.monotonic()
            try:
                response = completion(
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
            except Exception as exc:
                response = {
                    "text": "",
                    "latency_seconds": time.monotonic() - call_started,
                    "usage": {},
                    "raw_response": {"error": str(exc)},
                }
            grid = extract_grid_from_text(str(response.get("text") or ""))
            metrics = candidate_metrics(grid, instance)
            record = completion_record(
                instance_id=instance.instance_id,
                arm=arm,
                model=model,
                prompt=prompt,
                response=response,
                grid=grid,
                metrics=metrics,
            )
            transcripts.append({**record, "public_feedback": public_verifier_feedback(metrics)})
            records_by_instance[instance.instance_id].append(record)
            if arm == "27b_metta_trm_frozen":
                frozen_raw_text = str(response.get("text") or "")
                frozen_feedback = public_verifier_feedback(metrics)
    for index, instance in enumerate(instances, start=1):
        records = records_by_instance[instance.instance_id]
        base_rows.append(build_signature_route_row(instance, records, index))
        for record_index, record in enumerate(records, start=1):
            base_rows.extend(build_prediction_gate_rows(instance, record, index=index, record_index=record_index))
    return base_rows, transcripts, [record for records in records_by_instance.values() for record in records]


def summarize_arm_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_arm[str(record.get("arm") or "unknown")].append(record)
    summary = {}
    for arm, rows in sorted(by_arm.items()):
        exact = [bool(row["final"].get("exact_match")) for row in rows]
        public_valid = [bool(row["final"].get("public_valid")) for row in rows]
        cell = [float(row["final"].get("grid_cell_accuracy") or 0.0) for row in rows]
        parsed = [row["final"].get("output_status") == "parsed_grid" for row in rows]
        tokens = [int(row["final"].get("token_total") or 0) for row in rows]
        latency = [float(row["final"].get("latency_seconds") or 0.0) for row in rows]
        summary[arm] = {
            "count": len(rows),
            "parsed_grid_rate": float(np.mean(parsed)) if parsed else 0.0,
            "exact_match_rate": float(np.mean(exact)) if exact else 0.0,
            "public_valid_rate": float(np.mean(public_valid)) if public_valid else 0.0,
            "mean_cell_accuracy": float(np.mean(cell)) if cell else 0.0,
            "mean_tokens": float(np.mean(tokens)) if tokens else 0.0,
            "mean_latency_seconds": float(np.mean(latency)) if latency else 0.0,
        }
    return summary


def render_flex_report(summary: dict[str, Any], matrix_metrics: dict[str, Any] | None) -> str:
    lines = [
        "# Snacksack Qwen 27B MeTTa/TRM Flex",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        f"Endpoint: `{summary['endpoint']['base_url']}`",
        f"Model: `{summary['endpoint']['model']}`",
        "",
        "## Arm Metrics",
        "",
        "| Arm | n | parsed | public-valid | exact | mean cell | mean tokens | mean latency s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, metrics in summary["arm_metrics"].items():
        lines.append(
            f"| `{arm}` | {metrics['count']} | {metrics['parsed_grid_rate']:.3f} | "
            f"{metrics['public_valid_rate']:.3f} | {metrics['exact_match_rate']:.3f} | "
            f"{metrics['mean_cell_accuracy']:.3f} | {metrics['mean_tokens']:.1f} | "
            f"{metrics['mean_latency_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Matrix Result",
            "",
        ]
    )
    if matrix_metrics:
        variants = matrix_metrics.get("variant_results", {})
        for variant, result in variants.items():
            modular = result["split_metrics"]["modular"]["non_train"]
            baseline = result["split_metrics"]["monolithic_baseline"]["non_train"]
            lines.append(
                f"- `{variant}`: modular non-train accuracy `{modular.get('accuracy', 0.0):.4f}`, "
                f"baseline `{baseline.get('accuracy', 0.0):.4f}`, false commit `{modular.get('false_commit_rate', 0.0):.4f}`."
            )
        accepted = sum(1 for row in matrix_metrics.get("component_registry", []) if row.get("accepted_for_growth"))
        lines.append(f"- Accepted components: `{accepted}`.")
    else:
        lines.append("- Matrix metrics were not available.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This run uses Qwen 27B for proposal generation and verifier-visible MeTTa/TRM gates for control-plane analysis. "
            "The hidden target grid is used only for benchmark scoring and row labels, not in prompts or repair feedback.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Snacksack Qwen 27B MeTTa/TRM flex experiment.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model")
    parser.add_argument("--source", default=str(DEFAULT_INTELLECT3_SOURCE))
    parser.add_argument("--predictions", default=str(DEFAULT_INTELLECT3_PREDICTIONS))
    parser.add_argument("--component-registry", default=str(DEFAULT_COMPONENT_REGISTRY))
    parser.add_argument("--out-dir")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--storyworld-limit", type=int, default=2)
    parser.add_argument("--no-storyworld", action="store_true")
    parser.add_argument("--storyworld-play-root", default=str(DEFAULT_STORYWORLD_PLAY_ROOT))
    parser.add_argument("--storyworld-build-root", default=str(DEFAULT_STORYWORLD_BUILD_ROOT))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--endpoint-timeout", type=int, default=10)
    parser.add_argument("--matrix-epochs", type=int, default=35)
    parser.add_argument("--matrix-hidden-dim", type=int, default=24)
    parser.add_argument("--matrix-recursive-steps", type=int, default=2)
    parser.add_argument("--matrix-backend", choices=["python", "rust", "auto"], default="python")
    parser.add_argument("--arms", nargs="+", choices=ARM_IDS, default=list(ARM_IDS))
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "logs" / f"snacksack_qwen27b_metta_flex_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve_path(args.source, DEFAULT_INTELLECT3_SOURCE)
    predictions_path = resolve_path(args.predictions, DEFAULT_INTELLECT3_PREDICTIONS)
    component_registry = resolve_path(args.component_registry, DEFAULT_COMPONENT_REGISTRY)
    gate_cards = load_gate_cards(component_registry)
    model = args.model or fetch_model_id(args.base_url, timeout=args.endpoint_timeout)
    endpoint = {
        "base_url": normalize_base_url(args.base_url),
        "model": model,
        "models_payload": endpoint_json(args.base_url, "models", timeout=args.endpoint_timeout),
    }
    instances = select_logic_instances(source_path, predictions_path, args.limit)
    if not instances:
        raise RuntimeError(f"no Intellect-3 Logic instances loaded from {source_path}")
    base_rows, transcripts, records = run_logic_collection(
        instances=instances,
        base_url=args.base_url,
        model=model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.request_timeout,
        gate_cards=gate_cards,
        arms=tuple(args.arms),
    )
    if not args.no_storyworld and args.storyworld_limit > 0:
        play_root = resolve_path(args.storyworld_play_root, DEFAULT_STORYWORLD_PLAY_ROOT)
        build_root = resolve_path(args.storyworld_build_root, DEFAULT_STORYWORLD_BUILD_ROOT)
        base_rows.extend(build_storyworld_play_rows(play_root, max(1, args.storyworld_limit // 2)))
        base_rows.extend(build_storyworld_build_rows(build_root, max(1, args.storyworld_limit - args.storyworld_limit // 2)))
    matrix_rows = expand_metta_variants(base_rows)
    matrix_input = out_dir / "matrix_input.rows.jsonl"
    write_jsonl(matrix_input, matrix_rows)
    write_jsonl(out_dir / "qwen_transcripts.jsonl", transcripts)
    write_jsonl(out_dir / "qwen_records.jsonl", records)
    write_jsonl(out_dir / "base_control_rows.jsonl", base_rows)
    arm_metrics = summarize_arm_metrics(records)
    pre_summary = {
        "status": "collected",
        "generated_at_utc": utc_now(),
        "endpoint": endpoint,
        "source_path": str(source_path),
        "predictions_path": str(predictions_path),
        "component_registry_path": str(component_registry),
        "gate_card_count": len(gate_cards),
        "logic_instance_count": len(instances),
        "base_row_count": len(base_rows),
        "matrix_row_count": len(matrix_rows),
        "arm_metrics": arm_metrics,
    }
    write_json(out_dir / "snacksack_flex_collection_summary.json", pre_summary)
    matrix_code = matrix_run.main(
        [
            "--trace",
            str(matrix_input),
            "--out-dir",
            str(out_dir),
            "--backend",
            args.matrix_backend,
            "--epochs",
            str(args.matrix_epochs),
            "--hidden-dim",
            str(args.matrix_hidden_dim),
            "--recursive-steps",
            str(args.matrix_recursive_steps),
            "--no-storyworld",
        ]
    )
    matrix_metrics = read_json(out_dir / "metrics.json") if (out_dir / "metrics.json").exists() else None
    final_summary = {
        **pre_summary,
        "status": "completed" if matrix_code == 0 else "matrix_failed",
        "matrix_code": matrix_code,
        "matrix_metrics_path": str(out_dir / "metrics.json"),
        "component_registry_output": str(out_dir / "component_registry.json"),
    }
    write_json(out_dir / "snacksack_flex_summary.json", final_summary)
    (out_dir / "report.md").write_text(render_flex_report(final_summary, matrix_metrics), encoding="utf-8")
    return final_summary


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(sys.argv[1:] if argv is None else argv))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "out_dir": str(Path(summary["matrix_metrics_path"]).parent),
                "model": summary["endpoint"]["model"],
                "logic_instance_count": summary["logic_instance_count"],
                "arm_metrics": summary["arm_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from param_decomp.accel import score_rank_one_linear_components
from param_decomp.experiments.metta_keygate.data import (
    DECISIONS,
    DECISION_TO_ID,
    KeygateRow,
    encode_rows,
    load_trace_rows,
    sha256_file,
    split_indices,
    trace_summary,
)
from param_decomp.experiments.metta_keygate.model import KeygateMLP
from param_decomp.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT


DEFAULT_TRACE = Path("param_decomp/experiments/metta_keygate/fixtures/example_keygate_trace.jsonl")
DEFAULT_TRACE_SHA256 = "d0344bee007156269434b2c99963b54dda133d9d28917f9560bffec3c72546f5"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML object")
    return data


def resolve_path(raw: str | Path | None, default: Path) -> Path:
    path = Path(raw) if raw is not None else default
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_run_dir(config: dict[str, Any], override: str | None) -> Path:
    raw = override or config.get("out_dir")
    if raw:
        out_dir = Path(raw)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = PARAM_DECOMP_OUT_DIR / "metta_keygate_v1" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def train_model(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[KeygateMLP, list[dict[str, Any]]]:
    set_seed(seed)
    model = KeygateMLP(input_dim=features.shape[1], hidden_dim=hidden_dim, output_dim=len(DECISIONS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    events: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[train_indices])
        loss = loss_fn(logits, y[train_indices])
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            model.eval()
            with torch.no_grad():
                train_metrics = metrics_for_logits(labels[train_indices], model(x[train_indices]).numpy())
                val_metrics = metrics_for_logits(labels[val_indices], model(x[val_indices]).numpy()) if len(val_indices) else {}
            events.append(
                {
                    "ts": utc_now(),
                    "event": "checkpoint",
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu().item()),
                    "train_accuracy": train_metrics.get("accuracy"),
                    "validation_accuracy": val_metrics.get("accuracy"),
                }
            )
    return model, events


def metrics_for_logits(labels: np.ndarray, logits: np.ndarray, rows: list[KeygateRow] | None = None) -> dict[str, float]:
    if len(labels) == 0:
        return {"count": 0.0}
    preds = np.argmax(logits, axis=1)
    commit_id = DECISION_TO_ID["commit"]
    repair_id = DECISION_TO_ID["repair"]
    expected_commit = labels == commit_id
    expected_repair = labels == repair_id
    expected_non_commit = labels != commit_id
    metrics = {
        "count": float(len(labels)),
        "accuracy": float(np.mean(preds == labels)),
        "false_commit_rate": _safe_rate(np.logical_and(preds == commit_id, expected_non_commit), expected_non_commit),
        "false_reject_rate": _safe_rate(np.logical_and(preds != commit_id, expected_commit), expected_commit),
        "repair_preservation_rate": _safe_rate(np.logical_and(preds == repair_id, expected_repair), expected_repair),
    }
    if rows is not None:
        hard_negative = np.array([row.input_kind == "hard_negative" for row in rows], dtype=bool)
        partial = np.array([row.input_kind == "partial_positive_repair" for row in rows], dtype=bool)
        metrics["hard_negative_false_commit_rate"] = _safe_rate(preds == commit_id, hard_negative)
        metrics["partial_false_commit_rate"] = _safe_rate(preds == commit_id, partial)
    return metrics


def _safe_rate(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    denom = int(np.sum(denominator_mask))
    if denom == 0:
        return 0.0
    return float(np.sum(np.logical_and(numerator_mask, denominator_mask)) / denom)


def evaluate_splits(
    *,
    rows: list[KeygateRow],
    labels: np.ndarray,
    logits: np.ndarray,
    indices_by_split: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for split, indices in indices_by_split.items():
        split_rows = [rows[int(index)] for index in indices]
        result[split] = metrics_for_logits(labels[indices], logits[indices], split_rows)
    non_train = np.concatenate([indices for split, indices in indices_by_split.items() if split != "train" and len(indices)])
    result["non_train"] = metrics_for_logits(labels[non_train], logits[non_train], [rows[int(index)] for index in non_train])
    return result


def head_components(model: KeygateMLP) -> tuple[np.ndarray, np.ndarray, list[str], list[float]]:
    weight = model.head.weight.detach().cpu().numpy().astype(np.float32)
    u, singular_values, vh = np.linalg.svd(weight, full_matrices=False)
    svd_u = [u[:, index] * singular_values[index] for index in range(len(singular_values))]
    svd_v = [vh[index] for index in range(len(singular_values))]
    class_u: list[np.ndarray] = []
    class_v: list[np.ndarray] = []
    for class_index in range(weight.shape[0]):
        basis = np.zeros((weight.shape[0],), dtype=np.float32)
        basis[class_index] = 1.0
        class_u.append(basis)
        class_v.append(weight[class_index])
    components_u = np.stack([*svd_u, *class_u]).astype(np.float32)
    components_v = np.stack([*svd_v, *class_v]).astype(np.float32)
    component_ids = [
        *[f"head_svd:{index:04d}" for index in range(len(singular_values))],
        *[f"head_row:{decision}" for decision in DECISIONS],
    ]
    return components_u, components_v, component_ids, [float(value) for value in singular_values]


def build_slices(rows: list[KeygateRow], local_indices: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    local_by_global = {int(global_index): local_index for local_index, global_index in enumerate(local_indices)}
    slice_members: dict[str, list[int]] = {}
    for global_index in local_indices:
        row = rows[int(global_index)]
        for name in (f"split={row.split}", f"input_kind={row.input_kind}", f"decision={row.expected_decision}"):
            slice_members.setdefault(name, []).append(local_by_global[int(global_index)])
    names = sorted(slice_members)
    offsets = [0]
    values: list[int] = []
    for name in names:
        values.extend(slice_members[name])
        offsets.append(len(values))
    return names, np.array(offsets, dtype=np.int64), np.array(values, dtype=np.int64)


def component_effects(
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    rows: list[KeygateRow],
    reference_logits: np.ndarray,
    components_u: np.ndarray,
    components_v: np.ndarray,
    component_ids: list[str],
) -> list[dict[str, Any]]:
    base_metrics = metrics_for_logits(labels, reference_logits, rows)
    effects: list[dict[str, Any]] = []
    for index, component_id in enumerate(component_ids):
        contribution = (inputs @ components_v[index])[:, None] * components_u[index][None, :]
        ablated_logits = reference_logits - contribution
        ablated = metrics_for_logits(labels, ablated_logits, rows)
        effects.append(
            {
                "component_id": component_id,
                "accuracy_drop": round(base_metrics["accuracy"] - ablated["accuracy"], 6),
                "false_commit_delta": round(ablated["false_commit_rate"] - base_metrics["false_commit_rate"], 6),
                "false_reject_delta": round(ablated["false_reject_rate"] - base_metrics["false_reject_rate"], 6),
                "repair_preservation_delta": round(
                    ablated["repair_preservation_rate"] - base_metrics["repair_preservation_rate"],
                    6,
                ),
                "ablated_metrics": ablated,
            }
        )
    return effects


def random_control_effects(
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    rows: list[KeygateRow],
    reference_logits: np.ndarray,
    component_norm: float,
    hidden_dim: int,
    output_dim: int,
    count: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    components_u = rng.normal(size=(count, output_dim)).astype(np.float32)
    components_v = rng.normal(size=(count, hidden_dim)).astype(np.float32)
    for index in range(count):
        norm = np.linalg.norm(np.outer(components_u[index], components_v[index]))
        if norm > 0:
            components_u[index] *= np.sqrt(component_norm / norm)
            components_v[index] *= np.sqrt(component_norm / norm)
    effects = component_effects(
        inputs=inputs,
        labels=labels,
        rows=rows,
        reference_logits=reference_logits,
        components_u=components_u,
        components_v=components_v,
        component_ids=[f"random_control:{index:04d}" for index in range(count)],
    )
    return {
        "count": count,
        "mean_accuracy_drop": float(np.mean([effect["accuracy_drop"] for effect in effects])),
        "max_accuracy_drop": float(np.max([effect["accuracy_drop"] for effect in effects])),
        "mean_false_commit_delta": float(np.mean([effect["false_commit_delta"] for effect in effects])),
        "max_false_commit_delta": float(np.max([effect["false_commit_delta"] for effect in effects])),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config_path) if args.config_path else None)
    seed = int(args.seed if args.seed is not None else config.get("seed", 20260508))
    trace_override = args.trace is not None
    trace_path = resolve_path(args.trace or config.get("trace_path"), DEFAULT_TRACE)
    expected_sha = str(config.get("trace_sha256", "" if trace_override else DEFAULT_TRACE_SHA256)).lower()
    actual_sha = sha256_file(trace_path)
    if expected_sha and actual_sha.lower() != expected_sha:
        raise ValueError(f"trace SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")

    out_dir = build_run_dir(config, args.out_dir)
    shutil.copy2(trace_path, out_dir / "source_trace.jsonl")

    rows = load_trace_rows(trace_path)
    features, labels, feature_names = encode_rows(rows)
    split_map = split_indices(rows)
    train_indices = split_map["train"]
    val_indices = split_map["validation"]
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("trace must include non-empty train and validation splits")

    model, events = train_model(
        features=features,
        labels=labels,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=seed,
        hidden_dim=int(args.hidden_dim or config.get("hidden_dim", 32)),
        epochs=int(args.epochs or config.get("epochs", 250)),
        lr=float(args.lr or config.get("lr", 0.03)),
        weight_decay=float(args.weight_decay or config.get("weight_decay", 0.0001)),
    )
    x = torch.tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy().astype(np.float32)
        hidden = model.hidden(x).detach().cpu().numpy().astype(np.float32)

    split_metrics = evaluate_splits(rows=rows, labels=labels, logits=logits, indices_by_split=split_map)
    score_indices = np.concatenate([indices for split, indices in split_map.items() if split != "train" and len(indices)])
    score_rows = [rows[int(index)] for index in score_indices]
    components_u, components_v, component_ids, singular_values = head_components(model)
    slice_names, slice_offsets, slice_indices = build_slices(rows, score_indices)
    backend = str(args.backend or config.get("backend", "auto"))
    scorer_records = score_rank_one_linear_components(
        inputs=hidden[score_indices],
        labels=labels[score_indices],
        reference_logits=logits[score_indices],
        components_u=components_u,
        components_v=components_v,
        component_ids=component_ids,
        row_indices=score_indices,
        slice_names=slice_names,
        slice_offsets=slice_offsets,
        slice_indices=slice_indices,
        backend=backend,  # type: ignore[arg-type]
        rust_threads=int(args.rust_threads or config.get("rust_threads", 0)),
    )
    effects = component_effects(
        inputs=hidden[score_indices],
        labels=labels[score_indices],
        rows=score_rows,
        reference_logits=logits[score_indices],
        components_u=components_u,
        components_v=components_v,
        component_ids=component_ids,
    )
    top_effect = max(effects, key=lambda item: (item["accuracy_drop"], item["false_commit_delta"]))
    top_index = component_ids.index(top_effect["component_id"])
    random_controls = random_control_effects(
        inputs=hidden[score_indices],
        labels=labels[score_indices],
        rows=score_rows,
        reference_logits=logits[score_indices],
        component_norm=float(np.linalg.norm(np.outer(components_u[top_index], components_v[top_index]))),
        hidden_dim=hidden.shape[1],
        output_dim=len(DECISIONS),
        count=int(config.get("random_controls", 32)),
        seed=seed + 17,
    )

    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "experiment": "metta_keygate_v1",
        "seed": seed,
        "trace_path": str(trace_path),
        "trace_sha256": actual_sha,
        "out_dir": str(out_dir),
        "trace_summary": trace_summary(rows),
        "model": {
            "input_dim": int(features.shape[1]),
            "hidden_dim": int(hidden.shape[1]),
            "output_dim": len(DECISIONS),
            "param_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        "feature_names": feature_names,
        "split_metrics": split_metrics,
        "component_scoring": {
            "backend": backend,
            "score_row_count": int(len(score_indices)),
            "component_count": len(component_ids),
            "singular_values": singular_values,
            "top_effect": top_effect,
            "random_controls": random_controls,
        },
        "claim_boundary": (
            "This run tests a tiny typed gate over MeTTa-derived rows. "
            "It is not evidence that Qwen hidden-module SVD clusters are causal semantics."
        ),
    }

    write_outputs(out_dir, summary, events, scorer_records, effects)
    return summary


def write_outputs(
    out_dir: Path,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    scorer_records: list[dict[str, Any]],
    effects: list[dict[str, Any]],
) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "component_scores.json").write_text(
        json.dumps({"scorer_records": scorer_records, "behavior_effects": effects}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        for split, metrics in summary["split_metrics"].items():
            for metric, value in metrics.items():
                writer.writerow({"split": split, "metric": metric, "value": value})
    manifest = {
        "experiment": summary["experiment"],
        "generated_at_utc": summary["generated_at_utc"],
        "trace_path": summary["trace_path"],
        "trace_sha256": summary["trace_sha256"],
        "outputs": ["summary.json", "component_scores.json", "events.jsonl", "metrics.csv", "source_trace.jsonl"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MeTTa keygate VDP experiment.")
    parser.add_argument("config_path", nargs="?", help="Optional YAML config path, used by pd-local.")
    parser.add_argument("--trace", help="Trace JSONL path.")
    parser.add_argument("--out-dir", help="Output artifact directory.")
    parser.add_argument("--backend", choices=["python", "rust", "auto"], help="Component scoring backend.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--rust-threads", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = run_experiment(args)
    print(
        json.dumps(
            {"status": summary["status"], "out_dir": summary["out_dir"], "split_metrics": summary["split_metrics"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

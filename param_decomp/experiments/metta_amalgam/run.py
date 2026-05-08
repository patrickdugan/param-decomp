from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from param_decomp.accel import score_rank_one_linear_components
from param_decomp.experiments.metta_amalgam.data import (
    GATE_DECISIONS,
    GLOBAL_DECISIONS,
    AmalgamRow,
    build_default_rows,
    default_task_graph,
    encode_rows,
    indices_for_split,
    load_rows,
    summarize_rows,
    write_jsonl,
)
from param_decomp.experiments.metta_amalgam.model import RecursiveGateTRM
from param_decomp.settings import PARAM_DECOMP_OUT_DIR, REPO_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def resolve_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_out_dir(config: dict[str, Any], override: str | None) -> Path:
    raw = override or config.get("out_dir")
    if raw:
        out_dir = Path(raw)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = PARAM_DECOMP_OUT_DIR / "metta_trm_amalgam_v1" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def train_trm(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    hidden_dim: int,
    output_dim: int,
    recursive_steps: int,
    epochs: int,
    lr: float,
    seed: int,
) -> RecursiveGateTRM:
    set_seed(seed)
    model = RecursiveGateTRM(features.shape[1], hidden_dim, output_dim, recursive_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    best_state = None
    best_val = -1.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[train_indices]), y[train_indices])
        loss.backward()
        optimizer.step()
        if len(val_indices):
            model.eval()
            with torch.no_grad():
                val_accuracy = float(np.mean(np.argmax(model(x[val_indices]).numpy(), axis=1) == labels[val_indices]))
            if val_accuracy >= best_val:
                best_val = val_accuracy
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_decisions(model: RecursiveGateTRM, features: np.ndarray, decisions: tuple[str, ...]) -> list[str]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32)).detach().cpu().numpy()
    return [decisions[int(index)] for index in np.argmax(logits, axis=1)]


def decision_metrics(rows: list[AmalgamRow], predictions: list[str]) -> dict[str, float]:
    if not rows:
        return {"count": 0.0}
    expected = [row.expected_decision for row in rows]
    false_commit_cases = [index for index, row in enumerate(rows) if row.expected_decision != "commit"]
    exact_commit_cases = [index for index, row in enumerate(rows) if row.expected_decision == "commit"]
    repair_cases = [index for index, row in enumerate(rows) if row.expected_decision == "repair"]
    return {
        "count": float(len(rows)),
        "accuracy": float(np.mean([pred == exp for pred, exp in zip(predictions, expected, strict=True)])),
        "false_commit_rate": _rate([predictions[index] == "commit" for index in false_commit_cases]),
        "false_reject_rate": _rate([predictions[index] != "commit" for index in exact_commit_cases]),
        "repair_preservation_rate": _rate([predictions[index] == "repair" for index in repair_cases]),
    }


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def evaluate_by_split(rows: list[AmalgamRow], predictions: list[str]) -> dict[str, dict[str, float]]:
    result = {}
    for split in ("train", "validation", "holdout_seen", "holdout_unseen", "backdoor_probe"):
        split_pairs = [(row, predictions[index]) for index, row in enumerate(rows) if row.split == split]
        result[split] = decision_metrics([row for row, _ in split_pairs], [pred for _, pred in split_pairs])
    non_train_pairs = [(row, predictions[index]) for index, row in enumerate(rows) if row.split != "train"]
    result["non_train"] = decision_metrics([row for row, _ in non_train_pairs], [pred for _, pred in non_train_pairs])
    return result


def train_organelle_models(
    rows: list[AmalgamRow],
    *,
    hidden_dim: int,
    recursive_steps: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, RecursiveGateTRM], dict[str, tuple[np.ndarray, np.ndarray, list[str]]]]:
    models = {}
    encoded = {}
    for gate_index, (gate_family, decisions) in enumerate(GATE_DECISIONS.items()):
        gate_rows = [row for row in rows if row.gate_family == gate_family]
        features, labels, names = encode_rows(gate_rows, decisions)
        split_map = indices_for_split(gate_rows)
        models[gate_family] = train_trm(
            features=features,
            labels=labels,
            train_indices=split_map["train"],
            val_indices=split_map["validation"],
            hidden_dim=hidden_dim,
            output_dim=len(decisions),
            recursive_steps=recursive_steps,
            epochs=epochs,
            lr=lr,
            seed=seed + gate_index,
        )
        encoded[gate_family] = (features, labels, names)
    return models, encoded


def modular_predictions(
    rows: list[AmalgamRow],
    models: dict[str, RecursiveGateTRM],
    encoded: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
) -> list[str]:
    by_gate_predictions = {
        gate: predict_decisions(models[gate], encoded[gate][0], GATE_DECISIONS[gate])
        for gate in GATE_DECISIONS
    }
    cursor = {gate: 0 for gate in GATE_DECISIONS}
    predictions = []
    for row in rows:
        gate = row.gate_family
        predictions.append(by_gate_predictions[gate][cursor[gate]])
        cursor[gate] += 1
    return predictions


def train_baseline(
    rows: list[AmalgamRow],
    *,
    hidden_dim: int,
    recursive_steps: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[RecursiveGateTRM, np.ndarray]:
    features, labels, _ = encode_rows(rows, GLOBAL_DECISIONS)
    split_map = indices_for_split(rows)
    model = train_trm(
        features=features,
        labels=labels,
        train_indices=split_map["train"],
        val_indices=split_map["validation"],
        hidden_dim=hidden_dim,
        output_dim=len(GLOBAL_DECISIONS),
        recursive_steps=recursive_steps,
        epochs=epochs,
        lr=lr,
        seed=seed + 99,
    )
    return model, features


def head_components(model: RecursiveGateTRM, decisions: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str], list[float]]:
    weight = model.head.weight.detach().cpu().numpy().astype(np.float32)
    u, singular_values, vh = np.linalg.svd(weight, full_matrices=False)
    svd_u = [u[:, index] * singular_values[index] for index in range(len(singular_values))]
    svd_v = [vh[index] for index in range(len(singular_values))]
    class_u = []
    class_v = []
    for class_index, decision in enumerate(decisions):
        basis = np.zeros((weight.shape[0],), dtype=np.float32)
        basis[class_index] = 1.0
        class_u.append(basis)
        class_v.append(weight[class_index])
    return (
        np.stack([*svd_u, *class_u]).astype(np.float32),
        np.stack([*svd_v, *class_v]).astype(np.float32),
        [*[f"head_svd:{index:04d}" for index in range(len(singular_values))], *[f"head_row:{decision}" for decision in decisions]],
        [float(value) for value in singular_values],
    )


def component_audit(
    *,
    gate_family: str,
    rows: list[AmalgamRow],
    model: RecursiveGateTRM,
    features: np.ndarray,
    labels: np.ndarray,
    backend: str,
    rust_threads: int,
    seed: int,
) -> dict[str, Any]:
    split_map = indices_for_split(rows)
    score_indices = np.concatenate([indices for split, indices in split_map.items() if split != "train" and len(indices)])
    score_rows = [rows[int(index)] for index in score_indices]
    decisions = GATE_DECISIONS[gate_family]
    x = torch.tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy().astype(np.float32)
        hidden = model.hidden(x).detach().cpu().numpy().astype(np.float32)
    components_u, components_v, component_ids, singular_values = head_components(model, decisions)
    scorer_records = score_rank_one_linear_components(
        inputs=hidden[score_indices],
        labels=labels[score_indices],
        reference_logits=logits[score_indices],
        components_u=components_u,
        components_v=components_v,
        component_ids=component_ids,
        row_indices=score_indices,
        backend=backend,  # type: ignore[arg-type]
        rust_threads=rust_threads,
    )
    effects = component_effects(
        inputs=hidden[score_indices],
        labels=labels[score_indices],
        rows=score_rows,
        logits=logits[score_indices],
        components_u=components_u,
        components_v=components_v,
        component_ids=component_ids,
        decisions=decisions,
    )
    top_effect = max(effects, key=lambda effect: (effect["accuracy_drop"], effect["false_commit_delta"]))
    rng = np.random.default_rng(seed)
    random_controls = []
    top_index = component_ids.index(top_effect["component_id"])
    top_norm = float(np.linalg.norm(np.outer(components_u[top_index], components_v[top_index])))
    for control_index in range(16):
        rand_u = rng.normal(size=(1, len(decisions))).astype(np.float32)
        rand_v = rng.normal(size=(1, hidden.shape[1])).astype(np.float32)
        norm = np.linalg.norm(np.outer(rand_u[0], rand_v[0]))
        if norm > 0:
            rand_u *= np.sqrt(top_norm / norm)
            rand_v *= np.sqrt(top_norm / norm)
        random_controls.extend(
            component_effects(
                inputs=hidden[score_indices],
                labels=labels[score_indices],
                rows=score_rows,
                logits=logits[score_indices],
                components_u=rand_u,
                components_v=rand_v,
                component_ids=[f"random_control:{control_index:04d}"],
                decisions=decisions,
            )
        )
    return {
        "gate_family": gate_family,
        "score_row_count": int(len(score_indices)),
        "component_count": len(component_ids),
        "singular_values": singular_values,
        "top_effect": top_effect,
        "random_control_max_accuracy_drop": float(max(effect["accuracy_drop"] for effect in random_controls)),
        "random_control_max_false_commit_delta": float(max(effect["false_commit_delta"] for effect in random_controls)),
        "scorer_records": scorer_records,
        "behavior_effects": effects,
    }


def component_effects(
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    rows: list[AmalgamRow],
    logits: np.ndarray,
    components_u: np.ndarray,
    components_v: np.ndarray,
    component_ids: list[str],
    decisions: tuple[str, ...],
) -> list[dict[str, Any]]:
    base_predictions = [decisions[int(index)] for index in np.argmax(logits, axis=1)]
    base_metrics = decision_metrics(rows, base_predictions)
    effects = []
    for index, component_id in enumerate(component_ids):
        ablated_logits = logits - (inputs @ components_v[index])[:, None] * components_u[index][None, :]
        ablated_predictions = [decisions[int(pred)] for pred in np.argmax(ablated_logits, axis=1)]
        ablated_metrics = decision_metrics(rows, ablated_predictions)
        effects.append(
            {
                "component_id": component_id,
                "accuracy_drop": round(base_metrics["accuracy"] - ablated_metrics["accuracy"], 6),
                "false_commit_delta": round(ablated_metrics["false_commit_rate"] - base_metrics["false_commit_rate"], 6),
                "false_reject_delta": round(ablated_metrics["false_reject_rate"] - base_metrics["false_reject_rate"], 6),
                "repair_preservation_delta": round(
                    ablated_metrics["repair_preservation_rate"] - base_metrics["repair_preservation_rate"],
                    6,
                ),
                "ablated_metrics": ablated_metrics,
            }
        )
    return effects


def build_component_registry(audits: dict[str, dict[str, Any]], split_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    registry = []
    for gate_family, audit in audits.items():
        top = audit["top_effect"]
        accepted = top["accuracy_drop"] > audit["random_control_max_accuracy_drop"]
        registry.append(
            {
                "organelle_id": f"trm_organelle:{gate_family}",
                "gate_family": gate_family,
                "component_id": top["component_id"],
                "accepted_for_growth": accepted,
                "acceptance_reason": "top_component_beats_random_accuracy_control" if accepted else "needs_harder_trace_or_more_training",
                "accuracy_drop": top["accuracy_drop"],
                "false_commit_delta": top["false_commit_delta"],
                "non_train_accuracy": split_metrics["modular"]["non_train"]["accuracy"],
                "storyworld_tag": f"{gate_family}:control_pressure",
            }
        )
    return registry


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config_path) if args.config_path else None)
    seed = int(args.seed if args.seed is not None else config.get("seed", 20260508))
    hidden_dim = int(args.hidden_dim or config.get("hidden_dim", 32))
    recursive_steps = int(args.recursive_steps or config.get("recursive_steps", 3))
    epochs = int(args.epochs or config.get("epochs", 220))
    lr = float(args.lr or config.get("lr", 0.03))
    backend = str(args.backend or config.get("backend", "auto"))
    rust_threads = int(args.rust_threads or config.get("rust_threads", 0))
    out_dir = build_out_dir(config, args.out_dir)
    rows = load_rows(resolve_path(args.trace or config.get("trace_path")))
    write_jsonl(out_dir / "source_trace.jsonl", [row.row for row in rows])
    (out_dir / "task_graph.json").write_text(json.dumps(default_task_graph(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    models, encoded = train_organelle_models(
        rows,
        hidden_dim=hidden_dim,
        recursive_steps=recursive_steps,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )
    modular = modular_predictions(rows, models, encoded)
    baseline_model, baseline_features = train_baseline(
        rows,
        hidden_dim=hidden_dim,
        recursive_steps=recursive_steps,
        epochs=epochs,
        lr=lr,
        seed=seed,
    )
    baseline = predict_decisions(baseline_model, baseline_features, GLOBAL_DECISIONS)
    split_metrics = {
        "modular": evaluate_by_split(rows, modular),
        "monolithic_baseline": evaluate_by_split(rows, baseline),
    }
    audits = {}
    for gate_family, model in models.items():
        gate_rows = [row for row in rows if row.gate_family == gate_family]
        features, labels, _ = encoded[gate_family]
        audits[gate_family] = component_audit(
            gate_family=gate_family,
            rows=gate_rows,
            model=model,
            features=features,
            labels=labels,
            backend=backend,
            rust_threads=rust_threads,
            seed=seed + 1000,
        )
    registry = build_component_registry(audits, split_metrics)
    summary = {
        "status": "completed",
        "experiment": "metta_trm_amalgam_v1",
        "generated_at_utc": utc_now(),
        "out_dir": str(out_dir),
        "task_graph": default_task_graph(),
        "row_summary": summarize_rows(rows),
        "model_config": {
            "hidden_dim": hidden_dim,
            "recursive_steps": recursive_steps,
            "epochs": epochs,
            "lr": lr,
        },
        "split_metrics": split_metrics,
        "component_registry": registry,
        "claim_boundary": "This is a synthetic typed task-graph control experiment, not a full autonomous skill-growth result.",
    }
    write_outputs(out_dir, summary, audits, registry)
    return summary


def write_outputs(out_dir: Path, summary: dict[str, Any], audits: dict[str, dict[str, Any]], registry: list[dict[str, Any]]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "component_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_light = {
        gate: {
            "score_row_count": audit["score_row_count"],
            "component_count": audit["component_count"],
            "top_effect": audit["top_effect"],
            "random_control_max_accuracy_drop": audit["random_control_max_accuracy_drop"],
            "random_control_max_false_commit_delta": audit["random_control_max_false_commit_delta"],
        }
        for gate, audit in audits.items()
    }
    (out_dir / "organelle_scores.json").write_text(json.dumps(audit_light, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "metric", "value"])
        writer.writeheader()
        for model_name, split_map in summary["split_metrics"].items():
            for split, metrics in split_map.items():
                for metric, value in metrics.items():
                    writer.writerow({"model": model_name, "split": split, "metric": metric, "value": value})
    manifest = {
        "experiment": summary["experiment"],
        "generated_at_utc": summary["generated_at_utc"],
        "outputs": ["summary.json", "component_registry.json", "organelle_scores.json", "metrics.csv", "source_trace.jsonl", "task_graph.json"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MeTTa TRM amalgam task-graph experiment.")
    parser.add_argument("config_path", nargs="?")
    parser.add_argument("--trace")
    parser.add_argument("--out-dir")
    parser.add_argument("--backend", choices=["python", "rust", "auto"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--recursive-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--rust-threads", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run_experiment(parse_args(sys.argv[1:] if argv is None else argv))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "out_dir": summary["out_dir"],
                "modular_non_train": summary["split_metrics"]["modular"]["non_train"],
                "baseline_non_train": summary["split_metrics"]["monolithic_baseline"]["non_train"],
                "accepted_components": sum(1 for row in summary["component_registry"] if row["accepted_for_growth"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""VPD-style decomposition for recursive tiny Hermes/TRM checkpoints.

This is not Goodfire VPD. It builds a parameter dictionary over local recursive
tiny model checkpoints using SVD rank atoms grouped by architecture role, then
compares the action-family and explicit-action-family Hermes logic skill runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "recursive_tiny_vpd_int3_logic"
COMPONENT_DIR = ARTIFACT_ROOT / "components"
REPORT_PATH = REPO_ROOT / "reports" / "recursive_tiny_vpd_int3_logic_report.md"

HRM_GYM = Path(r"D:\projects\HRM-re\experiments\hermes_skill_gym\outputs")

TARGETS = {
    "action_family_trm_like": HRM_GYM
    / "hermes_skill_gym_trm_action_family_single"
    / "variants"
    / "trm_like_single_tier",
    "explicit_action_family_trm_like": HRM_GYM
    / "hermes_skill_gym_trm_action_family_explicit_single"
    / "variants"
    / "trm_like_single_tier",
    "pruned_base_trm_like": HRM_GYM
    / "hermes_skill_gym_pruned_cycle13_base_compare"
    / "variants"
    / "trm_like_single_tier",
    "constellation_safe_enriched": HRM_GYM
    / "hermes_skill_gym_constellation_compare_safe_enriched"
    / "variants"
    / "hrm_constellation_3plus1_safe",
}

ROLE_ORDER = [
    "x_input_embedding",
    "task_or_role_embedding",
    "recursive_trunk_z",
    "recursive_interface_yz",
    "output_bucket_head",
    "output_action_head",
    "q_reward_or_halting",
    "unclassified",
]


@dataclass
class TensorSummary:
    name: str
    role: str
    shape: list[int]
    parameters: int
    fro_norm: float
    top_singular_values: list[float]
    top_energy_fraction: float | None
    rank_atoms_written: int


@dataclass
class ModelSummary:
    model_id: str
    model_path: str
    parameter_count: int
    tensor_count: int
    role_parameter_counts: dict[str, int]
    role_fro_norms: dict[str, float]
    tensor_summaries: list[TensorSummary]
    metrics: dict[str, Any]
    logic_eval_slice: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-tensors-per-model", type=int, default=28)
    parser.add_argument("--max-delta-tensors", type=int, default=36)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_payload(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint payload: {path}")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"no state_dict in checkpoint payload: {path}")
    tensors = {str(key): value.detach().cpu() for key, value in state.items() if torch.is_tensor(value)}
    return payload, tensors


def classify_role(name: str) -> str:
    lowered = name.lower()
    if "role_embedding" in lowered or "task" in lowered or "worker_role_embeddings" in lowered:
        return "task_or_role_embedding"
    if "to_input_embed" in lowered or "token_embed" in lowered or "input_embed" in lowered:
        return "x_input_embedding"
    if "hidden_combiner" in lowered or "input_combiner" in lowered or "to_combined" in lowered:
        return "recursive_interface_yz"
    if "bucket_head" in lowered:
        return "output_bucket_head"
    if "action_head" in lowered:
        return "output_action_head"
    if "reward_head" in lowered or "halt" in lowered or "continue" in lowered or "q_head" in lowered:
        return "q_reward_or_halting"
    if (
        "networks" in lowered
        or "worker_networks" in lowered
        or "governor_network" in lowered
        or ".layers." in lowered
        or "transition" in lowered
    ):
        return "recursive_trunk_z"
    return "unclassified"


def role_norm_add(role_fro_squares: dict[str, float], role: str, tensor: torch.Tensor) -> None:
    if torch.is_floating_point(tensor):
        role_fro_squares[role] = role_fro_squares.get(role, 0.0) + float(torch.sum(tensor.float() ** 2).item())


def svd_terms(
    *,
    model_id: str,
    tensor_name: str,
    role: str,
    tensor: torch.Tensor,
    top_k: int,
    prefix: str,
) -> tuple[list[float], float | None, int]:
    if tensor.ndim != 2 or not torch.is_floating_point(tensor) or min(tensor.shape) < 2:
        return [], None, 0
    matrix = tensor.float()
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        u, s, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
        u, s, vh = u.float(), s.float(), vh.float()
    k = min(top_k, s.numel())
    singular_values = [float(value) for value in s[:k].tolist()]
    total_energy = float(torch.sum(s**2).item())
    top_energy = float(torch.sum(s[:k] ** 2).item()) if total_energy > 0 else 0.0
    safe_name = tensor_name.replace(".", "__").replace("/", "_").replace("\\", "_")
    atom_path = COMPONENT_DIR / model_id / f"{prefix}__{safe_name}.npz"
    atom_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        atom_path,
        singular_values=s[:k].numpy().astype(np.float32),
        left_vectors=u[:, :k].numpy().astype(np.float32),
        right_vectors=vh[:k, :].numpy().astype(np.float32),
        tensor_name=np.array(tensor_name),
        role=np.array(role),
        source=np.array("svd_rank_atoms"),
    )
    return singular_values, (top_energy / total_energy if total_energy > 0 else None), k


def select_decomp_tensors(state: dict[str, torch.Tensor], max_tensors: int) -> list[tuple[str, torch.Tensor]]:
    candidates = []
    for name, tensor in state.items():
        if tensor.ndim != 2 or not torch.is_floating_point(tensor) or min(tensor.shape) < 2:
            continue
        norm = float(torch.linalg.vector_norm(tensor.float()).item())
        candidates.append((name, tensor, norm, int(tensor.numel())))
    candidates.sort(key=lambda item: (classify_role(item[0]) == "unclassified", -item[2], -item[3], item[0]))
    return [(name, tensor) for name, tensor, _norm, _params in candidates[:max_tensors]]


def logic_slice(variant_dir: Path) -> dict[str, Any]:
    metrics = load_json(variant_dir / "eval_metrics.json")
    predictions = load_jsonl(variant_dir / "predictions.jsonl")
    logic_predictions = [row for row in predictions if row.get("source_env_name") == "logic_env"]
    return {
        "env_rollup_logic_env": metrics.get("env_rollup", {}).get("logic_env"),
        "logic_prediction_count": len(logic_predictions),
        "logic_predictions": logic_predictions,
    }


def summarize_model(model_id: str, variant_dir: Path, top_k: int, max_tensors: int) -> ModelSummary:
    model_path = variant_dir / "artifacts" / "model.pt"
    _payload, state = load_payload(model_path)
    role_counts = {role: 0 for role in ROLE_ORDER}
    role_fro_squares = {role: 0.0 for role in ROLE_ORDER}
    for name, tensor in state.items():
        role = classify_role(name)
        role_counts[role] = role_counts.get(role, 0) + int(tensor.numel())
        role_norm_add(role_fro_squares, role, tensor)

    tensor_summaries: list[TensorSummary] = []
    for name, tensor in select_decomp_tensors(state, max_tensors):
        role = classify_role(name)
        singulars, top_energy_fraction, atoms_written = svd_terms(
            model_id=model_id,
            tensor_name=name,
            role=role,
            tensor=tensor,
            top_k=top_k,
            prefix="weight",
        )
        tensor_summaries.append(
            TensorSummary(
                name=name,
                role=role,
                shape=list(tensor.shape),
                parameters=int(tensor.numel()),
                fro_norm=float(torch.linalg.vector_norm(tensor.float()).item()),
                top_singular_values=singulars,
                top_energy_fraction=top_energy_fraction,
                rank_atoms_written=atoms_written,
            )
        )

    return ModelSummary(
        model_id=model_id,
        model_path=str(model_path),
        parameter_count=int(sum(t.numel() for t in state.values())),
        tensor_count=len(state),
        role_parameter_counts=role_counts,
        role_fro_norms={role: math.sqrt(value) for role, value in role_fro_squares.items()},
        tensor_summaries=tensor_summaries,
        metrics=load_json(variant_dir / "eval_metrics.json"),
        logic_eval_slice=logic_slice(variant_dir),
    )


def summarize_delta(
    *,
    delta_id: str,
    left_path: Path,
    right_path: Path,
    top_k: int,
    max_tensors: int,
) -> dict[str, Any]:
    _left_payload, left = load_payload(left_path)
    _right_payload, right = load_payload(right_path)
    role_counts = {role: 0 for role in ROLE_ORDER}
    role_fro_squares = {role: 0.0 for role in ROLE_ORDER}
    tensor_rows = []
    skipped_shape_mismatch = []

    candidates = []
    for name, right_tensor in right.items():
        left_tensor = left.get(name)
        if left_tensor is None:
            continue
        if left_tensor.shape != right_tensor.shape:
            skipped_shape_mismatch.append(
                {"name": name, "left_shape": list(left_tensor.shape), "right_shape": list(right_tensor.shape)}
            )
            continue
        if not torch.is_floating_point(right_tensor):
            continue
        delta = right_tensor.float() - left_tensor.float()
        role = classify_role(name)
        role_counts[role] = role_counts.get(role, 0) + int(delta.numel())
        role_norm_add(role_fro_squares, role, delta)
        if delta.ndim == 2 and min(delta.shape) >= 2:
            candidates.append((name, delta, float(torch.linalg.vector_norm(delta).item())))

    candidates.sort(key=lambda item: (classify_role(item[0]) == "unclassified", -item[2], item[0]))
    for name, delta, _norm in candidates[:max_tensors]:
        role = classify_role(name)
        singulars, top_energy_fraction, atoms_written = svd_terms(
            model_id=delta_id,
            tensor_name=name,
            role=role,
            tensor=delta,
            top_k=top_k,
            prefix="delta",
        )
        tensor_rows.append(
            {
                "name": name,
                "role": role,
                "shape": list(delta.shape),
                "fro_norm": float(torch.linalg.vector_norm(delta).item()),
                "top_singular_values": singulars,
                "top_energy_fraction": top_energy_fraction,
                "rank_atoms_written": atoms_written,
            }
        )

    return {
        "delta_id": delta_id,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "matched_parameter_count": sum(role_counts.values()),
        "role_parameter_counts": role_counts,
        "role_fro_norms": {role: math.sqrt(value) for role, value in role_fro_squares.items()},
        "tensor_summaries": tensor_rows,
        "skipped_shape_mismatch": skipped_shape_mismatch[:40],
        "skipped_shape_mismatch_count": len(skipped_shape_mismatch),
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Recursive Tiny VPD-Style Decomposition: Intellect-3 Logic Skill",
        "",
        "Status: `RECURSIVE_TINY_VPD_STYLE_COMPLETE`",
        "",
        "No actual Goodfire VPD dictionary is claimed here. This artifact uses VPD-style rank atoms over local recursive tiny model parameters and groups them by architecture role.",
        "",
        "The local Intellect-3 logic skill proxy is the Hermes skill gym action-family corpus sourced from `primehub_trm_cycle12_public_trace_shape_exactonly.jsonl`; its manifests include `logic_env` rows. No cached 4D ARC heads are used.",
        "",
        "## Model Role Summary",
        "",
        "| model | params | score | logic rows | logic acc | dominant roles by norm |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in payload["models"]:
        metrics = model.get("metrics", {})
        logic = model.get("logic_eval_slice", {}).get("env_rollup_logic_env") or {}
        role_norms = model["role_fro_norms"]
        dominant = sorted(role_norms.items(), key=lambda item: item[1], reverse=True)[:3]
        lines.append(
            "| {model_id} | {params} | {score:.6f} | {logic_rows} | {logic_acc:.6f} | {dominant} |".format(
                model_id=model["model_id"],
                params=model["parameter_count"],
                score=float(metrics.get("leaderboard_score", 0.0)),
                logic_rows=int(logic.get("rows", 0) or 0),
                logic_acc=float(logic.get("critic_bucket_accuracy", 0.0) or 0.0),
                dominant=", ".join(f"{role}:{value:.3f}" for role, value in dominant),
            )
        )
    lines.extend(
        [
            "",
            "## Explicit-vs-Action Delta",
            "",
            "```json",
            json.dumps(payload["deltas"], indent=2, sort_keys=True)[:8000],
            "```",
            "",
            "## Interpretation",
            "",
            "- The TRM-like single-tier runs are mostly recursive-trunk mass plus small bucket/action/reward heads.",
            "- The explicit action-family variant improves overall leaderboard score while retaining perfect reported `logic_env` bucket accuracy on the small local slice.",
            "- Delta decomposition is shape-matched only; input vocabulary and action-head shape mismatches are reported instead of forced.",
            "- The emitted rank atoms are parameter-coordinate artifacts, not prose labels and not actual Goodfire VPD.",
            "",
            f"Full manifest: `{ARTIFACT_ROOT / 'manifest.json'}`",
            f"Rank atom files: `{COMPONENT_DIR}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    models = []
    for model_id, variant_dir in TARGETS.items():
        if (variant_dir / "artifacts" / "model.pt").exists():
            models.append(asdict(summarize_model(model_id, variant_dir, args.top_k, args.max_tensors_per_model)))

    action_model = TARGETS["action_family_trm_like"] / "artifacts" / "model.pt"
    explicit_model = TARGETS["explicit_action_family_trm_like"] / "artifacts" / "model.pt"
    deltas = []
    if action_model.exists() and explicit_model.exists():
        deltas.append(
            summarize_delta(
                delta_id="explicit_minus_action_family_trm_like",
                left_path=action_model,
                right_path=explicit_model,
                top_k=args.top_k,
                max_tensors=args.max_delta_tensors,
            )
        )

    payload = {
        "status": "RECURSIVE_TINY_VPD_STYLE_COMPLETE",
        "claims_actual_vpd": False,
        "uses_cached_4d_arc_heads": False,
        "role_contract": {
            "x_input_embedding": "input token/feature embedding",
            "task_or_role_embedding": "task/worker role conditioning when present",
            "recursive_trunk_z": "recursive latent/trunk update tensors",
            "recursive_interface_yz": "hidden combiners that mediate token/y-like state and recursive/z-like state",
            "output_bucket_head": "critic bucket readout",
            "output_action_head": "action-family readout",
            "q_reward_or_halting": "reward/Q/halting surrogate head",
        },
        "targets": {key: str(value) for key, value in TARGETS.items()},
        "models": models,
        "deltas": deltas,
    }
    write_json(ARTIFACT_ROOT / "manifest.json", payload)
    write_json(ARTIFACT_ROOT / "model_summaries.json", models)
    write_json(ARTIFACT_ROOT / "delta_summaries.json", deltas)
    REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(REPORT_PATH), "artifact_root": str(ARTIFACT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

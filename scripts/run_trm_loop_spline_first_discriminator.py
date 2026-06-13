"""First discriminator for the TRM loop-spline hypothesis.

Tests whether recurrence-step intervention profiles add predictive information
beyond static SVD/component features for held-out depth-transfer damage.

This is VPD-style/SVD rank-one component analysis over recursive tiny Hermes
checkpoints. It does not claim actual Goodfire VPD and does not use cached ARC
heads, policy code, RCPI, patch search, or a new algebra.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
HRM_REPO = Path(r"D:\projects\HRM-re")
if str(HRM_REPO) not in sys.path:
    sys.path.insert(0, str(HRM_REPO))

from experiments.hermes_skill_gym.hermes_dataset import (  # noqa: E402
    HermesSkillDataset,
    collate_batch,
    read_jsonl,
)
from experiments.hermes_skill_gym.train_hrm_hermes import (  # noqa: E402
    ConstellationGovernorHermesClassifier,
    HermesHRMClassifier,
    load_trained_model,
)


ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "recursive_vpd_style"
REPORT_PATH = REPO_ROOT / "reports" / "trm_loop_spline_first_discriminator.md"
DEVICE = "cpu"
MODEL_DIR = (
    Path(r"D:\projects\HRM-re\experiments\hermes_skill_gym\outputs")
    / "hermes_skill_gym_trm_action_family_explicit_single"
    / "variants"
    / "trm_like_single_tier"
)
MODEL_PATH = MODEL_DIR / "artifacts" / "model.pt"
TRAIN_PATH = MODEL_DIR.parents[1] / "dataset" / "train.jsonl"
EVAL_PATH = MODEL_DIR.parents[1] / "dataset" / "eval.jsonl"

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
class Component:
    component_id: str
    module_id: str
    role: str
    module_depth: int
    svd_rank_index: int
    scale: float
    u_norm: float
    v_norm: float
    uv_norm_product: float
    reconstruction_contribution: float
    u: torch.Tensor
    v: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--base-depth", type=int, default=4)
    parser.add_argument("--probe-target", type=int, default=256)
    parser.add_argument("--rank-atoms-per-module", type=int, default=4)
    parser.add_argument("--per-decile", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    if "networks" in lowered or "worker_networks" in lowered or ".layers." in lowered:
        return "recursive_trunk_z"
    return "unclassified"


def module_depth(name: str) -> int:
    parts = name.split(".")
    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts):
            with contextlib.suppress(ValueError):
                return int(parts[idx + 1])
    if "networks" in parts:
        idx = parts.index("networks")
        if idx + 1 < len(parts):
            with contextlib.suppress(ValueError):
                return int(parts[idx + 1])
    return -1


def load_rows(train: Path, eval_path: Path, seed: int, target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rows = []
    for path in [train, eval_path]:
        if path.exists():
            rows.extend(read_jsonl(str(path)))
    rng = random.Random(seed)
    rng.shuffle(rows)
    desired = target * 2
    status = "OK"
    if len(rows) < desired:
        status = "UNDERPOWERED_PROBES"
    rows = rows[: min(len(rows), desired)]
    split = len(rows) // 2
    return rows[:split], rows[split:], status


def build_loader(rows: list[dict[str, Any]], payload: dict[str, Any], batch_size: int) -> DataLoader:
    config = payload["config"]
    vocab = payload["vocab"]
    bucket_labels = list(config["bucket_labels"])
    action_labels = list(config["action_labels"])
    dataset = HermesSkillDataset(
        rows,
        vocab,
        {label: index for index, label in enumerate(bucket_labels)},
        {label: index for index, label in enumerate(action_labels)},
        max_tokens=int(config.get("max_tokens", 512)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, int(vocab["<PAD>"])),
    )


def build_components(model: torch.nn.Module, rank_atoms_per_module: int) -> list[Component]:
    components: list[Component] = []
    for name, param in model.named_parameters():
        if param.ndim != 2 or not torch.is_floating_point(param) or min(param.shape) < 2:
            continue
        matrix = param.detach().cpu().float()
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        total_energy = float(torch.sum(s**2).item())
        k = min(rank_atoms_per_module, s.numel())
        for rank_index in range(k):
            scale = float(s[rank_index].item())
            u_vec = u[:, rank_index].clone()
            v_vec = vh[rank_index, :].clone()
            role = classify_role(name)
            components.append(
                Component(
                    component_id=f"{name}::svd{rank_index:03d}",
                    module_id=name,
                    role=role,
                    module_depth=module_depth(name),
                    svd_rank_index=rank_index,
                    scale=scale,
                    u_norm=float(torch.linalg.vector_norm(u_vec).item()),
                    v_norm=float(torch.linalg.vector_norm(v_vec).item()),
                    uv_norm_product=float(torch.linalg.vector_norm(u_vec).item() * torch.linalg.vector_norm(v_vec).item()),
                    reconstruction_contribution=float((scale * scale) / total_energy) if total_energy > 0 else 0.0,
                    u=u_vec,
                    v=v_vec,
                )
            )
    return components


@contextlib.contextmanager
def intervened(model: torch.nn.Module, component: Component, alpha: float):
    params = dict(model.named_parameters())
    param = params[component.module_id]
    delta = (alpha - 1.0) * component.scale * torch.outer(
        component.u.to(param.device, dtype=param.dtype),
        component.v.to(param.device, dtype=param.dtype),
    )
    with torch.no_grad():
        param.add_(delta)
    try:
        yield
    finally:
        with torch.no_grad():
            param.sub_(delta)


@contextlib.contextmanager
def reasoning_depth(model: torch.nn.Module, depth: int):
    old_depth = getattr(model, "reasoning_steps", None)
    if old_depth is not None:
        model.reasoning_steps = depth
    try:
        yield
    finally:
        if old_depth is not None:
            model.reasoning_steps = old_depth


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


def head_metrics(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    bucket_logits = model.bucket_head(hidden)
    action_logits = model.action_head(hidden)
    reward_pred = model.reward_head(hidden).squeeze(-1)
    labels = batch["bucket_labels"].to(hidden.device)
    action_labels = batch["action_labels"].to(hidden.device)
    rewards = batch["rewards"].to(hidden.device)
    bucket_loss = F.cross_entropy(bucket_logits, labels, reduction="none")
    action_loss = F.cross_entropy(action_logits, action_labels, reduction="none")
    reward_loss = F.smooth_l1_loss(reward_pred, rewards, reduction="none")
    loss = bucket_loss + 0.5 * action_loss + 0.05 * reward_loss
    probs = torch.softmax(bucket_logits, dim=-1)
    pred = probs.argmax(dim=-1)
    true_logits = bucket_logits.gather(1, labels[:, None]).squeeze(1)
    other = bucket_logits.clone()
    other.scatter_(1, labels[:, None], -1e9)
    margin = true_logits - other.max(dim=-1).values
    return {
        "loss": loss,
        "pred": pred,
        "correct": pred.eq(labels),
        "margin": margin,
        "confidence": probs.max(dim=-1).values,
        "action_score": action_logits.gather(1, action_labels[:, None]).squeeze(1),
    }


def trace_hierarchical(
    model: HermesHRMClassifier,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch: dict[str, Any],
) -> list[dict[str, float]]:
    tokens = model.to_input_embed(input_ids)
    hiddens = torch.zeros_like(tokens).unsqueeze(0).repeat(model.num_networks, 1, 1, 1)
    hiddens_dict = {index: hidden for index, hidden in enumerate(hiddens)}
    total_low_steps = model.reasoning_steps * model.lowest_steps_per_reasoning_step
    rows = []
    prev_highest = None
    with torch.no_grad():
        for index in range(total_low_steps):
            iteration = index + 1
            for network_index, (network, hidden_combine, evaluate_network_at) in enumerate(
                zip(model.networks, model.hidden_combiners, model.evaluate_networks_at)
            ):
                if iteration % evaluate_network_at != 0:
                    continue
                combined_input = hidden_combine((tokens, *hiddens_dict.values()), network_index)
                hiddens_dict[network_index] = network(combined_input, mask=attention_mask)
            highest = hiddens_dict[model.num_networks - 1]
            pooled = masked_mean(highest, attention_mask)
            metrics = head_metrics(model, pooled, batch)
            if prev_highest is None:
                drift = torch.zeros((), device=highest.device)
            else:
                drift = torch.linalg.vector_norm((highest - prev_highest).reshape(highest.shape[0], -1), dim=1).mean()
            prev_highest = highest.clone()
            rows.append(
                {
                    "step": float(index + 1),
                    "loss": float(metrics["loss"].mean().item()),
                    "z_drift": float(drift.item()),
                    "logit_margin": float(metrics["margin"].mean().item()),
                    "confidence": float(metrics["confidence"].mean().item()),
                    "y_norm": float(torch.linalg.vector_norm(pooled, dim=1).mean().item()),
                    "z_norm": float(torch.linalg.vector_norm(highest.reshape(highest.shape[0], -1), dim=1).mean().item()),
                    "output_action_score": float(metrics["action_score"].mean().item()),
                }
            )
    return rows


def trace_model(model: torch.nn.Module, loader: DataLoader, depth: int, pad_id: int) -> tuple[list[dict[str, float]], dict[str, Any]]:
    step_acc: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    step_count: dict[int, int] = defaultdict(int)
    all_correct = []
    all_loss = []
    with reasoning_depth(model, depth):
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = input_ids.ne(pad_id)
            if isinstance(model, HermesHRMClassifier) and not isinstance(model, ConstellationGovernorHermesClassifier):
                trace_rows = trace_hierarchical(model, input_ids, attention_mask, batch)
            else:
                raise RuntimeError("BLOCKED_NO_RECURSION_TRACE: unsupported model class for per-step trace")
            for row in trace_rows:
                step = int(row["step"])
                for key, value in row.items():
                    if key != "step":
                        step_acc[step][key] += float(value)
                step_count[step] += 1
            final = trace_rows[-1]
            # Re-run final correctness with public forward for exact per-example correctness.
            bucket_logits, _action_logits, _reward_pred, _pooled = model(input_ids, attention_mask)
            labels = batch["bucket_labels"].to(DEVICE)
            loss = F.cross_entropy(bucket_logits, labels, reduction="none")
            all_loss.extend(float(x) for x in loss.tolist())
            all_correct.extend(bool(x) for x in bucket_logits.argmax(dim=-1).eq(labels).tolist())
    profile = []
    for step in sorted(step_acc):
        row = {"step": step}
        for key, value in step_acc[step].items():
            row[key] = value / max(1, step_count[step])
        profile.append(row)
    return profile, {"loss": float(np.mean(all_loss)), "correct": all_correct, "accuracy": float(np.mean(all_correct))}


def damage(base_correct: list[bool], patched_correct: list[bool]) -> int:
    return int(sum(1 for base, patched in zip(base_correct, patched_correct) if base and not patched))


def evaluate_component(
    model: torch.nn.Module,
    loader: DataLoader,
    component: Component | None,
    alpha: float,
    depth: int,
    pad_id: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if component is None:
        return trace_model(model, loader, depth, pad_id)
    with intervened(model, component, alpha):
        return trace_model(model, loader, depth, pad_id)


def static_feature_row(component: Component, static_damage: int, endpoint_loss: float) -> dict[str, Any]:
    return {
        "component_id": component.component_id,
        "module_id": component.module_id,
        "role": component.role,
        "module_depth": component.module_depth,
        "svd_rank_index": component.svd_rank_index,
        "scale": component.scale,
        "u_norm": component.u_norm,
        "v_norm": component.v_norm,
        "uv_norm_product": component.uv_norm_product,
        "reconstruction_contribution": component.reconstruction_contribution,
        "static_ablation_damage_T_probe_A": static_damage,
        "endpoint_final_step_loss_alpha0_probe_A": endpoint_loss,
    }


def assign_deciles(rows: list[dict[str, Any]]) -> dict[str, int]:
    values = np.array([row["static_ablation_damage_T_probe_A"] for row in rows], dtype=float)
    order = np.argsort(values, kind="stable")
    deciles = {}
    for rank, idx in enumerate(order):
        deciles[rows[int(idx)]["component_id"]] = min(9, int(10 * rank / max(1, len(rows))))
    return deciles


def stratified_sample(rows: list[dict[str, Any]], per_decile: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    deciles = assign_deciles(rows)
    grouped: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        grouped[deciles[row["component_id"]]].append(row["component_id"])
    sample = []
    for decile in range(10):
        ids = grouped.get(decile, [])
        rng.shuffle(ids)
        sample.extend(ids[:per_decile])
    return sample


def interpolate_profile(profile: list[dict[str, float]], channel: str) -> np.ndarray:
    return np.array([float(row.get(channel, 0.0)) for row in profile], dtype=float)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) < 2 or len(neg) < 2:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
        total += len(neg)
    return wins / total if total else None


def extract_shape_features(profile_rows: dict[str, Any]) -> dict[str, float]:
    channels = ["z_drift", "logit_margin", "confidence", "y_norm", "z_norm", "output_action_score"]
    out = {}
    steps = None
    for alpha_key in ["0.5", "1.5"]:
        profile = profile_rows[alpha_key]
        steps = np.arange(1, len(profile) + 1, dtype=float)
        for channel in channels:
            values = interpolate_profile(profile, channel)
            mass = np.abs(values)
            denom = float(mass.sum()) or 1.0
            out[f"a{alpha_key}_{channel}_centroid"] = float((steps * mass).sum() / denom)
            split = max(1, len(values) // 2)
            early = float(mass[:split].sum())
            late = float(mass[split:].sum())
            out[f"a{alpha_key}_{channel}_early_late"] = early / (late + 1e-9)
            peak = int(np.argmax(mass))
            out[f"a{alpha_key}_{channel}_peak_step"] = float(peak + 1)
            out[f"a{alpha_key}_{channel}_signed_peak"] = float(values[peak])
            out[f"a{alpha_key}_{channel}_monotonicity"] = float(np.corrcoef(steps, values)[0, 1]) if np.std(values) > 0 else 0.0
        drift = interpolate_profile(profile, "z_drift")
        if len(drift) >= 3:
            alternating = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(len(drift))])
            out[f"a{alpha_key}_z_drift_period2_power"] = float(abs(np.dot(drift - drift.mean(), alternating)) / (np.linalg.norm(drift) + 1e-9))
        else:
            out[f"a{alpha_key}_z_drift_period2_power"] = 0.0
    p05 = profile_rows["0.5"]
    p15 = profile_rows["1.5"]
    for channel in channels:
        v05 = interpolate_profile(p05, channel)
        v15 = interpolate_profile(p15, channel)
        out[f"alpha_asym_{channel}"] = float(np.mean(v15 - v05))
    return out


def ridge_cv_predict(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    preds = np.zeros_like(y, dtype=float)
    for test_idx in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        x_train = X[train_mask]
        y_train = y[train_mask]
        x_test = X[test_idx]
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0] = 1.0
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_train = np.column_stack([np.ones(len(x_train)), x_train])
        x_test = np.column_stack([np.ones(len(x_test)), x_test])
        lam = 1.0
        beta = np.linalg.pinv(x_train.T @ x_train + lam * np.eye(x_train.shape[1])) @ x_train.T @ y_train
        preds[test_idx] = x_test @ beta
    return preds


def build_design(static_rows: list[dict[str, Any]], shape_rows: list[dict[str, Any]] | None) -> tuple[np.ndarray, list[str]]:
    roles = sorted({row["role"] for row in static_rows})
    feature_names = ["module_depth", "svd_rank_index", "scale", "u_norm", "v_norm", "uv_norm_product", "reconstruction_contribution", "static_ablation_damage_T_probe_A", "endpoint_final_step_loss_alpha0_probe_A"]
    shape_names = sorted([key for key in shape_rows[0] if key != "component_id"]) if shape_rows else []
    names = feature_names + [f"role={role}" for role in roles] + shape_names
    shape_by_id = {row["component_id"]: row for row in shape_rows or []}
    rows = []
    for row in static_rows:
        values = [float(row.get(name, 0.0)) for name in feature_names]
        values.extend(1.0 if row["role"] == role else 0.0 for role in roles)
        if shape_rows:
            shape = shape_by_id[row["component_id"]]
            values.extend(float(shape.get(name, 0.0)) for name in shape_names)
        rows.append(values)
    return np.array(rows, dtype=float), names


def make_folds(n: int, seed: int, k: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return [part for part in np.array_split(idx, min(k, n)) if len(part) > 0]


def permutation_pvalue(
    static_rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
    y: np.ndarray,
    folds: list[np.ndarray],
    observed_delta: float,
    permutations: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    deciles = assign_deciles(static_rows)
    grouped = defaultdict(list)
    for row in shape_rows:
        grouped[deciles[row["component_id"]]].append(row)
    exceed = 0
    for _ in range(permutations):
        shuffled = []
        for decile, rows in grouped.items():
            ids = [row["component_id"] for row in rows]
            vals = [{k: v for k, v in row.items() if k != "component_id"} for row in rows]
            rng.shuffle(vals)
            for component_id, values in zip(ids, vals, strict=False):
                shuffled.append({"component_id": component_id, **values})
        xs, _ = build_design(static_rows, None)
        xp, _ = build_design(static_rows, shuffled)
        ps = ridge_cv_predict(xs, y, folds)
        pp = ridge_cv_predict(xp, y, folds)
        delta = spearman(y, pp) - spearman(y, ps)
        if not math.isnan(delta) and delta >= observed_delta:
            exceed += 1
    return (exceed + 1) / (permutations + 1)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TRM Loop-Spline First Discriminator",
        "",
        f"Final label: `{payload['final_label']}`",
        "",
        f"Model: `{payload['model_path']}`",
        f"Component count: `{payload['component_count']}`; sampled: `{payload['sampled_component_count']}`",
        f"Probe A/B counts: `{payload['probe_counts']['A']}` / `{payload['probe_counts']['B']}`",
        f"Base depth T: `{payload['T']}`; T_prime: `{payload['T_prime']}`",
        "",
        "No actual Goodfire VPD is claimed. This uses a VPD-style/SVD rank-one component dictionary over recursive tiny model tensors. Cached 4D ARC heads were not used.",
        "",
        "## Reliability",
        "",
        "```json",
        json.dumps(payload["reliability"], indent=2, sort_keys=True),
        "```",
        "",
        "## Nested Models",
        "",
        "```json",
        json.dumps(payload["nested_models"], indent=2, sort_keys=True),
        "```",
        "",
        "## Per-Decile Breakdown",
        "",
        "```json",
        json.dumps(payload["per_decile_breakdown"], indent=2, sort_keys=True),
        "```",
        "",
        "## Top Partial Features",
        "",
        "```json",
        json.dumps(payload["top_partial_features"], indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    global DEVICE, ARTIFACT_ROOT, REPORT_PATH
    DEVICE = args.device
    ARTIFACT_ROOT = args.artifact_root
    REPORT_PATH = args.report
    random.seed(args.seed)
    np.random.seed(args.seed)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload, model = load_trained_model(str(args.model), DEVICE)
    if not isinstance(model, HermesHRMClassifier) or isinstance(model, ConstellationGovernorHermesClassifier):
        final = {"final_label": "BLOCKED_NO_RECURSION_TRACE", "reason": type(model).__name__}
        write_json(ARTIFACT_ROOT / "manifest.json", final)
        REPORT_PATH.write_text(render_report({**final, "model_path": str(args.model), "component_count": 0, "sampled_component_count": 0, "probe_counts": {"A": 0, "B": 0}, "T": args.base_depth, "T_prime": args.base_depth * 2, "reliability": {}, "nested_models": {}, "per_decile_breakdown": {}, "top_partial_features": []}), encoding="utf-8")
        return 0

    probe_a, probe_b, probe_status = load_rows(args.train, args.eval, args.seed, args.probe_target)
    loader_a = build_loader(probe_a, payload, args.batch_size)
    loader_b = build_loader(probe_b, payload, args.batch_size)
    components = build_components(model, args.rank_atoms_per_module)
    by_id = {component.component_id: component for component in components}
    pad_id = int(payload["vocab"]["<PAD>"])
    print(f"[disc] components={len(components)} probe_a={len(probe_a)} probe_b={len(probe_b)} status={probe_status}", flush=True)

    base_profile_a, base_metrics_a = evaluate_component(model, loader_a, None, 1.0, args.base_depth, pad_id)
    base_profile_b, base_metrics_b = evaluate_component(model, loader_b, None, 1.0, args.base_depth, pad_id)
    _base_profile_2t_a, base_metrics_2t_a = evaluate_component(model, loader_a, None, 1.0, args.base_depth * 2, pad_id)
    _base_profile_2t_b, base_metrics_2t_b = evaluate_component(model, loader_b, None, 1.0, args.base_depth * 2, pad_id)

    print("[disc] base profiles done; running static ablations", flush=True)
    static_rows = []
    static_metrics = {}
    for index, component in enumerate(components):
        profile, metrics = evaluate_component(model, loader_a, component, 0.0, args.base_depth, pad_id)
        d = damage(base_metrics_a["correct"], metrics["correct"])
        static_metrics[component.component_id] = metrics
        static_rows.append(static_feature_row(component, d, float(profile[-1]["loss"])))
        if (index + 1) % 10 == 0 or index + 1 == len(components):
            print(f"[disc] static {index + 1}/{len(components)}", flush=True)

    write_json(ARTIFACT_ROOT / "static_features.json", static_rows)
    sample_ids = stratified_sample(static_rows, args.per_decile, args.seed)
    write_json(
        ARTIFACT_ROOT / "loop_spline_sample.json",
        {
            "sampled_component_ids": sample_ids,
            "sampled_count": len(sample_ids),
            "deciles": assign_deciles(static_rows),
            "strategy": f"up_to_{args.per_decile}_per_static_damage_decile",
        },
    )

    print(f"[disc] static done; profiling {len(sample_ids)} sampled components", flush=True)
    profiles_a = {}
    profiles_b = {}
    targets = []
    for sample_index, component_id in enumerate(sample_ids):
        component = by_id[component_id]
        profiles_a[component_id] = {}
        profiles_b[component_id] = {}
        for alpha in [0.0, 0.5, 1.5]:
            profile_a, metrics_a = evaluate_component(model, loader_a, component, alpha, args.base_depth, pad_id)
            profile_b, metrics_b = evaluate_component(model, loader_b, component, alpha, args.base_depth, pad_id)
            profiles_a[component_id][str(alpha)] = profile_a
            profiles_b[component_id][str(alpha)] = profile_b
            if alpha == 0.0:
                damage_t_a = damage(base_metrics_a["correct"], metrics_a["correct"])
                damage_t_b = damage(base_metrics_b["correct"], metrics_b["correct"])
        _profile_2t_a, metrics_2t_a = evaluate_component(model, loader_a, component, 0.0, args.base_depth * 2, pad_id)
        _profile_2t_b, metrics_2t_b = evaluate_component(model, loader_b, component, 0.0, args.base_depth * 2, pad_id)
        damage_2t_a = damage(base_metrics_2t_a["correct"], metrics_2t_a["correct"])
        damage_2t_b = damage(base_metrics_2t_b["correct"], metrics_2t_b["correct"])
        delta_a = damage_2t_a - damage_t_a
        delta_b = damage_2t_b - damage_t_b
        targets.append(
            {
                "component_id": component_id,
                "damage_T_probe_A": damage_t_a,
                "damage_2T_probe_A": damage_2t_a,
                "delta_damage_depth_probe_A": delta_a,
                "sign_delta_damage_probe_A": 0 if delta_a == 0 else (1 if delta_a > 0 else -1),
                "damage_T_probe_B": damage_t_b,
                "damage_2T_probe_B": damage_2t_b,
                "delta_damage_depth_probe_B": delta_b,
                "sign_delta_damage_probe_B": 0 if delta_b == 0 else (1 if delta_b > 0 else -1),
            }
        )
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == len(sample_ids):
            print(f"[disc] profiled {sample_index + 1}/{len(sample_ids)}", flush=True)

    write_json(ARTIFACT_ROOT / "loop_profiles_A.json", profiles_a)
    write_json(ARTIFACT_ROOT / "loop_profiles_B.json", profiles_b)
    write_json(ARTIFACT_ROOT / "depth_transfer_targets.json", targets)

    shape_a = [{"component_id": cid, **extract_shape_features(profiles_a[cid])} for cid in sample_ids]
    shape_b = [{"component_id": cid, **extract_shape_features(profiles_b[cid])} for cid in sample_ids]
    write_json(ARTIFACT_ROOT / "loop_shape_features.json", {"probe_A": shape_a, "probe_B": shape_b})

    feature_names = [key for key in shape_a[0] if key != "component_id"] if shape_a else []
    reliability_by_feature = {}
    for name in feature_names:
        xa = np.array([row[name] for row in shape_a], dtype=float)
        xb = np.array([row[name] for row in shape_b], dtype=float)
        reliability_by_feature[name] = spearman(xa, xb)
    finite = [v for v in reliability_by_feature.values() if not math.isnan(v)]
    overall_reliability = float(np.nanmedian(finite)) if finite else float("nan")
    static_by_id = {row["component_id"]: row for row in static_rows}
    damage_values = np.array([static_by_id[cid]["static_ablation_damage_T_probe_A"] for cid in sample_ids], dtype=float)
    threshold = np.quantile(damage_values, 0.75) if len(damage_values) else 0
    top_ids = [cid for cid in sample_ids if static_by_id[cid]["static_ablation_damage_T_probe_A"] >= threshold]
    top_rels = []
    for name in feature_names:
        xa = np.array([next(row[name] for row in shape_a if row["component_id"] == cid) for cid in top_ids], dtype=float)
        xb = np.array([next(row[name] for row in shape_b if row["component_id"] == cid) for cid in top_ids], dtype=float)
        value = spearman(xa, xb)
        if not math.isnan(value):
            top_rels.append(value)
    top_reliability = float(np.nanmedian(top_rels)) if top_rels else float("nan")
    reliability = {
        "overall_median_spearman": overall_reliability,
        "top_damage_quartile_median_spearman": top_reliability,
        "feature_spearman": reliability_by_feature,
        "gate_pass": bool(overall_reliability >= 0.5 and top_reliability >= 0.7),
    }

    static_sample = [static_by_id[cid] for cid in sample_ids]
    target_by_id = {row["component_id"]: row for row in targets}
    y = np.array([target_by_id[cid]["delta_damage_depth_probe_A"] for cid in sample_ids], dtype=float)
    folds = make_folds(len(sample_ids), args.seed)
    nested: dict[str, Any] = {"skipped": False}
    top_partial = []
    final_label = probe_status if probe_status == "UNDERPOWERED_PROBES" else "PROFILE_UNSTABLE"
    if reliability["gate_pass"] and len(set(y.tolist())) > 1 and len(sample_ids) >= 12:
        xs, static_names = build_design(static_sample, None)
        xp, phi_names = build_design(static_sample, shape_a)
        pred_s = ridge_cv_predict(xs, y, folds)
        pred_p = ridge_cv_predict(xp, y, folds)
        sp_s = spearman(y, pred_s)
        sp_p = spearman(y, pred_p)
        delta_sp = sp_p - sp_s
        signs = np.array([1 if value > 0 else 0 for value in y], dtype=int)
        auc_s = auc_score(signs, pred_s)
        auc_p = auc_score(signs, pred_p)
        delta_auc = None if auc_s is None or auc_p is None else auc_p - auc_s
        p_value = permutation_pvalue(static_sample, shape_a, y, folds, delta_sp, args.permutations, args.seed)
        nested = {
            "cv_spearman_static": sp_s,
            "cv_spearman_static_plus_phi": sp_p,
            "delta_spearman": delta_sp,
            "cv_auc_static": auc_s,
            "cv_auc_static_plus_phi": auc_p,
            "delta_auc": delta_auc,
            "permutations": args.permutations,
            "permutation_p_delta_metric": p_value,
            "auc_skipped_reason": None if auc_s is not None else "insufficient positive/negative class balance",
        }
        # Partial feature proxy: residualize target and each phi feature against statics.
        static_pred = ridge_cv_predict(xs, y, folds)
        y_resid = y - static_pred
        for idx, name in enumerate(phi_names[len(static_names) :], start=len(static_names)):
            feature = xp[:, idx]
            f_resid = feature - ridge_cv_predict(xs, feature, folds)
            top_partial.append({"feature": name, "partial_spearman_proxy": spearman(y_resid, f_resid)})
        top_partial = sorted(
            [row for row in top_partial if not math.isnan(row["partial_spearman_proxy"])],
            key=lambda row: abs(row["partial_spearman_proxy"]),
            reverse=True,
        )[:20]
        success = (delta_sp >= 0.15 or (delta_auc is not None and delta_auc >= 0.07)) and p_value < 0.01
        final_label = "LOOP_SPLINE_INCREMENTAL_SIGNAL" if success else "STATIC_ENDPOINT_SUFFICIENT"
    else:
        nested = {
            "skipped": True,
            "reason": "reliability_gate_failed_or_constant_target",
            "target_unique_values": sorted(set(y.tolist())),
        }

    deciles = assign_deciles(static_rows)
    per_decile = {}
    for cid in sample_ids:
        decile = str(deciles[cid])
        per_decile.setdefault(decile, {"count": 0, "mean_static_damage": 0.0, "mean_delta_damage": 0.0})
        per_decile[decile]["count"] += 1
        per_decile[decile]["mean_static_damage"] += static_by_id[cid]["static_ablation_damage_T_probe_A"]
        per_decile[decile]["mean_delta_damage"] += target_by_id[cid]["delta_damage_depth_probe_A"]
    for row in per_decile.values():
        row["mean_static_damage"] /= max(1, row["count"])
        row["mean_delta_damage"] /= max(1, row["count"])

    manifest = {
        "final_label": final_label,
        "model_path": str(args.model),
        "component_count": len(components),
        "sampled_component_count": len(sample_ids),
        "probe_counts": {"A": len(probe_a), "B": len(probe_b), "status": probe_status},
        "T": args.base_depth,
        "T_prime": args.base_depth * 2,
        "base_metrics": {"probe_A_T": base_metrics_a, "probe_B_T": base_metrics_b},
        "reliability": reliability,
        "nested_models": nested,
        "per_decile_breakdown": per_decile,
        "top_partial_features": top_partial,
        "claims_actual_vpd": False,
        "uses_cached_4d_arc_heads": False,
    }
    write_json(ARTIFACT_ROOT / "manifest.json", manifest)
    REPORT_PATH.write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"final_label": final_label, "report": str(REPORT_PATH), "artifact_root": str(ARTIFACT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

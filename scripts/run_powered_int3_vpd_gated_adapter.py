"""VPD-gated component adapter on the powered Intellect-3-Logic TRM.

This is a fine-tuning methodology test after retiring loop-spline: apply or
train scalar coefficients on selected rank-one component directions, with the
base powered critic frozen. The primary arm can select either high-damage
components or validation-scored signed VPD components; the control arm uses
damage/role-matched random components at the same count.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

HRM_CORE = Path(r"C:\projects\HRM-re")
HRM_REPO = Path(r"D:\projects\HRM-re")
for root in (HRM_CORE, HRM_REPO):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.hermes_skill_gym.hermes_dataset import (  # noqa: E402 # pyright: ignore[reportMissingImports]
    HermesSkillDataset,
    collate_batch,
    read_jsonl,
)

load_trained_model: Any = importlib.import_module(
    "experiments.hermes_skill_gym.train_hrm_hermes"
).load_trained_model

MODEL_PATH = Path("artifacts") / "powered_int3_logic" / "model"
TRAIN_PATH = Path("artifacts") / "powered_int3_logic" / "gym_out" / "dataset" / "train.jsonl"
EVAL_PATH = Path("artifacts") / "powered_int3_logic" / "gym_out" / "dataset" / "eval.jsonl"
STATIC_FEATURES = Path("artifacts") / "powered_int3_logic" / "discriminator" / "static_features.json"
OUT_DIR = Path(r"D:\Research_Engine\runs\trm_param_decomp\powered_int3_vpd_gated_adapter")
REPORT_PATH = Path("reports") / "powered_int3_vpd_gated_adapter_report.md"


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    module_path: str
    rank_index: int
    static_damage: float
    role: str
    scale: float
    mean_ci: float = 0.0
    alive_fraction: float = 1.0
    label: str = ""
    initial_coeff: float = 0.0
    validation_score: float | None = None
    validation_loss: float | None = None
    validation_drift: float | None = None
    basis: Tensor | None = None


class ComponentDeltaLinear(nn.Module):
    def __init__(self, base: nn.Linear, specs: list[ComponentSpec]) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.register_buffer("base_weight", base.weight.detach().clone())
        bias = getattr(base, "bias", None)
        self.register_buffer(
            "base_bias",
            None if bias is None else bias.detach().clone(),
        )
        self.specs = specs
        basis = []
        weight = base.weight.detach().float().cpu()
        u, _s, vh = torch.linalg.svd(weight, full_matrices=False)
        for spec in specs:
            if spec.basis is not None:
                basis.append(spec.basis.to(dtype=base.weight.dtype))
            else:
                idx = min(spec.rank_index, u.shape[1] - 1, vh.shape[0] - 1)
                basis.append(torch.outer(u[:, idx], vh[idx]).to(dtype=base.weight.dtype))
        self.register_buffer("basis", torch.stack(basis, dim=0))
        initial = torch.tensor([spec.initial_coeff for spec in specs], dtype=base.weight.dtype)
        self.coeff = nn.Parameter(initial)

    @override
    def forward(self, x: Tensor) -> Tensor:
        delta = torch.einsum("c,coi->oi", self.coeff, self.basis)
        base_weight = cast(Tensor, self.base_weight)
        base_bias = cast(Tensor | None, self.base_bias)
        return F.linear(x, base_weight + delta, base_bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--static-features", type=Path, default=STATIC_FEATURES)
    parser.add_argument("--vpd-manifest", type=Path, default=None)
    parser.add_argument("--spd-checkpoint", type=Path, default=None)
    parser.add_argument("--selection-source", choices=["static", "vpd"], default="static")
    parser.add_argument("--selection-method", choices=["damage", "validated"], default="damage")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-rows", type=int, default=256)
    parser.add_argument("--max-eval-rows", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--drift-weight", type=float, default=0.05)
    parser.add_argument("--candidate-pool-size", type=int, default=24)
    parser.add_argument("--min-alive-fraction", type=float, default=0.05)
    parser.add_argument("--max-alive-fraction", type=float, default=0.95)
    parser.add_argument("--probe-coeffs", default="-0.10,-0.05,0.05,0.10")
    parser.add_argument("--validation-rows", type=int, default=32)
    return parser.parse_args()


def module_for_param(param_name: str) -> str:
    return param_name[:-7] if param_name.endswith(".weight") else param_name


def load_static_specs(path: Path, model: nn.Module) -> list[ComponentSpec]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    modules = dict(model.named_modules())
    specs = []
    for row in rows:
        module_path = module_for_param(str(row["module_id"]))
        module = modules.get(module_path)
        if not isinstance(module, nn.Linear):
            continue
        specs.append(
            ComponentSpec(
                component_id=str(row["component_id"]),
                module_path=module_path,
                rank_index=int(row.get("svd_rank_index") or 0),
                static_damage=float(row.get("static_ablation_damage_T_probe_A") or 0.0),
                role=str(row.get("role") or ""),
                scale=float(row.get("scale") or 0.0),
                label=str(row.get("label") or ""),
            )
        )
    return specs


def safe_module_key(module_path: str) -> str:
    return module_path.replace(".", "-")


def component_role(module_path: str) -> str:
    if module_path.startswith("networks."):
        return "recursive_trunk_z"
    if module_path == "bucket_head":
        return "output_bucket_head"
    return "unclassified"


def load_vpd_specs(manifest_path: Path, checkpoint_path: Path, model: nn.Module) -> list[ComponentSpec]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    modules = dict(model.named_modules())
    specs = []
    for row in manifest["components"]:
        module_path = str(row["module"])
        module = modules.get(module_path)
        if not isinstance(module, nn.Linear):
            continue
        rank_index = int(row["rank_index"])
        key = safe_module_key(module_path)
        u = state[f"_components.{key}.U"][rank_index].detach().cpu().float()
        v = state[f"_components.{key}.V"][:, rank_index].detach().cpu().float()
        basis = torch.outer(u, v)
        specs.append(
            ComponentSpec(
                component_id=str(row["component_id"]),
                module_path=module_path,
                rank_index=rank_index,
                static_damage=float(row.get("ablation_output_l2_delta") or 0.0),
                role=component_role(module_path),
                scale=float(torch.linalg.vector_norm(basis).item()),
                mean_ci=float(row.get("mean_ci") or 0.0),
                alive_fraction=float(row.get("alive_fraction") or 0.0),
                label=str(row.get("label") or ""),
                basis=basis,
            )
        )
    return specs


def select_damage_specs(
    specs: list[ComponentSpec], *, top_k: int, seed: int
) -> tuple[list[ComponentSpec], list[ComponentSpec]]:
    trunk = [spec for spec in specs if spec.role == "recursive_trunk_z"]
    selected = sorted(trunk, key=lambda item: (item.static_damage, item.scale), reverse=True)[:top_k]
    control = select_control_specs(trunk, selected, top_k=top_k, seed=seed)
    return selected, control


def select_control_specs(
    pool: list[ComponentSpec], selected: list[ComponentSpec], *, top_k: int, seed: int
) -> list[ComponentSpec]:
    selected_ids = {spec.component_id for spec in selected}
    remaining = [spec for spec in pool if spec.component_id not in selected_ids]
    rng = random.Random(seed)
    pool_by_damage = sorted(remaining, key=lambda item: abs(item.static_damage - mean_damage(selected)))
    near_pool = pool_by_damage[: max(top_k * 4, top_k)]
    control = rng.sample(near_pool, k=min(top_k, len(near_pool)))
    return control


def parse_probe_coeffs(raw: str) -> list[float]:
    coeffs = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not coeffs:
        raise ValueError("--probe-coeffs must contain at least one coefficient")
    return coeffs


def component_basis(spec: ComponentSpec, module: nn.Linear) -> Tensor:
    if spec.basis is not None:
        return spec.basis.to(dtype=module.weight.dtype, device=module.weight.device)
    weight = module.weight.detach().float().cpu()
    u, _s, vh = torch.linalg.svd(weight, full_matrices=False)
    idx = min(spec.rank_index, u.shape[1] - 1, vh.shape[0] - 1)
    return torch.outer(u[:, idx], vh[idx]).to(dtype=module.weight.dtype, device=module.weight.device)


def score_component_probe(
    model: nn.Module,
    spec: ComponentSpec,
    coeff: float,
    loader: DataLoader[Any],
    device: str,
    base_logits: list[Tensor],
    drift_weight: float,
) -> tuple[float, dict[str, float]]:
    module = model.get_submodule(spec.module_path)
    if not isinstance(module, nn.Linear):
        raise TypeError(f"{spec.module_path} is not Linear")
    basis = component_basis(spec, module)
    original = module.weight.data.detach().clone()
    try:
        module.weight.data = original + float(coeff) * basis
        metrics = evaluate(model, loader, device, base_logits)
    finally:
        module.weight.data = original
    score = float(metrics["loss"]) + float(drift_weight) * float(metrics["bucket_logit_l2_drift"])
    return score, metrics


def select_validated_specs(
    specs: list[ComponentSpec],
    *,
    top_k: int,
    seed: int,
    model: nn.Module,
    loader: DataLoader[Any],
    device: str,
    base_logits: list[Tensor],
    drift_weight: float,
    candidate_pool_size: int,
    min_alive_fraction: float,
    max_alive_fraction: float,
    probe_coeffs: list[float],
) -> tuple[list[ComponentSpec], list[ComponentSpec], list[dict[str, Any]]]:
    filtered = [
        spec
        for spec in specs
        if spec.role == "recursive_trunk_z"
        and spec.alive_fraction >= min_alive_fraction
        and spec.alive_fraction <= max_alive_fraction
        and spec.label != "dead_component"
    ]
    candidates = sorted(filtered, key=lambda item: (item.static_damage, item.scale), reverse=True)[:candidate_pool_size]
    scored: list[ComponentSpec] = []
    probe_rows: list[dict[str, Any]] = []
    for spec in candidates:
        best_coeff = 0.0
        best_score = float("inf")
        best_metrics: dict[str, float] = {}
        for coeff in probe_coeffs:
            score, metrics = score_component_probe(
                model,
                spec,
                coeff,
                loader,
                device,
                base_logits,
                drift_weight,
            )
            probe_rows.append(
                {
                    "component_id": spec.component_id,
                    "coeff": coeff,
                    "score": round(score, 6),
                    "loss": metrics["loss"],
                    "bucket_accuracy": metrics["bucket_accuracy"],
                    "bucket_logit_l2_drift": metrics["bucket_logit_l2_drift"],
                }
            )
            if score < best_score:
                best_coeff = coeff
                best_score = score
                best_metrics = metrics
        scored.append(
            replace(
                spec,
                initial_coeff=best_coeff,
                validation_score=round(best_score, 6),
                validation_loss=best_metrics.get("loss"),
                validation_drift=best_metrics.get("bucket_logit_l2_drift"),
            )
        )
    selected = sorted(scored, key=lambda item: (item.validation_score or float("inf"), -item.static_damage))[:top_k]
    control = select_control_specs(candidates, selected, top_k=top_k, seed=seed)
    return selected, control, probe_rows


def mean_damage(specs: list[ComponentSpec]) -> float:
    return sum(spec.static_damage for spec in specs) / max(1, len(specs))


def spec_to_dict(spec: ComponentSpec) -> dict[str, Any]:
    return {
        "component_id": spec.component_id,
        "module_path": spec.module_path,
        "rank_index": spec.rank_index,
        "static_damage": spec.static_damage,
        "role": spec.role,
        "scale": spec.scale,
        "mean_ci": spec.mean_ci,
        "alive_fraction": spec.alive_fraction,
        "label": spec.label,
        "initial_coeff": spec.initial_coeff,
        "validation_score": spec.validation_score,
        "validation_loss": spec.validation_loss,
        "validation_drift": spec.validation_drift,
        "has_explicit_basis": spec.basis is not None,
    }


def set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent.get_submodule(part)
    setattr(parent, parts[-1], module)


def install_adapters(model: nn.Module, specs: list[ComponentSpec]) -> list[nn.Parameter]:
    by_module: dict[str, list[ComponentSpec]] = {}
    for spec in specs:
        by_module.setdefault(spec.module_path, []).append(spec)
    params: list[nn.Parameter] = []
    for module_path, module_specs in by_module.items():
        base = model.get_submodule(module_path)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{module_path} is not Linear")
        wrapped = ComponentDeltaLinear(base, module_specs)
        set_submodule(model, module_path, wrapped)
        params.append(wrapped.coeff)
    return params


def adapter_modules(model: nn.Module) -> list[ComponentDeltaLinear]:
    return [module for module in model.modules() if isinstance(module, ComponentDeltaLinear)]


def zero_coefficients(modules: list[ComponentDeltaLinear]) -> list[Tensor]:
    saved = [module.coeff.detach().clone() for module in modules]
    with torch.no_grad():
        for module in modules:
            module.coeff.zero_()
    return saved


def restore_coefficients(modules: list[ComponentDeltaLinear], saved: list[Tensor]) -> None:
    with torch.no_grad():
        for module, value in zip(modules, saved, strict=True):
            module.coeff.copy_(value)


def build_loader(rows: list[dict[str, Any]], payload: dict[str, Any], batch_size: int, *, shuffle: bool) -> DataLoader[Any]:
    config = payload["config"]
    vocab = payload["vocab"]
    dataset = HermesSkillDataset(
        rows,
        vocab,
        {label: index for index, label in enumerate(config["bucket_labels"])},
        {label: index for index, label in enumerate(config["action_labels"])},
        max_tokens=int(config.get("max_tokens", 512)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, int(vocab["<PAD>"])),
    )


def batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def forward_loss(model: nn.Module, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float], Tensor, Tensor]:
    input_ids = batch["input_ids"]
    attention_mask = input_ids.ne(0)
    bucket_logits, action_logits, reward_pred, _pooled = model(input_ids, attention_mask)
    bucket_labels = batch["bucket_labels"]
    action_labels = batch["action_labels"]
    rewards = batch["rewards"]
    action_weights = batch["action_weights"]
    bucket_loss = F.cross_entropy(bucket_logits, bucket_labels)
    action_per_example = F.cross_entropy(action_logits, action_labels, reduction="none")
    action_loss = (action_per_example * action_weights).sum() / action_weights.sum().clamp(min=1e-6)
    reward_loss = F.smooth_l1_loss(reward_pred, rewards)
    loss = bucket_loss + 0.5 * action_loss + 0.05 * reward_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "bucket_loss": float(bucket_loss.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "reward_loss": float(reward_loss.detach().cpu()),
        "bucket_accuracy": float(bucket_logits.argmax(dim=-1).eq(bucket_labels).float().mean().detach().cpu()),
    }
    return loss, metrics, bucket_logits, action_logits


def evaluate(model: nn.Module, loader: DataLoader[Any], device: str, base_logits: list[Tensor] | None = None) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    drift_total = 0.0
    drift_count = 0
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            batch = batch_to_device(raw, device)
            _loss, metrics, bucket_logits, action_logits = forward_loss(model, batch)
            n = int(batch["input_ids"].shape[0])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * n
            count += n
            if base_logits is not None and batch_index < len(base_logits):
                ref = base_logits[batch_index].to(bucket_logits.device)
                drift_total += float(torch.linalg.vector_norm(bucket_logits - ref, dim=-1).mean().cpu())
                drift_count += 1
            _ = action_logits
    out = {key: round(float(value) / max(1, count), 6) for key, value in totals.items()}
    out["bucket_logit_l2_drift"] = round(drift_total / max(1, drift_count), 6) if drift_count else 0.0
    return out


def collect_base_logits(model: nn.Module, loader: DataLoader[Any], device: str) -> list[Tensor]:
    model.eval()
    logits = []
    with torch.no_grad():
        for raw in loader:
            batch = batch_to_device(raw, device)
            input_ids = batch["input_ids"]
            attention_mask = input_ids.ne(0)
            bucket_logits, _action_logits, _reward_pred, _pooled = model(input_ids, attention_mask)
            logits.append(bucket_logits.detach().cpu())
    return logits


def cycle(loader: DataLoader[Any]) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def run_arm(
    *,
    name: str,
    specs: list[ComponentSpec],
    args: argparse.Namespace,
    payload: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    base_eval_logits: list[Tensor],
) -> dict[str, Any]:
    _payload, model = load_trained_model(str(args.model), args.device)
    model.requires_grad_(False)
    params = install_adapters(model, specs)
    modules = adapter_modules(model)
    for param in params:
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=float(args.lr))
    train_loader = build_loader(train_rows, payload, args.batch_size, shuffle=True)
    eval_loader = build_loader(eval_rows, payload, args.batch_size, shuffle=False)
    iterator = cycle(train_loader)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch = batch_to_device(next(iterator), args.device)
        optimizer.zero_grad()
        saved = zero_coefficients(modules)
        with torch.no_grad():
            _base_loss, _base_metrics, base_bucket_logits, _base_action_logits = forward_loss(model, batch)
        restore_coefficients(modules, saved)
        loss, metrics, bucket_logits, _action_logits = forward_loss(model, batch)
        coeff_penalty = torch.stack([(param * param).mean() for param in params]).sum()
        drift_penalty = F.mse_loss(bucket_logits, base_bucket_logits)
        train_loss = loss + float(args.drift_weight) * drift_penalty + 1e-3 * coeff_penalty
        train_loss.backward()
        optimizer.step()
        if step == 1 or step == args.steps:
            history.append(
                {
                    "step": step,
                    **metrics,
                    "coeff_l2": float(torch.sqrt(coeff_penalty).detach().cpu()),
                    "train_bucket_drift_mse": float(drift_penalty.detach().cpu()),
                }
            )
        _ = bucket_logits
    metrics = evaluate(model, eval_loader, args.device, base_eval_logits)
    coeffs = {
        spec.component_id: float(coeff.detach().cpu()[idx])
        for module in model.modules()
        if isinstance(module, ComponentDeltaLinear)
        for idx, spec in enumerate(module.specs)
        for coeff in [module.coeff]
    }
    return {
        "name": name,
        "component_count": len(specs),
        "components": [spec_to_dict(spec) for spec in specs],
        "history": history,
        "eval": metrics,
        "coefficients": coeffs,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Powered Intellect-3 VPD-Gated Adapter Report",
        "",
        f"Model: `{payload['model']}`",
        f"Train rows: `{payload['train_rows']}`; validation rows: `{payload['validation_rows']}`; "
        f"eval rows: `{payload['eval_rows']}`; steps: `{payload['steps']}`",
        f"Selection source: `{payload['selection_source']}`",
        f"Selection method: `{payload['selection_method']}`",
        "",
        "The selected arm applies and optionally trains scalar coefficients on chosen recursive-trunk component directions. "
        "The control arm uses the same number of random recursive-trunk component directions from the same source.",
        "",
        "## Results",
        "",
        "| arm | components | eval loss | bucket acc | bucket drift |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        ev = arm["eval"]
        lines.append(
            f"| {arm['name']} | {arm['component_count']} | {ev['loss']} | "
            f"{ev['bucket_accuracy']} | {ev['bucket_logit_l2_drift']} |"
        )
    if payload["selection_method"] == "validated":
        lines += [
            "",
            "## Validation Selection",
            "",
            "| component | init coeff | validation score | validation loss | validation drift | alive | damage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        selected_arm = next((arm for arm in payload["arms"] if arm["name"].endswith("_validated")), None)
        if selected_arm is not None:
            for comp in selected_arm["components"]:
                lines.append(
                    f"| `{comp['component_id']}` | {comp['initial_coeff']} | {comp['validation_score']} | "
                    f"{comp['validation_loss']} | {comp['validation_drift']} | "
                    f"{comp['alive_fraction']} | {comp['static_damage']} |"
                )
    lines += ["", "## Selected Components", ""]
    for arm in payload["arms"]:
        lines.append(f"### {arm['name']}")
        lines.append("")
        for comp in arm["components"][:12]:
            lines.append(
                f"- `{comp['component_id']}` damage={comp['static_damage']} "
                f"scale={round(float(comp['scale']), 6)} alive={comp['alive_fraction']} "
                f"init={comp['initial_coeff']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    payload, base_model = load_trained_model(str(args.model), args.device)
    if args.selection_source == "vpd":
        if args.vpd_manifest is None or args.spd_checkpoint is None:
            raise ValueError("--selection-source vpd requires --vpd-manifest and --spd-checkpoint")
        specs = load_vpd_specs(args.vpd_manifest, args.spd_checkpoint, base_model)
    else:
        specs = load_static_specs(args.static_features, base_model)
    all_train_rows = read_jsonl(str(args.train))[: args.max_train_rows]
    validation_count = min(args.validation_rows, max(0, len(all_train_rows) - 1))
    if args.selection_method == "validated" and validation_count > 0:
        train_rows = all_train_rows[:-validation_count]
        validation_rows = all_train_rows[-validation_count:]
    else:
        train_rows = all_train_rows
        validation_rows = []
    eval_rows = read_jsonl(str(args.eval))[: args.max_eval_rows]
    eval_loader = build_loader(eval_rows, payload, args.batch_size, shuffle=False)
    base_eval = evaluate(base_model, eval_loader, args.device)
    base_eval_logits = collect_base_logits(base_model, eval_loader, args.device)
    validation_probe_rows: list[dict[str, Any]] = []
    if args.selection_method == "validated":
        if not validation_rows:
            raise ValueError("--selection-method validated requires at least two train rows")
        validation_loader = build_loader(validation_rows, payload, args.batch_size, shuffle=False)
        validation_base_logits = collect_base_logits(base_model, validation_loader, args.device)
        selected, control, validation_probe_rows = select_validated_specs(
            specs,
            top_k=args.top_k,
            seed=args.seed,
            model=base_model,
            loader=validation_loader,
            device=args.device,
            base_logits=validation_base_logits,
            drift_weight=float(args.drift_weight),
            candidate_pool_size=args.candidate_pool_size,
            min_alive_fraction=args.min_alive_fraction,
            max_alive_fraction=args.max_alive_fraction,
            probe_coeffs=parse_probe_coeffs(args.probe_coeffs),
        )
    else:
        selected, control = select_damage_specs(specs, top_k=args.top_k, seed=args.seed)

    arms = [
        {
            "name": "base_frozen",
            "component_count": 0,
            "components": [],
            "history": [],
            "eval": base_eval,
            "coefficients": {},
        },
        run_arm(
            name=(
                f"{args.selection_source}_validated"
                if args.selection_method == "validated"
                else ("vpd_selected" if args.selection_source == "vpd" else "static_selected")
            ),
            specs=selected,
            args=args,
            payload=payload,
            train_rows=train_rows,
            eval_rows=eval_rows,
            base_eval_logits=base_eval_logits,
        ),
        run_arm(
            name="random_trunk_control",
            specs=control,
            args=args,
            payload=payload,
            train_rows=train_rows,
            eval_rows=eval_rows,
            base_eval_logits=base_eval_logits,
        ),
    ]
    result = {
        "model": str(args.model),
        "static_features": str(args.static_features),
        "selection_source": args.selection_source,
        "vpd_manifest": str(args.vpd_manifest) if args.vpd_manifest else None,
        "spd_checkpoint": str(args.spd_checkpoint) if args.spd_checkpoint else None,
        "selection_method": args.selection_method,
        "candidate_pool_size": args.candidate_pool_size,
        "min_alive_fraction": args.min_alive_fraction,
        "max_alive_fraction": args.max_alive_fraction,
        "probe_coeffs": parse_probe_coeffs(args.probe_coeffs),
        "validation_probe_rows": validation_probe_rows,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "eval_rows": len(eval_rows),
        "steps": args.steps,
        "top_k": args.top_k,
        "arms": arms,
    }
    (args.out_dir / "vpd_gated_adapter_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

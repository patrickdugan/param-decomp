"""Loop-spline discriminator over real Hermes SPD/VPD components.

This reruns the loop-spline experimental technique on the upgraded Hermes
component harness: components come from this repo's SPD checkpoint, not from SVD
rank atoms. The question is whether recurrence-step intervention profiles add
predictive information beyond static SPD component features for held-out
depth-transfer damage.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

HRM_CORE = Path(r"C:\projects\HRM-re")
if str(HRM_CORE) not in sys.path:
    sys.path.insert(0, str(HRM_CORE))

import scripts.run_trm_loop_spline_first_discriminator as disc  # noqa: E402
from scripts.trm_hermes_decomp import MODEL_PATH, OUT_DIR, TARGET_MODULES, TRAIN_PATH  # noqa: E402

ARTIFACT_ROOT = Path(r"D:\Research_Engine\runs\trm_param_decomp\hermes_logic_critic_100\vpd_loop_spline")
REPORT_PATH = Path("reports") / "hermes_vpd_loop_spline_report.md"


@dataclass
class VpdComponent:
    component_id: str
    module_id: str
    role: str
    module_depth: int
    component_index: int
    scale: float
    u_norm: float
    v_norm: float
    uv_norm_product: float
    reconstruction_contribution: float
    u: torch.Tensor
    v: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-model", type=Path, default=MODEL_PATH)
    parser.add_argument("--component-model", type=Path, default=OUT_DIR / "model_100.pth")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=TRAIN_PATH)
    parser.add_argument("--base-depth", type=int, default=3)
    parser.add_argument("--probe-target", type=int, default=32)
    parser.add_argument("--per-decile", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--reuse-static", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def safe_module_key(module_id: str) -> str:
    return module_id.replace(".", "-")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_vpd_components(path: Path, model: torch.nn.Module) -> list[VpdComponent]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    components: list[VpdComponent] = []
    for module_id, count in TARGET_MODULES:
        key = safe_module_key(module_id)
        u = state[f"_components.{key}.U"].float()
        v = state[f"_components.{key}.V"].float()
        linear = cast(torch.nn.Linear, model.get_submodule(module_id))
        weight = linear.weight.detach().cpu().float()
        weight_norm = float(torch.linalg.vector_norm(weight).item()) or 1.0
        role = disc.classify_role(module_id)
        for index in range(count):
            u_vec = u[index].clone()
            v_vec = v[:, index].clone()
            contribution = torch.outer(u_vec, v_vec)
            scale = float(torch.linalg.vector_norm(contribution).item())
            components.append(
                VpdComponent(
                    component_id=f"{module_id}:c{index:03d}",
                    module_id=module_id,
                    role=role,
                    module_depth=disc.module_depth(module_id),
                    component_index=index,
                    scale=scale,
                    u_norm=float(torch.linalg.vector_norm(u_vec).item()),
                    v_norm=float(torch.linalg.vector_norm(v_vec).item()),
                    uv_norm_product=float(
                        torch.linalg.vector_norm(u_vec).item() * torch.linalg.vector_norm(v_vec).item()
                    ),
                    reconstruction_contribution=float(torch.linalg.vector_norm(contribution).item() / weight_norm),
                    u=u_vec,
                    v=v_vec,
                )
            )
    return components


@contextlib.contextmanager
def intervened_vpd(model: torch.nn.Module, component: VpdComponent, alpha: float):
    param = dict(model.named_parameters())[f"{component.module_id}.weight"]
    delta = (alpha - 1.0) * torch.outer(
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


def evaluate_vpd_component(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    component: VpdComponent | None,
    alpha: float,
    depth: int,
    pad_id: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if component is None:
        return disc.trace_model(model, loader, depth, pad_id)
    with intervened_vpd(model, component, alpha):
        return disc.trace_model(model, loader, depth, pad_id)


def static_feature_row(component: VpdComponent, static_damage: int, endpoint_loss: float) -> dict[str, Any]:
    return {
        "component_id": component.component_id,
        "module_id": component.module_id,
        "role": component.role,
        "module_depth": component.module_depth,
        "svd_rank_index": component.component_index,
        "component_index": component.component_index,
        "scale": component.scale,
        "u_norm": component.u_norm,
        "v_norm": component.v_norm,
        "uv_norm_product": component.uv_norm_product,
        "reconstruction_contribution": component.reconstruction_contribution,
        "static_ablation_damage_T_probe_A": static_damage,
        "endpoint_final_step_loss_alpha0_probe_A": endpoint_loss,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Hermes Real-VPD Loop-Spline Discriminator",
        "",
        f"Final label: `{payload['final_label']}`",
        "",
        f"Critic model: `{payload['model_path']}`",
        f"SPD checkpoint: `{payload['component_model_path']}`",
        f"Component count: `{payload['component_count']}`; sampled: `{payload['sampled_component_count']}`",
        f"Probe A/B counts: `{payload['probe_counts']['A']}` / `{payload['probe_counts']['B']}`",
        f"Base depth T: `{payload['T']}`; T_prime: `{payload['T_prime']}`",
        "",
        "Components are real SPD/VPD rank components from this repo's `ComponentModel`, not SVD atoms.",
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
        "## Top Static VPD Components",
        "",
        "```json",
        json.dumps(payload["top_static_components"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    disc.DEVICE = args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    payload, model = disc.load_trained_model(str(args.critic_model), args.device)  # pyright: ignore[reportPrivateLocalImportUsage]
    if not isinstance(model, disc.HermesHRMClassifier) or isinstance(  # pyright: ignore[reportPrivateLocalImportUsage]
        model, disc.ConstellationGovernorHermesClassifier  # pyright: ignore[reportPrivateLocalImportUsage]
    ):
        raise RuntimeError(f"Unsupported critic class for recurrence trace: {type(model).__name__}")
    model.eval()
    model.requires_grad_(False)

    probe_a, probe_b, probe_status = disc.load_rows(args.train, args.eval, args.seed, args.probe_target)
    loader_a = disc.build_loader(probe_a, payload, args.batch_size)
    loader_b = disc.build_loader(probe_b, payload, args.batch_size)
    components = load_vpd_components(args.component_model, model)
    by_id = {component.component_id: component for component in components}
    pad_id = int(payload["vocab"]["<PAD>"])
    print(
        f"[vpd-disc] components={len(components)} probe_a={len(probe_a)} "
        f"probe_b={len(probe_b)} status={probe_status}",
        flush=True,
    )

    _base_profile_a, base_metrics_a = evaluate_vpd_component(model, loader_a, None, 1.0, args.base_depth, pad_id)
    _base_profile_b, base_metrics_b = evaluate_vpd_component(model, loader_b, None, 1.0, args.base_depth, pad_id)
    _base_profile_2t_a, base_metrics_2t_a = evaluate_vpd_component(
        model, loader_a, None, 1.0, args.base_depth * 2, pad_id
    )
    _base_profile_2t_b, base_metrics_2t_b = evaluate_vpd_component(
        model, loader_b, None, 1.0, args.base_depth * 2, pad_id
    )

    static_path = args.artifact_root / "static_features.json"
    if args.reuse_static and static_path.exists():
        with static_path.open(encoding="utf-8") as f:
            static_rows = json.load(f)
        print(f"[vpd-disc] reused static rows={len(static_rows)}", flush=True)
    else:
        static_rows = []
        for index, component in enumerate(components):
            profile, metrics = evaluate_vpd_component(model, loader_a, component, 0.0, args.base_depth, pad_id)
            d = disc.damage(base_metrics_a["correct"], metrics["correct"])
            static_rows.append(static_feature_row(component, d, float(profile[-1]["loss"])))
            if (index + 1) % 10 == 0 or index + 1 == len(components):
                print(f"[vpd-disc] static {index + 1}/{len(components)}", flush=True)
        write_json(static_path, static_rows)

    sample_ids = disc.stratified_sample(static_rows, args.per_decile, args.seed)
    if args.sample_limit is not None and args.sample_limit < len(sample_ids):
        indices = np.linspace(0, len(sample_ids) - 1, args.sample_limit, dtype=int)
        sample_ids = [sample_ids[int(index)] for index in indices]
    write_json(
        args.artifact_root / "loop_spline_sample.json",
        {
            "sampled_component_ids": sample_ids,
            "sampled_count": len(sample_ids),
            "deciles": disc.assign_deciles(static_rows),
            "strategy": f"up_to_{args.per_decile}_per_static_damage_decile",
        },
    )

    profiles_a: dict[str, Any] = {}
    profiles_b: dict[str, Any] = {}
    targets = []
    for sample_index, component_id in enumerate(sample_ids):
        component = by_id[component_id]
        profiles_a[component_id] = {}
        profiles_b[component_id] = {}
        damage_t_a = 0
        damage_t_b = 0
        for alpha in [0.0, 0.5, 1.5]:
            profile_a, metrics_a = evaluate_vpd_component(model, loader_a, component, alpha, args.base_depth, pad_id)
            profile_b, metrics_b = evaluate_vpd_component(model, loader_b, component, alpha, args.base_depth, pad_id)
            profiles_a[component_id][str(alpha)] = profile_a
            profiles_b[component_id][str(alpha)] = profile_b
            if alpha == 0.0:
                damage_t_a = disc.damage(base_metrics_a["correct"], metrics_a["correct"])
                damage_t_b = disc.damage(base_metrics_b["correct"], metrics_b["correct"])
        _profile_2t_a, metrics_2t_a = evaluate_vpd_component(
            model, loader_a, component, 0.0, args.base_depth * 2, pad_id
        )
        _profile_2t_b, metrics_2t_b = evaluate_vpd_component(
            model, loader_b, component, 0.0, args.base_depth * 2, pad_id
        )
        damage_2t_a = disc.damage(base_metrics_2t_a["correct"], metrics_2t_a["correct"])
        damage_2t_b = disc.damage(base_metrics_2t_b["correct"], metrics_2t_b["correct"])
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
            print(f"[vpd-disc] profiled {sample_index + 1}/{len(sample_ids)}", flush=True)

    write_json(args.artifact_root / "loop_profiles_A.json", profiles_a)
    write_json(args.artifact_root / "loop_profiles_B.json", profiles_b)
    write_json(args.artifact_root / "depth_transfer_targets.json", targets)

    shape_a = [{"component_id": cid, **disc.extract_shape_features(profiles_a[cid])} for cid in sample_ids]
    shape_b = [{"component_id": cid, **disc.extract_shape_features(profiles_b[cid])} for cid in sample_ids]
    write_json(args.artifact_root / "loop_shape_features.json", {"probe_A": shape_a, "probe_B": shape_b})

    feature_names = [key for key in shape_a[0] if key != "component_id"] if shape_a else []
    reliability_by_feature = {}
    for name in feature_names:
        xa = np.array([row[name] for row in shape_a], dtype=float)
        xb = np.array([row[name] for row in shape_b], dtype=float)
        reliability_by_feature[name] = disc.spearman(xa, xb)
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
        value = disc.spearman(xa, xb)
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
    folds = disc.make_folds(len(sample_ids), args.seed)
    nested: dict[str, Any]
    top_partial: list[dict[str, Any]] = []
    final_label = probe_status if probe_status == "UNDERPOWERED_PROBES" else "PROFILE_UNSTABLE"
    if reliability["gate_pass"] and len(set(y.tolist())) > 1 and len(sample_ids) >= 12:
        xs, static_names = disc.build_design(static_sample, None)
        xp, phi_names = disc.build_design(static_sample, shape_a)
        pred_s = disc.ridge_cv_predict(xs, y, folds)
        pred_p = disc.ridge_cv_predict(xp, y, folds)
        sp_s = disc.spearman(y, pred_s)
        sp_p = disc.spearman(y, pred_p)
        delta_sp = sp_p - sp_s
        signs = np.array([1 if value > 0 else 0 for value in y], dtype=int)
        auc_s = disc.auc_score(signs, pred_s)
        auc_p = disc.auc_score(signs, pred_p)
        delta_auc = None if auc_s is None or auc_p is None else auc_p - auc_s
        p_value = disc.permutation_pvalue(static_sample, shape_a, y, folds, delta_sp, args.permutations, args.seed)
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
        static_pred = disc.ridge_cv_predict(xs, y, folds)
        y_resid = y - static_pred
        for idx, name in enumerate(phi_names[len(static_names) :], start=len(static_names)):
            feature = xp[:, idx]
            f_resid = feature - disc.ridge_cv_predict(xs, feature, folds)
            top_partial.append({"feature": name, "partial_spearman_proxy": disc.spearman(y_resid, f_resid)})
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

    deciles = disc.assign_deciles(static_rows)
    per_decile: dict[str, Any] = {}
    for cid in sample_ids:
        decile = str(deciles[cid])
        per_decile.setdefault(decile, {"count": 0, "mean_static_damage": 0.0, "mean_delta_damage": 0.0})
        per_decile[decile]["count"] += 1
        per_decile[decile]["mean_static_damage"] += static_by_id[cid]["static_ablation_damage_T_probe_A"]
        per_decile[decile]["mean_delta_damage"] += target_by_id[cid]["delta_damage_depth_probe_A"]
    for row in per_decile.values():
        row["mean_static_damage"] /= max(1, row["count"])
        row["mean_delta_damage"] /= max(1, row["count"])

    top_static = sorted(static_rows, key=lambda row: row["static_ablation_damage_T_probe_A"], reverse=True)[:12]
    manifest = {
        "final_label": final_label,
        "model_path": str(args.critic_model),
        "component_model_path": str(args.component_model),
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
        "top_static_components": top_static,
        "claims_actual_vpd": True,
        "uses_svd_atoms": False,
        "uses_cached_4d_arc_heads": False,
    }
    write_json(args.artifact_root / "manifest.json", manifest)
    args.report.write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"final_label": final_label, "report": str(args.report), "artifact_root": str(args.artifact_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 3-5: extract real VPD artifacts from the trained TRM decomposition.

Loads the trained ComponentModel, then reports parameter-reconstruction
faithfulness, per-datapoint causal importance, per-component ablation damage on
real ARC inputs, and writes a manifest + cautiously labeled report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from param_decomp.models.batch_and_loss_fns import run_batch_first_element
from param_decomp.models.component_model import ComponentModel
from param_decomp.utils.module_utils import expand_module_patterns
from scripts.trm_real_decomp_run import ConveyorTRMConcat, build_config
from scripts.trm_real_decomp_stage0 import arc_prompts, load_trm

CI_ALIVE = 0.1


def _linear(model: nn.Module, path: str) -> nn.Linear:
    submodule = model.get_submodule(path)
    assert isinstance(submodule, nn.Linear)
    return submodule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"D:\Research_Engine\tesseract_persistent\data\models\conveyor\trm_arc_challenge.pt"),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(r"D:\Research_Engine\prime_envs\arc\ARC-Challenge\train-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(r"D:\Research_Engine\runs\trm_param_decomp\trm_arc_challenge_full\model_8000.pth"),
    )
    parser.add_argument("--n-prompts", type=int, default=1024)
    parser.add_argument("--c0", type=int, default=64)
    parser.add_argument("--c2", type=int, default=64)
    parser.add_argument(
        "--out-dir", type=Path, default=Path(r"D:\Research_Engine\runs\trm_param_decomp\trm_arc_challenge_full")
    )
    parser.add_argument(
        "--report-path", type=Path, default=Path("reports") / "trm_labeled_decomp_report.md"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = "cpu"
    base, vectorizer, _input_dim = load_trm(args.checkpoint)
    model: ConveyorTRMConcat = ConveyorTRMConcat(base)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    config = build_config(args.c0, args.c2, steps=1, batch_size=args.n_prompts)
    module_path_info = expand_module_patterns(model, config.all_module_info)
    cm = ComponentModel(
        target_model=model,
        run_batch=run_batch_first_element,
        module_path_info=module_path_info,
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
    ).to(device)
    state = torch.load(args.model, map_location=device, weights_only=False)
    cm.load_state_dict(state)
    cm.eval()

    prompts = arc_prompts(args.parquet, args.n_prompts)
    x = torch.tensor(np.asarray(vectorizer.transform(prompts).todense(), dtype=np.float32), device=device)
    batch = (x,)

    original_out = model(x).detach()
    with torch.no_grad():
        out = cm(batch, cache_type="input")
        ci = cm.calc_causal_importances(pre_weight_acts=out.cache, sampling=config.sampling)
    deltas = cm.calc_weight_deltas()

    # Install the faithful surrogate weights W = components + delta on every target.
    original_weights = {p: _linear(model, p).weight.data.clone() for p in cm.target_module_paths}
    w_full: dict[str, Tensor] = {}
    for path in cm.target_module_paths:
        w_full[path] = (cm.components[path].weight.detach() + deltas[path].detach()).to(device)
        _linear(model, path).weight.data = w_full[path]
    baseline_out = model(x).detach()
    behavioral_rel_error = float(
        torch.linalg.norm(baseline_out - original_out) / torch.linalg.norm(original_out)
    )

    layers: dict[str, dict[str, object]] = {}
    component_rows: list[dict[str, object]] = []
    for path in cm.target_module_paths:
        component = cm.components[path]
        w_components = component.weight.detach()
        w_orig = original_weights[path]
        component_capture = float(torch.linalg.norm(w_components) / torch.linalg.norm(w_orig))
        delta_fraction = float(torch.linalg.norm(deltas[path]) / torch.linalg.norm(w_orig))

        ci_layer = ci.lower_leaky[path].detach()  # [N, C]
        ci_mean = ci_layer.mean(dim=0)
        alive_frac = (ci_layer > CI_ALIVE).float().mean(dim=0)
        per_datapoint_l0 = (ci_layer > CI_ALIVE).float().sum(dim=1)

        linear = _linear(model, path)
        for c in range(component.C):
            contribution = torch.outer(component.U[c], component.V[:, c])
            linear.weight.data = w_full[path] - contribution
            with torch.no_grad():
                ablated_out = model(x).detach()
            linear.weight.data = w_full[path]
            damage = float(torch.linalg.norm(ablated_out - baseline_out, dim=-1).mean())
            component_rows.append(
                {
                    "component_id": f"{path}:c{c:03d}",
                    "module": path,
                    "rank_index": c,
                    "mean_ci": round(float(ci_mean[c]), 6),
                    "alive_fraction": round(float(alive_frac[c]), 6),
                    "ablation_output_l2_delta": round(damage, 6),
                    "label": _label(float(alive_frac[c]), damage),
                }
            )

        layers[path] = {
            "weight_shape": list(w_orig.shape),
            "component_count": int(component.C),
            "component_norm_capture_fraction": round(component_capture, 6),
            "delta_norm_fraction": round(delta_fraction, 6),
            "mean_l0_alive_per_datapoint": round(float(per_datapoint_l0.mean()), 6),
            "alive_components": int((alive_frac > 0.0).sum()),
        }

    for path in cm.target_module_paths:
        _linear(model, path).weight.data = original_weights[path]

    manifest: dict[str, object] = {
        "dictionary_type": "GOODFIRE_PARAM_DECOMP (this repo SPD)",
        "model": "ConveyorTRM trm_arc_challenge",
        "checkpoint": str(args.checkpoint),
        "trained_decomposition": str(args.model),
        "n_datapoints": len(prompts),
        "behavioral_faithfulness_rel_error": round(behavioral_rel_error, 8),
        "layers": layers,
        "components": component_rows,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(args.report_path, manifest)
    print(json.dumps({"layers": layers, "n_components": len(component_rows)}, indent=2))
    return 0


def _label(alive_fraction: float, damage: float) -> str:
    if alive_fraction < 0.01:
        return "dead_component"
    if damage >= 0.1:
        return "high_causal_importance_component"
    if alive_fraction >= 0.5:
        return "broad_low_damage_component"
    return "sparse_specific_component"


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    layers = manifest["layers"]
    assert isinstance(layers, dict)
    components = manifest["components"]
    assert isinstance(components, list)
    lines = [
        "# TRM Labeled Decomposition Report (Stages 3-5)",
        "",
        "Real Goodfire/param-decomp artifacts were used: this repo's SPD `optimize()` "
        "decomposed a genuine frozen TRM, not the 4D cached choice head.",
        "",
        f"Behavioral faithfulness (components+delta surrogate vs original TRM output, "
        f"relative L2): {manifest['behavioral_faithfulness_rel_error']}.",
        "",
        "## Per-layer decomposition",
        "",
        "component_norm_capture = how much of the weight Frobenius norm the sparse "
        "components carry; the rest is the shared delta component (input-independent).",
        "",
        "| module | weight shape | C | component capture | delta fraction | mean alive L0 / datapoint | alive components |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for module, info in layers.items():
        assert isinstance(info, dict)
        lines.append(
            f"| {module} | {info['weight_shape']} | {info['component_count']} | "
            f"{info['component_norm_capture_fraction']} | {info['delta_norm_fraction']} | "
            f"{info['mean_l0_alive_per_datapoint']} | {info['alive_components']} |"
        )
    top = sorted(components, key=lambda r: -float(r["ablation_output_l2_delta"]))[:12]
    lines += [
        "",
        "## Top causal components by ablation damage",
        "",
        "| component | module | mean CI | alive frac | ablation L2 Δ | label |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in top:
        lines.append(
            f"| {row['component_id']} | {row['module']} | {row['mean_ci']} | "
            f"{row['alive_fraction']} | {row['ablation_output_l2_delta']} | {row['label']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

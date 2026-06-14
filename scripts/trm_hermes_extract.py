"""Extract real VPD artifacts from Fable's recursive HRM logic critic SPD run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import scripts.trm_hermes_decomp as hermes_decomp
from param_decomp.configs import Config
from param_decomp.models.component_model import ComponentModel
from param_decomp.utils.module_utils import expand_module_patterns
from scripts.trm_real_decomp_run import build_config

CI_ALIVE = 0.1

MODEL_PATH = hermes_decomp.MODEL_PATH
OUT_DIR = hermes_decomp.OUT_DIR
TARGET_MODULES = hermes_decomp.TARGET_MODULES
TRAIN_PATH = hermes_decomp.TRAIN_PATH
build_loader = hermes_decomp.build_loader
make_run_batch = hermes_decomp.make_run_batch
load_trained_model: Any = hermes_decomp.load_trained_model  # pyright: ignore[reportPrivateLocalImportUsage]
read_jsonl: Any = hermes_decomp.read_jsonl  # pyright: ignore[reportPrivateLocalImportUsage]


def _linear(model: nn.Module, path: str) -> nn.Linear:
    submodule = model.get_submodule(path)
    assert isinstance(submodule, nn.Linear)
    return submodule


def _batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-model", type=Path, default=MODEL_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--model", type=Path, default=OUT_DIR / "model_500.pth")
    parser.add_argument("--n-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports") / "hermes_logic_critic_vpd_report.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload, model = load_trained_model(str(args.critic_model), device=args.device)
    model.requires_grad_(False)
    model.eval()

    pad_id = int(payload["vocab"]["<PAD>"])
    run_batch = make_run_batch(pad_id)
    config = build_config(
        TARGET_MODULES,
        steps=1,
        batch_size=args.batch_size,
        use_delta_component=False,
    )
    module_path_info = expand_module_patterns(model, config.all_module_info)
    cm = ComponentModel(
        target_model=model,
        run_batch=run_batch,
        module_path_info=module_path_info,
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
    ).to(args.device)
    state = torch.load(args.model, map_location=args.device, weights_only=False)
    cm.load_state_dict(state)
    cm.eval()

    rows = read_jsonl(args.train)[: args.n_rows]
    loader = build_loader(rows, payload, args.batch_size)
    batch = _batch_to_device(next(iter(loader)), args.device)
    run_extraction(
        model,
        cm,
        batch,
        run_batch,
        config,
        model_name="HermesHRMClassifier recursive logic critic",
        critic_model_path=args.critic_model,
        component_model_path=args.model,
        out_dir=args.out_dir,
        report_path=args.report_path,
    )
    return 0


def run_extraction(
    model: nn.Module,
    cm: ComponentModel,
    batch: dict[str, Any],
    run_batch: Any,
    config: Config,
    *,
    model_name: str,
    critic_model_path: Path,
    component_model_path: Path,
    out_dir: Path,
    report_path: Path,
) -> dict[str, object]:
    """Faithfulness, CI, and per-component ablation on a Hermes critic batch."""
    with torch.no_grad():
        original_out = run_batch(model, batch).detach()
        out = cm(batch, cache_type="input")
        ci = cm.calc_causal_importances(pre_weight_acts=out.cache, sampling=config.sampling)
    deltas = cm.calc_weight_deltas()

    original_weights = {path: _linear(model, path).weight.data.clone() for path in cm.target_module_paths}
    w_full: dict[str, Tensor] = {}
    for path in cm.target_module_paths:
        w_full[path] = cm.components[path].weight.detach() + deltas[path].detach()
        _linear(model, path).weight.data = w_full[path]

    with torch.no_grad():
        baseline_out = run_batch(model, batch).detach()
    behavioral_rel_error = float(
        torch.linalg.norm(baseline_out - original_out) / torch.linalg.norm(original_out)
    )

    layers: dict[str, dict[str, object]] = {}
    component_rows: list[dict[str, object]] = []
    for path in cm.target_module_paths:
        component = cm.components[path]
        w_orig = original_weights[path]
        component_capture = float(torch.linalg.norm(component.weight.detach()) / torch.linalg.norm(w_orig))
        delta_fraction = float(torch.linalg.norm(deltas[path]) / torch.linalg.norm(w_orig))

        ci_layer = ci.lower_leaky[path].detach()
        ci_flat = ci_layer.reshape(-1, component.C)
        ci_mean = ci_flat.mean(dim=0)
        alive_flat = (ci_flat > CI_ALIVE).float()
        alive_frac = alive_flat.mean(dim=0)
        per_site_l0 = alive_flat.sum(dim=1)

        linear = _linear(model, path)
        for c in range(component.C):
            contribution = torch.outer(component.U[c], component.V[:, c])
            linear.weight.data = w_full[path] - contribution
            with torch.no_grad():
                ablated_out = run_batch(model, batch).detach()
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
            "mean_l0_alive_per_activation_site": round(float(per_site_l0.mean()), 6),
            "alive_components": int((alive_frac > 0.0).sum()),
        }

    for path in cm.target_module_paths:
        _linear(model, path).weight.data = original_weights[path]

    manifest: dict[str, object] = {
        "dictionary_type": "GOODFIRE_PARAM_DECOMP (this repo SPD)",
        "model": model_name,
        "critic_model": str(critic_model_path),
        "trained_decomposition": str(component_model_path),
        "n_datapoints": int(batch["input_ids"].shape[0]),
        "behavioral_faithfulness_rel_error": round(behavioral_rel_error, 8),
        "layers": layers,
        "components": component_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hermes_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(report_path, manifest)
    print(json.dumps({"layers": layers, "n_components": len(component_rows)}, indent=2))
    return manifest


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
        "# Hermes Logic Critic VPD Report",
        "",
        "This report uses this repo's SPD `optimize()` on Fable's recursive HRM logic critic. "
        "It is not the earlier SVD loop-spline atom lane.",
        "",
        f"Behavioral faithfulness relative L2: {manifest['behavioral_faithfulness_rel_error']}.",
        "",
        "## Per-layer decomposition",
        "",
        "| module | weight shape | C | component capture | delta fraction | mean alive L0 / activation site | alive components |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for module, row in layers.items():
        assert isinstance(row, dict)
        lines.append(
            f"| `{module}` | {row['weight_shape']} | {row['component_count']} | "
            f"{row['component_norm_capture_fraction']} | {row['delta_norm_fraction']} | "
            f"{row['mean_l0_alive_per_activation_site']} | {row['alive_components']} |"
        )

    lines += [
        "",
        "## Top components by ablation damage",
        "",
        "| component | mean CI | alive fraction | output L2 damage | label |",
        "|---|---:|---:|---:|---|",
    ]
    for row in sorted(
        components,
        key=lambda item: float(item["ablation_output_l2_delta"]),
        reverse=True,
    )[:25]:
        lines.append(
            f"| `{row['component_id']}` | {row['mean_ci']} | {row['alive_fraction']} | "
            f"{row['ablation_output_l2_delta']} | {row['label']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

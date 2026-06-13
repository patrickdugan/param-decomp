"""Stage 2: run this repo's SPD optimize() on a real frozen TRM.

Decomposes the conveyor TRM's shared Linear layers (`shared.0`, `shared.2`) into
parameter components with stochastic-mask faithfulness, reconstruction, and
per-datapoint causal importance — real VPD artifacts on a genuine model, over
TF-IDF features of real ARC prompts.
"""

from __future__ import annotations

import argparse
import typing
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.configs import Config
from param_decomp.models.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp.run_param_decomp import optimize
from scripts.trm_real_decomp_stage0 import arc_prompts, load_trm


class ConveyorTRMConcat(nn.Module):
    """Frozen TRM that returns its heads concatenated as a single tensor (B, 5)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.shared: nn.Module = base.get_submodule("shared")
        self.category_head: nn.Module = base.get_submodule("category_head")
        self.risk_head: nn.Module = base.get_submodule("risk_head")
        self.success_head: nn.Module = base.get_submodule("success_head")

    @typing.override
    def forward(self, x: Tensor) -> Tensor:
        h = self.shared(x)
        return torch.cat([self.category_head(h), self.risk_head(h), self.success_head(h)], dim=-1)


def build_config(
    module_info: list[tuple[str, int]] | None = None,
    *,
    c0: int = 64,
    c2: int = 64,
    steps: int,
    batch_size: int,
    use_delta_component: bool = True,
) -> Config:
    info = module_info or [("shared.0", c0), ("shared.2", c2)]
    config_dict = {
        "wandb_project": None,
        "wandb_run_name": None,
        "seed": 0,
        "n_mask_samples": 1,
        "ci_config": {"mode": "layerwise", "fn_type": "mlp", "hidden_dims": [16]},
        "sigmoid_type": "leaky_hard",
        "module_info": [{"module_pattern": pattern, "C": c} for pattern, c in info],
        "identity_module_info": None,
        "use_delta_component": use_delta_component,
        "loss_metric_configs": [
            {"classname": "ImportanceMinimalityLoss", "coeff": 1e-5, "pnorm": 2.0, "beta": 0},
            {"classname": "StochasticReconLayerwiseLoss", "coeff": 1.0},
            {"classname": "StochasticReconLoss", "coeff": 1.0},
        ],
        "batch_size": batch_size,
        "eval_batch_size": batch_size,
        "steps": steps,
        "lr_schedule": {"start_val": 2e-3, "fn_type": "constant", "warmup_pct": 0.0},
        "faithfulness_warmup_steps": 100,
        "faithfulness_warmup_lr": 0.01,
        "faithfulness_warmup_weight_decay": 0.1,
        "train_log_freq": 100,
        "eval_freq": 500,
        "n_eval_steps": 20,
        "slow_eval_freq": 5_000,
        "slow_eval_on_first_step": False,
        "save_freq": None,
        "ci_alive_threshold": 0.1,
        "eval_metric_configs": [
            {"classname": "CI_L0", "groups": None},
            {"classname": "CIMeanPerComponent"},
        ],
        "pretrained_model_class": "scripts.trm_real_decomp_run.ConveyorTRMConcat",
        "pretrained_model_path": "local:trm_arc_challenge",
        "task_config": {
            "task_name": "resid_mlp",
            "feature_probability": 0.01,
            "data_generation_type": "at_least_zero_active",
        },
    }
    return Config.model_validate(config_dict)


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
    parser.add_argument("--n-prompts", type=int, default=1024)
    parser.add_argument("--c0", type=int, default=64)
    parser.add_argument("--c2", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"D:\Research_Engine\runs\trm_param_decomp\trm_arc_challenge"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base, vectorizer, input_dim = load_trm(args.checkpoint)
    model = ConveyorTRMConcat(base).to(args.device)
    model.eval()
    model.requires_grad_(False)

    prompts = arc_prompts(args.parquet, args.n_prompts)
    features = np.asarray(vectorizer.transform(prompts).todense(), dtype=np.float32)
    x = torch.tensor(features, device=args.device)
    assert x.shape[1] == input_dim

    generator = torch.Generator(device="cpu").manual_seed(0)
    dataset = TensorDataset(x)
    train_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    eval_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    config = build_config(c0=args.c0, c2=args.c2, steps=args.steps, batch_size=args.batch_size)
    optimize(
        target_model=model,
        config=config,
        device=args.device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        out_dir=args.out_dir,
    )
    print(f"decomposition written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

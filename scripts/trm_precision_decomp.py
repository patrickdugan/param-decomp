"""Scale real VPD to precision_trm_mcq (4.3M): SPD decomposition + extraction.

precision_trm_mcq is a frozen TF-IDF -> MLP 10-way MCQ answer classifier
(net.0: 8192->512, net.3: 512->256, net.6: 256->10), trained on arc_easy,
arc_challenge, mmlu_professional_law. We decompose the two large layers
(net.0, net.3) over the model's own training prompts and extract the same VPD
artifacts as the conveyor TRM.
"""

from __future__ import annotations

import argparse
import typing
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.models.batch_and_loss_fns import recon_loss_kl, run_batch_first_element
from param_decomp.models.component_model import ComponentModel
from param_decomp.run_param_decomp import optimize
from param_decomp.utils.module_utils import expand_module_patterns
from scripts.trm_decomp_extract import run_extraction
from scripts.trm_real_decomp_run import build_config

CHECKPOINT = Path(
    r"D:\Research_Engine\tesseract_persistent\data\models\precision_trm\precision_trm_mcq_precision.pt"
)
OUT_DIR = Path(r"D:\Research_Engine\runs\trm_param_decomp\precision_trm_mcq")


class PrecisionTRM(nn.Module):
    """Frozen TF-IDF MLP 10-way classifier: Linear-ReLU-Dropout x2 then Linear."""

    def __init__(self, input_dim: int, hidden0: int, hidden1: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden0),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden0, hidden1),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden1, n_classes),
        )

    @typing.override
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def load_precision_trm(checkpoint_path: Path) -> tuple[PrecisionTRM, Any, list[str]]:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    model = PrecisionTRM(
        input_dim=int(ck["input_dim"]),
        hidden0=sd["net.0.weight"].shape[0],
        hidden1=sd["net.3.weight"].shape[0],
        n_classes=sd["net.6.weight"].shape[0],
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, ck["vectorizer"], list(ck["prompt_to_answer"].keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--n-prompts", type=int, default=2048)
    parser.add_argument("--c0", type=int, default=128)
    parser.add_argument("--c3", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--report-path", type=Path, default=Path("reports") / "trm_precision_labeled_decomp_report.md"
    )
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    model, vectorizer, prompts = load_precision_trm(args.checkpoint)
    model.to(device)
    prompts = prompts[: args.n_prompts]
    x = torch.tensor(np.asarray(vectorizer.transform(prompts).todense(), dtype=np.float32), device=device)

    module_info = [("net.0", args.c0), ("net.3", args.c3)]
    config = build_config(module_info, steps=args.steps, batch_size=args.batch_size)
    model_path = args.out_dir / f"model_{args.steps}.pth"

    if not args.extract_only:
        generator = torch.Generator(device="cpu").manual_seed(0)
        train_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
            TensorDataset(x), batch_size=args.batch_size, shuffle=True, generator=generator
        )
        eval_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
            TensorDataset(x), batch_size=args.batch_size, shuffle=False
        )
        optimize(
            target_model=model,
            config=config,
            device=device,
            train_loader=train_loader,
            eval_loader=eval_loader,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_kl,
            out_dir=args.out_dir,
        )

    cm = ComponentModel(
        target_model=model,
        run_batch=run_batch_first_element,
        module_path_info=expand_module_patterns(model, config.all_module_info),
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
    ).to(device)
    cm.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    cm.eval()

    run_extraction(
        model, cm, x, config,
        model_name="PrecisionTRM mcq_precision (4.3M)",
        checkpoint=args.checkpoint,
        model_path=model_path,
        out_dir=args.out_dir,
        report_path=args.report_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

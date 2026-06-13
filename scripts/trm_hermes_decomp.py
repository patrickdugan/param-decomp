"""Real SPD decomposition of Fable's recursive HRM logic critic.

The loop-spline discriminator (commit 626440ec) decomposed this critic into
SVD atoms and explicitly "does not claim actual Goodfire VPD". This builds the
new version: run this repo's real SPD optimize() on the recursive HermesHRM
logic critic, producing genuine causally-important parameter components instead
of SVD atoms, over the critic's own logic dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from param_decomp.models.batch_and_loss_fns import recon_loss_kl
from param_decomp.run_param_decomp import optimize
from scripts.trm_real_decomp_run import build_config

HRM_REPO = Path(r"D:\projects\HRM-re")
HRM_CORE = Path(r"C:\projects\HRM-re")  # HRM.hrm package lives here
for _root in (HRM_CORE, HRM_REPO):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments.hermes_skill_gym.hermes_dataset import (  # noqa: E402 # pyright: ignore[reportMissingImports]
    HermesSkillDataset,
    collate_batch,
    read_jsonl,
)
from experiments.hermes_skill_gym.train_hrm_hermes import (  # noqa: E402 # pyright: ignore[reportMissingImports]
    load_trained_model,
)

MODEL_DIR = (
    HRM_REPO
    / "experiments/hermes_skill_gym/outputs"
    / "hermes_skill_gym_trm_action_family_explicit_single"
    / "variants"
    / "trm_like_single_tier"
)
MODEL_PATH = MODEL_DIR / "artifacts" / "model.pt"
TRAIN_PATH = MODEL_DIR.parents[1] / "dataset" / "train.jsonl"
OUT_DIR = Path(r"D:\Research_Engine\runs\trm_param_decomp\hermes_logic_critic")

# Recursive-trunk Linears (applied every reasoning step) plus the bucket readout.
TARGET_MODULES = [
    ("networks.0.layers.0.1.to_out", 32),
    ("networks.0.layers.1.1.ff.2", 32),
    ("bucket_head", 16),
]


def make_run_batch(pad_id: int):
    def run_batch_hermes(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
        input_ids = batch["input_ids"]
        attention_mask = input_ids.ne(pad_id)
        return model(input_ids, attention_mask)[0]  # bucket_logits

    return run_batch_hermes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--n-rows", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def build_loader(rows: list[dict[str, Any]], payload: dict[str, Any], batch_size: int) -> DataLoader[Any]:
    config = payload["config"]
    vocab = payload["vocab"]
    dataset = HermesSkillDataset(
        rows,
        vocab,
        {label: index for index, label in enumerate(config["bucket_labels"])},
        {label: index for index, label in enumerate(config["action_labels"])},
        max_tokens=int(config.get("max_tokens", 512)),
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate_batch(batch, int(vocab["<PAD>"])),
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload, model = load_trained_model(str(args.model), device=args.device)
    model.requires_grad_(False)
    pad_id = int(payload["vocab"]["<PAD>"])

    rows = read_jsonl(args.train)[: args.n_rows]
    train_loader = build_loader(rows, payload, args.batch_size)
    eval_loader = build_loader(rows, payload, args.batch_size)

    config = build_config(
        TARGET_MODULES, steps=args.steps, batch_size=args.batch_size, use_delta_component=False
    )
    optimize(
        target_model=model,
        config=config,
        device=args.device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=make_run_batch(pad_id),
        reconstruction_loss=recon_loss_kl,
        out_dir=args.out_dir,
    )
    print(f"hermes critic decomposition written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

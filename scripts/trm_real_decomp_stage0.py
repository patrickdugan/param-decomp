"""Stage 0/1 of real parameter decomposition on the local TRMs.

Reconstruct a conveyor TRM (a frozen TF-IDF -> MLP classifier) from its
checkpoint, rebuild its input features with the stored TfidfVectorizer over real
task prompts, and verify a faithful forward pass. This is the foundation for
running this repo's SPD `optimize()` on a genuine model instead of the 4D cached
choice head.

No SPD yet: this proves the model + data + module targets are real and wrappable.
"""

from __future__ import annotations

import argparse
import json
import typing
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

CONVEYOR_DIR = Path(r"D:\Research_Engine\tesseract_persistent\data\models\conveyor")
ARC_PARQUET = Path(r"D:\Research_Engine\prime_envs\arc\ARC-Challenge\train-00000-of-00001.parquet")


class ConveyorTRM(nn.Module):
    """Frozen TF-IDF MLP classifier matching the conveyor checkpoint state dict.

    shared: Linear(in, 256) -> ReLU -> Linear(256, 128) -> ReLU
    heads: category (in->3), risk (in->1), success (in->1)
    """

    def __init__(self, input_dim: int, hidden0: int, hidden1: int, n_category: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden0),
            nn.ReLU(),
            nn.Linear(hidden0, hidden1),
            nn.ReLU(),
        )
        self.category_head = nn.Linear(hidden1, n_category)
        self.risk_head = nn.Linear(hidden1, 1)
        self.success_head = nn.Linear(hidden1, 1)

    @typing.override
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "category": self.category_head(h),
            "risk": self.risk_head(h),
            "success": self.success_head(h),
        }


def load_trm(checkpoint_path: Path) -> tuple[ConveyorTRM, Any, int]:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    input_dim = int(ck["input_dim"])
    hidden0 = sd["shared.0.weight"].shape[0]
    hidden1 = sd["shared.2.weight"].shape[0]
    n_category = sd["category_head.weight"].shape[0]
    model = ConveyorTRM(input_dim, hidden0, hidden1, n_category)
    model.load_state_dict(sd, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ck["vectorizer"], input_dim


def arc_prompts(parquet_path: Path, n: int) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    prompts: list[str] = []
    for index in range(len(df)):
        row = df.iloc[index]
        labels = [str(label) for label in row["choices"]["label"]]
        texts = [str(text) for text in row["choices"]["text"]]
        options = "\n".join(f"{label}: {text}" for label, text in zip(labels, texts, strict=True))
        prompts.append(f"{row['question']}\n{options}")
        if len(prompts) >= n:
            break
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CONVEYOR_DIR / "trm_arc_challenge.pt")
    parser.add_argument("--parquet", type=Path, default=ARC_PARQUET)
    parser.add_argument("--n-prompts", type=int, default=256)
    parser.add_argument(
        "--audit-path", type=Path, default=Path("reports") / "trm_param_decomp_audit.md"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, vectorizer, input_dim = load_trm(args.checkpoint)
    n_params = sum(int(p.numel()) for p in model.parameters())

    prompts = arc_prompts(args.parquet, args.n_prompts)
    features = vectorizer.transform(prompts)
    x = torch.tensor(np.asarray(features.todense()), dtype=torch.float32)
    assert x.shape[1] == input_dim, f"feature dim {x.shape[1]} != model input_dim {input_dim}"

    with torch.no_grad():
        out = model(x)
    category_pred = out["category"].argmax(dim=1)
    linear0, linear2 = model.shared[0], model.shared[2]
    assert isinstance(linear0, nn.Linear) and isinstance(linear2, nn.Linear)
    decomposable = {
        "shared.0": tuple(linear0.weight.shape),
        "shared.2": tuple(linear2.weight.shape),
        "category_head": tuple(model.category_head.weight.shape),
    }
    feature_density = float((x != 0).float().mean())

    audit = {
        "checkpoint": str(args.checkpoint),
        "model_class": "ConveyorTRM (TF-IDF -> MLP classifier)",
        "parameter_count": n_params,
        "input_dim": input_dim,
        "vectorizer": type(vectorizer).__name__,
        "n_prompts": len(prompts),
        "feature_density": round(feature_density, 6),
        "forward_ok": True,
        "category_logit_std": round(float(out["category"].std()), 6),
        "risk_std": round(float(out["risk"].std()), 6),
        "success_std": round(float(out["success"].std()), 6),
        "category_pred_distribution": {
            int(k): int(v)
            for k, v in zip(*np.unique(category_pred.numpy(), return_counts=True), strict=True)
        },
        "decomposable_linear_modules": {k: list(v) for k, v in decomposable.items()},
        "candidate_modules_for_decomp": ["shared.0", "shared.2"],
        "tooling_status": "THIS_REPO_SPD_AVAILABLE (param_decomp ComponentModel wraps nn.Linear)",
        "behavioral_variation_note": (
            "category argmax is constant on ARC-only prompts, but category logits "
            "and the continuous risk/success heads vary per datapoint (see *_std), "
            "so SPD has output variation to attribute. For richer multi-class "
            "variation, also consider precision_trm_mcq (4.3M, 10-way) or diverse "
            "multi-env prompts."
        ),
    }

    lines = [
        "# TRM Parameter-Decomposition Audit (Stage 0/1)",
        "",
        "Real model located and wrappable. This replaces the 4D cached choice-head "
        "proxy with a genuine frozen TRM whose Linear layers can be decomposed by "
        "this repo's SPD framework.",
        "",
        "```json",
        json.dumps(audit, indent=2),
        "```",
        "",
        "## Next: SPD decomposition",
        "",
        "Wrap `ConveyorTRM` in `ComponentModel` with targets `shared.0`, `shared.2` "
        "and run `param_decomp.run_param_decomp.optimize()` (resid_mlp template) to "
        "produce real parameter components with reconstruction error and per-"
        "datapoint causal importance — the actual VPD artifacts.",
    ]
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

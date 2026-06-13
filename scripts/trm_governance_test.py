"""Stage 6: governance/patch test using the real decomposition components.

Train a dirty low-rank diff on precision_trm_mcq's net.3 that lifts a target env
(mmlu_professional_law) but regresses a non-target env (arc_challenge). Then test
whether the real parameter-decomposition components let us identify and remove the
regression-carrying part of the diff better than norm/random/SVD controls.

A diff is decomposed by projecting it onto the component basis; a selector drops
the projection onto a chosen component subset. The decomposition-informed
selector ranks components by |projection| x causal-importance-on-non-target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from param_decomp.models.batch_and_loss_fns import run_batch_first_element
from param_decomp.models.component_model import ComponentModel
from param_decomp.utils.module_utils import expand_module_patterns
from scripts.trm_decomp_extract import _linear
from scripts.trm_precision_decomp import CHECKPOINT, OUT_DIR, load_precision_trm
from scripts.trm_real_decomp_run import build_config

TARGET_ENV = "mmlu_professional_law"
NONTARGET_ENV = "arc_challenge"
PATH = "net.3"


def _slice(prompt_to_answer: dict[str, str], label_vocab: list[str], env: str, vectorizer: Any, n: int):
    rows = [(p.split("|||", 1)[1], a) for p, a in prompt_to_answer.items() if p.startswith(env + "|||")]
    rows = rows[:n]
    texts = [t for t, _ in rows]
    y = torch.tensor([label_vocab.index(str(a)) for _, a in rows], dtype=torch.long)
    x = torch.tensor(np.asarray(vectorizer.transform(texts).todense(), dtype=np.float32))  # type: ignore[attr-defined]
    return x, y


def _accuracy(model: nn.Module, x: Tensor, y: Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(dim=-1) == y).float().mean())


def train_dirty_diff(
    model: nn.Module, path: str, x: Tensor, y: Tensor, *, rank: int, steps: int, lr: float
) -> Tensor:
    d_out, d_in = _linear(model, path).weight.shape
    base = _linear(model, path).weight.data.clone()
    a = (torch.randn(rank, d_in) * 0.01).requires_grad_(True)
    b = (torch.randn(d_out, rank) * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = torch.func.functional_call(model, {f"{path}.weight": base + b @ a}, (x,))
        torch.nn.functional.cross_entropy(logits, y).backward()
        opt.step()
    return (b @ a).detach()


def _project(diff: Tensor, weights: Tensor) -> Tensor:
    """Projection coefficient of diff onto each component weight (Frobenius)."""
    flat_diff = diff.reshape(-1)
    flat_w = weights.reshape(weights.shape[0], -1)
    denom = (flat_w * flat_w).sum(dim=1).clamp_min(1e-12)
    return (flat_w @ flat_diff) / denom  # [C]


def _cleaned_diff(diff: Tensor, weights: Tensor, alpha: Tensor, drop: list[int]) -> Tensor:
    cleaned = diff.clone()
    for c in drop:
        cleaned = cleaned - alpha[c] * weights[c]
    return cleaned


def _apply_and_measure(
    model: nn.Module, path: str, base: Tensor, diff: Tensor, slices: dict[str, tuple[Tensor, Tensor]]
) -> dict[str, float]:
    linear = _linear(model, path)
    linear.weight.data = base + diff
    out = {env: _accuracy(model, x, y) for env, (x, y) in slices.items()}
    linear.weight.data = base
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--model", type=Path, default=OUT_DIR / "model_8000.pth")
    parser.add_argument("--n-per-env", type=int, default=400)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--drop-k", type=int, default=8)
    parser.add_argument("--c0", type=int, default=128)
    parser.add_argument("--c3", type=int, default=128)
    parser.add_argument("--report-path", type=Path, default=Path("reports") / "trm_governance_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = "cpu"
    model, vectorizer, _ = load_precision_trm(args.checkpoint)
    model.to(device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prompt_to_answer, label_vocab = ck["prompt_to_answer"], list(ck["label_vocab"])

    xt, yt = _slice(prompt_to_answer, label_vocab, TARGET_ENV, vectorizer, args.n_per_env)
    xn, yn = _slice(prompt_to_answer, label_vocab, NONTARGET_ENV, vectorizer, args.n_per_env)
    slices = {"target": (xt, yt), "nontarget": (xn, yn)}
    base = _linear(model, PATH).weight.data.clone()
    baseline = {env: _accuracy(model, x, y) for env, (x, y) in slices.items()}

    diff = train_dirty_diff(model, PATH, xt, yt, rank=args.rank, steps=args.steps, lr=args.lr)
    raw = _apply_and_measure(model, PATH, base, diff, slices)

    config = build_config([("net.0", args.c0), ("net.3", args.c3)], steps=1, batch_size=args.n_per_env)
    cm = ComponentModel(
        target_model=model,
        run_batch=run_batch_first_element,
        module_path_info=expand_module_patterns(model, config.all_module_info),
        ci_config=config.ci_config,
        sigmoid_type=config.sigmoid_type,
    ).to(device)
    cm.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    cm.eval()

    component = cm.components[PATH]
    weights = torch.stack(
        [torch.outer(component.U[c], component.V[:, c]) for c in range(component.C)]
    )  # [C, d_out, d_in]
    alpha = _project(diff, weights)

    with torch.no_grad():
        ci_t = cm.calc_causal_importances(
            pre_weight_acts=cm((xt,), cache_type="input").cache, sampling=config.sampling
        ).lower_leaky[PATH]
        ci_n = cm.calc_causal_importances(
            pre_weight_acts=cm((xn,), cache_type="input").cache, sampling=config.sampling
        ).lower_leaky[PATH]
    importance_nontarget = (ci_n > 0.1).float().mean(dim=0)  # [C]
    importance_target = (ci_t > 0.1).float().mean(dim=0)

    abs_alpha = alpha.abs()
    decomp_score = abs_alpha * importance_nontarget * (1.0 - importance_target).clamp_min(0.0)
    norm_score = abs_alpha * weights.reshape(component.C, -1).norm(dim=1)
    rng = np.random.default_rng(0)

    selectors: dict[str, list[int]] = {
        "decomp_ci_informed": torch.topk(decomp_score, args.drop_k).indices.tolist(),
        "norm_only": torch.topk(norm_score, args.drop_k).indices.tolist(),
        "random": [int(i) for i in rng.choice(component.C, size=args.drop_k, replace=False)],
    }
    # SVD control: remove top-k SVD directions of the diff itself.
    u, s, vt = torch.linalg.svd(diff, full_matrices=False)
    svd_removed = diff.clone()
    for k in range(min(args.drop_k, s.shape[0])):
        svd_removed = svd_removed - s[k] * torch.outer(u[:, k], vt[k])

    results: dict[str, dict[str, float]] = {"raw_dirty": raw}
    for name, drop in selectors.items():
        cleaned = _cleaned_diff(diff, weights, alpha, drop)
        results[name] = _apply_and_measure(model, PATH, base, cleaned, slices)
    results["svd_topk_removed"] = _apply_and_measure(model, PATH, base, svd_removed, slices)

    payload = _classify(baseline, results)
    _write_report(args.report_path, baseline, results, payload, args.drop_k)
    print(json.dumps(payload, indent=2))
    return 0


def _classify(baseline: dict[str, float], results: dict[str, dict[str, float]]) -> dict[str, object]:
    raw = results["raw_dirty"]
    target_gain = raw["target"] - baseline["target"]
    raw_regression = baseline["nontarget"] - raw["nontarget"]

    def retention(name: str) -> float:
        gain = results[name]["target"] - baseline["target"]
        return gain / target_gain if target_gain > 0 else 0.0

    def regression(name: str) -> float:
        return baseline["nontarget"] - results[name]["nontarget"]

    decomp_reg = regression("decomp_ci_informed")
    control_names = ("norm_only", "random", "svd_topk_removed")
    controls = {n: regression(n) for n in control_names}
    control_retention = {n: round(retention(n), 6) for n in control_names}
    # A control that also dropped the whole patch (no target retained) is degenerate;
    # its low regression is not a real win.
    valid_controls = {n: r for n, r in controls.items() if control_retention[n] >= 0.5}
    best_control = min(valid_controls.values()) if valid_controls else raw_regression
    decomp_retention = retention("decomp_ci_informed")
    if target_gain <= 0 or raw_regression <= 0:
        verdict = "BLOCKED_NO_DIRTY_DIFF"
    elif decomp_retention >= 0.5 and decomp_reg < best_control and decomp_reg < raw_regression:
        verdict = "REAL_DECOMP_SELECTOR_POSITIVE"
    elif decomp_retention >= 0.5 and decomp_reg < raw_regression:
        verdict = "REAL_DECOMP_EXISTENCE_POSITIVE"
    else:
        verdict = "REAL_DECOMP_NEGATIVE"
    return {
        "target_gain": round(target_gain, 6),
        "raw_regression": round(raw_regression, 6),
        "decomp_regression": round(decomp_reg, 6),
        "decomp_target_retention": round(decomp_retention, 6),
        "control_regressions": {k: round(v, 6) for k, v in controls.items()},
        "control_target_retention": control_retention,
        "verdict": verdict,
    }


def _write_report(
    path: Path,
    baseline: dict[str, float],
    results: dict[str, dict[str, float]],
    payload: dict[str, object],
    drop_k: int,
) -> None:
    lines = [
        "# TRM Governance / Patch Test (Stage 6)",
        "",
        f"Real decomposition of precision_trm_mcq net.3. A rank-edited dirty diff "
        f"lifts {TARGET_ENV} and regresses {NONTARGET_ENV}; each selector drops the "
        f"diff's projection onto {drop_k} components.",
        "",
        f"Baseline accuracy: target {baseline['target']:.4f}, nontarget {baseline['nontarget']:.4f}.",
        "",
        "| arm | target acc | nontarget acc |",
        "|---|---:|---:|",
    ]
    for name, res in results.items():
        lines.append(f"| {name} | {res['target']:.4f} | {res['nontarget']:.4f} |")
    lines += ["", "## Verdict", "", "```json", json.dumps(payload, indent=2), "```"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

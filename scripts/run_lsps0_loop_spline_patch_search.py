"""LSPS-0: Loop-Spline Patch Search v0.

Tests whether spline-derived proposal scores beat static-derived proposal
scores at finding accepted sparse rank-one component patches on a frozen TRM.

This is one frozen TRM, one SVD rank-one component dictionary, cached loop
profiles, and a greedy scored patch search. There is no policy network, archive
learner, consolidation, dual variables, or RCPI here. The base model is never
updated; patches are applied transiently and reverted.

This uses a VPD-style/SVD rank-one component dictionary. It does not claim
actual Goodfire VPD and does not use cached 4D ARC heads.

Precondition: the loop-spline discriminator must not have failed. If its final
label is one of {PROFILE_UNSTABLE, STATIC_ENDPOINT_SUFFICIENT, UNDERPOWERED_PROBES,
BLOCKED_NO_RECURSION_TRACE} we write a blocker section and do not claim LSPS-0
evidence (unless --override is passed).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import run_trm_loop_spline_first_discriminator as disc
from run_trm_loop_spline_first_discriminator import (
    ARTIFACT_ROOT,
    EVAL_PATH,
    MODEL_PATH,
    TRAIN_PATH,
    Component,
    ConstellationGovernorHermesClassifier,
    HermesHRMClassifier,
    build_components,
    build_loader,
    head_metrics,
    masked_mean,
    read_jsonl,
    reasoning_depth,
)
from run_trm_loop_spline_first_discriminator import load_trained_model  # noqa: E402


REPORT_PATH = disc.REPORT_PATH.parent / "lsps0_loop_spline_patch_search.md"
RESULTS_PATH = ARTIFACT_ROOT / "lsps0_results.json"
DISCRIMINATOR_MANIFEST = ARTIFACT_ROOT / "manifest.json"

BLOCKING_DISCRIMINATOR_LABELS = {
    "PROFILE_UNSTABLE",
    "STATIC_ENDPOINT_SUFFICIENT",
    "UNDERPOWERED_PROBES",
    "BLOCKED_NO_RECURSION_TRACE",
}

KAPPA_MAX = 8
SHORTLIST_B = 4
ALPHA_PROBE_MAG = 0.25
ETA_STEP = 1e-4
LINE_SEARCH_GRID = [0.1, 0.25, 0.5, 1.0]
LINE_SEARCH_SWEEPS = 2
TRUST_REGION_RHO = 0.05
ACCEPT_DELTA = 0.05
ACCEPT_EPSILON = 0.0
MIN_FAMILIES = 5
MIN_FAMILY_ROWS = 6
PROPOSAL_TARGET = 64
METHODS = ["LSPS0", "B1_random", "B2_static_norm", "B3_static_damage", "B4_gradient_only"]


@dataclass
class Family:
    name: str
    proposal_rows: list[dict[str, Any]]
    holdout_rows: list[dict[str, Any]]
    regression_suites: dict[str, list[dict[str, Any]]]
    base_proposal_loss: float = 0.0
    base_holdout_loss: float = 0.0
    base_proposal_errors: int = 0


@dataclass
class PatchResult:
    method: str
    family: str
    support: list[str] = field(default_factory=list)
    signs: dict[str, float] = field(default_factory=dict)
    coeffs: dict[str, float] = field(default_factory=dict)
    proposal_gain: float = 0.0
    holdout_gain: float = 0.0
    regression_deltas: dict[str, float] = field(default_factory=dict)
    regression_max_ucb: float = 0.0
    trust_region_max: float = 0.0
    evals: int = 0
    interaction_flag: bool = False
    accepted: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--base-depth", type=int, default=3)
    parser.add_argument("--rank-atoms-per-module", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=104)
    parser.add_argument("--max-families", type=int, default=0, help="0 = no cap")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--override",
        action="store_true",
        help="Run LSPS-0 even if the discriminator label is blocking. Evidence is still gated honestly.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for output filenames, to keep descriptive override runs separate from canonical artifacts.",
    )
    return parser.parse_args()


def output_paths(tag: str) -> tuple[Path, Path]:
    if not tag:
        return REPORT_PATH, RESULTS_PATH
    report = REPORT_PATH.with_name(f"{REPORT_PATH.stem}.{tag}{REPORT_PATH.suffix}")
    results = RESULTS_PATH.with_name(f"{RESULTS_PATH.stem}.{tag}{RESULTS_PATH.suffix}")
    return report, results


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_discriminator_label() -> str | None:
    if not DISCRIMINATOR_MANIFEST.exists():
        return None
    manifest = json.loads(DISCRIMINATOR_MANIFEST.read_text(encoding="utf-8"))
    return manifest.get("final_label")


def load_all_rows(train: Path, eval_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [train, eval_path]:
        if path.exists():
            rows.extend(read_jsonl(str(path)))
    return rows


def forward_losses(
    model: HermesHRMClassifier, loader: DataLoader, depth: int, pad_id: int
) -> tuple[float, list[bool]]:
    losses: list[float] = []
    correct: list[bool] = []
    with reasoning_depth(model, depth), torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = input_ids.ne(pad_id)
            bucket_logits, _action, _reward, _pooled = model(input_ids, attention_mask)
            labels = batch["bucket_labels"]
            loss = F.cross_entropy(bucket_logits, labels, reduction="none")
            losses.extend(float(x) for x in loss.tolist())
            correct.extend(bool(x) for x in bucket_logits.argmax(dim=-1).eq(labels).tolist())
    return float(np.mean(losses)) if losses else 0.0, correct


def grad_step_weights(
    model: HermesHRMClassifier, loader: DataLoader, depth: int, pad_id: int
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    """Per-step loss sensitivity w_t and dL/dW for every 2D module on the family.

    w_t mirrors g_t(tau) = E[d loss / d Psi(s_t)] via the loss-gradient channel:
    the mean norm of the gradient of the family loss w.r.t. the pooled step state.
    Param grads provide the gradient-only baseline projection.
    """
    param_grads: dict[str, torch.Tensor] = {}
    step_weight_sum: dict[int, float] = defaultdict(float)
    step_count: dict[int, int] = defaultdict(int)
    model.zero_grad(set_to_none=True)
    with reasoning_depth(model, depth):
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = input_ids.ne(pad_id)
            tokens = model.to_input_embed(input_ids)
            hiddens = torch.zeros_like(tokens).unsqueeze(0).repeat(model.num_networks, 1, 1, 1)
            hiddens_dict = {index: hidden for index, hidden in enumerate(hiddens)}
            total_low_steps = model.reasoning_steps * model.lowest_steps_per_reasoning_step
            highest_steps: list[torch.Tensor] = []
            for index in range(total_low_steps):
                iteration = index + 1
                for network_index, (network, hidden_combine, evaluate_at) in enumerate(
                    zip(model.networks, model.hidden_combiners, model.evaluate_networks_at)
                ):
                    if iteration % evaluate_at != 0:
                        continue
                    combined = hidden_combine((tokens, *hiddens_dict.values()), network_index)
                    hiddens_dict[network_index] = network(combined, mask=attention_mask)
                highest = hiddens_dict[model.num_networks - 1]
                highest.retain_grad()
                highest_steps.append(highest)
            final_pooled = masked_mean(highest_steps[-1], attention_mask)
            metrics = head_metrics(model, final_pooled, batch)
            loss = metrics["loss"].mean()
            loss.backward()
            for step_index, highest in enumerate(highest_steps):
                if highest.grad is None:
                    continue
                grad = highest.grad.reshape(highest.shape[0], -1)
                weight = float(torch.linalg.vector_norm(grad, dim=1).mean().item())
                step_weight_sum[step_index + 1] += weight
                step_count[step_index + 1] += 1
            for name, param in model.named_parameters():
                if param.grad is None or param.ndim != 2:
                    continue
                grad = param.grad.detach().cpu().float().clone()
                param_grads[name] = param_grads.get(name, torch.zeros_like(grad)) + grad
            model.zero_grad(set_to_none=True)
    steps = sorted(step_weight_sum)
    weights = np.array([step_weight_sum[s] / max(1, step_count[s]) for s in steps], dtype=float)
    norm = float(np.sum(weights)) or 1.0
    return weights / norm, param_grads


def trace_loss_profile(
    model: HermesHRMClassifier, loader: DataLoader, depth: int, pad_id: int
) -> np.ndarray:
    """Per-step mean bucket loss profile over the loader (channel for Phi)."""
    step_sum: dict[int, float] = defaultdict(float)
    step_count: dict[int, int] = defaultdict(int)
    with reasoning_depth(model, depth), torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = input_ids.ne(pad_id)
            rows = disc.trace_hierarchical(model, input_ids, attention_mask, batch)
            for row in rows:
                step_sum[int(row["step"])] += float(row["loss"])
                step_count[int(row["step"])] += 1
    steps = sorted(step_sum)
    return np.array([step_sum[s] / max(1, step_count[s]) for s in steps], dtype=float)


def cached_loss_deltas(
    model: HermesHRMClassifier,
    loader: DataLoader,
    components: list[Component],
    depth: int,
    pad_id: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Phi_k(t, alpha) loss-channel deltas vs base for alpha in {0.5, 1.5}.

    These are the cached loop-effect profiles. The base profile is alpha=1.0.
    """
    base = trace_loss_profile(model, loader, depth, pad_id)
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for component in components:
        deltas: dict[str, np.ndarray] = {}
        for alpha in [0.5, 1.5]:
            with disc.intervened(model, component, alpha):
                profile = trace_loss_profile(model, loader, depth, pad_id)
            deltas[str(alpha)] = profile - base
        profiles[component.component_id] = deltas
    return profiles


def spline_scores(
    profiles: dict[str, dict[str, np.ndarray]],
    weights: np.ndarray,
    components: list[Component],
) -> dict[str, tuple[float, float]]:
    """LSPS-0 proposal scores a_k = max(a_k_plus, a_k_minus) with chosen sign."""
    scores: dict[str, tuple[float, float]] = {}
    for component in components:
        deltas = profiles[component.component_id]
        dot_plus = deltas["1.5"] / 0.5
        dot_minus = deltas["0.5"] / -0.5
        n = min(len(weights), len(dot_plus), len(dot_minus))
        a_plus = -float(np.dot(weights[:n], dot_plus[:n]))
        a_minus = -float(np.dot(weights[:n], dot_minus[:n]))
        if a_plus >= a_minus:
            scores[component.component_id] = (a_plus, 1.0)
        else:
            scores[component.component_id] = (a_minus, -1.0)
    return scores


def static_norm_scores(components: list[Component]) -> dict[str, float]:
    return {c.component_id: c.scale * c.u_norm * c.v_norm for c in components}


def gradient_projection_scores(
    components: list[Component], param_grads: dict[str, torch.Tensor]
) -> dict[str, tuple[float, float]]:
    scores: dict[str, tuple[float, float]] = {}
    for component in components:
        grad = param_grads.get(component.module_id)
        if grad is None:
            scores[component.component_id] = (0.0, 1.0)
            continue
        projection = float(component.u @ grad @ component.v) * component.scale
        scores[component.component_id] = (abs(projection), -1.0 if projection > 0 else 1.0)
    return scores


def static_damage_scores(
    model: HermesHRMClassifier,
    loader: DataLoader,
    components: list[Component],
    base_correct: list[bool],
    depth: int,
    pad_id: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for component in components:
        with disc.intervened(model, component, 0.0):
            _loss, correct = forward_losses(model, loader, depth, pad_id)
        scores[component.component_id] = float(disc.damage(base_correct, correct))
    return scores


@contextlib.contextmanager
def apply_patch(
    model: HermesHRMClassifier,
    coeffs: dict[str, float],
    by_id: dict[str, Component],
    rho: float,
):
    """Apply rank-one patch deltas grouped per module, clipped to trust region."""
    params = dict(model.named_parameters())
    deltas_by_module: dict[str, torch.Tensor] = {}
    for component_id, coeff in coeffs.items():
        if coeff == 0.0:
            continue
        component = by_id[component_id]
        param = params[component.module_id]
        delta = coeff * component.scale * torch.outer(
            component.u.to(param.device, dtype=param.dtype),
            component.v.to(param.device, dtype=param.dtype),
        )
        deltas_by_module[component.module_id] = deltas_by_module.get(
            component.module_id, torch.zeros_like(param)
        ) + delta
    applied: dict[str, torch.Tensor] = {}
    ratios: dict[str, float] = {}
    for module_id, delta in deltas_by_module.items():
        param = params[module_id]
        param_norm = float(torch.linalg.matrix_norm(param)) or 1.0
        delta_norm = float(torch.linalg.matrix_norm(delta))
        max_norm = rho * param_norm
        if delta_norm > max_norm and delta_norm > 0:
            delta = delta * (max_norm / delta_norm)
            delta_norm = max_norm
        applied[module_id] = delta
        ratios[module_id] = delta_norm / param_norm
    with torch.no_grad():
        for module_id, delta in applied.items():
            params[module_id].add_(delta)
    try:
        yield ratios
    finally:
        with torch.no_grad():
            for module_id, delta in applied.items():
                params[module_id].sub_(delta)


def patch_loss(
    model: HermesHRMClassifier,
    loader: DataLoader,
    coeffs: dict[str, float],
    by_id: dict[str, Component],
    depth: int,
    pad_id: int,
    rho: float,
) -> tuple[float, dict[str, float]]:
    with apply_patch(model, coeffs, by_id, rho) as ratios:
        loss, _correct = forward_losses(model, loader, depth, pad_id)
    return loss, ratios


def greedy_select(
    model: HermesHRMClassifier,
    family: Family,
    loader: DataLoader,
    scored_signs: dict[str, tuple[float, float]],
    by_id: dict[str, Component],
    depth: int,
    pad_id: int,
) -> tuple[list[str], dict[str, float], int]:
    ranked = sorted(scored_signs.items(), key=lambda kv: kv[1][0], reverse=True)
    ranked_ids = [cid for cid, _ in ranked]
    support: list[str] = []
    coeffs: dict[str, float] = {}
    evals = 0
    base_loss = family.base_proposal_loss
    while len(support) < KAPPA_MAX:
        shortlist = [cid for cid in ranked_ids if cid not in support][:SHORTLIST_B]
        if not shortlist:
            break
        best_cid = None
        best_gain = ETA_STEP
        best_coeff = 0.0
        current_loss, _ = patch_loss(model, loader, coeffs, by_id, depth, pad_id, TRUST_REGION_RHO)
        for cid in shortlist:
            sign = scored_signs[cid][1]
            trial = dict(coeffs)
            trial[cid] = ALPHA_PROBE_MAG * sign
            loss, _ = patch_loss(model, loader, trial, by_id, depth, pad_id, TRUST_REGION_RHO)
            evals += 1
            gain = current_loss - loss
            if gain > best_gain:
                best_gain = gain
                best_cid = cid
                best_coeff = ALPHA_PROBE_MAG * sign
        if best_cid is None:
            break
        support.append(best_cid)
        coeffs[best_cid] = best_coeff
    _ = base_loss
    return support, coeffs, evals


def line_search(
    model: HermesHRMClassifier,
    loader: DataLoader,
    support: list[str],
    coeffs: dict[str, float],
    scored_signs: dict[str, tuple[float, float]],
    by_id: dict[str, Component],
    depth: int,
    pad_id: int,
) -> tuple[dict[str, float], int, bool]:
    coeffs = dict(coeffs)
    evals = 0
    history: list[dict[str, float]] = []
    for _sweep in range(LINE_SEARCH_SWEEPS):
        for cid in support:
            sign = scored_signs[cid][1]
            best_coeff = coeffs[cid]
            best_loss, _ = patch_loss(model, loader, coeffs, by_id, depth, pad_id, TRUST_REGION_RHO)
            for magnitude in LINE_SEARCH_GRID:
                trial = dict(coeffs)
                trial[cid] = magnitude * sign
                loss, _ = patch_loss(model, loader, trial, by_id, depth, pad_id, TRUST_REGION_RHO)
                evals += 1
                if loss < best_loss:
                    best_loss = loss
                    best_coeff = magnitude * sign
            coeffs[cid] = best_coeff
        history.append(dict(coeffs))
    interaction_flag = False
    if len(history) >= 2:
        drift = sum(abs(history[-1][cid] - history[-2][cid]) for cid in support)
        interaction_flag = drift > 1e-6 + 0.1 * sum(abs(history[-1][cid]) for cid in support)
    return coeffs, evals, interaction_flag


def evaluate_patch(
    model: HermesHRMClassifier,
    family: Family,
    holdout_loader: DataLoader,
    regression_loaders: dict[str, DataLoader],
    coeffs: dict[str, float],
    by_id: dict[str, Component],
    depth: int,
    pad_id: int,
    eta: float,
) -> tuple[float, dict[str, float], float, float, bool, int]:
    evals = 0
    holdout_loss, ratios = patch_loss(model, holdout_loader, coeffs, by_id, depth, pad_id, TRUST_REGION_RHO)
    evals += 1
    holdout_gain = family.base_holdout_loss - holdout_loss
    trust_max = max(ratios.values()) if ratios else 0.0
    regression_deltas: dict[str, float] = {}
    regression_ucbs: list[float] = []
    for suite_name, suite_loader in regression_loaders.items():
        base_loss, _ = forward_losses(model, suite_loader, depth, pad_id)
        patched_loss, _ = patch_loss(model, suite_loader, coeffs, by_id, depth, pad_id, TRUST_REGION_RHO)
        evals += 2
        delta = patched_loss - base_loss
        regression_deltas[suite_name] = delta
        n = sum(len(b["bucket_labels"]) for b in suite_loader)
        ucb = delta + 1.0 / math.sqrt(max(1, n))
        regression_ucbs.append(ucb)
    regression_max_ucb = max(regression_ucbs) if regression_ucbs else 0.0
    accepted = (
        holdout_gain >= eta
        and regression_max_ucb <= ACCEPT_EPSILON
        and trust_max <= TRUST_REGION_RHO + 1e-6
    )
    return holdout_gain, regression_deltas, regression_max_ucb, trust_max, accepted, evals


def build_families(
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    model: HermesHRMClassifier,
    depth: int,
    pad_id: int,
    batch_size: int,
    seed: int,
) -> tuple[list[Family], list[str]]:
    rng = random.Random(seed)
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_env[row["source_env_name"]].append(row)
    notes: list[str] = []
    families: list[Family] = []
    for env, env_rows in sorted(by_env.items()):
        if len(env_rows) < MIN_FAMILY_ROWS:
            continue
        loader = build_loader(env_rows, payload, batch_size)
        loss, correct = forward_losses(model, loader, depth, pad_id)
        errors = sum(1 for c in correct if not c)
        if errors == 0:
            continue
        shuffled = list(env_rows)
        rng.shuffle(shuffled)
        split = len(shuffled) // 2
        proposal_rows = shuffled[:split]
        holdout_rows = shuffled[split:]
        regression_pool = [r for other, other_rows in by_env.items() if other != env for r in other_rows]
        rng.shuffle(regression_pool)
        regression_suites = {"regression_global": regression_pool[: min(64, len(regression_pool))]}
        families.append(
            Family(
                name=env,
                proposal_rows=proposal_rows,
                holdout_rows=holdout_rows,
                regression_suites=regression_suites,
            )
        )
    if len(families) < MIN_FAMILIES:
        notes.append("UNDERPOWERED_FAMILY_COUNT")
    if any(len(f.proposal_rows) < PROPOSAL_TARGET for f in families):
        notes.append("PROPOSAL_BATCH_BELOW_TARGET")
    return families, notes


def run_methods_on_family(
    model: HermesHRMClassifier,
    family: Family,
    payload: dict[str, Any],
    components: list[Component],
    by_id: dict[str, Component],
    depth: int,
    pad_id: int,
    batch_size: int,
    seed: int,
    eta: float,
) -> list[PatchResult]:
    proposal_loader = build_loader(family.proposal_rows, payload, batch_size)
    holdout_loader = build_loader(family.holdout_rows, payload, batch_size)
    regression_loaders = {
        name: build_loader(suite, payload, batch_size) for name, suite in family.regression_suites.items()
    }
    family.base_proposal_loss, proposal_correct = forward_losses(model, proposal_loader, depth, pad_id)
    family.base_holdout_loss, _ = forward_losses(model, holdout_loader, depth, pad_id)
    family.base_proposal_errors = sum(1 for c in proposal_correct if not c)

    weights, param_grads = grad_step_weights(model, proposal_loader, depth, pad_id)
    profiles = cached_loss_deltas(model, proposal_loader, components, depth, pad_id)

    lsps0 = spline_scores(profiles, weights, components)
    static_norm = static_norm_scores(components)
    grad_only = gradient_projection_scores(components, param_grads)
    damage = static_damage_scores(model, proposal_loader, components, proposal_correct, depth, pad_id)

    rng = random.Random(seed + hash(family.name) % 10_000)

    def signed_by_probe(raw: dict[str, float]) -> dict[str, tuple[float, float]]:
        ranked = sorted(raw, key=lambda cid: raw[cid], reverse=True)
        probe_ids = set(ranked[: KAPPA_MAX * SHORTLIST_B])
        signed: dict[str, tuple[float, float]] = {}
        for cid in raw:
            if cid in probe_ids:
                loss_plus, _ = patch_loss(model, proposal_loader, {cid: ALPHA_PROBE_MAG}, by_id, depth, pad_id, TRUST_REGION_RHO)
                loss_minus, _ = patch_loss(model, proposal_loader, {cid: -ALPHA_PROBE_MAG}, by_id, depth, pad_id, TRUST_REGION_RHO)
                sign = 1.0 if loss_plus <= loss_minus else -1.0
            else:
                sign = 1.0
            signed[cid] = (raw[cid], sign)
        return signed

    method_scores: dict[str, dict[str, tuple[float, float]]] = {
        "LSPS0": lsps0,
        "B1_random": {c.component_id: (rng.random(), rng.choice([-1.0, 1.0])) for c in components},
        "B2_static_norm": signed_by_probe(static_norm),
        "B3_static_damage": signed_by_probe(damage),
        "B4_gradient_only": grad_only,
    }

    results: list[PatchResult] = []
    for method in METHODS:
        scored = method_scores[method]
        support, coeffs, greedy_evals = greedy_select(
            model, family, proposal_loader, scored, by_id, depth, pad_id
        )
        result = PatchResult(method=method, family=family.name, support=support)
        result.signs = {cid: scored[cid][1] for cid in support}
        if support:
            coeffs, ls_evals, interaction = line_search(
                model, proposal_loader, support, coeffs, scored, by_id, depth, pad_id
            )
            result.interaction_flag = interaction
            proposal_loss, _ = patch_loss(model, proposal_loader, coeffs, by_id, depth, pad_id, TRUST_REGION_RHO)
            result.proposal_gain = family.base_proposal_loss - proposal_loss
            holdout_gain, reg_deltas, reg_ucb, trust_max, accepted, eval_evals = evaluate_patch(
                model, family, holdout_loader, regression_loaders, coeffs, by_id, depth, pad_id, eta
            )
            result.holdout_gain = holdout_gain
            result.regression_deltas = reg_deltas
            result.regression_max_ucb = reg_ucb
            result.trust_region_max = trust_max
            result.accepted = accepted
            result.evals = greedy_evals + ls_evals + eval_evals
        else:
            result.evals = greedy_evals
        result.coeffs = coeffs
        results.append(result)
    return results


def summarize(results: list[PatchResult], families: list[Family]) -> dict[str, Any]:
    by_family: dict[str, dict[str, PatchResult]] = defaultdict(dict)
    for result in results:
        by_family[result.family][result.method] = result
    beats_b3 = {family: 0 for family in by_family}
    lsps0_accept = 0
    b3_accept = 0
    any_accept_families = 0
    for family, methods in by_family.items():
        lsps0 = methods.get("LSPS0")
        b3 = methods.get("B3_static_damage")
        if lsps0 and b3 and lsps0.holdout_gain > b3.holdout_gain:
            beats_b3[family] = 1
        if lsps0 and lsps0.accepted:
            lsps0_accept += 1
        if b3 and b3.accepted:
            b3_accept += 1
        if any(r.accepted for r in methods.values()):
            any_accept_families += 1
    n_families = len(by_family)
    beat_count = sum(beats_b3.values())
    return {
        "n_families": n_families,
        "lsps0_beats_b3_families": beat_count,
        "lsps0_accept_families": lsps0_accept,
        "b3_accept_families": b3_accept,
        "any_accept_families": any_accept_families,
        "beats_b3_by_family": beats_b3,
    }


def decide_label(summary: dict[str, Any], families: list[Family], notes: list[str]) -> str:
    n = summary["n_families"]
    if n < MIN_FAMILIES or "UNDERPOWERED_FAMILY_COUNT" in notes or "PROPOSAL_BATCH_BELOW_TARGET" in notes:
        return "UNDERPOWERED_LSPS0"
    if summary["any_accept_families"] < 3:
        return "COORDINATE_SYSTEM_VOID"
    lsps0 = summary["lsps0_accept_families"]
    b3 = summary["b3_accept_families"]
    beats = summary["lsps0_beats_b3_families"]
    negative = sum(
        1 for fam, b in summary["beats_b3_by_family"].items() if b == 0
    )
    if negative >= 3 and lsps0 <= b3:
        return "SPLINE_PROPOSAL_NEGATIVE"
    if beats >= 4 and lsps0 >= 1.5 * max(1, b3):
        return "LSPS0_SPLINE_PROPOSAL_POSITIVE"
    return "SPLINE_PROPOSAL_NEGATIVE"


def render_table(results: list[PatchResult], summary: dict[str, Any]) -> list[str]:
    header = (
        "| family | method | accepted | holdout_gain | regression_max_ucb | support_size "
        "| trust_region_max | evals | beats_B3 | interaction_flag |"
    )
    sep = "| ------ | ------ | -------- | -----------: | -----------------: | -----------: | ---------------: | ----: | -------- | ---------------- |"
    lines = [header, sep]
    beats = summary["beats_b3_by_family"]
    for result in sorted(results, key=lambda r: (r.family, r.method)):
        beats_b3 = "yes" if result.method == "LSPS0" and beats.get(result.family) == 1 else (
            "—" if result.method != "LSPS0" else "no"
        )
        lines.append(
            f"| {result.family} | {result.method} | {result.accepted} | "
            f"{result.holdout_gain:+.4f} | {result.regression_max_ucb:+.4f} | {len(result.support)} | "
            f"{result.trust_region_max:.4f} | {result.evals} | {beats_b3} | {result.interaction_flag} |"
        )
    return lines


def write_blocker(label: str, discriminator_label: str | None, report_path: Path) -> None:
    lines = [
        "# LSPS-0: Loop-Spline Patch Search v0",
        "",
        f"Final label: `{label}`",
        "",
        "## Blocker",
        "",
        "LSPS-0 was **not run for evidence** because the loop-spline discriminator "
        "precondition was not satisfied.",
        "",
        f"- Discriminator final label: `{discriminator_label}`",
        "- Blocking labels: `PROFILE_UNSTABLE`, `STATIC_ENDPOINT_SUFFICIENT`, "
        "`UNDERPOWERED_PROBES`, `BLOCKED_NO_RECURSION_TRACE`.",
        "",
        "Per the LSPS-0 protocol, when the discriminator label is blocking we write this "
        "blocker section and do not claim LSPS-0 evidence. No patch-search comparison "
        "between spline-derived and static-derived proposal scores is asserted here.",
        "",
        "To run the machinery anyway for descriptive (non-evidential) numbers, re-run with "
        "`--override`. The result will still be reported honestly and cannot claim a spline "
        "advantage.",
        "",
        "No actual Goodfire VPD is claimed. Cached 4D ARC heads were not used. The base "
        "model was not updated; no policy, RCPI, or consolidation was built.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(
    label: str,
    discriminator_label: str | None,
    summary: dict[str, Any],
    results: list[PatchResult],
    families: list[Family],
    notes: list[str],
    eta: float,
    eta_note: str,
) -> str:
    lines = [
        "# LSPS-0: Loop-Spline Patch Search v0",
        "",
        f"Final label: `{label}`",
        "",
        f"Discriminator final label: `{discriminator_label}` "
        f"({'override active' if discriminator_label in BLOCKING_DISCRIMINATOR_LABELS else 'precondition path'})",
        "",
        "VPD-style/SVD rank-one component dictionary over a frozen TRM. No actual Goodfire "
        "VPD is claimed; no cached 4D ARC heads; base model never updated; no policy/RCPI/"
        "consolidation; this is not self-improvement.",
        "",
        "## Setup",
        "",
        f"- Families (source_env with >= {MIN_FAMILY_ROWS} rows and >= 1 model error): "
        f"`{summary['n_families']}`",
        f"- Required families: `{MIN_FAMILIES}`; proposal target: `{PROPOSAL_TARGET}` examples/family",
        f"- kappa_max=`{KAPPA_MAX}`, shortlist B=`{SHORTLIST_B}`, alpha_probe_mag=`{ALPHA_PROBE_MAG}`",
        f"- Trust region rho=`{TRUST_REGION_RHO}`, accept delta(eta)=`{eta:.4f}` ({eta_note}), "
        f"accept epsilon_j=`{ACCEPT_EPSILON}`",
        f"- Power notes: `{notes or ['none']}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Results",
        "",
    ]
    lines.extend(render_table(results, summary))
    lines.extend(
        [
            "",
            "## Honesty gates",
            "",
            "- B4 gradient-only is implemented via rank-one projection u^T (dL/dW) v "
            "(not blocked).",
            "- g_t(tau) uses the loss-gradient channel (per-step pooled-state gradient norm); "
            "full Psi gradients are not exposed by this checkpoint. Reported as a limitation.",
            "- Spline advantage is only claimed if LSPS-0 beats B3 static-damage; see final label.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    report_path, results_path = output_paths(args.tag)
    discriminator_label = read_discriminator_label()
    blocked = discriminator_label in BLOCKING_DISCRIMINATOR_LABELS or discriminator_label is None

    if blocked and not args.override:
        label = "UNDERPOWERED_LSPS0"
        write_blocker(label, discriminator_label, report_path)
        write_json(
            results_path,
            {
                "final_label": label,
                "blocked": True,
                "discriminator_label": discriminator_label,
                "reason": "discriminator precondition not satisfied; LSPS-0 evidence gated",
                "claims_actual_vpd": False,
                "uses_cached_4d_arc_heads": False,
                "base_model_updated": False,
            },
        )
        print(json.dumps({"final_label": label, "blocked": True, "report": str(report_path)}, indent=2))
        return 0

    payload, model = load_trained_model(str(args.model), "cpu")
    if not isinstance(model, HermesHRMClassifier) or isinstance(model, ConstellationGovernorHermesClassifier):
        label = "BLOCKED_NO_GRADIENT_PROFILE"
        write_blocker(label, discriminator_label, report_path)
        write_json(results_path, {"final_label": label, "reason": type(model).__name__})
        return 0

    pad_id = int(payload["vocab"]["<PAD>"])
    rows = load_all_rows(args.train, args.eval)
    components = build_components(model, args.rank_atoms_per_module)[: args.max_candidates]
    by_id = {c.component_id: c for c in components}

    families, notes = build_families(rows, payload, model, args.base_depth, pad_id, args.batch_size, args.seed)
    if args.max_families > 0:
        families = families[: args.max_families]
        notes = list(dict.fromkeys(notes + ["FAMILY_CAP_APPLIED"]))
    eta = ACCEPT_DELTA
    eta_note = "NO_LORA_YARDSTICK fallback (no matched LoRA baseline available)"

    all_results: list[PatchResult] = []
    for family_index, family in enumerate(families):
        print(f"[lsps0] family {family_index + 1}/{len(families)}: {family.name}", flush=True)
        all_results.extend(
            run_methods_on_family(
                model, family, payload, components, by_id, args.base_depth, pad_id, args.batch_size, args.seed, eta
            )
        )

    summary = summarize(all_results, families)
    label = decide_label(summary, families, notes)
    if discriminator_label in BLOCKING_DISCRIMINATOR_LABELS:
        label = "UNDERPOWERED_LSPS0"
        notes = list(dict.fromkeys(notes + ["DISCRIMINATOR_OVERRIDE_NO_EVIDENCE"]))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(label, discriminator_label, summary, all_results, families, notes, eta, eta_note),
        encoding="utf-8",
    )
    write_json(
        results_path,
        {
            "final_label": label,
            "discriminator_label": discriminator_label,
            "override": args.override,
            "notes": notes,
            "eta": eta,
            "eta_note": eta_note,
            "summary": summary,
            "families": [
                {
                    "name": f.name,
                    "proposal_rows": len(f.proposal_rows),
                    "holdout_rows": len(f.holdout_rows),
                    "regression_rows": {k: len(v) for k, v in f.regression_suites.items()},
                    "base_proposal_loss": f.base_proposal_loss,
                    "base_holdout_loss": f.base_holdout_loss,
                    "base_proposal_errors": f.base_proposal_errors,
                }
                for f in families
            ],
            "results": [
                {
                    "method": r.method,
                    "family": r.family,
                    "support_size": len(r.support),
                    "support": r.support,
                    "coeffs": r.coeffs,
                    "proposal_gain": r.proposal_gain,
                    "holdout_gain": r.holdout_gain,
                    "regression_deltas": r.regression_deltas,
                    "regression_max_ucb": r.regression_max_ucb,
                    "trust_region_max": r.trust_region_max,
                    "evals": r.evals,
                    "interaction_flag": r.interaction_flag,
                    "accepted": r.accepted,
                }
                for r in all_results
            ],
            "claims_actual_vpd": False,
            "uses_cached_4d_arc_heads": False,
            "base_model_updated": False,
        },
    )
    print(json.dumps({"final_label": label, "report": str(report_path), "n_families": summary["n_families"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

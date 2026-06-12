"""Best-of-N + extractor comparison CLI (roadmap items 3 & 4).

Over powered cells: report clean-subpatch existence and best-of-N search success
(N in {4,16,64}, sparsity-stratified, with norm-matched random controls), then
compare the greedy entry extractor against best-of-N at matched eval budget via a
single pooled sign test.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from param_decomp.experiments.trm_basis_constrained_edits.cell_base import load_pooled_suites
from param_decomp.experiments.trm_basis_constrained_edits.cell_search import (
    DEFAULT_REPEATS,
    N_GRID,
    CellSearch,
    PooledExtractorTest,
    build_search_base,
)
from param_decomp.experiments.trm_basis_constrained_edits.common import write_json

DEFAULT_SCORE_ROOT = Path(r"D:\Research_Engine\runs\eval_power_harness")
DEFAULT_SUITES = {
    "arc_challenge": DEFAULT_SCORE_ROOT
    / "powered_arc_challenge_1024_seed23"
    / "choice_constrained_sample_scores.jsonl",
    "arc_easy": DEFAULT_SCORE_ROOT
    / "powered_arc_easy_1024_seed23"
    / "choice_constrained_sample_scores.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default=None, help="Comma-separated suite=score_file pairs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCORE_ROOT / "cell_search")
    parser.add_argument("--min-target", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--report-path", type=Path, default=Path("reports") / "cell_search_report.md"
    )
    return parser.parse_args()


def _parse_suites(raw: str | None) -> dict[str, Path]:
    if raw is None:
        return {key: path for key, path in DEFAULT_SUITES.items() if path.exists()}
    suites: dict[str, Path] = {}
    for index, part in enumerate(raw.split(",")):
        part = part.strip()
        if not part:
            continue
        key, value = part.split("=", 1) if "=" in part else (f"suite{index}", part)
        suites[key.strip()] = Path(value.strip())
    return suites


def build_report(
    cells: tuple[CellSearch, ...], pooled: PooledExtractorTest, suites: dict[str, Path]
) -> str:
    lines = [
        "# Best-of-N + Extractor Report (roadmap items 3 & 4)",
        "",
        f"Suites: {', '.join(suites)}. Cells: {len(cells)}. Search space: ΔW entry "
        "coordinates (rank-1 E_ij atoms). A clean subpatch retains >=50% of the raw "
        "target lift with zero measured regression.",
        "",
        "## Item 3 — best-of-N existence and success rate",
        "",
        "Per cell and sparsity stratum (subset L0): existence of a strict zero-"
        "regression clean subpatch and of a regression-reducing one (<=0.5x raw), "
        f"and the best-of-N success rate at N in {list(N_GRID)} for the regression-"
        "reducing criterion (control = norm-matched random entries).",
        "",
        "| cell | L0 | clean | reducing | "
        + " | ".join(f"succ@{n}" for n in N_GRID)
        + " | "
        + " | ".join(f"ctrl@{n}" for n in N_GRID)
        + " |",
        "|---|---:|:--:|:--:|" + "---:|" * (2 * len(N_GRID)),
    ]
    for cell in cells:
        for support in cell.supports:
            succ = " | ".join(f"{support.success_rate_by_n[n]:.2f}" for n in N_GRID)
            ctrl = " | ".join(f"{support.control_success_rate_by_n[n]:.2f}" for n in N_GRID)
            lines.append(
                f"| {cell.family_key} | {support.support} | "
                f"{'yes' if support.clean_exists else 'no'} | "
                f"{'yes' if support.improved_exists else 'no'} | {succ} | {ctrl} |"
            )

    lines += [
        "",
        "## Item 4 — extractor vs best-of-N at matched eval budget",
        "",
        "| cell | extractor regression | extractor evals | best-of-budget regression | "
        "extractor − best-of-N |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell.family_key} | {cell.extractor.regression:.4f} | "
            f"{cell.extractor.eval_budget} | {cell.matched_best_of_n_regression:.4f} | "
            f"{cell.extractor_minus_best_of_n:.4f} |"
        )
    lines += [
        "",
        "## Pooled verdict",
        "",
        f"Cells compared: {pooled.n_cells}. Extractor better: {pooled.extractor_better}; "
        f"best-of-N better: {pooled.best_of_n_better}; ties: {pooled.ties}. "
        f"Median (extractor − best-of-N) regression: "
        f"{pooled.median_extractor_minus_best_of_n}. Two-sided sign-test p = "
        f"{pooled.sign_test_p}.",
        "",
        f"**{pooled.verdict}**",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suites = _parse_suites(args.scores)
    assert suites, "no suite score files found"
    pooled_samples = load_pooled_suites(suites)
    cells, pooled = build_search_base(
        pooled_samples, min_target=args.min_target, repeats=args.repeats
    )
    write_json(
        args.out_dir / "cell_search.json",
        {
            "suites": {key: str(value) for key, value in suites.items()},
            "pooled_extractor_test": asdict(pooled),
            "cells": [asdict(cell) for cell in cells],
        },
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(build_report(cells, pooled, suites), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_cells": pooled.n_cells,
                "verdict": pooled.verdict,
                "sign_test_p": pooled.sign_test_p,
                "report": str(args.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

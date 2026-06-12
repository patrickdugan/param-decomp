"""Powered cell-base builder CLI (roadmap item 2).

Pools scored suites, builds eval-powered `top_target` cells, trains an ordinary
low-rank adapter per cell, and counts regression-positive (and bootstrap-
recurrent) cells. Goal: >= 20 raw-LoRA regression-positive cells, to resolve
whether the single `B->A` cell is representative or lucky.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from param_decomp.experiments.trm_basis_constrained_edits.cell_base import (
    DEFAULT_BOOTSTRAPS,
    build_cell_base,
    load_pooled_suites,
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
    parser.add_argument(
        "--scores",
        default=None,
        help="Comma-separated suite=score_file pairs. Defaults to the powered "
        "ARC-Challenge + ARC-Easy slices.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCORE_ROOT / "cell_base")
    parser.add_argument("--min-target", type=int, default=50)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--report-path", type=Path, default=Path("reports") / "cell_base_report.md")
    return parser.parse_args()


def _parse_suites(raw: str | None) -> dict[str, Path]:
    if raw is None:
        return {key: path for key, path in DEFAULT_SUITES.items() if path.exists()}
    suites: dict[str, Path] = {}
    for index, part in enumerate(raw.split(",")):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        else:
            key, value = f"suite{index}", part
        suites[key.strip()] = Path(value.strip())
    return suites


def build_report(report: object, suites: dict[str, Path], min_target: int) -> str:
    from param_decomp.experiments.trm_basis_constrained_edits.cell_base import CellBaseReport

    assert isinstance(report, CellBaseReport)
    lines = [
        "# Powered Cell-Base Report (roadmap item 2)",
        "",
        f"Suites pooled: {', '.join(suites)} ({report.n_pooled_samples} samples).",
        f"Cell granularity: `top_target` (gate-compatible), powered at "
        f">={min_target} target/family.",
        "",
        f"Powered families: {report.n_powered_families}. Cells: {report.n_cells}. "
        f"Regression-positive: {report.n_regression_positive}. "
        f"Bootstrap-recurrent regression-positive: "
        f"{report.n_recurrent_regression_positive}.",
        "",
        "| cell | n_target | n_nontarget | raw target Δ | raw non-target Δ | damage | "
        "regression+ | bootstrap recurrence |",
        "|---|---:|---:|---:|---:|---:|:--:|---:|",
    ]
    for cell in report.cells:
        lines.append(
            f"| {cell.family_key} | {cell.n_target} | {cell.n_nontarget} | "
            f"{cell.raw_target_delta:.4f} | {cell.raw_non_target_delta:.4f} | "
            f"{cell.raw_damage_count} | "
            f"{'yes' if cell.regression_positive else 'no'} | "
            f"{cell.bootstrap_recurrence:.3f} |"
        )
    target_met = report.n_regression_positive >= 20
    lines += [
        "",
        "## Verdict",
        "",
        (
            f"{report.n_regression_positive} regression-positive cells >= 20 target: "
            "cell base is sufficient to test representativeness."
            if target_met
            else f"{report.n_regression_positive} regression-positive cells < 20. "
            "Add more suites (mmlu_formal_logic, gsm8k choice variants) or lower the "
            "granularity floor."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suites = _parse_suites(args.scores)
    assert suites, "no suite score files found"
    pooled = load_pooled_suites(suites)
    report = build_cell_base(
        pooled,
        min_target=args.min_target,
        rank=args.rank,
        steps=args.steps,
        lr=args.lr,
        n_bootstraps=args.bootstraps,
    )
    write_json(
        args.out_dir / "cell_base.json",
        {"suites": {key: str(value) for key, value in suites.items()}, **asdict(report)},
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(build_report(report, suites, args.min_target), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_pooled_samples": report.n_pooled_samples,
                "n_powered_families": report.n_powered_families,
                "n_regression_positive": report.n_regression_positive,
                "n_recurrent_regression_positive": report.n_recurrent_regression_positive,
                "report": str(args.report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Eval-power harness CLI (roadmap item 1).

Measures the achieved per-family resolution of the cached constrained-choice
lane, projects how many freshly scored items reach >=50 target failures per
family at each granularity, and emits a ready-to-run powered card slice plus the
exact constrained-decode probe invocations. It does NOT load the model or run
scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from param_decomp.experiments.trm_basis_constrained_edits.common import (
    DEFAULT_SEED_SCORE_FILES,
    load_seed_samples,
    pooled_samples,
    write_json,
    write_jsonl,
)
from param_decomp.experiments.trm_basis_constrained_edits.eval_power import (
    GRANULARITIES,
    MAX_RESOLVABLE_DELTA,
    MIN_NONTARGET,
    MIN_TARGET_PER_FAMILY,
    PowerReport,
    ScaleProjection,
    build_challenge_cards,
    measure_power,
    probe_commands,
    project_scale,
)

DEFAULT_PARQUET = Path(
    r"D:\Research_Engine\prime_envs\arc\ARC-Challenge\train-00000-of-00001.parquet"
)
DEFAULT_TESSERACT_ROOT = Path(r"C:\projects\Tesseract\Tesseract")
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\eval_power_harness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--tesseract-root", type=Path, default=DEFAULT_TESSERACT_ROOT)
    parser.add_argument("--env-id", default="arc_challenge")
    parser.add_argument("--planned-items", type=int, default=256)
    parser.add_argument("--seeds", default="23,37,101")
    parser.add_argument(
        "--report-path", type=Path, default=Path("reports") / "eval_power_report.md"
    )
    parser.add_argument(
        "--skip-cards",
        action="store_true",
        help="Skip materializing the card slice (measure + project only).",
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="Comma-separated freshly scored choice_constrained_sample_scores.jsonl "
        "files to measure instead of the cached default lane (the post-run "
        "validation path).",
    )
    return parser.parse_args()


def _fmt_delta(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.4f}"


def _artifact_lines(
    card_path: Path | None, spec_path: Path | None, planned_items: int, seeds: list[int]
) -> list[str]:
    if card_path is None or spec_path is None:
        return []
    return [
        "",
        "## Emitted scoring artifacts",
        "",
        f"- Card slice: `{card_path}` ({planned_items} cards)",
        f"- Scoring spec: `{spec_path}` (seeds {seeds})",
        "",
        "Run each per-seed command in the spec under the existing memory-cap "
        "wrapper, then re-run this harness with `--scores <score_file>` to validate.",
    ]


def build_report(
    *,
    source_label: str,
    measured: bool,
    pooled_unique_items: int,
    pooled_fail_rate: float,
    pooled_reports: list[PowerReport],
    projections: list[ScaleProjection],
    planned_items: int,
    seeds: list[int],
    card_path: Path | None,
    spec_path: Path | None,
) -> str:
    claim = (
        f"Claim boundary: per-family resolution measured on a freshly scored set "
        f"({source_label}); projection columns are advisory."
        if measured
        else "Claim boundary: measurement + projection only. No model was loaded; "
        "the powered probe sets still require running the constrained-decode probe."
    )
    section = "## Measured resolution" if measured else "## Cached lane resolution (current)"
    lines = [
        "# Eval-Power Harness Report (roadmap item 1)",
        "",
        f"Thresholds: >={MIN_TARGET_PER_FAMILY} target/family, >={MIN_NONTARGET} "
        f"non-target/family, resolvable delta <= {MAX_RESOLVABLE_DELTA}.",
        "",
        f"Source: {source_label}.",
        "",
        claim,
        "",
        section,
        "",
        f"Pooled unique items: {pooled_unique_items}; pooled fail rate: {pooled_fail_rate:.3f}.",
        "",
        "| granularity | families | target-powered | fully-powered | worst resolvable delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for report in pooled_reports:
        lines.append(
            f"| {report.granularity} | {report.n_families} | "
            f"{report.n_families_target_powered} | "
            f"{report.n_families_fully_powered} | "
            f"{_fmt_delta(report.worst_resolvable_target_delta)} |"
        )

    lines += [
        "",
        f"## Projection to {planned_items} scored items",
        "",
        "Estimator: expected_target = n_items * cached_fail_rate * family_share. "
        "Seeds replicate the same items for cross-seed recurrence and do not add "
        "unique target samples; item count is the power driver.",
        "",
        "| granularity | families | req. items (top family) | req. items (all >=2) | "
        f"{planned_items} powers top? | {planned_items} powers all? |",
        "|---|---:|---:|---:|:--:|:--:|",
    ]
    for proj in projections:
        lines.append(
            f"| {proj.granularity} | {proj.n_families} | "
            f"{proj.required_items_top_family} | "
            f"{proj.required_items_all_families} | "
            f"{'yes' if proj.planned_powers_top_family else 'NO'} | "
            f"{'yes' if proj.planned_powers_all_families else 'NO'} |"
        )

    lines += ["", "## Verdict", ""]
    if measured:
        passing = [r for r in pooled_reports if r.passes]
        if passing:
            details = "; ".join(
                f"`{r.granularity}` ({r.n_families_fully_powered}/{r.n_families} "
                f"families, worst delta {_fmt_delta(r.worst_resolvable_target_delta)})"
                for r in passing
            )
            coarsest_fail = next(
                (r for r in pooled_reports if not r.passes and r.granularity != passing[-1].granularity),
                None,
            )
            note = (
                f" Finer granularities stay underpowered on this suite "
                f"(e.g. `{coarsest_fail.granularity}` tops out at "
                f"{coarsest_fail.families[0].n_target} target/family); powering them "
                "needs a second task suite."
                if coarsest_fail and coarsest_fail.families
                else ""
            )
            lines.append(
                f"Item 1 achieved at: {details}. Define powered cells at this "
                f"granularity.{note}"
            )
        else:
            best = max(pooled_reports, key=lambda r: r.n_families_fully_powered)
            lines.append(
                f"No granularity reaches full power on this suite. Closest is "
                f"`{best.granularity}` with {best.n_families_fully_powered} fully "
                "powered families. Add a second task suite or more items."
            )
        return "\n".join(lines + _artifact_lines(card_path, spec_path, planned_items, seeds)) + "\n"

    feasible = [p for p in projections if p.planned_powers_all_families]
    if feasible:
        names = ", ".join(p.granularity for p in feasible)
        lines.append(
            f"{planned_items} items fully powers all supported families at "
            f"granularity: {names}. Use the coarsest granularity that the paper's "
            "family claims tolerate."
        )
    else:
        coarsest = projections[-1]
        fine = projections[0]
        lines.append(
            f"{planned_items} items does NOT power the slowest family at any "
            "granularity. Top-family requirement (the coarsening-sensitive number) "
            f"falls from {fine.required_items_top_family} items at "
            f"`{fine.granularity}` to {coarsest.required_items_top_family} at "
            f"`{coarsest.granularity}`; even the coarsest still exceeds "
            f"{planned_items}. The `all families` column "
            f"({coarsest.required_items_all_families}) is dominated by rare "
            "2-count tail families whose share is unreliable from only "
            f"{pooled_unique_items} cached items, so treat it as a "
            "loose upper bound. Either raise --planned-items toward the top-family "
            "requirement or restrict the cell base to the head families."
        )

    return "\n".join(lines + _artifact_lines(card_path, spec_path, planned_items, seeds)) + "\n"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]

    if args.scores:
        score_files = {
            f"scored{index}": Path(part.strip())
            for index, part in enumerate(args.scores.split(","))
            if part.strip()
        }
    else:
        score_files = DEFAULT_SEED_SCORE_FILES
    per_seed = load_seed_samples(score_files)
    pooled = pooled_samples(per_seed)

    pooled_reports: list[PowerReport] = []
    projections: list[ScaleProjection] = []
    for granularity in GRANULARITIES:
        pooled_reports.append(measure_power(pooled, granularity=granularity))
        projections.append(
            project_scale(pooled, granularity=granularity, planned_items=args.planned_items)
        )

    n_fail = sum(1 for sample in pooled if not sample.baseline_hit)
    write_json(
        args.out_dir / "eval_power_cached.json",
        {
            "pooled_unique_items": len(pooled),
            "pooled_fail_rate": n_fail / len(pooled),
            "pooled_reports": [asdict(report) for report in pooled_reports],
        },
    )
    write_json(
        args.out_dir / "eval_power_projection.json",
        {"granularities": [asdict(proj) for proj in projections]},
    )

    card_path: Path | None = None
    spec_path: Path | None = None
    if not args.skip_cards:
        cards = build_challenge_cards(
            args.parquet, n_items=args.planned_items, seed=seeds[0], env_id=args.env_id
        )
        cards_file = args.out_dir / "powered_choice_cards.jsonl"
        write_jsonl(cards_file, cards)
        commands = probe_commands(
            card_file=cards_file,
            env_id=args.env_id,
            n_items=args.planned_items,
            seeds=seeds,
            score_root=args.out_dir,
            tesseract_root=args.tesseract_root,
        )
        spec = {
            "claim_boundary": "Scoring spec only; commands must run under the "
            "existing memory-cap wrapper. Item count drives power; seeds give "
            "cross-seed recurrence.",
            "env_id": args.env_id,
            "n_items": args.planned_items,
            "seeds": seeds,
            "card_file": str(cards_file),
            "min_target_per_family": MIN_TARGET_PER_FAMILY,
            "min_nontarget": MIN_NONTARGET,
            "probe_commands": commands,
        }
        spec_file = args.out_dir / "eval_power_scoring_spec.json"
        write_json(spec_file, spec)
        card_path = cards_file
        spec_path = spec_file

    source_label = (
        ", ".join(str(path) for path in score_files.values())
        if args.scores
        else "cached default lane (DEFAULT_SEED_SCORE_FILES)"
    )
    report_text = build_report(
        source_label=source_label,
        measured=bool(args.scores),
        pooled_unique_items=len(pooled),
        pooled_fail_rate=n_fail / len(pooled),
        pooled_reports=pooled_reports,
        projections=projections,
        planned_items=args.planned_items,
        seeds=seeds,
        card_path=card_path,
        spec_path=spec_path,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "report": str(args.report_path),
                "pooled_unique_items": len(pooled),
                "planned_items": args.planned_items,
                "feasible_granularities": [
                    proj.granularity for proj in projections if proj.planned_powers_all_families
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

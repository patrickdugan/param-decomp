"""Leave-one-score-file-out validation for conditional gain policies."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import (
    dedupe_samples,
    parse_floats,
    parse_paths,
    read_jsonl,
    write_jsonl,
)
from param_decomp.experiments.trm_gain_policy_condition_miner import (
    candidate_policies,
    score_policy,
)


DEFAULT_SCORE_FILES = (
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604\choice_constrained_sample_scores.jsonl;"
    r"D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604\choice_constrained_sample_scores.jsonl"
)
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_condition_cv")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate mined conditional gain policies by held-out score file.")
    parser.add_argument("--score-files", default=DEFAULT_SCORE_FILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--penalties", default="0.25,0.5,1")
    parser.add_argument("--margins", default="0.25,0.5,1")
    parser.add_argument("--touch-penalty", type=float, default=0.05)
    parser.add_argument("--complexity-penalty", type=float, default=0.003)
    parser.add_argument("--max-prompt-tokens", type=int, default=1200)
    return parser.parse_args()


def load_file(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def rank_conditions(
    samples: list[dict[str, Any]],
    *,
    penalties: list[float],
    margins: list[float],
    touch_penalty: float,
    complexity_penalty: float,
) -> list[dict[str, Any]]:
    policies = candidate_policies(samples, penalties, margins)
    rows = [score_policy(samples, policy, touch_penalty=touch_penalty, complexity_penalty=complexity_penalty) for policy in policies]
    rows.sort(key=lambda row: (row["accepted"], row["efficiency_reward"], row["reward"], -row["touch_rate"]), reverse=True)
    return rows


def best_fixed_control(samples: list[dict[str, Any]], *, penalties: list[float], touch_penalty: float, complexity_penalty: float) -> dict[str, Any] | None:
    actions = sorted({str(score["action"]) for sample in samples for score in sample.get("scores", [])})
    controls = []
    for action in actions:
        for penalty in penalties:
            controls.append(
                score_policy(
                    samples,
                    {"type": "fixed_action_penalty", "action": action, "penalty": penalty},
                    touch_penalty=touch_penalty,
                    complexity_penalty=complexity_penalty,
                )
            )
    controls.sort(key=lambda row: (row["reward"], row["delta"], -row["touch_rate"]), reverse=True)
    return controls[0] if controls else None


def fold_result(
    *,
    heldout_path: Path,
    train_paths: list[Path],
    penalties: list[float],
    margins: list[float],
    touch_penalty: float,
    complexity_penalty: float,
) -> dict[str, Any]:
    train_samples = dedupe_samples([sample for path in train_paths for sample in load_file(path)])
    heldout_samples = dedupe_samples(load_file(heldout_path))
    ranked = rank_conditions(
        train_samples,
        penalties=penalties,
        margins=margins,
        touch_penalty=touch_penalty,
        complexity_penalty=complexity_penalty,
    )
    train_best = ranked[0] if ranked else None
    heldout_score = (
        score_policy(
            heldout_samples,
            train_best["policy"],
            touch_penalty=touch_penalty,
            complexity_penalty=complexity_penalty,
        )
        if train_best
        else None
    )
    control = best_fixed_control(
        heldout_samples,
        penalties=penalties,
        touch_penalty=touch_penalty,
        complexity_penalty=complexity_penalty,
    )
    beats_control = bool(
        heldout_score
        and control
        and heldout_score["accepted"]
        and float(heldout_score["reward"]) > float(control["reward"])
        and float(heldout_score["delta"]) >= float(control["delta"])
    )
    ties_control_lower_touch = bool(
        heldout_score
        and control
        and heldout_score["accepted"]
        and float(heldout_score["reward"]) == float(control["reward"])
        and float(heldout_score["delta"]) == float(control["delta"])
        and float(heldout_score["touch_rate"]) < float(control["touch_rate"])
    )
    return {
        "fold_id": heldout_path.parent.name,
        "heldout_score_file": str(heldout_path),
        "train_score_files": [str(path) for path in train_paths],
        "train_sample_count": len(train_samples),
        "heldout_sample_count": len(heldout_samples),
        "train_best_candidate_id": train_best["candidate_id"] if train_best else None,
        "train_best_delta": train_best["delta"] if train_best else None,
        "train_best_reward": train_best["reward"] if train_best else None,
        "train_best_efficiency_reward": train_best["efficiency_reward"] if train_best else None,
        "heldout_delta": heldout_score["delta"] if heldout_score else None,
        "heldout_reward": heldout_score["reward"] if heldout_score else None,
        "heldout_efficiency_reward": heldout_score["efficiency_reward"] if heldout_score else None,
        "heldout_rescue_count": heldout_score["rescue_count"] if heldout_score else 0,
        "heldout_damage_count": heldout_score["damage_count"] if heldout_score else 0,
        "heldout_touch_rate": heldout_score["touch_rate"] if heldout_score else None,
        "heldout_accepted": heldout_score["accepted"] if heldout_score else False,
        "best_control_candidate_id": control["candidate_id"] if control else None,
        "best_control_delta": control["delta"] if control else None,
        "best_control_reward": control["reward"] if control else None,
        "best_control_touch_rate": control["touch_rate"] if control else None,
        "beats_control": beats_control,
        "ties_control_lower_touch": ties_control_lower_touch,
        "claim_boundary": "Held-out cached score validation for controller predicates; not a VPD feature edit.",
    }


def compact_packet(summary: dict[str, Any], best_fold: dict[str, Any] | None) -> str:
    lines = [
        "TASK: Decide whether conditional gain policy generalizes across held-out score files.",
        "CURRENT STATE:",
        f"- run: {summary['run_dir']}",
        f"- folds: {summary['fold_count']}",
        f"- heldout accepted folds: {summary['heldout_accepted_fold_count']}",
        f"- beats-control folds: {summary['beats_control_fold_count']}",
        f"- tie-lower-touch folds: {summary['ties_control_lower_touch_fold_count']}",
        "BEST HELDOUT FOLD:",
        f"- fold: {best_fold.get('fold_id') if best_fold else None}",
        f"- condition: {best_fold.get('train_best_candidate_id') if best_fold else None}",
        f"- heldout delta/reward: {best_fold.get('heldout_delta') if best_fold else None}/{best_fold.get('heldout_reward') if best_fold else None}",
        f"- rescues/damages: {best_fold.get('heldout_rescue_count') if best_fold else 0}/{best_fold.get('heldout_damage_count') if best_fold else 0}",
        f"- touch_rate: {best_fold.get('heldout_touch_rate') if best_fold else None}",
        "NEXT ACTION:",
        "- If heldout beats-control folds exist, map that predicate to VPD feature candidates.",
        "- Otherwise collect fresh score cards before feature promotion.",
    ]
    return "\n".join(lines) + "\n"


def run_cv(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_files = parse_paths(args.score_files)
    penalties = parse_floats(args.penalties)
    margins = parse_floats(args.margins)
    folds = []
    for heldout in score_files:
        train = [path for path in score_files if path != heldout]
        if train:
            folds.append(
                fold_result(
                    heldout_path=heldout,
                    train_paths=train,
                    penalties=penalties,
                    margins=margins,
                    touch_penalty=args.touch_penalty,
                    complexity_penalty=args.complexity_penalty,
                )
            )
    folds.sort(
        key=lambda row: (
            row["beats_control"],
            row["ties_control_lower_touch"],
            row["heldout_accepted"],
            row["heldout_reward"] if row["heldout_reward"] is not None else -999.0,
            -(row["heldout_touch_rate"] if row["heldout_touch_rate"] is not None else 999.0),
        ),
        reverse=True,
    )
    best_fold = folds[0] if folds else None
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "score_files": [str(path) for path in score_files],
        "fold_count": len(folds),
        "heldout_accepted_fold_count": sum(int(row["heldout_accepted"]) for row in folds),
        "beats_control_fold_count": sum(int(row["beats_control"]) for row in folds),
        "ties_control_lower_touch_fold_count": sum(int(row["ties_control_lower_touch"]) for row in folds),
        "best_fold_id": best_fold["fold_id"] if best_fold else None,
        "best_heldout_delta": best_fold["heldout_delta"] if best_fold else None,
        "best_heldout_reward": best_fold["heldout_reward"] if best_fold else None,
        "claim_boundary": "Leave-one-score-file-out cached validation only; selected predicates remain controller priors.",
        "outputs": {
            "summary": str(args.out_dir / "condition_cv_summary.json"),
            "folds": str(args.out_dir / "condition_cv_folds.jsonl"),
            "prompt_packet": str(args.out_dir / "prompt_packet.txt"),
        },
    }
    write_jsonl(args.out_dir / "condition_cv_folds.jsonl", folds)
    packet = compact_packet(summary, best_fold)
    (args.out_dir / "prompt_packet.txt").write_text(packet, encoding="utf-8")
    summary["prompt_packet_est_tokens"] = len(packet) // 4
    (args.out_dir / "condition_cv_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_cv(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

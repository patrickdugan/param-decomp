"""Build a powered Hermes-format Intellect-3 logic corpus with bucket variation.

Adapts the normalized Intellect-3 logic trajectories into primehub trace-shape
RAW rows the Hermes gym consumes (gym_make_dataset.build_dataset_from_config ->
prepare_rows -> train_hrm_hermes). The source file is correct-solutions-only
(all reward=1.0), which gives no failures. To make the critic task non-trivial,
we manufacture bucket variation by perturbing each known solution -- the same
methodology behind the existing 214-row augmented logic file:

- exact_positive: model_action = the true solution
- weak_positive:  model_action = lightly corrupted solution (near miss)
- negative:       model_action = heavily corrupted solution or a foreign solution

Families are `action.task` puzzle types. arc_agi is EXCLUDED (no ARC). No actual
Goodfire VPD is claimed; no cached 4D ARC heads are used.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HRM_REPO = Path(r"D:\projects\HRM-re")
if str(HRM_REPO) not in sys.path:
    sys.path.insert(0, str(HRM_REPO))

from experiments.hermes_skill_gym.hermes_dataset import write_jsonl  # noqa: E402

SOURCE = Path(r"D:\Research_Engine\tesseract_persistent\data\normalized_trajectories\intellect_3_logic.jsonl")
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "powered_int3_logic"
EXCLUDED_FAMILIES = {"arc_agi"}
SOLUTION_KEYS = ("grid", "solution", "answer", "result", "candidates")
ALPHABET = string.ascii_letters + string.digits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-items-per-family", type=int, default=220)
    parser.add_argument("--weak-frac", type=float, default=0.10)
    parser.add_argument("--neg-frac", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def compact_solution(game_data: dict[str, Any]) -> str:
    metadata = game_data.get("metadata") or {}
    for key in SOLUTION_KEYS:
        if key in metadata and metadata[key] not in (None, "", []):
            return json.dumps(metadata[key])[:400]
    answer = game_data.get("answer")
    return json.dumps(answer)[:400] if answer else ""


def corrupt(text: str, frac: float, rng: random.Random) -> str:
    chars = list(text)
    positions = [i for i, c in enumerate(chars) if c.isalnum()]
    rng.shuffle(positions)
    for i in positions[: int(len(positions) * frac)]:
        chars[i] = rng.choice(ALPHABET)
    return "".join(chars)


def base_row(family: str, task: str, observation: str, model_action: str, bucket: str) -> dict[str, Any]:
    has_target = bucket != "negative"
    return {
        "source_env_name": family,
        "source_env_type": "intellect_3_logic",
        "task_family": "logic",
        "task": task,
        "observation": observation[:2000],
        "model_action": model_action,
        "reasoning_mode": "off",
        "reasoning_trace": [],
        "reasoning_summary": "",
        "output_status": "completed",
        "visible_output_emitted": True,
        "valid_action": True,
        "symbolic_channels": {},
        "constitutional_score": 1.0,
        "latent_goal_conflict": 0.0,
        "rationalization_gap": 0.0,
        "bucket": bucket,
        "target_action_family": family if has_target else "",
        "supervision_weight": 1.0 if bucket == "exact_positive" else (0.6 if bucket == "weak_positive" else 0.3),
        "reward": 1.0 if bucket == "exact_positive" else (0.5 if bucket == "weak_positive" else 0.0),
        "meta": {"model_name": "intellect_3_logic_perturbed"},
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    items: list[tuple[str, str, str, str]] = []  # family, task, observation, solution
    excluded = 0
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            trajectory = json.loads(line)
            action = json.loads(trajectory["action"])
            task = str(action.get("task") or "").strip()
            if not task or task in EXCLUDED_FAMILIES:
                excluded += 1
                continue
            game_data = json.loads(action["game_data_str"])
            solution = compact_solution(game_data)
            if not solution:
                excluded += 1
                continue
            observation = str(game_data.get("question") or trajectory.get("state_prompt") or "").strip()
            items.append((f"logic_{task}", task, observation, solution))

    by_family: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for item in items:
        by_family[item[0]].append(item)
    for family in by_family:
        rng.shuffle(by_family[family])
        by_family[family] = by_family[family][: args.max_items_per_family]

    all_solutions = [item[3] for item in items]
    raw_rows: list[dict[str, Any]] = []
    for family, family_items in by_family.items():
        for _, task, observation, solution in family_items:
            raw_rows.append(base_row(family, task, observation, solution, "exact_positive"))
            raw_rows.append(base_row(family, task, observation, corrupt(solution, args.weak_frac, rng), "weak_positive"))
            if rng.random() < 0.5:
                negative_action = corrupt(solution, args.neg_frac, rng)
            else:
                negative_action = rng.choice(all_solutions)
            raw_rows.append(base_row(family, task, observation, negative_action, "negative"))

    rng.shuffle(raw_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw_logic_traces.jsonl"
    write_jsonl(str(raw_path), raw_rows)

    bucket_counts: dict[str, int] = defaultdict(int)
    for row in raw_rows:
        bucket_counts[row["bucket"]] += 1
    manifest = {
        "source": str(args.source),
        "raw_path": str(raw_path),
        "excluded_arc_or_unparseable": excluded,
        "n_families": len(by_family),
        "n_items": sum(len(v) for v in by_family.values()),
        "n_rows": len(raw_rows),
        "bucket_counts": dict(bucket_counts),
        "family_item_counts": {f: len(v) for f, v in sorted(by_family.items(), key=lambda kv: -len(kv[1]))},
        "weak_frac": args.weak_frac,
        "neg_frac": args.neg_frac,
        "excluded_families": sorted(EXCLUDED_FAMILIES),
        "claims_actual_vpd": False,
        "uses_cached_4d_arc_heads": False,
    }
    (args.out_dir / "raw_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_families": len(by_family), "n_rows": len(raw_rows), "buckets": dict(bucket_counts), "excluded": excluded}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

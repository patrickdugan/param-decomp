"""Summarize Hermes SPD/VPD extraction stability across row windows."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        required=True,
        help="Hermes extraction manifests to compare.",
    )
    parser.add_argument(
        "--loop-spline-report",
        type=Path,
        default=Path("reports") / "trm_loop_spline_first_discriminator.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(r"D:\Research_Engine\runs\trm_param_decomp\hermes_logic_critic_100")
        / "hermes_stability.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports") / "hermes_logic_critic_vpd_stability_report.md",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    payload["_path"] = str(path)
    return payload


def component_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["component_id"]): row for row in manifest["components"]}


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denom if denom > 0 else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def is_trunk(module: str) -> bool:
    return module.startswith("networks.")


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["components"]:
        groups["trunk" if is_trunk(str(row["module"])) else "head"].append(row)
    summary: dict[str, Any] = {
        "path": manifest["_path"],
        "row_start": manifest.get("row_start", 0),
        "n_datapoints": manifest.get("n_datapoints"),
        "behavioral_faithfulness_rel_error": manifest.get("behavioral_faithfulness_rel_error"),
        "top10": [
            row["component_id"]
            for row in sorted(
                manifest["components"],
                key=lambda item: float(item["ablation_output_l2_delta"]),
                reverse=True,
            )[:10]
        ],
    }
    for name, rows in groups.items():
        damages = [float(row["ablation_output_l2_delta"]) for row in rows]
        alive = [float(row["alive_fraction"]) for row in rows]
        summary[name] = {
            "component_count": len(rows),
            "mean_damage": round(sum(damages) / len(damages), 6) if damages else 0.0,
            "max_damage": round(max(damages), 6) if damages else 0.0,
            "mean_alive_fraction": round(sum(alive) / len(alive), 6) if alive else 0.0,
            "top5": [
                row["component_id"]
                for row in sorted(rows, key=lambda item: float(item["ablation_output_l2_delta"]), reverse=True)[:5]
            ],
        }
    return summary


def pairwise(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    maps = [component_map(manifest) for manifest in manifests]
    for i in range(len(manifests)):
        for j in range(i + 1, len(manifests)):
            common = sorted(set(maps[i]) & set(maps[j]))
            xi = [float(maps[i][key]["ablation_output_l2_delta"]) for key in common]
            xj = [float(maps[j][key]["ablation_output_l2_delta"]) for key in common]
            top_i = set(
                key
                for key, _row in sorted(
                    maps[i].items(),
                    key=lambda item: float(item[1]["ablation_output_l2_delta"]),
                    reverse=True,
                )[:10]
            )
            top_j = set(
                key
                for key, _row in sorted(
                    maps[j].items(),
                    key=lambda item: float(item[1]["ablation_output_l2_delta"]),
                    reverse=True,
                )[:10]
            )
            rows.append(
                {
                    "left": manifests[i]["_path"],
                    "right": manifests[j]["_path"],
                    "common_components": len(common),
                    "damage_spearman": round(spearman(xi, xj), 6),
                    "top10_overlap": len(top_i & top_j),
                    "top10_jaccard": round(len(top_i & top_j) / len(top_i | top_j), 6),
                }
            )
    return rows


def loop_spline_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    text = path.read_text(encoding="utf-8")
    label = re.search(r"Final label: `([^`]+)`", text)
    delta = re.search(r'"delta_spearman": ([^,\n]+)', text)
    pval = re.search(r'"permutation_p_delta_metric": ([^,\n]+)', text)
    return {
        "available": True,
        "path": str(path),
        "final_label": label.group(1) if label else None,
        "delta_spearman": float(delta.group(1)) if delta else None,
        "permutation_p_delta_metric": float(pval.group(1)) if pval else None,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Hermes Logic Critic VPD Stability Report",
        "",
        "This compares real SPD/VPD extraction manifests across Hermes critic row windows and "
        "places the result next to the earlier SVD loop-spline discriminator.",
        "",
        "## Inputs",
        "",
    ]
    for item in payload["manifest_summaries"]:
        lines.append(
            f"- `{item['path']}`: rows start {item['row_start']}, n={item['n_datapoints']}, "
            f"faithfulness={item['behavioral_faithfulness_rel_error']}"
        )
    lines += [
        "",
        "## Pairwise Stability",
        "",
        "| left start | right start | common | damage Spearman | top-10 overlap | top-10 Jaccard |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    starts = {item["path"]: item["row_start"] for item in payload["manifest_summaries"]}
    for row in payload["pairwise"]:
        lines.append(
            f"| {starts[row['left']]} | {starts[row['right']]} | {row['common_components']} | "
            f"{row['damage_spearman']} | {row['top10_overlap']} | {row['top10_jaccard']} |"
        )
    lines += ["", "## Trunk vs Head", ""]
    for item in payload["manifest_summaries"]:
        lines.append(f"### Row start {item['row_start']}")
        lines.append("")
        lines.append(
            f"- Trunk: mean damage {item['trunk']['mean_damage']}, max damage {item['trunk']['max_damage']}, "
            f"mean alive fraction {item['trunk']['mean_alive_fraction']}"
        )
        lines.append(
            f"- Head: mean damage {item['head']['mean_damage']}, max damage {item['head']['max_damage']}, "
            f"mean alive fraction {item['head']['mean_alive_fraction']}"
        )
        lines.append(f"- Top trunk components: {', '.join(f'`{x}`' for x in item['trunk']['top5'])}")
        lines.append(f"- Top head components: {', '.join(f'`{x}`' for x in item['head']['top5'])}")
        lines.append("")
    loop = payload["loop_spline"]
    lines += [
        "## Against SVD Loop-Spline",
        "",
        f"- Prior SVD discriminator label: `{loop.get('final_label')}`",
        f"- Prior static+phi delta Spearman: `{loop.get('delta_spearman')}`",
        f"- Prior permutation p-value: `{loop.get('permutation_p_delta_metric')}`",
        "",
        "Interpretation: the SVD lane judged static endpoint features sufficient. The real SPD lane "
        "does not directly refute that label yet, but it changes the evidence class: high-damage "
        "components now exist inside the recursive trunk with real component-model faithfulness, so "
        "the loop-spline question can be rerun on actual VPD components instead of SVD atoms.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifests = [load_manifest(path) for path in args.manifests]
    payload = {
        "manifest_summaries": [summarize_manifest(manifest) for manifest in manifests],
        "pairwise": pairwise(manifests),
        "loop_spline": loop_spline_summary(args.loop_spline_report),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Capture activation stats for validated gain-policy contrast packets."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from param_decomp.experiments.trm_choice_rl_feedback_loop import read_jsonl, write_jsonl


DEFAULT_TESSERACT_ROOT = Path(r"C:\projects\Tesseract\Tesseract")
DEFAULT_DATA_ROOT = Path(r"D:\Research_Engine\tesseract_persistent\data")
DEFAULT_CARD_FILE = Path(r"D:\Research_Engine\runs\trm_choice_candidate_cards_20260604\choice_candidate_cards.jsonl")
DEFAULT_CONTRAST_RUN = Path(r"D:\Research_Engine\runs\trm_gain_policy_feature_search_packet_arc_20260604")
DEFAULT_PROBE_RUN = Path(r"D:\Research_Engine\runs\trm_gain_policy_activation_contrast_arc_20260604")
DEFAULT_OUT_DIR = Path(r"D:\Research_Engine\runs\trm_gain_policy_activation_capture_arc_20260604")
DEFAULT_ENV_ID = "arc_challenge"
DEFAULT_MODEL_BUCKET = "2B"
DEFAULT_ADAPTER_BUCKET = "2026-03-12-overnight"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture activation stats for validated gain-policy contrast rows.")
    parser.add_argument("--tesseract-root", type=Path, default=DEFAULT_TESSERACT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--card-file", type=Path, default=DEFAULT_CARD_FILE)
    parser.add_argument("--contrast-run", type=Path, default=DEFAULT_CONTRAST_RUN)
    parser.add_argument("--probe-run", type=Path, default=DEFAULT_PROBE_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--model-bucket", default=DEFAULT_MODEL_BUCKET)
    parser.add_argument("--adapter-bucket", default=DEFAULT_ADAPTER_BUCKET)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sample_id(env_id: str, trajectory_id: str) -> str:
    return f"{env_id}:{trajectory_id}"


def card_lookup(cards: list[dict[str, Any]], env_id: str) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for card in cards:
        if str(card.get("env_id")) != env_id:
            continue
        if str(card.get("status")) != "matched":
            continue
        trajectory_id = str(card.get("trajectory_id") or "")
        if not trajectory_id:
            continue
        lookup[canonical_sample_id(env_id, trajectory_id)] = card
    return lookup


def build_selected_rows(
    contrast_rows: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
    *,
    sample_limit: int = 0,
) -> list[dict[str, Any]]:
    selected = []
    for row in contrast_rows:
        sample_id = str(row.get("sample_id") or "")
        card = lookup.get(sample_id)
        if card is None:
            continue
        selected.append({**row, "card": card})
        if sample_limit > 0 and len(selected) >= sample_limit:
            break
    return selected


def module_paths(probe_rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["module_path"]) for row in probe_rows if row.get("module_path")]


def first_tensor(bench: Any, output: Any) -> Any | None:
    if isinstance(output, (tuple, list)) and output:
        output = output[0]
    if not bench.torch.is_tensor(output):
        return None
    if output.numel() == 0:
        return None
    return output


def mean_abs_activation(bench: Any, output: Any) -> float | None:
    tensor = first_tensor(bench, output)
    if tensor is None:
        return None
    return float(tensor.detach().float().abs().mean().cpu())


def format_prompt(bench: Any, tokenizer: Any, env_id: str, prompt: str) -> str:
    return bench.format_chat_prompt(
        tokenizer,
        bench.format_adapter_prompt(prompt, env_id),
        env_id,
        enable_thinking=False,
        append_instruction=False,
    )


def adapter_path(data_root: Path, model_bucket: str, adapter_bucket: str, env_id: str) -> Path:
    return data_root / "models" / "adapters" / model_bucket / adapter_bucket / env_id


def capture_activation_stats(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.tesseract_root / "scripts"))
    import comprehensive_bench as bench  # noqa: PLC0415

    cards = read_jsonl(args.card_file)
    contrast_rows = read_jsonl(args.contrast_run / "contrast_sets.jsonl")
    probe_rows = read_jsonl(args.probe_run / "module_probe_requests.jsonl")
    lookup = card_lookup(cards, args.env_id)
    selected_rows = build_selected_rows(contrast_rows, lookup, sample_limit=args.sample_limit)
    probe_module_paths = module_paths(probe_rows)
    if not selected_rows:
        raise RuntimeError(f"No matched contrast rows found for env {args.env_id}")
    if not probe_module_paths:
        raise RuntimeError(f"No probe module paths found in {args.probe_run}")

    tokenizer = bench.ensure_pad_token(bench.AutoTokenizer.from_pretrained(str(bench.BASE_2B_BASE_ID)))
    adapter = bench.sanitize_adapter_dir(
        adapter_path(args.data_root, args.model_bucket, args.adapter_bucket, args.env_id),
        args.out_dir / "cleaned_adapters",
    )
    base_model, routed_model = bench.load_routed_model(bench.BASE_2B_BASE_ID, adapter)
    routed_model.eval()
    device = getattr(routed_model, "device", None)
    if device is None:
        device = next(routed_model.parameters()).device

    stats: dict[str, list[dict[str, float]]] = {
        module_path: [{"sum": 0.0, "count": 0.0} for _ in selected_rows] for module_path in probe_module_paths
    }
    handles = []
    current_sample = {"index": None}
    for module_path in probe_module_paths:
        module = bench.module_by_path(routed_model, module_path)

        def _capture(_module, _inputs, output, module_path=module_path):
            sample_index = current_sample["index"]
            if sample_index is None:
                return
            metric = mean_abs_activation(bench, output)
            if metric is None:
                return
            stats[module_path][sample_index]["sum"] += metric
            stats[module_path][sample_index]["count"] += 1.0

        handles.append(module.register_forward_hook(_capture))

    try:
        for sample_index, row in enumerate(selected_rows):
            current_sample["index"] = sample_index
            card = row["card"]
            prompt = format_prompt(bench, tokenizer, args.env_id, str(card["candidate_prompt"]))
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            full_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with bench.torch.no_grad():
                _ = routed_model(input_ids=full_ids, attention_mask=attention_mask)
            del encoded, full_ids, attention_mask
            gc.collect()
            if bench.torch.cuda.is_available():
                bench.torch.cuda.empty_cache()
    finally:
        current_sample["index"] = None
        for handle in handles:
            handle.remove()

    rows = []
    for sample_index, row in enumerate(selected_rows):
        sample_id = str(row["sample_id"])
        for module_path in probe_module_paths:
            cell = stats[module_path][sample_index]
            value = cell["sum"] / cell["count"] if cell["count"] else 0.0
            rows.append(
                {
                    "sample_id": sample_id,
                    "trajectory_id": row["card"]["trajectory_id"],
                    "env_id": args.env_id,
                    "module_path": module_path,
                    "activation_value": round(value, 6),
                    "mean_abs_activation": round(value, 6),
                    "label": row.get("label"),
                    "claim_boundary": "Activation capture only; no runtime edit has been tested.",
                }
            )

    write_jsonl(args.out_dir / "activation_stats.jsonl", rows)
    summary = {
        "status": "completed",
        "generated_at_utc": utc_now(),
        "run_dir": str(args.out_dir),
        "env_id": args.env_id,
        "card_file": str(args.card_file),
        "contrast_run": str(args.contrast_run),
        "probe_run": str(args.probe_run),
        "sample_count": len(selected_rows),
        "module_count": len(probe_module_paths),
        "activation_row_count": len(rows),
        "claim_boundary": "Activation capture only; statistics are ready for ranked bridge mode.",
        "outputs": {
            "summary": str(args.out_dir / "activation_capture_summary.json"),
            "activation_stats": str(args.out_dir / "activation_stats.jsonl"),
        },
    }
    (args.out_dir / "activation_capture_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    del routed_model, base_model, tokenizer
    gc.collect()
    if bench.torch.cuda.is_available():
        bench.torch.cuda.empty_cache()
    return summary


def main() -> int:
    capture_activation_stats(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

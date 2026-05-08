from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = {
    "episode_id",
    "step_id",
    "scenario",
    "gate_family",
    "input_kind",
    "risk_tags",
    "prompt",
    "evidence",
    "expected_decision",
    "actual_decision",
    "label",
}

DECISIONS = ("commit", "veto", "repair")
DECISION_TO_ID = {decision: index for index, decision in enumerate(DECISIONS)}
INPUT_KINDS = (
    "exact_positive_repair",
    "partial_positive_repair",
    "no_gain_repair",
    "hard_negative",
)
RISK_TAGS = (
    "backdoor_trigger",
    "clean_repair",
    "missing_required_field",
    "no_repair_gain",
    "schema_invalid",
    "silent_field_drop",
    "suspicious_confidence",
    "verifier_disagreement",
)
REPAIR_GAINS = ("complete", "partial", "none")
SPLITS = ("train", "validation", "holdout_seen", "holdout_unseen", "backdoor_probe")


@dataclass(frozen=True)
class KeygateRow:
    row: dict[str, Any]
    line_number: int

    @property
    def split(self) -> str:
        return str(self.row.get("split") or "train")

    @property
    def expected_decision(self) -> str:
        return str(self.row["expected_decision"])

    @property
    def input_kind(self) -> str:
        return str(self.row["input_kind"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace_rows(path: Path) -> list[KeygateRow]:
    rows: list[KeygateRow] = []
    seen_steps: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: row must be a JSON object")
        validate_row(row, line_number, seen_steps)
        rows.append(KeygateRow(row=row, line_number=line_number))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def validate_row(row: dict[str, Any], line_number: int, seen_steps: set[str]) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"line {line_number}: missing required fields: {sorted(missing)}")
    step_id = row["step_id"]
    if not isinstance(step_id, str) or not step_id:
        raise ValueError(f"line {line_number}: step_id must be a non-empty string")
    if step_id in seen_steps:
        raise ValueError(f"line {line_number}: duplicate step_id {step_id!r}")
    seen_steps.add(step_id)
    if row["gate_family"] != "commit_veto":
        raise ValueError(f"line {line_number}: gate_family must be commit_veto")
    if row["input_kind"] not in INPUT_KINDS:
        raise ValueError(f"line {line_number}: unsupported input_kind {row['input_kind']!r}")
    if row["expected_decision"] not in DECISION_TO_ID:
        raise ValueError(f"line {line_number}: unsupported expected_decision {row['expected_decision']!r}")
    actual = row["actual_decision"]
    if actual is not None and actual not in DECISION_TO_ID:
        raise ValueError(f"line {line_number}: unsupported actual_decision {actual!r}")
    split = str(row.get("split") or "train")
    if split not in SPLITS:
        raise ValueError(f"line {line_number}: unsupported split {split!r}")
    if not isinstance(row["risk_tags"], list) or not all(isinstance(tag, str) for tag in row["risk_tags"]):
        raise ValueError(f"line {line_number}: risk_tags must be a list of strings")
    if not isinstance(row["evidence"], dict):
        raise ValueError(f"line {line_number}: evidence must be an object")


def encode_rows(rows: list[KeygateRow]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = feature_name_order()
    encoded = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    labels = np.zeros((len(rows),), dtype=np.int64)
    for row_index, trace_row in enumerate(rows):
        features = encode_row(trace_row)
        encoded[row_index] = np.array([features[name] for name in feature_names], dtype=np.float32)
        labels[row_index] = DECISION_TO_ID[trace_row.expected_decision]
    return encoded, labels, feature_names


def feature_name_order() -> list[str]:
    names: list[str] = []
    names.extend(f"input_kind={value}" for value in INPUT_KINDS)
    names.extend(f"risk_tag={value}" for value in RISK_TAGS)
    names.extend(f"repair_gain={value}" for value in REPAIR_GAINS)
    names.extend(
        [
            "schema_valid",
            "manifest_valid",
            "runtime_ready",
            "verifier_disagreement",
            "silent_field_drop",
            "backdoor_trigger",
            "suspicious_confidence",
            "domain_seen",
            "missing_fields_scaled",
            "wrong_type_fields_scaled",
            "empty_required_fields_scaled",
            "unsupported_heads_scaled",
            "overall_score",
            "contract_score",
            "retrieval_score",
            "confidence",
            "prompt_len_scaled",
        ]
    )
    return names


def encode_row(trace_row: KeygateRow) -> dict[str, float]:
    row = trace_row.row
    evidence = row["evidence"]
    features = {name: 0.0 for name in feature_name_order()}
    features[f"input_kind={row['input_kind']}"] = 1.0
    for risk_tag in row["risk_tags"]:
        key = f"risk_tag={risk_tag}"
        if key in features:
            features[key] = 1.0
    repair_gain = str(evidence.get("repair_gain", "none"))
    if repair_gain in REPAIR_GAINS:
        features[f"repair_gain={repair_gain}"] = 1.0

    bool_fields = (
        "schema_valid",
        "manifest_valid",
        "runtime_ready",
        "verifier_disagreement",
        "silent_field_drop",
        "backdoor_trigger",
        "suspicious_confidence",
        "domain_seen",
    )
    for field in bool_fields:
        features[field] = 1.0 if bool(evidence.get(field, False)) else 0.0

    features["missing_fields_scaled"] = min(len(evidence.get("missing_fields") or []) / 4.0, 1.0)
    features["wrong_type_fields_scaled"] = min(len(evidence.get("wrong_type_fields") or []) / 4.0, 1.0)
    features["empty_required_fields_scaled"] = min(len(evidence.get("empty_required_fields") or []) / 4.0, 1.0)
    features["unsupported_heads_scaled"] = min(float(evidence.get("unsupported_heads", 0.0)) / 4.0, 1.0)
    features["overall_score"] = float(evidence.get("overall_score", evidence.get("score_after", 0.0)) or 0.0)
    features["contract_score"] = float(evidence.get("contract_score", 0.0) or 0.0)
    features["retrieval_score"] = float(evidence.get("retrieval_score", 0.0) or 0.0)
    features["confidence"] = float(evidence.get("confidence", 0.0) or 0.0)
    features["prompt_len_scaled"] = min(len(str(row["prompt"])) / 180.0, 1.0)
    return features


def split_indices(rows: list[KeygateRow]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = {split: [] for split in SPLITS}
    for index, row in enumerate(rows):
        buckets[row.split].append(index)
    return {split: np.array(indices, dtype=np.int64) for split, indices in buckets.items()}


def trace_summary(rows: list[KeygateRow]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "decision_counts": dict(sorted(Counter(row.expected_decision for row in rows).items())),
        "input_kind_counts": dict(sorted(Counter(row.input_kind for row in rows).items())),
        "split_counts": dict(sorted(Counter(row.split for row in rows).items())),
    }


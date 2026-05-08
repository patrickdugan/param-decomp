from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

METTA_VARIANTS = ("facts_only", "typed_gate_graph", "rewrite_repair_rules", "task_dag_organelle")
ENV_FAMILIES = ("intellect3_logic", "storyworld_play", "storyworld_build")
GATE_FAMILIES = (
    "signature_route",
    "candidate_verify",
    "repair_step",
    "format_commit",
    "legal_action_router",
    "secret_gate_navigator",
    "memory_use_gate",
    "encounter_repair_gate",
    "ending_reachability_gate",
    "quality_commit_gate",
)
GATE_DECISIONS: dict[str, tuple[str, ...]] = {
    "signature_route": ("route_trm", "route_plain", "abstain"),
    "candidate_verify": ("commit", "veto", "repair"),
    "repair_step": ("repair", "preserve", "abstain"),
    "format_commit": ("commit", "repair", "veto"),
    "legal_action_router": ("commit", "veto", "repair"),
    "secret_gate_navigator": ("commit", "repair", "veto"),
    "memory_use_gate": ("use_memory", "ignore_memory", "abstain"),
    "encounter_repair_gate": ("commit", "repair", "veto"),
    "ending_reachability_gate": ("commit", "repair", "veto"),
    "quality_commit_gate": ("commit", "repair", "veto"),
}
GLOBAL_DECISIONS = tuple(dict.fromkeys(decision for decisions in GATE_DECISIONS.values() for decision in decisions))
SPLITS = ("train", "validation", "holdout_seen", "holdout_unseen")
ARMS = ("vanilla", "generic_skill", "logic_skill", "logic_skill_trm", "unknown")
TRANSFORMS = ("none", "original", "c_repair", "dual_repair")
INPUT_KINDS = (
    "logic_signature_sparse",
    "logic_signature_dense",
    "logic_candidate_valid",
    "logic_candidate_repairable",
    "logic_candidate_invalid",
    "logic_format_valid",
    "logic_format_invalid",
    "story_action_legal",
    "story_action_illegal",
    "story_secret_progress",
    "story_memory_helpful",
    "story_build_quality_pass",
    "story_build_quality_repair",
    "story_build_ending_repair",
    "story_build_encounter_repair",
)
NUMERIC_FEATURES = (
    "height_scaled",
    "width_scaled",
    "area_scaled",
    "tree_density",
    "row_constraint_mean",
    "col_constraint_mean",
    "format_valid",
    "shape_valid",
    "row_signature_l1_scaled",
    "col_signature_l1_scaled",
    "c_count_delta_scaled",
    "camp_adjacent_ok",
    "camps_non_touching_ok",
    "tree_count_match",
    "public_valid",
    "token_total_scaled",
    "latency_scaled",
    "visible_output",
    "metta_atom_count_scaled",
    "metta_rule_count_scaled",
    "metta_typed_edges_scaled",
    "metta_dag_depth_scaled",
    "legal_action_count_scaled",
    "legal_action_valid",
    "matched_nav",
    "diary_used",
    "score_delta_scaled",
    "generation_seconds_scaled",
    "benchmark_pass",
    "ending_effective_scaled",
    "secret_reachability",
    "ending_entropy_scaled",
    "options_per_encounter_scaled",
    "reactions_per_option_scaled",
    "effects_per_reaction_scaled",
    "theme_coherence",
    "text_uniqueness",
)

DEFAULT_INTELLECT3_SOURCE = Path(r"C:\projects\Tesseract\Tesseract\data\normalized_trajectories\intellect_3_logic.jsonl")
DEFAULT_INTELLECT3_PREDICTIONS = Path(
    r"C:\projects\trm_observability_harness\data\qwen27b_intellect3_logic_hybrid_200\predictions.jsonl"
)
DEFAULT_STORYWORLD_PLAY_ROOT = Path(
    r"C:\projects\metta-storyworld\metta-storyworld\metta-storyworld\traces\local_llm_storyworld"
)
DEFAULT_STORYWORLD_BUILD_ROOT = Path(r"C:\projects\GPTStoryworld\hermes-skills\storyworld-conveyor\working_worlds")
SYMBOLS = {"T", "C", "X"}


@dataclass(frozen=True)
class MatrixRow:
    row: dict[str, Any]
    line_number: int

    @property
    def env_family(self) -> str:
        return str(self.row["env_family"])

    @property
    def gate_family(self) -> str:
        return str(self.row["gate_family"])

    @property
    def metta_variant(self) -> str:
        return str(self.row["metta_variant"])

    @property
    def split(self) -> str:
        return str(self.row["split"])

    @property
    def expected_decision(self) -> str:
        return str(self.row["expected_decision"])

    @property
    def input_kind(self) -> str:
        return str(self.row["input_kind"])


@dataclass(frozen=True)
class LogicInstance:
    instance_id: str
    prompt: str
    grid: list[list[str]]
    solution: list[list[str]]
    row_constraints: list[int]
    col_constraints: list[int]
    source_path: str


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def load_rows(path: Path) -> list[MatrixRow]:
    rows = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        validate_row(row, line_number, seen_ids)
        rows.append(MatrixRow(row=row, line_number=line_number))
    return rows


def build_rows(
    *,
    source_path: Path = DEFAULT_INTELLECT3_SOURCE,
    predictions_path: Path | None = DEFAULT_INTELLECT3_PREDICTIONS,
    limit: int | None = None,
    include_storyworld: bool = True,
    storyworld_limit: int = 96,
    storyworld_play_root: Path = DEFAULT_STORYWORLD_PLAY_ROOT,
    storyworld_build_root: Path = DEFAULT_STORYWORLD_BUILD_ROOT,
) -> list[MatrixRow]:
    base_rows = build_intellect3_logic_rows(source_path=source_path, predictions_path=predictions_path, limit=limit)
    if include_storyworld:
        base_rows.extend(build_storyworld_play_rows(storyworld_play_root, storyworld_limit // 2))
        base_rows.extend(build_storyworld_build_rows(storyworld_build_root, storyworld_limit // 2))
    if not base_rows:
        base_rows = build_synthetic_rows()
    expanded = expand_metta_variants(base_rows)
    seen_ids: set[str] = set()
    rows = []
    for line_number, row in enumerate(expanded, start=1):
        validate_row(row, line_number, seen_ids)
        rows.append(MatrixRow(row=row, line_number=line_number))
    return rows


def build_intellect3_logic_rows(*, source_path: Path, predictions_path: Path | None, limit: int | None) -> list[dict[str, Any]]:
    if not source_path.exists():
        return []
    instances = load_logic_instances(source_path, limit)
    if not instances:
        return []
    prediction_groups = load_prediction_groups(predictions_path, set(instances)) if predictions_path and predictions_path.exists() else {}
    rows: list[dict[str, Any]] = []
    for index, instance_id in enumerate(sorted(instances), start=1):
        instance = instances[instance_id]
        predictions = prediction_groups.get(instance_id, [])
        rows.append(build_signature_route_row(instance, predictions, index))
        for record_index, record in enumerate(predictions, start=1):
            rows.extend(build_prediction_gate_rows(instance, record, index=index, record_index=record_index))
    return rows


def load_logic_instances(source_path: Path, limit: int | None) -> dict[str, LogicInstance]:
    instances: dict[str, LogicInstance] = {}
    with source_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if limit is not None and len(instances) >= limit:
                break
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            action = _json_loads_maybe(raw.get("action"))
            game_data = _json_loads_maybe(action.get("game_data_str") if isinstance(action, dict) else None)
            metadata = game_data.get("metadata") if isinstance(game_data, dict) else None
            if not isinstance(metadata, dict):
                continue
            grid = parse_grid(metadata.get("grid"))
            solution = parse_grid(metadata.get("solution"))
            row_constraints = _int_list(metadata.get("row_constraints"))
            col_constraints = _int_list(metadata.get("col_constraints"))
            instance_id = str(raw.get("trajectory_id") or raw.get("row_id") or f"logic_{len(instances):04d}")
            if grid is None or solution is None or not row_constraints or not col_constraints:
                continue
            instances[instance_id] = LogicInstance(
                instance_id=instance_id,
                prompt=str(raw.get("state_prompt") or ""),
                grid=grid,
                solution=solution,
                row_constraints=row_constraints,
                col_constraints=col_constraints,
                source_path=str(source_path),
            )
    return instances


def load_prediction_groups(predictions_path: Path | None, allowed_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if predictions_path is None or not predictions_path.exists():
        return groups
    for row in _read_jsonl(predictions_path):
        row_id = str(row.get("row_id") or "")
        if row_id in allowed_ids:
            groups[row_id].append(row)
    return groups


def build_signature_route_row(instance: LogicInstance, predictions: list[dict[str, Any]], index: int) -> dict[str, Any]:
    arm_scores: dict[str, list[float]] = defaultdict(list)
    arm_exact: dict[str, bool] = defaultdict(bool)
    for record in predictions:
        grid = prediction_grid(record)
        metrics = candidate_metrics(grid, instance)
        arm = str(record.get("arm") or "unknown")
        arm_scores[arm].append(float(metrics["cell_accuracy"]))
        arm_exact[arm] = arm_exact[arm] or bool(metrics["exact_match"])
    trm_best = max(arm_scores.get("logic_skill_trm", [0.0]))
    plain_best = max([score for arm, scores in arm_scores.items() if arm != "logic_skill_trm" for score in scores] or [0.0])
    if arm_exact.get("logic_skill_trm") or trm_best >= plain_best + 0.01:
        decision = "route_trm"
    elif plain_best > trm_best + 0.01:
        decision = "route_plain"
    else:
        decision = "abstain"
    evidence = logic_puzzle_evidence(instance)
    evidence.update({"arm": "unknown", "transform": "none", "trm_best_cell": trm_best, "plain_best_cell": plain_best})
    return base_row(
        env_family="intellect3_logic",
        instance_id=instance.instance_id,
        gate_family="signature_route",
        input_kind="logic_signature_dense" if evidence["tree_density"] >= 0.18 else "logic_signature_sparse",
        expected_decision=decision,
        evidence=evidence,
        score_signals={"trm_best_cell_accuracy": trm_best, "plain_best_cell_accuracy": plain_best},
        split=split_for_index(index),
        source_path=instance.source_path,
    )


def build_prediction_gate_rows(instance: LogicInstance, record: dict[str, Any], *, index: int, record_index: int) -> list[dict[str, Any]]:
    grid = prediction_grid(record)
    metrics = candidate_metrics(grid, instance)
    evidence = logic_candidate_evidence(instance, record, metrics)
    score_signals = {
        "exact_match": metrics["exact_match"],
        "cell_accuracy": metrics["cell_accuracy"],
        "public_valid": metrics["public_valid"],
    }
    if metrics["public_valid"]:
        verify_decision = "commit"
        repair_decision = "preserve"
        verify_kind = "logic_candidate_valid"
    elif metrics["format_valid"] and metrics["shape_valid"] and (metrics["row_signature_l1"] + metrics["col_signature_l1"]) <= 3:
        verify_decision = "repair"
        repair_decision = "repair"
        verify_kind = "logic_candidate_repairable"
    else:
        verify_decision = "veto"
        repair_decision = "abstain"
        verify_kind = "logic_candidate_invalid"
    if metrics["format_valid"] and metrics["shape_valid"]:
        format_decision = "commit"
        format_kind = "logic_format_valid"
    elif metrics["format_valid"]:
        format_decision = "veto"
        format_kind = "logic_format_invalid"
    else:
        format_decision = "repair"
        format_kind = "logic_format_invalid"
    suffix = f"{instance.instance_id}:{record.get('arm', 'unknown')}:{record_index}"
    split = split_for_index(index + record_index)
    return [
        base_row(
            env_family="intellect3_logic",
            instance_id=f"{suffix}:candidate_verify",
            gate_family="candidate_verify",
            input_kind=verify_kind,
            expected_decision=verify_decision,
            evidence=evidence,
            score_signals=score_signals,
            split=split,
            source_path=str(record.get("source_path") or instance.source_path),
        ),
        base_row(
            env_family="intellect3_logic",
            instance_id=f"{suffix}:repair_step",
            gate_family="repair_step",
            input_kind=verify_kind,
            expected_decision=repair_decision,
            evidence=evidence,
            score_signals=score_signals,
            split=split,
            source_path=str(record.get("source_path") or instance.source_path),
        ),
        base_row(
            env_family="intellect3_logic",
            instance_id=f"{suffix}:format_commit",
            gate_family="format_commit",
            input_kind=format_kind,
            expected_decision=format_decision,
            evidence=evidence,
            score_signals=score_signals,
            split=split,
            source_path=str(record.get("source_path") or instance.source_path),
        ),
    ]


def build_storyworld_play_rows(root: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.rglob("turns.jsonl")):
            if len(rows) >= max_rows:
                break
            for turn_index, raw in enumerate(_read_jsonl(path), start=1):
                if len(rows) >= max_rows:
                    break
                rows.extend(storyworld_play_gate_rows(raw, path, turn_index))
    return rows[:max_rows] if rows else synthetic_storyworld_play_rows()


def storyworld_play_gate_rows(raw: dict[str, Any], path: Path, turn_index: int) -> list[dict[str, Any]]:
    legal_actions = raw.get("legal_actions") if isinstance(raw.get("legal_actions"), list) else []
    parsed = str(raw.get("parsed_action") or raw.get("selected_action") or raw.get("action") or "")
    legal_valid = bool(parsed and parsed in {str(action) for action in legal_actions})
    matched_nav = bool(raw.get("matched_metta_nav") or raw.get("metta_nav_match"))
    diary_used = bool(raw.get("play_diary_context_used") or raw.get("diary_context_used"))
    score_delta = extract_score_delta(raw.get("immediate_action_effects"))
    evidence = {
        "legal_action_count_scaled": min(len(legal_actions) / 6.0, 1.0),
        "legal_action_valid": legal_valid,
        "matched_nav": matched_nav,
        "diary_used": diary_used,
        "score_delta_scaled": max(-1.0, min(score_delta / 5.0, 1.0)),
        "generation_seconds_scaled": min(float(raw.get("generation_seconds") or 0.0) / 60.0, 1.0),
        "arm": "unknown",
        "transform": "none",
    }
    if legal_valid:
        legal_decision = "commit"
    elif parsed:
        legal_decision = "repair"
    else:
        legal_decision = "veto"
    secret_decision = "commit" if matched_nav or score_delta > 0 else ("repair" if legal_valid else "veto")
    memory_decision = "use_memory" if diary_used and (matched_nav or score_delta >= 0) else ("ignore_memory" if legal_valid else "abstain")
    base = f"{path.parent.name}:turn_{turn_index:04d}"
    split = split_for_index(turn_index)
    return [
        base_row(
            env_family="storyworld_play",
            instance_id=f"{base}:legal",
            gate_family="legal_action_router",
            input_kind="story_action_legal" if legal_valid else "story_action_illegal",
            expected_decision=legal_decision,
            evidence=evidence,
            score_signals={"score_delta": score_delta, "matched_nav": matched_nav},
            split=split,
            source_path=str(path),
        ),
        base_row(
            env_family="storyworld_play",
            instance_id=f"{base}:secret",
            gate_family="secret_gate_navigator",
            input_kind="story_secret_progress" if matched_nav else "story_action_legal",
            expected_decision=secret_decision,
            evidence=evidence,
            score_signals={"score_delta": score_delta, "matched_nav": matched_nav},
            split=split,
            source_path=str(path),
        ),
        base_row(
            env_family="storyworld_play",
            instance_id=f"{base}:memory",
            gate_family="memory_use_gate",
            input_kind="story_memory_helpful" if diary_used else "story_action_legal",
            expected_decision=memory_decision,
            evidence=evidence,
            score_signals={"score_delta": score_delta, "diary_used": diary_used},
            split=split,
            source_path=str(path),
        ),
    ]


def build_storyworld_build_rows(root: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.rglob("verifier_quality_vector.json")):
            if len(rows) >= max_rows:
                break
            payload = _json_loads_maybe(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows.extend(storyworld_build_gate_rows(payload, path, len(rows) + 1))
    return rows[:max_rows] if rows else synthetic_storyworld_build_rows()


def storyworld_build_gate_rows(payload: dict[str, Any], path: Path, index: int) -> list[dict[str, Any]]:
    ranked = payload.get("ranked") if isinstance(payload.get("ranked"), list) else []
    item = ranked[0] if ranked and isinstance(ranked[0], dict) else payload
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    benchmark_pass = bool(item.get("benchmark_pass") or payload.get("pass_count"))
    ending_effective = _float(metrics.get("ending_effective"))
    secret_reachability = _float(metrics.get("secret_reachability"))
    ending_entropy = _float(metrics.get("ending_entropy"))
    options = _float(metrics.get("options_per_encounter"))
    reactions = _float(metrics.get("reactions_per_option"))
    effects = _float(metrics.get("effects_per_reaction"))
    theme = max(_float(metrics.get("encounter_theme_semantic_coherence")), _float(metrics.get("reaction_theme_semantic_coherence")))
    uniqueness = min(
        1.0,
        _float(metrics.get("encounter_text_uniqueness_ratio"), 1.0),
        _float(metrics.get("reaction_text_uniqueness_ratio"), 1.0),
    )
    evidence = {
        "benchmark_pass": benchmark_pass,
        "ending_effective_scaled": min(ending_effective / 8.0, 1.0),
        "secret_reachability": min(secret_reachability, 1.0),
        "ending_entropy_scaled": min(max(ending_entropy, 0.0) / 3.0, 1.0),
        "options_per_encounter_scaled": min(options / 4.0, 1.0),
        "reactions_per_option_scaled": min(reactions / 4.0, 1.0),
        "effects_per_reaction_scaled": min(effects / 6.0, 1.0),
        "theme_coherence": min(max(theme, 0.0), 1.0),
        "text_uniqueness": uniqueness,
        "arm": "unknown",
        "transform": "none",
    }
    quality_decision = "commit" if benchmark_pass else "repair"
    ending_ok = 0.03 <= secret_reachability <= 0.18 and ending_effective >= 3.0
    ending_decision = "commit" if ending_ok else "repair"
    encounter_ok = options >= 3.0 and reactions >= 2.0 and effects >= 3.0 and uniqueness >= 0.95
    encounter_decision = "commit" if encounter_ok else "repair"
    split = split_for_index(index)
    base = path.parent.name
    return [
        base_row(
            env_family="storyworld_build",
            instance_id=f"{base}:quality",
            gate_family="quality_commit_gate",
            input_kind="story_build_quality_pass" if benchmark_pass else "story_build_quality_repair",
            expected_decision=quality_decision,
            evidence=evidence,
            score_signals={"benchmark_pass": benchmark_pass, "ending_effective": ending_effective},
            split=split,
            source_path=str(path),
        ),
        base_row(
            env_family="storyworld_build",
            instance_id=f"{base}:ending",
            gate_family="ending_reachability_gate",
            input_kind="story_build_quality_pass" if ending_ok else "story_build_ending_repair",
            expected_decision=ending_decision,
            evidence=evidence,
            score_signals={"secret_reachability": secret_reachability, "ending_effective": ending_effective},
            split=split,
            source_path=str(path),
        ),
        base_row(
            env_family="storyworld_build",
            instance_id=f"{base}:encounter",
            gate_family="encounter_repair_gate",
            input_kind="story_build_quality_pass" if encounter_ok else "story_build_encounter_repair",
            expected_decision=encounter_decision,
            evidence=evidence,
            score_signals={"options_per_encounter": options, "reactions_per_option": reactions, "effects_per_reaction": effects},
            split=split,
            source_path=str(path),
        ),
    ]


def synthetic_storyworld_play_rows() -> list[dict[str, Any]]:
    rows = []
    cases = [
        ("legal_good", True, True, True, 3.0),
        ("legal_blind", True, False, False, -1.0),
        ("illegal_repair", False, False, True, 0.0),
    ]
    for index, (name, legal, nav, diary, score_delta) in enumerate(cases, start=1):
        raw = {
            "legal_actions": ["listen", "ask", "wait"],
            "parsed_action": "listen" if legal else "invent",
            "matched_metta_nav": nav,
            "play_diary_context_used": diary,
            "immediate_action_effects": {"score": score_delta},
        }
        rows.extend(storyworld_play_gate_rows(raw, Path(f"synthetic_play/{name}/turns.jsonl"), index))
    return rows


def synthetic_storyworld_build_rows() -> list[dict[str, Any]]:
    payloads = [
        {
            "ranked": [
                {
                    "benchmark_pass": True,
                    "metrics": {
                        "ending_effective": 4.2,
                        "secret_reachability": 0.08,
                        "ending_entropy": 1.8,
                        "options_per_encounter": 3.4,
                        "reactions_per_option": 3.0,
                        "effects_per_reaction": 5.0,
                        "encounter_text_uniqueness_ratio": 1.0,
                        "reaction_text_uniqueness_ratio": 1.0,
                    },
                }
            ]
        },
        {
            "ranked": [
                {
                    "benchmark_pass": False,
                    "metrics": {
                        "ending_effective": 0.0,
                        "secret_reachability": 1.0,
                        "ending_entropy": 0.0,
                        "options_per_encounter": 2.0,
                        "reactions_per_option": 1.0,
                        "effects_per_reaction": 1.0,
                        "encounter_text_uniqueness_ratio": 0.8,
                        "reaction_text_uniqueness_ratio": 0.8,
                    },
                }
            ]
        },
    ]
    rows = []
    for index, payload in enumerate(payloads, start=1):
        rows.extend(storyworld_build_gate_rows(payload, Path(f"synthetic_build/world_{index}/verifier_quality_vector.json"), index))
    return rows


def build_synthetic_rows() -> list[dict[str, Any]]:
    rows = synthetic_storyworld_play_rows()
    rows.extend(synthetic_storyworld_build_rows())
    return rows


def expand_metta_variants(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for row in base_rows:
        for variant in METTA_VARIANTS:
            item = json.loads(json.dumps(row))
            item["metta_variant"] = variant
            item["row_id"] = f"{row['instance_id']}:{row['gate_family']}:{variant}"
            item["metta_atoms"] = metta_atoms_for(item, variant)
            item["evidence"].update(metta_variant_features(item, variant))
            expanded.append(item)
    return expanded


def base_row(
    *,
    env_family: str,
    instance_id: str,
    gate_family: str,
    input_kind: str,
    expected_decision: str,
    evidence: dict[str, Any],
    score_signals: dict[str, Any],
    split: str,
    source_path: str,
) -> dict[str, Any]:
    return {
        "row_id": f"{instance_id}:{gate_family}",
        "env_family": env_family,
        "instance_id": instance_id,
        "gate_family": gate_family,
        "metta_variant": "",
        "input_kind": input_kind,
        "evidence": evidence,
        "score_signals": score_signals,
        "expected_decision": expected_decision,
        "split": split,
        "source_path": source_path,
        "metta_atoms": [],
    }


def validate_row(row: dict[str, Any], line_number: int, seen_ids: set[str]) -> None:
    required = {
        "row_id",
        "env_family",
        "instance_id",
        "gate_family",
        "metta_variant",
        "input_kind",
        "evidence",
        "score_signals",
        "expected_decision",
        "split",
        "source_path",
        "metta_atoms",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"line {line_number}: missing required fields: {sorted(missing)}")
    row_id = str(row["row_id"])
    if row_id in seen_ids:
        raise ValueError(f"line {line_number}: duplicate row_id {row_id!r}")
    seen_ids.add(row_id)
    if row["env_family"] not in ENV_FAMILIES:
        raise ValueError(f"line {line_number}: unsupported env_family {row['env_family']!r}")
    gate = str(row["gate_family"])
    if gate not in GATE_DECISIONS:
        raise ValueError(f"line {line_number}: unsupported gate_family {gate!r}")
    if row["metta_variant"] not in METTA_VARIANTS:
        raise ValueError(f"line {line_number}: unsupported metta_variant {row['metta_variant']!r}")
    if row["expected_decision"] not in GATE_DECISIONS[gate]:
        raise ValueError(f"line {line_number}: decision {row['expected_decision']!r} is invalid for {gate!r}")
    if row["split"] not in SPLITS:
        raise ValueError(f"line {line_number}: unsupported split {row['split']!r}")
    if not isinstance(row["evidence"], dict):
        raise ValueError(f"line {line_number}: evidence must be an object")
    if not isinstance(row["score_signals"], dict):
        raise ValueError(f"line {line_number}: score_signals must be an object")
    if not isinstance(row["metta_atoms"], list):
        raise ValueError(f"line {line_number}: metta_atoms must be a list")


def feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"env={value}" for value in ENV_FAMILIES)
    names.extend(f"gate={value}" for value in GATE_FAMILIES)
    names.extend(f"variant={value}" for value in METTA_VARIANTS)
    names.extend(f"input={value}" for value in INPUT_KINDS)
    names.extend(f"arm={value}" for value in ARMS)
    names.extend(f"transform={value}" for value in TRANSFORMS)
    names.extend(NUMERIC_FEATURES)
    return names


def encode_rows(rows: list[MatrixRow], decisions: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = feature_names()
    decision_to_id = {decision: index for index, decision in enumerate(decisions)}
    features = np.zeros((len(rows), len(names)), dtype=np.float32)
    labels = np.zeros((len(rows),), dtype=np.int64)
    for index, row in enumerate(rows):
        encoded = encode_row(row)
        features[index] = np.array([encoded[name] for name in names], dtype=np.float32)
        labels[index] = decision_to_id[row.expected_decision]
    return features, labels, names


def encode_row(row: MatrixRow) -> dict[str, float]:
    values = {name: 0.0 for name in feature_names()}
    values[f"env={row.env_family}"] = 1.0
    values[f"gate={row.gate_family}"] = 1.0
    values[f"variant={row.metta_variant}"] = 1.0
    if f"input={row.input_kind}" in values:
        values[f"input={row.input_kind}"] = 1.0
    evidence = row.row["evidence"]
    arm = str(evidence.get("arm") or "unknown")
    transform = str(evidence.get("transform") or "none")
    values[f"arm={arm if arm in ARMS else 'unknown'}"] = 1.0
    if f"transform={transform}" in values:
        values[f"transform={transform}"] = 1.0
    for name in NUMERIC_FEATURES:
        values[name] = _float(evidence.get(name), 0.0)
    return values


def indices_for_split(rows: list[MatrixRow]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = {split: [] for split in SPLITS}
    for index, row in enumerate(rows):
        buckets[row.split].append(index)
    return {split: np.array(indices, dtype=np.int64) for split, indices in buckets.items()}


def summarize_rows(rows: list[MatrixRow]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "env_counts": dict(sorted(Counter(row.env_family for row in rows).items())),
        "variant_counts": dict(sorted(Counter(row.metta_variant for row in rows).items())),
        "gate_counts": dict(sorted(Counter(row.gate_family for row in rows).items())),
        "decision_counts": dict(sorted(Counter(row.expected_decision for row in rows).items())),
        "split_counts": dict(sorted(Counter(row.split for row in rows).items())),
    }


def parse_grid(value: Any) -> list[list[str]] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
    if not isinstance(value, list) or not value:
        return None
    grid: list[list[str]] = []
    width: int | None = None
    for raw_row in value:
        if isinstance(raw_row, str):
            row = list(raw_row.strip().upper())
        elif isinstance(raw_row, list):
            row = [str(cell).strip().upper() for cell in raw_row]
        else:
            return None
        if not row or any(cell not in SYMBOLS for cell in row):
            return None
        if width is None:
            width = len(row)
        elif len(row) != width:
            return None
        grid.append(row)
    return grid


def prediction_grid(record: dict[str, Any]) -> list[list[str]] | None:
    final = record.get("final") if isinstance(record.get("final"), dict) else {}
    for field in ("action", "raw_action", "raw_text"):
        grid = parse_grid(final.get(field))
        if grid is not None:
            return grid
    return None


def candidate_metrics(candidate: list[list[str]] | None, instance: LogicInstance) -> dict[str, Any]:
    format_valid = candidate is not None
    shape_valid = bool(
        candidate
        and len(candidate) == len(instance.grid)
        and all(len(row) == len(instance.grid[0]) for row in candidate)
    )
    if not candidate or not shape_valid:
        return {
            "format_valid": format_valid,
            "shape_valid": False,
            "row_signature_l1": 999.0,
            "col_signature_l1": 999.0,
            "c_count_delta": 999.0,
            "camp_adjacent_ok": False,
            "camps_non_touching_ok": False,
            "tree_count_match": False,
            "public_valid": False,
            "exact_match": False,
            "cell_accuracy": 0.0,
        }
    rows = row_c_counts(candidate)
    cols = col_c_counts(candidate)
    row_l1 = sum(abs(a - b) for a, b in zip(rows, instance.row_constraints, strict=True))
    col_l1 = sum(abs(a - b) for a, b in zip(cols, instance.col_constraints, strict=True))
    c_count = sum(rows)
    tree_count = len(tree_cells(instance.grid))
    adjacent = every_camp_adjacent_to_tree(candidate, instance.grid)
    non_touching = camps_non_touching(candidate)
    tree_match = c_count == tree_count
    public_valid = row_l1 == 0 and col_l1 == 0 and adjacent and non_touching and tree_match and preserves_trees(candidate, instance.grid)
    exact = candidate == instance.solution
    return {
        "format_valid": True,
        "shape_valid": True,
        "row_signature_l1": float(row_l1),
        "col_signature_l1": float(col_l1),
        "c_count_delta": float(abs(c_count - tree_count)),
        "camp_adjacent_ok": adjacent,
        "camps_non_touching_ok": non_touching,
        "tree_count_match": tree_match,
        "public_valid": public_valid,
        "exact_match": exact,
        "cell_accuracy": cell_accuracy(candidate, instance.solution),
    }


def logic_puzzle_evidence(instance: LogicInstance) -> dict[str, Any]:
    height = len(instance.grid)
    width = len(instance.grid[0])
    area = height * width
    trees = len(tree_cells(instance.grid))
    return {
        "height_scaled": min(height / 10.0, 1.0),
        "width_scaled": min(width / 10.0, 1.0),
        "area_scaled": min(area / 100.0, 1.0),
        "tree_density": trees / max(1, area),
        "row_constraint_mean": float(np.mean(instance.row_constraints)) / max(1, width),
        "col_constraint_mean": float(np.mean(instance.col_constraints)) / max(1, height),
    }


def logic_candidate_evidence(instance: LogicInstance, record: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    evidence = logic_puzzle_evidence(instance)
    final = record.get("final") if isinstance(record.get("final"), dict) else {}
    evidence.update(
        {
            "arm": str(record.get("arm") or "unknown"),
            "transform": "original",
            "format_valid": 1.0 if metrics["format_valid"] else 0.0,
            "shape_valid": 1.0 if metrics["shape_valid"] else 0.0,
            "row_signature_l1_scaled": min(float(metrics["row_signature_l1"]) / 10.0, 1.0),
            "col_signature_l1_scaled": min(float(metrics["col_signature_l1"]) / 10.0, 1.0),
            "c_count_delta_scaled": min(float(metrics["c_count_delta"]) / 10.0, 1.0),
            "camp_adjacent_ok": 1.0 if metrics["camp_adjacent_ok"] else 0.0,
            "camps_non_touching_ok": 1.0 if metrics["camps_non_touching_ok"] else 0.0,
            "tree_count_match": 1.0 if metrics["tree_count_match"] else 0.0,
            "public_valid": 1.0 if metrics["public_valid"] else 0.0,
            "token_total_scaled": min(float(final.get("token_total") or 0.0) / 2000.0, 1.0),
            "latency_scaled": min(float(final.get("latency_seconds") or 0.0) / 60.0, 1.0),
            "visible_output": 1.0 if final.get("visible_output_emitted") else 0.0,
        }
    )
    return evidence


def metta_atoms_for(row: dict[str, Any], variant: str) -> list[str]:
    instance = row["instance_id"]
    gate = row["gate_family"]
    env = row["env_family"]
    decision = row["expected_decision"]
    if variant == "facts_only":
        return [f"(env {instance} {env})", f"(gate {instance} {gate})", f"(expected {instance} {decision})"]
    if variant == "typed_gate_graph":
        return [
            f"(: {gate} GateFamily)",
            f"(GateInstance {instance} {gate})",
            f"(EnvFamily {instance} {env})",
            f"(InputKind {instance} {row['input_kind']})",
        ]
    if variant == "rewrite_repair_rules":
        return [
            f"(= (needs-repair {instance}) (if (defect-visible {instance}) repair preserve))",
            f"(= (commit-policy {instance}) (if (safe-to-commit {instance}) commit veto))",
            f"(expected {instance} {decision})",
        ]
    if variant == "task_dag_organelle":
        return [
            f"(Organelle {gate})",
            f"(Node {instance} {gate})",
            f"(MetricSink {instance} {env})",
            f"(GrowthCandidate {instance} {decision})",
        ]
    raise ValueError(f"unsupported MeTTa variant {variant!r}")


def metta_variant_features(row: dict[str, Any], variant: str) -> dict[str, float]:
    atoms = metta_atoms_for(row, variant)
    rule_count = sum(1 for atom in atoms if atom.startswith("(="))
    typed_edges = sum(1 for atom in atoms if "Gate" in atom or "Node" in atom or "Organelle" in atom)
    dag_depth = 3 if variant == "task_dag_organelle" else (2 if variant in {"typed_gate_graph", "rewrite_repair_rules"} else 1)
    return {
        "metta_atom_count_scaled": min(len(atoms) / 8.0, 1.0),
        "metta_rule_count_scaled": min(rule_count / 4.0, 1.0),
        "metta_typed_edges_scaled": min(typed_edges / 6.0, 1.0),
        "metta_dag_depth_scaled": min(dag_depth / 4.0, 1.0),
    }


def row_c_counts(grid: list[list[str]]) -> list[int]:
    return [sum(1 for cell in row if cell == "C") for row in grid]


def col_c_counts(grid: list[list[str]]) -> list[int]:
    return [sum(1 for row in grid if row[column] == "C") for column in range(len(grid[0]))]


def tree_cells(grid: list[list[str]]) -> set[tuple[int, int]]:
    return {(row, col) for row, values in enumerate(grid) for col, value in enumerate(values) if value == "T"}


def camp_cells(grid: list[list[str]]) -> set[tuple[int, int]]:
    return {(row, col) for row, values in enumerate(grid) for col, value in enumerate(values) if value == "C"}


def neighbors4(row: int, col: int, height: int, width: int) -> list[tuple[int, int]]:
    values = []
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_row = row + delta_row
        next_col = col + delta_col
        if 0 <= next_row < height and 0 <= next_col < width:
            values.append((next_row, next_col))
    return values


def every_camp_adjacent_to_tree(candidate: list[list[str]], puzzle_grid: list[list[str]]) -> bool:
    trees = tree_cells(puzzle_grid)
    height = len(candidate)
    width = len(candidate[0])
    return all(any(neighbor in trees for neighbor in neighbors4(row, col, height, width)) for row, col in camp_cells(candidate))


def camps_non_touching(grid: list[list[str]]) -> bool:
    camps = camp_cells(grid)
    for row, col in camps:
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                if (row + delta_row, col + delta_col) in camps:
                    return False
    return True


def preserves_trees(candidate: list[list[str]], puzzle_grid: list[list[str]]) -> bool:
    return all(candidate[row][col] == "T" for row, col in tree_cells(puzzle_grid))


def cell_accuracy(candidate: list[list[str]], expected: list[list[str]]) -> float:
    total = len(expected) * len(expected[0])
    correct = sum(1 for row in range(len(expected)) for col in range(len(expected[0])) if candidate[row][col] == expected[row][col])
    return correct / max(1, total)


def split_for_index(index: int) -> str:
    return ("train", "train", "train", "validation", "holdout_seen", "holdout_unseen")[index % 6]


def extract_score_delta(value: Any) -> float:
    if isinstance(value, dict):
        total = 0.0
        for item in value.values():
            total += extract_score_delta(item)
        return total
    if isinstance(value, list):
        return sum(extract_score_delta(item) for item in value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return []
    return out


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

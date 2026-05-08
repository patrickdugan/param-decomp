from __future__ import annotations

import json
from pathlib import Path

from scripts.run_snacksack_metta_27b_flex import (
    extract_grid_from_text,
    prompt_for_arm,
    select_logic_instances,
    summarize_arm_metrics,
)


def _write_logic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "intellect_3_logic.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = []
    pred_rows = []
    for index in range(4):
        trajectory_id = f"intellect_3_logic_fixture_{index}"
        metadata = {
            "grid": [["T", "X"], ["X", "X"]],
            "solution": [["T", "C"], ["X", "X"]],
            "row_constraints": [1, 0],
            "col_constraints": [0, 1],
        }
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "state_prompt": "Place one camp next to the tree.",
                "action": json.dumps({"game_data_str": json.dumps({"metadata": metadata})}),
            }
        )
        pred_rows.extend(
            [
                {
                    "row_id": trajectory_id,
                    "arm": "logic_skill_trm",
                    "final": {"action": repr([["T", "C"], ["X", "X"]]) if index % 2 else repr([["T", "X"], ["X", "X"]])},
                },
                {
                    "row_id": trajectory_id,
                    "arm": "generic_skill",
                    "final": {"action": repr([["T", "X"], ["X", "X"]])},
                },
            ]
        )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    predictions.write_text("\n".join(json.dumps(row) for row in pred_rows) + "\n", encoding="utf-8")
    return source, predictions


def test_extract_grid_from_text_finds_final_grid_after_reasoning() -> None:
    text = "notes before\nFINAL_GRID = [['T', 'C'], ['X', 'X']]\n"

    assert extract_grid_from_text(text) == [["T", "C"], ["X", "X"]]


def test_prompts_do_not_include_hidden_solution() -> None:
    prompt = prompt_for_arm(
        arm="27b_metta_trm_adaptive",
        state_prompt="Puzzle prompt without answer.",
        gate_cards=[],
        previous_text="[['T', 'X'], ['X', 'X']]",
        verifier_feedback={"row_signature_l1": 1},
    )

    assert "Puzzle prompt without answer." in prompt
    assert "row_signature_l1" in prompt
    assert "[['T', 'C'], ['X', 'X']]" not in prompt


def test_select_logic_instances_returns_requested_limit(tmp_path: Path) -> None:
    source, predictions = _write_logic_fixture(tmp_path)

    selected = select_logic_instances(source, predictions, 3)

    assert len(selected) == 3
    assert len({row.instance_id for row in selected}) == 3


def test_summarize_arm_metrics_handles_records() -> None:
    summary = summarize_arm_metrics(
        [
            {
                "arm": "27b_direct",
                "final": {
                    "exact_match": True,
                    "public_valid": True,
                    "grid_cell_accuracy": 1.0,
                    "output_status": "parsed_grid",
                    "token_total": 100,
                    "latency_seconds": 2.0,
                },
            },
            {
                "arm": "27b_direct",
                "final": {
                    "exact_match": False,
                    "public_valid": False,
                    "grid_cell_accuracy": 0.5,
                    "output_status": "unparsed",
                    "token_total": 200,
                    "latency_seconds": 4.0,
                },
            },
        ]
    )

    assert summary["27b_direct"]["count"] == 2
    assert summary["27b_direct"]["exact_match_rate"] == 0.5
    assert summary["27b_direct"]["mean_cell_accuracy"] == 0.75

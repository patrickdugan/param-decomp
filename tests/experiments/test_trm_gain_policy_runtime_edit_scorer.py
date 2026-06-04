from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_runtime_edit_scorer import run_runtime_score


CONDITION_ID = "cond_top_margin:top_D:runner_A:bucket_any:max_1_0:penalty_1_0"


def _feature_map_run(tmp_path: Path) -> Path:
    run = tmp_path / "feature_map"
    run.mkdir()
    (run / "activation_feature_map_summary.json").write_text(
        json.dumps({"condition_id": CONDITION_ID}) + "\n",
        encoding="utf-8",
    )
    (run / "activation_feature_map.json").write_text(
        json.dumps({"entries": [{"key": "m.good"}]}) + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        run / "activation_edit_trials.jsonl",
        [
            {
                "trial_id": "activation_edit_trial:0000:m.good",
                "module_path": "m.good",
                "feature_name": "good",
                "scale": 1.25,
            }
        ],
    )
    return run


def test_runtime_score_promotes_activation_gated_rescue_over_fixed_control(tmp_path: Path) -> None:
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "p1",
                "target_action": "A",
                "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.5}],
            },
            {
                "env_id": "arc",
                "trajectory_id": "n1",
                "target_action": "D",
                "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.4}],
            },
        ],
    )
    stats = tmp_path / "activation_stats.jsonl"
    write_jsonl(
        stats,
        [
            {"sample_id": "arc:p1", "module_path": "m.good", "activation_value": 2.0, "label": "positive_rescue"},
            {"sample_id": "arc:n1", "module_path": "m.good", "activation_value": 1.0, "label": "negative_correct_touched"},
        ],
    )

    summary = run_runtime_score(
        Namespace(
            feature_map_run=_feature_map_run(tmp_path),
            activation_stats=stats,
            score_files=str(score_file),
            out_dir=tmp_path / "out",
            controller_touch_rate=0.5,
            max_prompt_tokens=1200,
        )
    )

    assert summary["promotion_ready"] is True
    assert summary["best_trial_delta"] == 0.5
    assert summary["best_control_delta"] == 0.0
    assert (tmp_path / "out" / "runtime_edit_scores.jsonl").exists()


def test_runtime_score_rejects_when_activation_gate_has_no_rescue(tmp_path: Path) -> None:
    score_file = tmp_path / "scores.jsonl"
    write_jsonl(
        score_file,
        [
            {
                "env_id": "arc",
                "trajectory_id": "n1",
                "target_action": "D",
                "scores": [{"action": "D", "logprob": -1.0}, {"action": "A", "logprob": -1.4}],
            }
        ],
    )
    stats = tmp_path / "activation_stats.jsonl"
    write_jsonl(
        stats,
        [{"sample_id": "arc:n1", "module_path": "m.good", "activation_value": 1.0, "label": "negative_correct_touched"}],
    )

    summary = run_runtime_score(
        Namespace(
            feature_map_run=_feature_map_run(tmp_path),
            activation_stats=stats,
            score_files=str(score_file),
            out_dir=tmp_path / "out",
            controller_touch_rate=1.0,
            max_prompt_tokens=1200,
        )
    )

    assert summary["promotion_ready"] is False
    assert summary["best_trial_delta"] <= 0.0

from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_activation_feature_map import run_feature_map


CONDITION_ID = "cond_top_margin:top_D:runner_A:bucket_any:max_1_0:penalty_1_0"


def _contrast_packet(tmp_path):
    contrast_run = tmp_path / "contrast"
    contrast_run.mkdir()
    (contrast_run / "feature_search_manifest.json").write_text(
        '{"condition_id": "' + CONDITION_ID + '"}\n',
        encoding="utf-8",
    )
    (contrast_run / "activation_probe_contract.json").write_text(
        '{"condition_id": "' + CONDITION_ID + '", "mcp_resources": {"contrast_sets": "contrast_sets.jsonl"}}\n',
        encoding="utf-8",
    )
    write_jsonl(
        contrast_run / "contrast_sets.jsonl",
        [
            {"sample_id": "p1", "label": "positive_rescue"},
            {"sample_id": "p2", "label": "positive_rescue"},
            {"sample_id": "n1", "label": "negative_same_pair_inactive"},
        ],
    )
    return contrast_run


def _probe_packet(tmp_path):
    probe_run = tmp_path / "probe"
    probe_run.mkdir()
    write_jsonl(
        probe_run / "module_probe_requests.jsonl",
        [
            {"request_id": "activation_probe:0000", "module_path": "m.a"},
            {"request_id": "activation_probe:0001", "module_path": "m.b"},
        ],
    )
    return probe_run


def test_run_feature_map_probe_only_writes_probe_map(tmp_path) -> None:
    contrast_run = _contrast_packet(tmp_path)
    probe_run = _probe_packet(tmp_path)

    summary = run_feature_map(
        Namespace(
            contrast_run=contrast_run,
            probe_run=probe_run,
            out_dir=tmp_path / "out",
            activation_stats=None,
            top_k=4,
            max_prompt_tokens=1200,
        )
    )

    assert summary["status"] == "probe_only"
    assert summary["feature_map_entry_count"] == 2
    assert (tmp_path / "out" / "activation_feature_map.json").exists()
    assert (tmp_path / "out" / "activation_edit_trials.jsonl").exists()


def test_run_feature_map_with_stats_ranks_modules(tmp_path) -> None:
    contrast_run = _contrast_packet(tmp_path)
    probe_run = _probe_packet(tmp_path)
    stats = tmp_path / "stats.jsonl"
    write_jsonl(
        stats,
        [
            {"sample_id": "p1", "module_path": "m.a", "activation_value": 4.0},
            {"sample_id": "p2", "module_path": "m.a", "activation_value": 5.0},
            {"sample_id": "n1", "module_path": "m.a", "activation_value": 1.0},
            {"sample_id": "p1", "module_path": "m.b", "activation_value": 1.5},
            {"sample_id": "n1", "module_path": "m.b", "activation_value": 1.0},
        ],
    )

    summary = run_feature_map(
        Namespace(
            contrast_run=contrast_run,
            probe_run=probe_run,
            out_dir=tmp_path / "out_ranked",
            activation_stats=stats,
            top_k=4,
            max_prompt_tokens=1200,
        )
    )

    assert summary["status"] == "ranked"
    assert summary["feature_map_entry_count"] == 2
    assert summary["edit_trial_count"] == 2

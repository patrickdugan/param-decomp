from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_choice_rl_feedback_loop import write_jsonl
from param_decomp.experiments.trm_gain_policy_activation_contrast_ranker import (
    build_probe_requests,
    rank_activation_contrast,
    run_ranker,
)


def test_rank_activation_contrast_orders_by_abs_difference() -> None:
    rows = [
        {"sample_id": "p1", "module_path": "m1", "activation_value": 3.0},
        {"sample_id": "p2", "module_path": "m1", "activation_value": 5.0},
        {"sample_id": "n1", "module_path": "m1", "activation_value": 1.0},
        {"sample_id": "n2", "module_path": "m1", "activation_value": 1.0},
        {"sample_id": "p1", "module_path": "m2", "activation_value": 2.0},
        {"sample_id": "n1", "module_path": "m2", "activation_value": 1.5},
    ]

    ranked = rank_activation_contrast(rows, positive_ids={"p1", "p2"}, negative_ids={"n1", "n2"})

    assert ranked[0]["module_path"] == "m1"
    assert ranked[0]["positive_minus_negative"] == 3.0
    assert ranked[0]["suggested_edit_direction"] == "suppress_when_positive_high"


def test_build_probe_requests_carries_sample_sets() -> None:
    contrast_rows = [
        {"sample_id": "p1", "label": "positive_rescue"},
        {"sample_id": "n1", "label": "negative_same_pair_inactive"},
    ]
    requests = build_probe_requests(["module.a"], contrast_rows, {"condition_id": "cond"})

    assert requests[0]["positive_sample_ids"] == ["p1"]
    assert requests[0]["negative_sample_ids"] == ["n1"]
    assert requests[0]["module_path"] == "module.a"


def test_run_ranker_without_stats_writes_probe_requests(tmp_path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "feature_search_manifest.json").write_text('{"condition_id": "cond"}\n', encoding="utf-8")
    (packet / "activation_probe_contract.json").write_text('{"condition_id": "cond"}\n', encoding="utf-8")
    write_jsonl(
        packet / "contrast_sets.jsonl",
        [
            {"sample_id": "p1", "label": "positive_rescue"},
            {"sample_id": "n1", "label": "negative_same_pair_inactive"},
        ],
    )

    summary = run_ranker(
        Namespace(
            packet_run=packet,
            out_dir=tmp_path / "out",
            activation_stats=None,
            module_paths="module.a,module.b",
            top_k=8,
            max_prompt_tokens=1200,
        )
    )

    assert summary["status"] == "probe_requests_ready"
    assert summary["probe_request_count"] == 2
    assert (tmp_path / "out" / "module_probe_requests.jsonl").exists()


def test_run_ranker_with_stats_writes_ranking(tmp_path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "feature_search_manifest.json").write_text('{"condition_id": "cond"}\n', encoding="utf-8")
    (packet / "activation_probe_contract.json").write_text('{"condition_id": "cond"}\n', encoding="utf-8")
    write_jsonl(
        packet / "contrast_sets.jsonl",
        [
            {"sample_id": "p1", "label": "positive_rescue"},
            {"sample_id": "n1", "label": "negative_same_pair_inactive"},
        ],
    )
    stats = tmp_path / "stats.jsonl"
    write_jsonl(
        stats,
        [
            {"sample_id": "p1", "module_path": "module.a", "activation_value": 4.0},
            {"sample_id": "n1", "module_path": "module.a", "activation_value": 1.0},
        ],
    )

    summary = run_ranker(
        Namespace(
            packet_run=packet,
            out_dir=tmp_path / "out",
            activation_stats=stats,
            module_paths="module.a",
            top_k=8,
            max_prompt_tokens=1200,
        )
    )

    assert summary["status"] == "ranked"
    assert summary["ranked_module_count"] == 1

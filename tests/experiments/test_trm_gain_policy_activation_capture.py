from __future__ import annotations

from argparse import Namespace

from param_decomp.experiments.trm_gain_policy_activation_capture import (
    build_selected_rows,
    card_lookup,
)


def test_build_selected_rows_pairs_contrast_rows_with_cards() -> None:
    cards = [
        {
            "env_id": "arc_challenge",
            "trajectory_id": "arc_challenge_34",
            "status": "matched",
            "candidate_prompt": "prompt-34",
        },
        {
            "env_id": "arc_challenge",
            "trajectory_id": "arc_challenge_49",
            "status": "matched",
            "candidate_prompt": "prompt-49",
        },
        {
            "env_id": "arc_easy",
            "trajectory_id": "arc_easy_0",
            "status": "matched",
            "candidate_prompt": "ignore",
        },
    ]
    lookup = card_lookup(cards, "arc_challenge")
    rows = build_selected_rows(
        [
            {"sample_id": "arc_challenge:arc_challenge_34", "label": "positive_rescue"},
            {"sample_id": "arc_challenge:arc_challenge_49", "label": "positive_rescue"},
            {"sample_id": "arc_challenge:missing", "label": "negative_same_pair_inactive"},
        ],
        lookup,
    )

    assert [row["sample_id"] for row in rows] == [
        "arc_challenge:arc_challenge_34",
        "arc_challenge:arc_challenge_49",
    ]
    assert rows[0]["card"]["candidate_prompt"] == "prompt-34"
    assert rows[1]["card"]["candidate_prompt"] == "prompt-49"

# VPD-TRM Edit Discovery Protocol

Yes. The protocol should be a closed-loop edit discovery system, but with hard separation between:

1. eval problem isolation,
2. feature correlation,
3. edit proposal,
4. controlled hill-climb validation.

The mistake to avoid is letting "feature seems relevant" become "edit works." Our recent runs proved why that separation matters.

## Protocol Shape

### 1. Eval Problem Isolation

Freeze an eval set and turn failures into compact cards:

```json
{
  "eval_id": "intellect_3_logic",
  "failure_cluster": "format_commit_failure",
  "sample_ids": ["..."],
  "expected_behavior": "valid logic grid / valid commit format",
  "observed_failure": "bad format / wrong route / repair miss",
  "editable_gate": "format_commit",
  "primary_metric": "format_success_rate",
  "guardrails": ["no false_commit increase", "no accuracy drop"],
  "holdout_split": "cluster_holdout_A"
}
```

Output artifacts:

- `eval_problem_cards.jsonl`
- `failure_clusters.jsonl`
- `holdout_eval_cards.jsonl`
- `baseline_eval_scores.json`

This stage is not about edits. It only defines what exact failure we are trying to hill climb.

### 2. Feature Correlation Scan

For each organelle TRM and decomposition vectorization, scan features against the isolated failure cards.

Candidate vectorizations:

- `head_row:*`
- `head_svd:*`
- low-rank SVD slices
- source-target graft vectors
- gate-family aggregates
- decision-label rows like `repair`, `veto`, `commit`, `route_plain`

For each feature, measure:

- `activation_on_failures`
- `activation_on_successes`
- `ablation_delta`
- `boost_delta`
- `eval_delta_estimate`
- `matched_random_p95_delta`
- `specificity`
- `guardrail_delta`
- `recurrence_across_seeds`

Output:

- `feature_correlation_scan.jsonl`
- `logic_efficacy_feature_map.json`
- `feature_family_summary.csv`

Important: this is still correlational. A feature can rank highly and still fail under edit controls.

### 3. Feature Identification RL Loop

Make the outer loop a contextual bandit, not vague agent learning.

State:

```json
{
  "failure_cluster": "...",
  "top_correlated_features": ["..."],
  "prior_failed_edits": ["..."],
  "self_model": {
    "families_that_failed": ["signature_route:head_row:route_plain"],
    "families_promising_but_unproven": ["format_commit:head_svd:0001"]
  }
}
```

Actions:

- `ablate_feature`
- `boost_feature`
- `scale_sweep`
- `low_rank_refine`
- `combine_features`
- `add_replay_slice`
- `generate_matched_random_control`
- `abstain`

Reward:

```text
reward =
  eval_delta
  - matched_random_p95_delta
  - guardrail_penalty
  - holdout_regression_penalty
  + recurrence_bonus
```

If a local adapter update is used as a prefilter before live eval, the prefilter
should not reuse the same batch it trained on. A disjoint micro-holdout batch,
even another 8 tokens, is enough to turn the prefilter into a crude
generalization probe without increasing the memory footprint in a meaningful
way. That is the minimum safeguard against reading a train-on-test loss drop as
evidence of an actual edit.

Output:

- `rl_feature_policy_state.json`
- `edit_action_trials.jsonl`
- `reward_history.jsonl`
- `self_model.json`

This gives us real learning: the loop learns which feature families are worth trying and which are dead ends.

### 4. Controlled Edit Hill Climb

Only now do edits get proposed.

Each edit candidate must include:

```json
{
  "candidate_id": "...",
  "feature_keys": ["..."],
  "edit_type": "scale_sweep | low_rank_slice | graft | ablation",
  "scale": 1.25,
  "expected_metric": "format_success_rate",
  "source_evidence": "feature_correlation_scan",
  "matched_random_controls": true,
  "runtime_only": true
}
```

Acceptance rule:

```text
accept only if:
  eval_delta > matched_random_p95_delta
  efficiency > matched_random_p95_efficiency
  specificity >= matched_random_p95_specificity
  holdout_delta >= 0
  guardrails pass
```

Output:

- `edit_candidates.jsonl`
- `matched_random_controls.jsonl`
- `accepted_edits.jsonl`
- `rejected_near_misses.jsonl`
- `hill_climb_summary.json`

## What This Gives Us

This turns the work into a clean scientific loop:

```text
failure -> feature correlation -> edit hypothesis -> controlled intervention -> reward -> updated feature policy
```

The current state says:

- We can isolate failures.
- We can rank proxy features.
- We can generate plausible edits.
- We have not yet found controlled eval-positive edits.

The next protocol step is to implement the correlation scan plus bandit state over the organelle decomps, then feed only the top-ranked feature families into hill climb.

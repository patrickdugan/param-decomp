# Opus Handoff: TRM Loop-Spline First Discriminator

## Current Goal

Implement the first discriminator for the TRM loop-spline hypothesis:

> Do recurrence-step intervention profiles add predictive information beyond static SVD/component features for held-out depth-transfer damage?

This must remain a discriminator only. Do not build policy code, patch search, RCPI, or new patch algebra.

## Guardrails

- Do not claim actual Goodfire VPD.
- Say `VPD-style/SVD rank-one component dictionary`.
- Do not use cached 4D ARC heads.
- Do not use final-step alpha=0 ablation loss as a loop-shape feature.
- Do not infer loop roles from norm dominance.
- Final labels allowed:
  - `LOOP_SPLINE_INCREMENTAL_SIGNAL`
  - `PROFILE_UNSTABLE`
  - `STATIC_ENDPOINT_SUFFICIENT`
  - `UNDERPOWERED_PROBES`
  - `BLOCKED_NO_RECURSION_TRACE`

## Files Added This Session

### Completed VPD-style recursive tiny decomposition

- `scripts/run_recursive_tiny_vpd_int3_logic.py`
- `reports/recursive_tiny_vpd_int3_logic_report.md`
- `artifacts/recursive_tiny_vpd_int3_logic/manifest.json`
- `artifacts/recursive_tiny_vpd_int3_logic/model_summaries.json`
- `artifacts/recursive_tiny_vpd_int3_logic/delta_summaries.json`
- `artifacts/recursive_tiny_vpd_int3_logic/components/**/*.npz`

Status from that pass:

```text
RECURSIVE_TINY_VPD_STYLE_COMPLETE
```

Summary:

- 4 local recursive tiny/Hermes models decomposed.
- 148 rank-atom `.npz` files emitted.
- Used Hermes skill gym action-family corpus with `logic_env` rows as the local Intellect-3 logic skill proxy.
- No cached 4D ARC heads used.
- No actual Goodfire VPD claimed.

Key local model:

```text
D:\projects\HRM-re\experiments\hermes_skill_gym\outputs\hermes_skill_gym_trm_action_family_explicit_single\variants\trm_like_single_tier\artifacts\model.pt
```

This is the preferred checkpoint for the loop-spline discriminator because it has the best score among the two direct TRM-like action-family variants:

```text
explicit_action_family_trm_like score: 0.755967
action_family_trm_like score:          0.659180
logic_env rows: 5
logic_env bucket accuracy: 1.0
```

### Loop-spline discriminator runner

- `scripts/run_trm_loop_spline_first_discriminator.py`

Compile passed:

```powershell
python -m py_compile scripts\run_trm_loop_spline_first_discriminator.py
```

Full run attempted:

```powershell
python scripts\run_trm_loop_spline_first_discriminator.py
```

It timed out after 15 minutes before writing final artifacts:

```text
command timed out after 904048 milliseconds
```

No final manifest exists:

```text
artifacts/recursive_vpd_style/manifest.json
```

is absent.

## Why It Timed Out

The first implementation evaluates too many interventions:

- Builds SVD components from all eligible 2D tensors, default 4 rank atoms per module.
- Computes static alpha=0 ablation damage for every component.
- Samples up to 25 components per static-damage decile.
- For every sampled component, runs alpha `{0.0, 0.5, 1.5}` on probe A and B at depth `T=4`.
- Also runs alpha=0 at `T_prime=8` on A and B.
- Uses full available probe split from train+eval, around 195 rows total.

This is conceptually right but too slow for the first run.

## Recommended Continuation

First make a smoke-sized run complete, then scale.

Suggested fast command:

```powershell
python scripts\run_trm_loop_spline_first_discriminator.py `
  --probe-target 32 `
  --rank-atoms-per-module 2 `
  --per-decile 3 `
  --base-depth 3 `
  --batch-size 64 `
  --permutations 50
```

If that completes, inspect:

```text
artifacts/recursive_vpd_style/static_features.json
artifacts/recursive_vpd_style/loop_spline_sample.json
artifacts/recursive_vpd_style/loop_profiles_A.json
artifacts/recursive_vpd_style/loop_profiles_B.json
artifacts/recursive_vpd_style/depth_transfer_targets.json
artifacts/recursive_vpd_style/loop_shape_features.json
artifacts/recursive_vpd_style/manifest.json
reports/trm_loop_spline_first_discriminator.md
```

Then increase in this order:

1. `--probe-target 64`
2. `--per-decile 5`
3. `--rank-atoms-per-module 4`
4. `--permutations 200`
5. only then consider `--probe-target 128` or higher

## Important Implementation Notes

The runner mirrors the `HermesHRMClassifier` recurrence loop to expose per-step traces:

- `loss`
- `z_drift`
- `logit_margin`
- `confidence`
- `y_norm`
- `z_norm`
- `output_action_score`

It sets recurrence depth by temporarily changing:

```python
model.reasoning_steps = T
```

The trained checkpoint has `reasoning_steps=1`, which gives no temporal shape. The runner therefore defaults to:

```text
T = 4
T_prime = 8
```

This is intentional. With `T=1`, the discriminator should be treated as `BLOCKED_NO_RECURSION_TRACE` or structurally underpowered.

## Possible Bug To Check First

In `trace_model`, the attention mask assumes `<PAD>` id is 0:

```python
pad_id = 0
attention_mask = input_ids.ne(pad_id)
```

For this corpus, `<PAD>` is expected to be 0, but Opus should make this explicit by passing `pad_id` into `trace_model` from `payload["vocab"]["<PAD>"]`.

## Performance Fixes To Apply

Best quick fix:

- Add `--max-static-components` and stratify before running expensive static ablations.
- For static features, compute static proxy features for all components but only run static ablation on a capped random stratified pre-sample.
- Or cache each component/depth/alpha result immediately to JSONL so timeout does not lose progress.

Minimum useful cache files:

```text
artifacts/recursive_vpd_style/events.jsonl
artifacts/recursive_vpd_style/static_features.partial.jsonl
artifacts/recursive_vpd_style/profiles.partial.jsonl
```

## Expected Honest Outcomes

Because local probes are small, likely outcomes are:

- `UNDERPOWERED_PROBES`, if probe split remains below target 256+256.
- `PROFILE_UNSTABLE`, if split-half reliability fails.
- `STATIC_ENDPOINT_SUFFICIENT`, if reliability passes but nested model improvement fails.

Do not force `LOOP_SPLINE_INCREMENTAL_SIGNAL`; require:

```text
delta Spearman >= +0.15 OR delta AUC >= +0.07
permutation p < 0.01
```

## Current Git State Relevant To This Work

Untracked relevant files:

```text
artifacts/recursive_tiny_vpd_int3_logic/
reports/recursive_tiny_vpd_int3_logic_report.md
scripts/run_recursive_tiny_vpd_int3_logic.py
scripts/run_trm_loop_spline_first_discriminator.py
massoff.md
```

There are many unrelated dirty files in the repo from prior work. Do not revert them.

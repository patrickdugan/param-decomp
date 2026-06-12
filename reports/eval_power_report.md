# Eval-Power Harness Report (roadmap item 1)

Thresholds: >=50 target/family, >=200 non-target/family, resolvable delta <= 0.02.

Source: D:\Research_Engine\runs\eval_power_harness\powered_arc_challenge_1024_seed23\choice_constrained_sample_scores.jsonl.

Claim boundary: per-family resolution measured on a freshly scored set (D:\Research_Engine\runs\eval_power_harness\powered_arc_challenge_1024_seed23\choice_constrained_sample_scores.jsonl); projection columns are advisory.

## Measured resolution

Pooled unique items: 1024; pooled fail rate: 0.252.

| granularity | families | target-powered | fully-powered | worst resolvable delta |
|---|---:|---:|---:|---:|
| top_target_runner | 36 | 0 | 0 | inf |
| top_target | 12 | 0 | 0 | inf |
| target | 4 | 3 | 3 | 0.0182 |

## Projection to 1024 scored items

Estimator: expected_target = n_items * cached_fail_rate * family_share. Seeds replicate the same items for cross-seed recurrence and do not add unique target samples; item count is the power driver.

| granularity | families | req. items (top family) | req. items (all >=2) | 1024 powers top? | 1024 powers all? |
|---|---:|---:|---:|:--:|:--:|
| top_target_runner | 36 | 2439 | 25600 | NO | NO |
| top_target | 12 | 1191 | 25600 | NO | NO |
| target | 4 | 534 | 1138 | yes | NO |

## Verdict

Item 1 achieved at: `target` (3/4 families, worst delta 0.0182). Define powered cells at this granularity. Finer granularities stay underpowered on this suite (e.g. `top_target_runner` tops out at 21 target/family); powering them needs a second task suite.

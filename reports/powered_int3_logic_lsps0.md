# LSPS-0: Loop-Spline Patch Search v0

Final label: `COORDINATE_SYSTEM_VOID`

Discriminator final label: `STATIC_ENDPOINT_SUFFICIENT` (override active)

VPD-style/SVD rank-one component dictionary over a frozen TRM. No actual Goodfire VPD is claimed; no cached 4D ARC heads; base model never updated; no policy/RCPI/consolidation; this is not self-improvement.

## Setup

- Families (source_env with >= 6 rows and >= 1 model error): `5`
- Required families: `5`; proposal target: `24` examples/family
- kappa_max=`8`, shortlist B=`4`, alpha_probe_mag=`0.25`
- Trust region rho=`0.05`, accept delta(eta)=`0.0500` (NO_LORA_YARDSTICK fallback (no matched LoRA baseline available)), accept epsilon_j=`0.0`
- Power notes: `['FAMILY_CAP_APPLIED', 'DISCRIMINATOR_GATE_BLOCKED:STATIC_ENDPOINT_SUFFICIENT']`

## Summary

```json
{
  "any_accept_families": 1,
  "b3_accept_families": 1,
  "beats_b3_by_family": {
    "logic_arrow_maze": 0,
    "logic_boolean_expressions": 1,
    "logic_buggy_tables": 1,
    "logic_calcudoko": 1,
    "logic_campsite": 1
  },
  "lsps0_accept_families": 0,
  "lsps0_beats_b3_families": 4,
  "n_families": 5
}
```

## Results

| family | method | accepted | holdout_gain | regression_max_ucb | support_size | trust_region_max | evals | beats_B3 | interaction_flag |
| ------ | ------ | -------- | -----------: | -----------------: | -----------: | ---------------: | ----: | -------- | ---------------- |
| logic_arrow_maze | B1_random | False | +0.3676 | +0.0024 | 2 | 0.0500 | 31 | — | False |
| logic_arrow_maze | B2_static_norm | True | +0.7379 | -0.0436 | 8 | 0.0500 | 99 | — | False |
| logic_arrow_maze | B3_static_damage | True | +0.9225 | -0.1228 | 8 | 0.0500 | 99 | — | False |
| logic_arrow_maze | B4_gradient_only | True | +0.9478 | -0.1259 | 8 | 0.0500 | 99 | — | False |
| logic_arrow_maze | LSPS0 | False | +0.0000 | +0.0000 | 0 | 0.0000 | 4 | no | False |
| logic_boolean_expressions | B1_random | False | +0.0775 | +0.0590 | 3 | 0.0500 | 43 | — | False |
| logic_boolean_expressions | B2_static_norm | False | +0.1179 | +0.0298 | 8 | 0.0500 | 99 | — | False |
| logic_boolean_expressions | B3_static_damage | False | +0.0846 | +0.0521 | 3 | 0.0500 | 43 | — | False |
| logic_boolean_expressions | B4_gradient_only | False | +0.1188 | +0.0678 | 7 | 0.0500 | 91 | — | False |
| logic_boolean_expressions | LSPS0 | False | +0.0974 | +0.0398 | 5 | 0.0500 | 67 | yes | False |
| logic_buggy_tables | B1_random | False | +0.0000 | +0.0000 | 0 | 0.0000 | 4 | — | False |
| logic_buggy_tables | B2_static_norm | False | -0.0064 | +0.1160 | 5 | 0.0500 | 67 | — | False |
| logic_buggy_tables | B3_static_damage | False | -0.0073 | +0.1209 | 7 | 0.0500 | 91 | — | False |
| logic_buggy_tables | B4_gradient_only | False | -0.0052 | +0.3130 | 8 | 0.0500 | 99 | — | False |
| logic_buggy_tables | LSPS0 | False | -0.0000 | +0.1677 | 3 | 0.0500 | 43 | yes | False |
| logic_calcudoko | B1_random | False | +0.0079 | +0.1311 | 4 | 0.0500 | 55 | — | False |
| logic_calcudoko | B2_static_norm | False | +0.1048 | +0.0714 | 8 | 0.0500 | 99 | — | False |
| logic_calcudoko | B3_static_damage | False | +0.0954 | +0.0738 | 8 | 0.0500 | 99 | — | False |
| logic_calcudoko | B4_gradient_only | False | +0.1088 | +0.0750 | 8 | 0.0500 | 99 | — | False |
| logic_calcudoko | LSPS0 | False | +0.0980 | +0.0841 | 8 | 0.0500 | 99 | yes | False |
| logic_campsite | B1_random | False | +0.0002 | +0.1130 | 3 | 0.0500 | 43 | — | False |
| logic_campsite | B2_static_norm | False | +0.0323 | +0.0705 | 4 | 0.0500 | 55 | — | False |
| logic_campsite | B3_static_damage | False | +0.0266 | +0.0502 | 1 | 0.0500 | 19 | — | False |
| logic_campsite | B4_gradient_only | False | +0.0266 | +0.0502 | 1 | 0.0500 | 19 | — | False |
| logic_campsite | LSPS0 | False | +0.0432 | +0.0448 | 5 | 0.0500 | 67 | yes | False |

## Honesty gates

- B4 gradient-only is implemented via rank-one projection u^T (dL/dW) v (not blocked).
- g_t(tau) uses the loss-gradient channel (per-step pooled-state gradient norm); full Psi gradients are not exposed by this checkpoint. Reported as a limitation.
- Spline advantage is only claimed if LSPS-0 beats B3 static-damage; see final label.

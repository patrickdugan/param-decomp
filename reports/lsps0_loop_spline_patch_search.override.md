# LSPS-0: Loop-Spline Patch Search v0

Final label: `UNDERPOWERED_LSPS0`

Discriminator final label: `STATIC_ENDPOINT_SUFFICIENT` (override active)

VPD-style/SVD rank-one component dictionary over a frozen TRM. No actual Goodfire VPD is claimed; no cached 4D ARC heads; base model never updated; no policy/RCPI/consolidation; this is not self-improvement.

## Setup

- Families (source_env with >= 6 rows and >= 1 model error): `2`
- Required families: `5`; proposal target: `64` examples/family
- kappa_max=`8`, shortlist B=`4`, alpha_probe_mag=`0.25`
- Trust region rho=`0.05`, accept delta(eta)=`0.0500` (NO_LORA_YARDSTICK fallback (no matched LoRA baseline available)), accept epsilon_j=`0.0`
- Power notes: `['UNDERPOWERED_FAMILY_COUNT', 'PROPOSAL_BATCH_BELOW_TARGET', 'FAMILY_CAP_APPLIED', 'DISCRIMINATOR_OVERRIDE_NO_EVIDENCE']`

## Summary

```json
{
  "any_accept_families": 0,
  "b3_accept_families": 0,
  "beats_b3_by_family": {
    "aime2024": 0,
    "psycho_bench": 1
  },
  "lsps0_accept_families": 0,
  "lsps0_beats_b3_families": 1,
  "n_families": 2
}
```

## Results

| family | method | accepted | holdout_gain | regression_max_ucb | support_size | trust_region_max | evals | beats_B3 | interaction_flag |
| ------ | ------ | -------- | -----------: | -----------------: | -----------: | ---------------: | ----: | -------- | ---------------- |
| aime2024 | B1_random | False | +0.0100 | +0.1229 | 1 | 0.0500 | 19 | — | False |
| aime2024 | B2_static_norm | False | +0.5770 | +0.1056 | 8 | 0.0500 | 99 | — | False |
| aime2024 | B3_static_damage | False | +0.5972 | +0.1187 | 8 | 0.0500 | 99 | — | False |
| aime2024 | B4_gradient_only | False | +0.6559 | +0.1051 | 8 | 0.0500 | 99 | — | False |
| aime2024 | LSPS0 | False | +0.0317 | +0.1395 | 3 | 0.0500 | 43 | no | False |
| psycho_bench | B1_random | False | +0.1289 | +0.1121 | 1 | 0.0388 | 19 | — | False |
| psycho_bench | B2_static_norm | False | +0.0383 | +0.1107 | 1 | 0.0500 | 19 | — | False |
| psycho_bench | B3_static_damage | False | +0.0000 | +0.0000 | 0 | 0.0000 | 4 | — | False |
| psycho_bench | B4_gradient_only | False | +0.0000 | +0.0000 | 0 | 0.0000 | 4 | — | False |
| psycho_bench | LSPS0 | False | +0.0927 | +0.0999 | 2 | 0.0500 | 31 | yes | False |

## Honesty gates

- B4 gradient-only is implemented via rank-one projection u^T (dL/dW) v (not blocked).
- g_t(tau) uses the loss-gradient channel (per-step pooled-state gradient norm); full Psi gradients are not exposed by this checkpoint. Reported as a limitation.
- Spline advantage is only claimed if LSPS-0 beats B3 static-damage; see final label.

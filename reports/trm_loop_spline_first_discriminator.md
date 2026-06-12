# TRM Loop-Spline First Discriminator

Final label: `STATIC_ENDPOINT_SUFFICIENT`

Model: `D:\projects\HRM-re\experiments\hermes_skill_gym\outputs\hermes_skill_gym_trm_action_family_explicit_single\variants\trm_like_single_tier\artifacts\model.pt`
Component count: `52`; sampled: `20`
Probe A/B counts: `16` / `16`
Base depth T: `3`; T_prime: `6`

No actual Goodfire VPD is claimed. This uses a VPD-style/SVD rank-one component dictionary over recursive tiny model tensors. Cached 4D ARC heads were not used.

## Reliability

```json
{
  "feature_spearman": {
    "a0.5_confidence_centroid": 0.6300751879699248,
    "a0.5_confidence_early_late": 0.7954887218045112,
    "a0.5_confidence_monotonicity": 0.7082706766917292,
    "a0.5_confidence_peak_step": 0.6857142857142857,
    "a0.5_confidence_signed_peak": 0.8842105263157894,
    "a0.5_logit_margin_centroid": 0.6075187969924812,
    "a0.5_logit_margin_early_late": 0.5007518796992482,
    "a0.5_logit_margin_monotonicity": 0.14436090225563908,
    "a0.5_logit_margin_peak_step": NaN,
    "a0.5_logit_margin_signed_peak": 0.3744360902255639,
    "a0.5_output_action_score_centroid": 0.5849624060150376,
    "a0.5_output_action_score_early_late": 0.5864661654135337,
    "a0.5_output_action_score_monotonicity": 0.5293233082706766,
    "a0.5_output_action_score_peak_step": NaN,
    "a0.5_output_action_score_signed_peak": 0.3278195488721804,
    "a0.5_y_norm_centroid": 0.6360902255639098,
    "a0.5_y_norm_early_late": 0.9263157894736842,
    "a0.5_y_norm_monotonicity": 0.6496240601503759,
    "a0.5_y_norm_peak_step": NaN,
    "a0.5_y_norm_signed_peak": 0.9097744360902256,
    "a0.5_z_drift_centroid": 0.8781954887218045,
    "a0.5_z_drift_early_late": NaN,
    "a0.5_z_drift_monotonicity": 0.8781954887218045,
    "a0.5_z_drift_peak_step": NaN,
    "a0.5_z_drift_period2_power": 0.8781954887218045,
    "a0.5_z_drift_signed_peak": 0.6962406015037594,
    "a0.5_z_norm_centroid": 0.5849624060150376,
    "a0.5_z_norm_early_late": 0.7548872180451127,
    "a0.5_z_norm_monotonicity": 0.7819548872180451,
    "a0.5_z_norm_peak_step": 1.0,
    "a0.5_z_norm_signed_peak": 0.7157894736842104,
    "a1.5_confidence_centroid": 0.5684210526315789,
    "a1.5_confidence_early_late": 0.6060150375939849,
    "a1.5_confidence_monotonicity": 0.5413533834586466,
    "a1.5_confidence_peak_step": NaN,
    "a1.5_confidence_signed_peak": 0.8345864661654135,
    "a1.5_logit_margin_centroid": 0.6045112781954887,
    "a1.5_logit_margin_early_late": 0.22406015037593985,
    "a1.5_logit_margin_monotonicity": 0.2270676691729323,
    "a1.5_logit_margin_peak_step": NaN,
    "a1.5_logit_margin_signed_peak": 0.09473684210526315,
    "a1.5_output_action_score_centroid": 0.8556390977443609,
    "a1.5_output_action_score_early_late": 0.7654135338345864,
    "a1.5_output_action_score_monotonicity": 0.23759398496240602,
    "a1.5_output_action_score_peak_step": 0.6902255639097743,
    "a1.5_output_action_score_signed_peak": 0.2270676691729323,
    "a1.5_y_norm_centroid": 0.9879699248120299,
    "a1.5_y_norm_early_late": 0.9849624060150375,
    "a1.5_y_norm_monotonicity": 0.9037593984962405,
    "a1.5_y_norm_peak_step": NaN,
    "a1.5_y_norm_signed_peak": 0.8586466165413533,
    "a1.5_z_drift_centroid": 0.8947368421052632,
    "a1.5_z_drift_early_late": NaN,
    "a1.5_z_drift_monotonicity": 0.8947368421052632,
    "a1.5_z_drift_peak_step": NaN,
    "a1.5_z_drift_period2_power": 0.8947368421052632,
    "a1.5_z_drift_signed_peak": 0.8796992481203006,
    "a1.5_z_norm_centroid": 0.7924812030075187,
    "a1.5_z_norm_early_late": 0.7819548872180451,
    "a1.5_z_norm_monotonicity": 0.8451127819548873,
    "a1.5_z_norm_peak_step": NaN,
    "a1.5_z_norm_signed_peak": 0.6857142857142857,
    "alpha_asym_confidence": 0.9022556390977443,
    "alpha_asym_logit_margin": 0.45563909774436084,
    "alpha_asym_output_action_score": 0.8796992481203006,
    "alpha_asym_y_norm": 0.9503759398496241,
    "alpha_asym_z_drift": 0.9849624060150375,
    "alpha_asym_z_norm": 0.8932330827067668
  },
  "gate_pass": true,
  "overall_median_spearman": 0.7548872180451127,
  "top_damage_quartile_median_spearman": 0.7548872180451127
}
```

## Nested Models

```json
{
  "auc_skipped_reason": "insufficient positive/negative class balance",
  "cv_auc_static": null,
  "cv_auc_static_plus_phi": null,
  "cv_spearman_static": -0.02857142857142857,
  "cv_spearman_static_plus_phi": 0.4796992481203006,
  "delta_auc": null,
  "delta_spearman": 0.5082706766917292,
  "permutation_p_delta_metric": 0.6190476190476191,
  "permutations": 20
}
```

## Per-Decile Breakdown

```json
{
  "0": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "1": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "2": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "3": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "4": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "5": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "6": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "7": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "8": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "9": {
    "count": 2,
    "mean_delta_damage": 2.0,
    "mean_static_damage": 1.5
  }
}
```

## Top Partial Features

```json
[
  {
    "feature": "a1.5_logit_margin_peak_step",
    "partial_spearman_proxy": -0.9864661654135337
  },
  {
    "feature": "a1.5_logit_margin_centroid",
    "partial_spearman_proxy": -0.9172932330827066
  },
  {
    "feature": "a0.5_output_action_score_monotonicity",
    "partial_spearman_proxy": 0.900751879699248
  },
  {
    "feature": "alpha_asym_logit_margin",
    "partial_spearman_proxy": 0.900751879699248
  },
  {
    "feature": "a1.5_logit_margin_early_late",
    "partial_spearman_proxy": 0.8406015037593983
  },
  {
    "feature": "a0.5_logit_margin_signed_peak",
    "partial_spearman_proxy": -0.825563909774436
  },
  {
    "feature": "a1.5_logit_margin_monotonicity",
    "partial_spearman_proxy": -0.7819548872180451
  },
  {
    "feature": "a1.5_confidence_monotonicity",
    "partial_spearman_proxy": -0.6330827067669172
  },
  {
    "feature": "a0.5_logit_margin_peak_step",
    "partial_spearman_proxy": -0.6030075187969924
  },
  {
    "feature": "a1.5_output_action_score_early_late",
    "partial_spearman_proxy": 0.5954887218045113
  },
  {
    "feature": "a1.5_confidence_centroid",
    "partial_spearman_proxy": -0.5909774436090225
  },
  {
    "feature": "a0.5_y_norm_centroid",
    "partial_spearman_proxy": 0.5804511278195488
  },
  {
    "feature": "a0.5_y_norm_early_late",
    "partial_spearman_proxy": -0.524812030075188
  },
  {
    "feature": "a0.5_z_norm_peak_step",
    "partial_spearman_proxy": 0.4992481203007519
  },
  {
    "feature": "a1.5_confidence_early_late",
    "partial_spearman_proxy": 0.49323308270676686
  },
  {
    "feature": "a0.5_z_norm_centroid",
    "partial_spearman_proxy": 0.4766917293233082
  },
  {
    "feature": "a1.5_output_action_score_centroid",
    "partial_spearman_proxy": -0.4646616541353384
  },
  {
    "feature": "a0.5_z_norm_early_late",
    "partial_spearman_proxy": -0.45714285714285713
  },
  {
    "feature": "alpha_asym_y_norm",
    "partial_spearman_proxy": 0.45112781954887216
  },
  {
    "feature": "a1.5_output_action_score_signed_peak",
    "partial_spearman_proxy": -0.38496240601503756
  }
]
```

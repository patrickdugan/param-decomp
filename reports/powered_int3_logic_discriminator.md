# TRM Loop-Spline First Discriminator

Final label: `STATIC_ENDPOINT_SUFFICIENT`

Model: `artifacts\powered_int3_logic\model`
Component count: `52`; sampled: `30`
Probe A/B counts: `48` / `48`
Base depth T: `3`; T_prime: `6`

No actual Goodfire VPD is claimed. This uses a VPD-style/SVD rank-one component dictionary over recursive tiny model tensors. Cached 4D ARC heads were not used.

## Reliability

```json
{
  "feature_spearman": {
    "a0.5_confidence_centroid": 0.6560622914349278,
    "a0.5_confidence_early_late": 0.7846496106785317,
    "a0.5_confidence_monotonicity": 0.3837597330367074,
    "a0.5_confidence_peak_step": 0.6422691879866518,
    "a0.5_confidence_signed_peak": 0.418020022246941,
    "a0.5_logit_margin_centroid": 0.6698553948832036,
    "a0.5_logit_margin_early_late": 0.6471635150166852,
    "a0.5_logit_margin_monotonicity": 0.7041156840934372,
    "a0.5_logit_margin_peak_step": 0.6204671857619577,
    "a0.5_logit_margin_signed_peak": -0.6097886540600668,
    "a0.5_output_action_score_centroid": 0.4905450500556173,
    "a0.5_output_action_score_early_late": 0.18798665183537264,
    "a0.5_output_action_score_monotonicity": 0.800222469410456,
    "a0.5_output_action_score_peak_step": NaN,
    "a0.5_output_action_score_signed_peak": 0.2849833147942158,
    "a0.5_y_norm_centroid": 0.7757508342602891,
    "a0.5_y_norm_early_late": 0.7365962180200223,
    "a0.5_y_norm_monotonicity": 0.4901001112347052,
    "a0.5_y_norm_peak_step": NaN,
    "a0.5_y_norm_signed_peak": 0.593325917686318,
    "a0.5_z_drift_centroid": 0.8816462736373749,
    "a0.5_z_drift_early_late": NaN,
    "a0.5_z_drift_monotonicity": 0.8816462736373749,
    "a0.5_z_drift_peak_step": NaN,
    "a0.5_z_drift_period2_power": 0.8816462736373749,
    "a0.5_z_drift_signed_peak": 0.7619577308120132,
    "a0.5_z_norm_centroid": 0.821579532814238,
    "a0.5_z_norm_early_late": 0.9612903225806452,
    "a0.5_z_norm_monotonicity": 0.6160177975528365,
    "a0.5_z_norm_peak_step": NaN,
    "a0.5_z_norm_signed_peak": 0.803337041156841,
    "a1.5_confidence_centroid": 0.7477196885428253,
    "a1.5_confidence_early_late": 0.6631813125695216,
    "a1.5_confidence_monotonicity": -0.07675194660734148,
    "a1.5_confidence_peak_step": 0.5804226918798664,
    "a1.5_confidence_signed_peak": 0.49454949944382637,
    "a1.5_logit_margin_centroid": 0.575083426028921,
    "a1.5_logit_margin_early_late": 0.6845383759733037,
    "a1.5_logit_margin_monotonicity": 0.689432703003337,
    "a1.5_logit_margin_peak_step": NaN,
    "a1.5_logit_margin_signed_peak": -0.16662958843159065,
    "a1.5_output_action_score_centroid": 0.5301446051167964,
    "a1.5_output_action_score_early_late": 0.4384872080088987,
    "a1.5_output_action_score_monotonicity": 0.7339265850945496,
    "a1.5_output_action_score_peak_step": NaN,
    "a1.5_output_action_score_signed_peak": 0.6925472747497218,
    "a1.5_y_norm_centroid": 0.8162402669632925,
    "a1.5_y_norm_early_late": 0.6889877641824249,
    "a1.5_y_norm_monotonicity": 0.7815350389321468,
    "a1.5_y_norm_peak_step": NaN,
    "a1.5_y_norm_signed_peak": 0.7793103448275861,
    "a1.5_z_drift_centroid": 0.84293659621802,
    "a1.5_z_drift_early_late": NaN,
    "a1.5_z_drift_monotonicity": 0.84293659621802,
    "a1.5_z_drift_peak_step": NaN,
    "a1.5_z_drift_period2_power": 0.84293659621802,
    "a1.5_z_drift_signed_peak": 0.8745272525027808,
    "a1.5_z_norm_centroid": 0.8576195773081201,
    "a1.5_z_norm_early_late": 0.9119021134593993,
    "a1.5_z_norm_monotonicity": 0.5301446051167964,
    "a1.5_z_norm_peak_step": NaN,
    "a1.5_z_norm_signed_peak": 0.7899888765294771,
    "alpha_asym_confidence": 0.7486095661846496,
    "alpha_asym_logit_margin": 0.11368186874304784,
    "alpha_asym_output_action_score": 0.5265850945494994,
    "alpha_asym_y_norm": 0.9692992213570634,
    "alpha_asym_z_drift": 0.8558398220244716,
    "alpha_asym_z_norm": 0.9448275862068966
  },
  "gate_pass": true,
  "overall_median_spearman": 0.6925472747497218,
  "top_damage_quartile_median_spearman": 0.7333333333333334
}
```

## Nested Models

```json
{
  "auc_skipped_reason": null,
  "cv_auc_static": 0.5416666666666666,
  "cv_auc_static_plus_phi": 0.4930555555555556,
  "cv_spearman_static": 0.20489432703003335,
  "cv_spearman_static_plus_phi": 0.18620689655172412,
  "delta_auc": -0.04861111111111105,
  "delta_spearman": -0.01868743047830923,
  "permutation_p_delta_metric": 0.49019607843137253,
  "permutations": 50
}
```

## Per-Decile Breakdown

```json
{
  "0": {
    "count": 3,
    "mean_delta_damage": 0.6666666666666666,
    "mean_static_damage": 0.0
  },
  "1": {
    "count": 3,
    "mean_delta_damage": 0.3333333333333333,
    "mean_static_damage": 0.0
  },
  "2": {
    "count": 3,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "3": {
    "count": 3,
    "mean_delta_damage": 0.6666666666666666,
    "mean_static_damage": 0.0
  },
  "4": {
    "count": 3,
    "mean_delta_damage": 0.6666666666666666,
    "mean_static_damage": 0.0
  },
  "5": {
    "count": 3,
    "mean_delta_damage": -0.6666666666666666,
    "mean_static_damage": 1.0
  },
  "6": {
    "count": 3,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 1.0
  },
  "7": {
    "count": 3,
    "mean_delta_damage": -1.6666666666666667,
    "mean_static_damage": 2.0
  },
  "8": {
    "count": 3,
    "mean_delta_damage": -1.0,
    "mean_static_damage": 2.6666666666666665
  },
  "9": {
    "count": 3,
    "mean_delta_damage": -2.3333333333333335,
    "mean_static_damage": 9.333333333333334
  }
}
```

## Top Partial Features

```json
[
  {
    "feature": "a1.5_z_norm_peak_step",
    "partial_spearman_proxy": 0.6849833147942158
  },
  {
    "feature": "a0.5_y_norm_monotonicity",
    "partial_spearman_proxy": -0.6342602892102336
  },
  {
    "feature": "alpha_asym_z_drift",
    "partial_spearman_proxy": -0.611123470522803
  },
  {
    "feature": "a1.5_y_norm_signed_peak",
    "partial_spearman_proxy": 0.6062291434927697
  },
  {
    "feature": "a1.5_confidence_centroid",
    "partial_spearman_proxy": -0.5937708565072302
  },
  {
    "feature": "a0.5_z_drift_centroid",
    "partial_spearman_proxy": 0.5911012235817574
  },
  {
    "feature": "a0.5_z_drift_period2_power",
    "partial_spearman_proxy": -0.5911012235817574
  },
  {
    "feature": "a0.5_z_drift_monotonicity",
    "partial_spearman_proxy": 0.5808676307007786
  },
  {
    "feature": "a1.5_z_drift_centroid",
    "partial_spearman_proxy": -0.5808676307007786
  },
  {
    "feature": "a1.5_z_drift_monotonicity",
    "partial_spearman_proxy": -0.5808676307007786
  },
  {
    "feature": "a1.5_z_drift_period2_power",
    "partial_spearman_proxy": 0.5808676307007786
  },
  {
    "feature": "a0.5_z_norm_monotonicity",
    "partial_spearman_proxy": -0.5670745272525028
  },
  {
    "feature": "a0.5_confidence_peak_step",
    "partial_spearman_proxy": 0.5194660734149055
  },
  {
    "feature": "a1.5_z_drift_signed_peak",
    "partial_spearman_proxy": -0.5181312569521691
  },
  {
    "feature": "a1.5_confidence_peak_step",
    "partial_spearman_proxy": 0.5083426028921022
  },
  {
    "feature": "a1.5_output_action_score_centroid",
    "partial_spearman_proxy": -0.4905450500556173
  },
  {
    "feature": "a1.5_z_norm_monotonicity",
    "partial_spearman_proxy": 0.4771968854282535
  },
  {
    "feature": "a0.5_y_norm_signed_peak",
    "partial_spearman_proxy": -0.4384872080088987
  },
  {
    "feature": "a1.5_y_norm_monotonicity",
    "partial_spearman_proxy": 0.43804226918798667
  },
  {
    "feature": "a0.5_y_norm_peak_step",
    "partial_spearman_proxy": -0.3890989988876529
  }
]
```

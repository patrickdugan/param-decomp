# Hermes Real-VPD Loop-Spline Discriminator

Final label: `PROFILE_UNSTABLE`

Critic model: `D:\projects\HRM-re\experiments\hermes_skill_gym\outputs\hermes_skill_gym_trm_action_family_explicit_single\variants\trm_like_single_tier\artifacts\model.pt`
SPD checkpoint: `D:\Research_Engine\runs\trm_param_decomp\hermes_logic_critic_100\model_100.pth`
Component count: `80`; sampled: `12`
Probe A/B counts: `12` / `12`
Base depth T: `3`; T_prime: `6`

Components are real SPD/VPD rank components from this repo's `ComponentModel`, not SVD atoms.

## Reliability

```json
{
  "feature_spearman": {
    "a0.5_confidence_centroid": -0.2517482517482518,
    "a0.5_confidence_early_late": -0.2937062937062937,
    "a0.5_confidence_monotonicity": 0.6853146853146854,
    "a0.5_confidence_peak_step": NaN,
    "a0.5_confidence_signed_peak": 0.8181818181818183,
    "a0.5_logit_margin_centroid": 0.7202797202797203,
    "a0.5_logit_margin_early_late": 0.7902097902097903,
    "a0.5_logit_margin_monotonicity": 0.5524475524475525,
    "a0.5_logit_margin_peak_step": NaN,
    "a0.5_logit_margin_signed_peak": 0.7132867132867133,
    "a0.5_output_action_score_centroid": 0.9790209790209792,
    "a0.5_output_action_score_early_late": 0.9720279720279721,
    "a0.5_output_action_score_monotonicity": -0.4055944055944056,
    "a0.5_output_action_score_peak_step": NaN,
    "a0.5_output_action_score_signed_peak": 0.6503496503496504,
    "a0.5_y_norm_centroid": 0.4545454545454546,
    "a0.5_y_norm_early_late": 0.3846153846153847,
    "a0.5_y_norm_monotonicity": 0.8811188811188813,
    "a0.5_y_norm_peak_step": NaN,
    "a0.5_y_norm_signed_peak": 0.8951048951048951,
    "a0.5_z_drift_centroid": 0.3636363636363637,
    "a0.5_z_drift_early_late": NaN,
    "a0.5_z_drift_monotonicity": 0.3636363636363637,
    "a0.5_z_drift_peak_step": NaN,
    "a0.5_z_drift_period2_power": 0.3636363636363637,
    "a0.5_z_drift_signed_peak": 0.965034965034965,
    "a0.5_z_norm_centroid": 0.37762237762237766,
    "a0.5_z_norm_early_late": 0.6223776223776225,
    "a0.5_z_norm_monotonicity": 0.1398601398601399,
    "a0.5_z_norm_peak_step": NaN,
    "a0.5_z_norm_signed_peak": 0.965034965034965,
    "a1.5_confidence_centroid": 0.5874125874125874,
    "a1.5_confidence_early_late": 0.31468531468531474,
    "a1.5_confidence_monotonicity": 0.4265734265734266,
    "a1.5_confidence_peak_step": NaN,
    "a1.5_confidence_signed_peak": 0.9160839160839163,
    "a1.5_logit_margin_centroid": 0.7622377622377624,
    "a1.5_logit_margin_early_late": 0.7132867132867133,
    "a1.5_logit_margin_monotonicity": 0.49650349650349657,
    "a1.5_logit_margin_peak_step": NaN,
    "a1.5_logit_margin_signed_peak": 0.8671328671328673,
    "a1.5_output_action_score_centroid": 0.9860139860139862,
    "a1.5_output_action_score_early_late": 0.9790209790209792,
    "a1.5_output_action_score_monotonicity": 0.07692307692307693,
    "a1.5_output_action_score_peak_step": NaN,
    "a1.5_output_action_score_signed_peak": 0.5734265734265735,
    "a1.5_y_norm_centroid": 0.3636363636363637,
    "a1.5_y_norm_early_late": 0.3426573426573427,
    "a1.5_y_norm_monotonicity": 0.6503496503496504,
    "a1.5_y_norm_peak_step": NaN,
    "a1.5_y_norm_signed_peak": 0.8951048951048951,
    "a1.5_z_drift_centroid": 0.5874125874125874,
    "a1.5_z_drift_early_late": NaN,
    "a1.5_z_drift_monotonicity": 0.5874125874125874,
    "a1.5_z_drift_peak_step": NaN,
    "a1.5_z_drift_period2_power": 0.5804195804195805,
    "a1.5_z_drift_signed_peak": 0.965034965034965,
    "a1.5_z_norm_centroid": 0.5804195804195805,
    "a1.5_z_norm_early_late": 0.5734265734265735,
    "a1.5_z_norm_monotonicity": 0.5174825174825175,
    "a1.5_z_norm_peak_step": NaN,
    "a1.5_z_norm_signed_peak": 0.965034965034965,
    "alpha_asym_confidence": 0.46153846153846156,
    "alpha_asym_logit_margin": 0.9020979020979022,
    "alpha_asym_output_action_score": 0.8741258741258742,
    "alpha_asym_y_norm": 0.9090909090909092,
    "alpha_asym_z_drift": 0.7132867132867133,
    "alpha_asym_z_norm": 0.7692307692307694
  },
  "gate_pass": false,
  "overall_median_spearman": 0.6363636363636365,
  "top_damage_quartile_median_spearman": 0.6363636363636365
}
```

## Nested Models

```json
{
  "reason": "reliability_gate_failed_or_constant_target",
  "skipped": true,
  "target_unique_values": [
    0.0
  ]
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
    "count": 1,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "2": {
    "count": 1,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "3": {
    "count": 1,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "4": {
    "count": 1,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "5": {
    "count": 1,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "6": {
    "count": 2,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "7": {
    "count": 1,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "8": {
    "count": 1,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  },
  "9": {
    "count": 1,
    "mean_delta_damage": 0.0,
    "mean_static_damage": 0.0
  }
}
```

## Top Static VPD Components

```json
[
  {
    "component_id": "networks.0.layers.0.1.to_out:c005",
    "component_index": 5,
    "endpoint_final_step_loss_alpha0_probe_A": 2.5492103695869446,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.19868801845048262,
    "role": "recursive_trunk_z",
    "scale": 1.3146227598190308,
    "static_ablation_damage_T_probe_A": 4,
    "svd_rank_index": 5,
    "u_norm": 1.418868899345398,
    "uv_norm_product": 1.3146235914900473,
    "v_norm": 0.9265292882919312
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c004",
    "component_index": 4,
    "endpoint_final_step_loss_alpha0_probe_A": 1.6390071213245392,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.18746307881978774,
    "role": "recursive_trunk_z",
    "scale": 1.240352749824524,
    "static_ablation_damage_T_probe_A": 3,
    "svd_rank_index": 4,
    "u_norm": 1.08670973777771,
    "uv_norm_product": 1.240353242917081,
    "v_norm": 1.1413841247558594
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c025",
    "component_index": 25,
    "endpoint_final_step_loss_alpha0_probe_A": 1.5978287160396576,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.1681063449622484,
    "role": "recursive_trunk_z",
    "scale": 1.1122785806655884,
    "static_ablation_damage_T_probe_A": 3,
    "svd_rank_index": 25,
    "u_norm": 1.3282885551452637,
    "uv_norm_product": 1.1122796470106096,
    "v_norm": 0.8373780250549316
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c030",
    "component_index": 30,
    "endpoint_final_step_loss_alpha0_probe_A": 1.2653471529483795,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.21175495427557087,
    "role": "recursive_trunk_z",
    "scale": 1.4010803699493408,
    "static_ablation_damage_T_probe_A": 3,
    "svd_rank_index": 30,
    "u_norm": 1.3755261898040771,
    "uv_norm_product": 1.4010812795284266,
    "v_norm": 1.0185784101486206
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c000",
    "component_index": 0,
    "endpoint_final_step_loss_alpha0_probe_A": 1.2905755639076233,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.19227568732220449,
    "role": "recursive_trunk_z",
    "scale": 1.2721954584121704,
    "static_ablation_damage_T_probe_A": 0,
    "svd_rank_index": 0,
    "u_norm": 1.0960233211517334,
    "uv_norm_product": 1.272196430872441,
    "v_norm": 1.160738468170166
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c001",
    "component_index": 1,
    "endpoint_final_step_loss_alpha0_probe_A": 1.2922166287899017,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.20067478062413938,
    "role": "recursive_trunk_z",
    "scale": 1.3277682065963745,
    "static_ablation_damage_T_probe_A": 0,
    "svd_rank_index": 1,
    "u_norm": 1.324721097946167,
    "uv_norm_product": 1.3277694095794743,
    "v_norm": 1.0023010969161987
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c002",
    "component_index": 2,
    "endpoint_final_step_loss_alpha0_probe_A": 0.9121609032154083,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.20769046248217135,
    "role": "recursive_trunk_z",
    "scale": 1.3741875886917114,
    "static_ablation_damage_T_probe_A": 0,
    "svd_rank_index": 2,
    "u_norm": 1.213789701461792,
    "uv_norm_product": 1.374188748042684,
    "v_norm": 1.1321473121643066
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c003",
    "component_index": 3,
    "endpoint_final_step_loss_alpha0_probe_A": 1.2682653069496155,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.17045151779246853,
    "role": "recursive_trunk_z",
    "scale": 1.1277954578399658,
    "static_ablation_damage_T_probe_A": 0,
    "svd_rank_index": 3,
    "u_norm": 1.4137749671936035,
    "uv_norm_product": 1.127796503837999,
    "v_norm": 0.7977199554443359
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c006",
    "component_index": 6,
    "endpoint_final_step_loss_alpha0_probe_A": 1.3801473677158356,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.1882086911764568,
    "role": "recursive_trunk_z",
    "scale": 1.2452861070632935,
    "static_ablation_damage_T_probe_A": 0,
    "svd_rank_index": 6,
    "u_norm": 1.3522645235061646,
    "uv_norm_product": 1.2452866802040035,
    "v_norm": 0.9208898544311523
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c007",
    "component_index": 7,
    "endpoint_final_step_loss_alpha0_probe_A": 1.302201360464096,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.1949490023777294,
    "role": "recursive_trunk_z",
    "scale": 1.2898834943771362,
    "static_ablation_damage_T_probe_A": 0,
    "svd_rank_index": 7,
    "u_norm": 1.1090819835662842,
    "uv_norm_product": 1.2898845449005023,
    "v_norm": 1.1630200147628784
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c008",
    "component_index": 8,
    "endpoint_final_step_loss_alpha0_probe_A": 1.1397700011730194,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.19548441128946184,
    "role": "recursive_trunk_z",
    "scale": 1.2934260368347168,
    "static_ablation_damage_T_probe_A": 0,
    "svd_rank_index": 8,
    "u_norm": 1.4794307947158813,
    "uv_norm_product": 1.2934274605373375,
    "v_norm": 0.8742737174034119
  },
  {
    "component_id": "networks.0.layers.0.1.to_out:c009",
    "component_index": 9,
    "endpoint_final_step_loss_alpha0_probe_A": 0.9719125330448151,
    "module_depth": 0,
    "module_id": "networks.0.layers.0.1.to_out",
    "reconstruction_contribution": 0.1787354829540807,
    "role": "recursive_trunk_z",
    "scale": 1.1826064586639404,
    "static_ablation_damage_T_probe_A": 0,
    "svd_rank_index": 9,
    "u_norm": 1.397910714149475,
    "uv_norm_product": 1.1826072396100784,
    "v_norm": 0.8459819555282593
  }
]
```

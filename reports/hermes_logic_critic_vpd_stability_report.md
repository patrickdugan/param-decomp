# Hermes Logic Critic VPD Stability Report

This compares real SPD/VPD extraction manifests across Hermes critic row windows and places the result next to the earlier SVD loop-spline discriminator.

## Inputs

- `D:\Research_Engine\runs\trm_param_decomp\hermes_logic_critic_100\hermes_manifest.json`: rows start 0, n=32, faithfulness=6e-08
- `D:\Research_Engine\runs\trm_param_decomp\hermes_logic_critic_100\hermes_manifest_rows32_63.json`: rows start 32, n=32, faithfulness=9e-08

## Pairwise Stability

| left start | right start | common | damage Spearman | top-10 overlap | top-10 Jaccard |
|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 80 | 0.991866 | 9 | 0.818182 |

## Trunk vs Head

### Row start 0

- Trunk: mean damage 0.093773, max damage 0.655078, mean alive fraction 0.032453
- Head: mean damage 0.362372, max damage 0.657506, mean alive fraction 0.71875
- Top trunk components: `networks.0.layers.0.1.to_out:c019`, `networks.0.layers.0.1.to_out:c005`, `networks.0.layers.0.1.to_out:c018`, `networks.0.layers.0.1.to_out:c030`, `networks.0.layers.0.1.to_out:c029`
- Top head components: `bucket_head:c014`, `bucket_head:c012`, `bucket_head:c007`, `bucket_head:c013`, `bucket_head:c015`

### Row start 32

- Trunk: mean damage 0.092968, max damage 0.638263, mean alive fraction 0.03241
- Head: mean damage 0.365909, max damage 0.620344, mean alive fraction 0.726562
- Top trunk components: `networks.0.layers.0.1.to_out:c018`, `networks.0.layers.0.1.to_out:c030`, `networks.0.layers.0.1.to_out:c019`, `networks.0.layers.0.1.to_out:c005`, `networks.0.layers.0.1.to_out:c029`
- Top head components: `bucket_head:c014`, `bucket_head:c007`, `bucket_head:c012`, `bucket_head:c002`, `bucket_head:c015`

## Against SVD Loop-Spline

- Prior SVD discriminator label: `STATIC_ENDPOINT_SUFFICIENT`
- Prior static+phi delta Spearman: `0.5082706766917292`
- Prior permutation p-value: `0.6190476190476191`

Interpretation: the SVD lane judged static endpoint features sufficient. The real SPD lane does not directly refute that label yet, but it changes the evidence class: high-damage components now exist inside the recursive trunk with real component-model faithfulness, so the loop-spline question can be rerun on actual VPD components instead of SVD atoms.

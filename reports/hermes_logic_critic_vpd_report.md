# Hermes Logic Critic VPD Report

This report uses this repo's SPD `optimize()` on Fable's recursive HRM logic critic. It is not the earlier SVD loop-spline atom lane.

Behavioral faithfulness relative L2: 6e-08.

## Per-layer decomposition

| module | weight shape | C | component capture | delta fraction | mean alive L0 / activation site | alive components |
|---|---|---:|---:|---:|---:|---:|
| `networks.0.layers.0.1.to_out` | [128, 512] | 32 | 0.779661 | 0.836769 | 1.301453 | 13 |
| `networks.0.layers.1.1.ff.2` | [128, 512] | 32 | 0.656777 | 0.783892 | 0.775513 | 18 |
| `bucket_head` | [3, 128] | 16 | 1.108805 | 0.191921 | 11.5 | 14 |

## Top components by ablation damage

| component | mean CI | alive fraction | output L2 damage | label |
|---|---:|---:|---:|---|
| `bucket_head:c014` | 0.555939 | 0.90625 | 0.657506 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c019` | 0.070632 | 0.257141 | 0.655078 | high_causal_importance_component |
| `bucket_head:c012` | 0.909483 | 0.96875 | 0.617847 | high_causal_importance_component |
| `bucket_head:c007` | 0.156938 | 0.40625 | 0.528384 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c005` | 0.105615 | 0.535461 | 0.524316 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c018` | 0.009847 | 0.0 | 0.522466 | dead_component |
| `networks.0.layers.0.1.to_out:c030` | 0.012398 | 0.0 | 0.497136 | dead_component |
| `bucket_head:c013` | 0.528001 | 0.8125 | 0.436184 | high_causal_importance_component |
| `bucket_head:c015` | 0.623382 | 0.9375 | 0.43394 | high_causal_importance_component |
| `bucket_head:c002` | 0.672311 | 0.75 | 0.406738 | high_causal_importance_component |
| `bucket_head:c004` | 0.481005 | 0.96875 | 0.404347 | high_causal_importance_component |
| `bucket_head:c003` | 0.309569 | 0.6875 | 0.369478 | high_causal_importance_component |
| `bucket_head:c000` | 0.0 | 0.0 | 0.369347 | dead_component |
| `bucket_head:c009` | 0.712559 | 0.90625 | 0.358085 | high_causal_importance_component |
| `bucket_head:c010` | 0.443648 | 0.96875 | 0.290328 | high_causal_importance_component |
| `bucket_head:c005` | 0.40311 | 0.75 | 0.279373 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c029` | 0.018606 | 0.042053 | 0.265126 | high_causal_importance_component |
| `bucket_head:c011` | 0.740383 | 0.84375 | 0.253472 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c020` | 0.009165 | 0.0 | 0.236776 | dead_component |
| `networks.0.layers.0.1.to_out:c025` | 0.018624 | 0.058167 | 0.226255 | high_causal_importance_component |
| `bucket_head:c008` | 0.392058 | 0.65625 | 0.218736 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c031` | 0.0 | 0.0 | 0.17145 | dead_component |
| `bucket_head:c001` | 0.639113 | 0.9375 | 0.166325 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c023` | 0.0 | 0.0 | 0.158756 | dead_component |
| `networks.0.layers.0.1.to_out:c000` | 0.005261 | 0.018127 | 0.14951 | high_causal_importance_component |

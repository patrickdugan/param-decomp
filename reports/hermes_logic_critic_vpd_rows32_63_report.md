# Hermes Logic Critic VPD Report

This report uses this repo's SPD `optimize()` on Fable's recursive HRM logic critic. It is not the earlier SVD loop-spline atom lane.

Behavioral faithfulness relative L2: 9e-08.

## Per-layer decomposition

| module | weight shape | C | component capture | delta fraction | mean alive L0 / activation site | alive components |
|---|---|---:|---:|---:|---:|---:|
| `networks.0.layers.0.1.to_out` | [128, 512] | 32 | 0.779661 | 0.836769 | 1.213684 | 12 |
| `networks.0.layers.1.1.ff.2` | [128, 512] | 32 | 0.656777 | 0.783892 | 0.860535 | 18 |
| `bucket_head` | [3, 128] | 16 | 1.108805 | 0.191921 | 11.625 | 14 |

## Top components by ablation damage

| component | mean CI | alive fraction | output L2 damage | label |
|---|---:|---:|---:|---|
| `networks.0.layers.0.1.to_out:c018` | 0.006381 | 0.0 | 0.638263 | dead_component |
| `bucket_head:c014` | 0.547172 | 0.90625 | 0.620344 | high_causal_importance_component |
| `bucket_head:c007` | 0.247475 | 0.59375 | 0.559384 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c030` | 0.009866 | 0.0 | 0.550732 | dead_component |
| `bucket_head:c012` | 0.842477 | 0.9375 | 0.542556 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c019` | 0.057111 | 0.172852 | 0.532387 | high_causal_importance_component |
| `bucket_head:c002` | 0.785043 | 0.8125 | 0.463178 | high_causal_importance_component |
| `bucket_head:c015` | 0.649501 | 0.9375 | 0.461204 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c005` | 0.118764 | 0.659485 | 0.460244 | high_causal_importance_component |
| `bucket_head:c009` | 0.816338 | 0.96875 | 0.41577 | high_causal_importance_component |
| `bucket_head:c000` | 0.0 | 0.0 | 0.40477 | dead_component |
| `bucket_head:c013` | 0.42309 | 0.78125 | 0.398942 | high_causal_importance_component |
| `bucket_head:c004` | 0.419624 | 0.9375 | 0.376352 | high_causal_importance_component |
| `bucket_head:c003` | 0.248577 | 0.59375 | 0.327434 | high_causal_importance_component |
| `bucket_head:c005` | 0.503083 | 0.8125 | 0.324449 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c029` | 0.030478 | 0.076172 | 0.304787 | high_causal_importance_component |
| `bucket_head:c010` | 0.471779 | 0.90625 | 0.304557 | high_causal_importance_component |
| `bucket_head:c011` | 0.862389 | 0.96875 | 0.295322 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c020` | 0.008239 | 0.0 | 0.206953 | dead_component |
| `bucket_head:c008` | 0.302749 | 0.5 | 0.197637 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c025` | 0.004298 | 0.008606 | 0.191063 | dead_component |
| `networks.0.layers.0.1.to_out:c031` | 0.0 | 0.0 | 0.17206 | dead_component |
| `networks.0.layers.0.1.to_out:c023` | 0.0 | 0.0 | 0.167823 | dead_component |
| `networks.0.layers.0.1.to_out:c015` | 0.000704 | 0.0 | 0.160854 | dead_component |
| `bucket_head:c001` | 0.646107 | 0.96875 | 0.15361 | high_causal_importance_component |

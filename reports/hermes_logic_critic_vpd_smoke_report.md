# Hermes Logic Critic VPD Report

This report uses this repo's SPD `optimize()` on Fable's recursive HRM logic critic. It is not the earlier SVD loop-spline atom lane.

Behavioral faithfulness relative L2: 1.9e-07.

## Per-layer decomposition

| module | weight shape | C | component capture | delta fraction | mean alive L0 / activation site | alive components |
|---|---|---:|---:|---:|---:|---:|
| `networks.0.layers.0.1.to_out` | [128, 512] | 32 | 0.771142 | 0.826344 | 2.468994 | 18 |
| `networks.0.layers.1.1.ff.2` | [128, 512] | 32 | 0.657222 | 0.778201 | 3.975586 | 28 |
| `bucket_head` | [3, 128] | 16 | 1.081808 | 0.20583 | 6.375 | 10 |

## Top components by ablation damage

| component | mean CI | alive fraction | output L2 damage | label |
|---|---:|---:|---:|---|
| `bucket_head:c008` | 1.0 | 1.0 | 1.475208 | high_causal_importance_component |
| `bucket_head:c009` | 0.598906 | 1.0 | 0.643467 | high_causal_importance_component |
| `bucket_head:c002` | 0.0 | 0.0 | 0.618045 | dead_component |
| `networks.0.layers.0.1.to_out:c026` | 0.000347 | 0.0 | 0.4615 | dead_component |
| `networks.0.layers.0.1.to_out:c009` | 0.002594 | 0.0 | 0.433467 | dead_component |
| `bucket_head:c005` | 0.57185 | 0.875 | 0.403858 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c001` | 0.005973 | 0.025391 | 0.363177 | high_causal_importance_component |
| `bucket_head:c004` | 0.06129 | 0.125 | 0.351688 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c007` | 0.052906 | 0.188232 | 0.34626 | high_causal_importance_component |
| `bucket_head:c013` | 0.526409 | 0.625 | 0.346049 | high_causal_importance_component |
| `bucket_head:c010` | 0.0 | 0.0 | 0.314422 | dead_component |
| `bucket_head:c011` | 0.0 | 0.0 | 0.290258 | dead_component |
| `networks.0.layers.0.1.to_out:c018` | 0.038398 | 0.138916 | 0.206125 | high_causal_importance_component |
| `networks.0.layers.1.1.ff.2:c015` | 0.0 | 0.0 | 0.201549 | dead_component |
| `networks.0.layers.0.1.to_out:c022` | 0.000193 | 0.0 | 0.196799 | dead_component |
| `networks.0.layers.1.1.ff.2:c006` | 0.080568 | 0.259766 | 0.193195 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c013` | 0.002335 | 0.002686 | 0.191339 | dead_component |
| `networks.0.layers.0.1.to_out:c008` | 0.028691 | 0.128174 | 0.181844 | high_causal_importance_component |
| `bucket_head:c001` | 0.035129 | 0.125 | 0.176515 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c028` | 0.014112 | 0.0271 | 0.169249 | high_causal_importance_component |
| `bucket_head:c006` | 0.0 | 0.0 | 0.164707 | dead_component |
| `networks.0.layers.0.1.to_out:c010` | 0.02057 | 0.101807 | 0.159916 | high_causal_importance_component |
| `bucket_head:c003` | 0.205657 | 0.75 | 0.158446 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c003` | 0.012155 | 0.0 | 0.156578 | dead_component |
| `networks.0.layers.0.1.to_out:c002` | 0.012117 | 0.009033 | 0.148265 | dead_component |

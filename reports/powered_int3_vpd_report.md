# Hermes Logic Critic VPD Report

This report uses this repo's SPD `optimize()` on a recursive HRM/Hermes logic critic. It is not the earlier SVD loop-spline atom lane.

Behavioral faithfulness relative L2: 8e-08.

## Per-layer decomposition

| module | weight shape | C | component capture | delta fraction | mean alive L0 / activation site | alive components |
|---|---|---:|---:|---:|---:|---:|
| `networks.0.layers.0.1.to_out` | [128, 512] | 32 | 0.786827 | 0.783704 | 7.716858 | 25 |
| `networks.0.layers.1.1.ff.2` | [128, 512] | 32 | 0.687251 | 0.798766 | 2.388123 | 24 |
| `bucket_head` | [3, 128] | 16 | 1.09006 | 0.203684 | 8.03125 | 13 |

## Top components by ablation damage

| component | mean CI | alive fraction | output L2 damage | label |
|---|---:|---:|---:|---|
| `bucket_head:c009` | 0.454176 | 0.75 | 0.511991 | high_causal_importance_component |
| `bucket_head:c007` | 0.538367 | 0.78125 | 0.434641 | high_causal_importance_component |
| `bucket_head:c002` | 0.10127 | 0.125 | 0.347795 | high_causal_importance_component |
| `bucket_head:c000` | 0.00175 | 0.0 | 0.326188 | dead_component |
| `networks.0.layers.0.1.to_out:c000` | 0.346717 | 0.764282 | 0.302251 | high_causal_importance_component |
| `bucket_head:c011` | 0.610395 | 0.875 | 0.296673 | high_causal_importance_component |
| `bucket_head:c006` | 0.74233 | 0.75 | 0.292488 | high_causal_importance_component |
| `bucket_head:c012` | 0.504987 | 0.59375 | 0.280129 | high_causal_importance_component |
| `bucket_head:c004` | 0.512031 | 0.78125 | 0.253958 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c007` | 0.082636 | 0.189148 | 0.247017 | high_causal_importance_component |
| `bucket_head:c001` | 0.457157 | 0.90625 | 0.241125 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c026` | 0.083947 | 0.391357 | 0.228342 | high_causal_importance_component |
| `bucket_head:c015` | 0.473822 | 0.5625 | 0.225351 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c021` | 0.165441 | 0.35498 | 0.22149 | high_causal_importance_component |
| `bucket_head:c010` | 0.006709 | 0.0 | 0.181244 | dead_component |
| `bucket_head:c008` | 0.143583 | 0.28125 | 0.180753 | high_causal_importance_component |
| `bucket_head:c013` | 0.0 | 0.0 | 0.177156 | dead_component |
| `networks.0.layers.0.1.to_out:c015` | 0.092791 | 0.333984 | 0.167758 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c016` | 0.211296 | 0.399719 | 0.15847 | high_causal_importance_component |
| `bucket_head:c014` | 0.152531 | 0.4375 | 0.1464 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c010` | 0.13541 | 0.432312 | 0.14466 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c003` | 0.128068 | 0.290405 | 0.140283 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c006` | 0.038393 | 0.168091 | 0.137143 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c017` | 0.091302 | 0.250854 | 0.129122 | high_causal_importance_component |
| `networks.0.layers.0.1.to_out:c008` | 0.108395 | 0.376648 | 0.123793 | high_causal_importance_component |

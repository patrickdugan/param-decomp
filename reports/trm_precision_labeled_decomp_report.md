# TRM Labeled Decomposition Report (Stages 3-5)

Real Goodfire/param-decomp artifacts were used: this repo's SPD `optimize()` decomposed a genuine frozen TRM, not the 4D cached choice head.

Behavioral faithfulness (components+delta surrogate vs original TRM output, relative L2): 4e-08.

## Per-layer decomposition

component_norm_capture = how much of the weight Frobenius norm the sparse components carry; the rest is the shared delta component (input-independent).

| module | weight shape | C | component capture | delta fraction | mean alive L0 / datapoint | alive components |
|---|---|---:|---:|---:|---:|---:|
| net.0 | [512, 8192] | 128 | 2.025717 | 1.964933 | 10.70752 | 102 |
| net.3 | [256, 512] | 128 | 1.287465 | 0.983549 | 18.13916 | 41 |

## Top causal components by ablation damage

| component | module | mean CI | alive frac | ablation L2 Δ | label |
|---|---|---:|---:|---:|---|
| net.3:c065 | net.3 | 0.854873 | 0.924805 | 2.260691 | high_causal_importance_component |
| net.0:c045 | net.0 | 0.402093 | 0.666016 | 2.103099 | high_causal_importance_component |
| net.0:c056 | net.0 | 0.812274 | 0.999512 | 1.702772 | high_causal_importance_component |
| net.0:c125 | net.0 | 0.764786 | 0.970215 | 1.570782 | high_causal_importance_component |
| net.0:c101 | net.0 | 0.87168 | 1.0 | 1.491521 | high_causal_importance_component |
| net.3:c041 | net.3 | 0.988301 | 1.0 | 1.183678 | high_causal_importance_component |
| net.0:c113 | net.0 | 0.75072 | 0.933594 | 1.180132 | high_causal_importance_component |
| net.3:c047 | net.3 | 0.815937 | 0.95166 | 1.022977 | high_causal_importance_component |
| net.0:c070 | net.0 | 0.859241 | 0.996094 | 0.985498 | high_causal_importance_component |
| net.3:c026 | net.3 | 0.769262 | 0.818359 | 0.940399 | high_causal_importance_component |
| net.3:c083 | net.3 | 0.706092 | 1.0 | 0.885154 | high_causal_importance_component |
| net.3:c127 | net.3 | 0.619691 | 0.850586 | 0.782187 | high_causal_importance_component |

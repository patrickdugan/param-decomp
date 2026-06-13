# TRM Labeled Decomposition Report (Stages 3-5)

Real Goodfire/param-decomp artifacts were used: this repo's SPD `optimize()` decomposed a genuine frozen TRM, not the 4D cached choice head.

Behavioral faithfulness (components+delta surrogate vs original TRM output, relative L2): 2e-08.

## Per-layer decomposition

component_norm_capture = how much of the weight Frobenius norm the sparse components carry; the rest is the shared delta component (input-independent).

| module | weight shape | C | component capture | delta fraction | mean alive L0 / datapoint | alive components |
|---|---|---:|---:|---:|---:|---:|
| shared.0 | [256, 1024] | 64 | 0.997733 | 0.880871 | 4.777344 | 17 |
| shared.2 | [128, 256] | 64 | 1.031727 | 0.624285 | 17.026367 | 18 |

## Top causal components by ablation damage

| component | module | mean CI | alive frac | ablation L2 Δ | label |
|---|---|---:|---:|---:|---|
| shared.0:c058 | shared.0 | 1.0 | 1.0 | 2.4539 | high_causal_importance_component |
| shared.0:c008 | shared.0 | 0.999557 | 1.0 | 1.223578 | high_causal_importance_component |
| shared.2:c019 | shared.2 | 1.0 | 1.0 | 1.217293 | high_causal_importance_component |
| shared.2:c015 | shared.2 | 1.0 | 1.0 | 0.861621 | high_causal_importance_component |
| shared.2:c055 | shared.2 | 1.0 | 1.0 | 0.838795 | high_causal_importance_component |
| shared.2:c005 | shared.2 | 0.999979 | 1.0 | 0.837074 | high_causal_importance_component |
| shared.2:c004 | shared.2 | 0.999998 | 1.0 | 0.822207 | high_causal_importance_component |
| shared.0:c040 | shared.0 | 0.996586 | 1.0 | 0.763955 | high_causal_importance_component |
| shared.2:c000 | shared.2 | 1.0 | 1.0 | 0.597329 | high_causal_importance_component |
| shared.2:c025 | shared.2 | 1.0 | 1.0 | 0.501434 | high_causal_importance_component |
| shared.2:c001 | shared.2 | 0.999998 | 1.0 | 0.449098 | high_causal_importance_component |
| shared.2:c011 | shared.2 | 0.999994 | 1.0 | 0.36217 | high_causal_importance_component |

# TRM Governance / Patch Test (Stage 6)

Real decomposition of precision_trm_mcq net.3. A rank-edited dirty diff lifts mmlu_professional_law and regresses arc_challenge; each selector drops the diff's projection onto 8 components.

Baseline accuracy: target 0.6500, nontarget 0.6775.

| arm | target acc | nontarget acc |
|---|---:|---:|
| raw_dirty | 1.0000 | 0.2975 |
| decomp_ci_informed | 1.0000 | 0.0100 |
| norm_only | 0.9500 | 0.0100 |
| random | 1.0000 | 0.2675 |
| svd_topk_removed | 0.6500 | 0.6775 |

## Verdict

```json
{
  "target_gain": 0.35,
  "raw_regression": 0.38,
  "decomp_regression": 0.6675,
  "decomp_target_retention": 1.0,
  "control_regressions": {
    "norm_only": 0.6675,
    "random": 0.41,
    "svd_topk_removed": 0.0
  },
  "control_target_retention": {
    "norm_only": 0.857143,
    "random": 1.0,
    "svd_topk_removed": 0.0
  },
  "verdict": "REAL_DECOMP_NEGATIVE"
}
```

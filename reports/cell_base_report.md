# Powered Cell-Base Report (roadmap item 2)

Suites pooled: arc_challenge, arc_easy (2048 samples).
Cell granularity: `top_target` (gate-compatible), powered at >=50 target/family.

Powered families: 2. Cells: 2. Regression-positive: 2. Bootstrap-recurrent regression-positive: 2.

| cell | n_target | n_nontarget | raw target Δ | raw non-target Δ | damage | regression+ | bootstrap recurrence |
|---|---:|---:|---:|---:|---:|:--:|---:|
| B->A:letter | 73 | 1975 | 0.7534 | -0.0309 | 61 | yes | 1.000 |
| D->A:letter | 60 | 1988 | 0.6333 | -0.0262 | 52 | yes | 1.000 |

## Verdict

2 regression-positive cells < 20. Add more suites (mmlu_formal_logic, gsm8k choice variants) or lower the granularity floor.

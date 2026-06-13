# TRM Parameter-Decomposition Audit (Stage 0/1)

Real model located and wrappable. This replaces the 4D cached choice-head proxy with a genuine frozen TRM whose Linear layers can be decomposed by this repo's SPD framework.

```json
{
  "checkpoint": "D:\\Research_Engine\\tesseract_persistent\\data\\models\\conveyor\\trm_arc_challenge.pt",
  "model_class": "ConveyorTRM (TF-IDF -> MLP classifier)",
  "parameter_count": 295941,
  "input_dim": 1024,
  "vectorizer": "TfidfVectorizer",
  "n_prompts": 256,
  "feature_density": 0.020065,
  "forward_ok": true,
  "category_logit_std": 0.441732,
  "risk_std": 0.078068,
  "success_std": 0.94053,
  "category_pred_distribution": {
    "1": 256
  },
  "decomposable_linear_modules": {
    "shared.0": [
      256,
      1024
    ],
    "shared.2": [
      128,
      256
    ],
    "category_head": [
      3,
      128
    ]
  },
  "candidate_modules_for_decomp": [
    "shared.0",
    "shared.2"
  ],
  "tooling_status": "THIS_REPO_SPD_AVAILABLE (param_decomp ComponentModel wraps nn.Linear)",
  "behavioral_variation_note": "category argmax is constant on ARC-only prompts, but category logits and the continuous risk/success heads vary per datapoint (see *_std), so SPD has output variation to attribute. For richer multi-class variation, also consider precision_trm_mcq (4.3M, 10-way) or diverse multi-env prompts."
}
```

## Next: SPD decomposition

Wrap `ConveyorTRM` in `ComponentModel` with targets `shared.0`, `shared.2` and run `param_decomp.run_param_decomp.optimize()` (resid_mlp template) to produce real parameter components with reconstruction error and per-datapoint causal importance — the actual VPD artifacts.

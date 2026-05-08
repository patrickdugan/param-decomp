# MeTTa Keygate VDP Experiment

`metta_keygate_v1` is the first narrow MeTTa-to-VDP bridge in this fork. It consumes typed gate traces from the `metta-storyworld` repo, trains a tiny post-repair gate, and scores final-head rank-one component ablations through the existing VDP Rust/Python acceleration seam.

The v1 gate asks whether a controller should `commit`, `veto`, or `repair` after a repair attempt. The important target is false-commit avoidance: do not commit packages with verifier-visible defects such as silent field drops, missing required fields, verifier disagreement, or backdoor triggers.

## Run

From `C:\projects\VDP`:

```powershell
python -m param_decomp.experiments.metta_keygate.run param_decomp\experiments\metta_keygate\metta_keygate_v1.yaml --backend auto
```

The registry entry also supports:

```powershell
pd-local metta_keygate_v1 --cpu
```

Outputs are written under `PARAM_DECOMP_OUT_DIR\metta_keygate_v1\...` unless `--out-dir` is provided.

## Source Trace

Upstream MeTTa source:

```text
C:\projects\metta-storyworld\metta-storyworld\metta-storyworld\research\vpd_keygate_village\traces\example_keygate_trace.jsonl
```

VDP fixture used by `metta_keygate_v1`:

```text
param_decomp\experiments\metta_keygate\fixtures\example_keygate_trace.jsonl
```

Expected SHA-256:

```text
d0344bee007156269434b2c99963b54dda133d9d28917f9560bffec3c72546f5
```

Each run copies the trace into its artifact directory as `source_trace.jsonl` and records the hash in `manifest.json`.

## Claim Boundary

Allowed claim:

```text
Typed MeTTa traces can define a small causal gate benchmark where rank-one components of a tiny learned gate can be ablated and compared against random controls.
```

Not allowed:

```text
This proves Qwen hidden-module SVD clusters are causal semantic mechanisms.
```

## Metrics

Primary metrics:

- joint decision accuracy
- false-commit rate
- false-reject rate
- repair-preservation rate
- top-component ablation effect versus random controls

Splits are reported separately for `train`, `validation`, `holdout_seen`, `holdout_unseen`, and `backdoor_probe`.

The initial 40-row fixture is intentionally easy. A clean v1 run should show that top head components can collapse held-out accuracy relative to random controls. It may not yet show false-commit increase under ablation; that requires a harder trace set with closer commit/veto margins.

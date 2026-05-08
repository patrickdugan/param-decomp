# MeTTa TRM Amalgam

`metta_trm_amalgam_v1` is a synthetic task-graph experiment for the "protist amalgam" architecture: small recursive TRM organelles enfold around typed MeTTa task nodes, then VDP-style component scoring audits which learned head components support graph execution.

The v1 graph has three nodes:

- `route_to_tool`: route to tool, answer directly, or abstain.
- `repair_or_abstain`: repair, preserve, or abstain.
- `commit_veto`: commit, veto, or request repair.

The experiment trains one tiny recursive gate model per node and compares the modular organism against a monolithic baseline trained on all rows. Each organelle head is decomposed into SVD and class-row components, scored through the existing VDP Rust/Python rank-one ablation seam, then summarized into `component_registry.json`.

## Run

```powershell
python -m param_decomp.experiments.metta_amalgam.run param_decomp\experiments\metta_amalgam\metta_trm_amalgam_v1.yaml --backend auto
```

Registry path:

```powershell
pd-local metta_trm_amalgam_v1 --cpu
```

## Outputs

- `summary.json`: modular versus monolithic split metrics and claim boundary.
- `component_registry.json`: accepted top components per organelle.
- `organelle_scores.json`: per-gate top effects and random-control maxima.
- `task_graph.json`: graph nodes and edges.
- `source_trace.jsonl`: deterministic synthetic task rows.

## Claim Boundary

Allowed claim:

```text
Typed MeTTa-style task graphs can be decomposed into small recursive gate models whose head components can be causally audited and stored as growth candidates.
```

Not allowed yet:

```text
The system autonomously grows new skills, or the component registry transfers to real Hermes skill performance without downstream benchmark evidence.
```

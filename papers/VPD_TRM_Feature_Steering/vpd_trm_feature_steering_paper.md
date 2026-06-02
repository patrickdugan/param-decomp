# VPD-Abetted Feature Steering in Tiny Recursive Models

Working draft, started 2026-06-01.

## Abstract

We study whether Vector/Variational/Virtual Parameter Decomposition (VPD) can expose reusable control features inside small task-recursive models (TRMs). The motivating use case is not broad language-model steering, but local workflow control: routing, verifier-aware repair, context curation, and transfer of compact decision motifs between related micro-models. We treat each TRM as a small organelle: a learned gate with explicit task state, a narrow action vocabulary, and a measurable safety or correctness boundary.

The larger thesis is that TRMs may be powerful when trained well, but isolated TRM training is still ordinary supervised learning. The system becomes reinforcement-like only when TRMs are placed inside a feedback loop: a larger controller, such as an LLM-managed harness, proposes goals, evaluates failures, selects edits or new training slices, and specializes the TRMs toward better downstream behavior. In this view, VPD is the editing and attribution layer that lets the outer architecture turn broad generalization pressure into targeted specialization.

The current evidence supports a limited claim. VPD can identify a small set of runtime-editable components that behave like portable control motifs across related TRM gates. These components sometimes improve hard-slice behavior while preserving guardrails and outperforming matched random controls. The evidence does not yet show general-purpose feature steering, permanent checkpoint-level skill transfer, or reliable transfer outside the current MeTTa-derived task families.

## Central Framing

TRMs are supervised specialists by default. A router, verifier, repair gate, or context-curation gate can be trained from labeled traces, but that only gives a static model of prior examples. The research target here is the next layer up: a goal-directed loop that observes where those specialists fail, uses VPD to identify editable control features, and then either patches, grafts, or retrains the relevant organelles.

The intended pipeline is:

```text
general controller -> task/goal selection -> TRM execution -> scored outcome
  -> failure analysis -> VPD component selection -> edit/graft/retrain
  -> specialized TRM -> new evaluation
```

This resembles reinforcement learning at the system level because behavior is improved through feedback over outcomes. The current implementation is not yet a full RL algorithm: there is no learned policy over edits and no long-horizon reward optimizer. The paper should therefore describe it as a reinforcement-style or goal-directed specialization loop, with VPD providing the mechanism for interpretable intervention.

The key claim is not that a tiny TRM magically generalizes. The key claim is that a larger architecture can manage a population of tiny TRMs, use explicit feedback to discover where each one should specialize, and use VPD to make that specialization less blind than ordinary retraining.

## Development Phases

The project has three separable phases. The paper should preserve that order because each phase tightened the experimental claim.

| Phase | System | Main question | Primary metric | Current status |
| --- | --- | --- | --- | --- |
| 1 | Router gate | Can typed workflow traces produce a causal VPD steering benchmark? | false-commit rate, decision accuracy | completed as narrow keygate benchmark |
| 2 | Repair optimization | Can a workflow be decomposed into auditable recursive repair gates? | repair preservation, false-commit avoidance | completed as Amalgam task graph |
| 3 | Organelle transfer | Can VPD components transfer between related TRM organelles? | accepted graft count, guardrail pass, beats-random rate | strict replication completed |
| 4 | Eval-aligned editing | Do proxy-positive VPD edits improve pinned downstream Intellect eval failures? | eval-aligned accept count, target eval delta, holdout/guardrail delta | first strict run completed; zero accepted |

### Phase 1: Router Gate

The first phase built a narrow router-style TRM around typed MeTTa traces. The task was intentionally small: decide whether a repaired package should be committed, vetoed, or sent back for repair. This gave the project a concrete false-commit boundary and a causal component test.

Key artifact:

```text
C:\projects\VDP\docs\metta_keygate_vdp.md
```

Core contribution:

- Converted typed MeTTa traces into a compact gate benchmark.
- Trained a tiny post-repair decision model.
- Scored final-head rank-one components through the VPD ablation path.
- Compared top VPD components against random controls.

Claim boundary:

The router phase shows that typed workflow traces can define a small causal gate benchmark where VPD components can be ablated and compared. It does not yet show transferable steering or autonomous skill growth.

Paper role:

This phase should appear as the minimal causal benchmark. It establishes the steering vocabulary: gates, guardrails, top components, ablation, and random controls.

### Phase 2: Repair Optimization

The second phase generalized from a single router into a repair-oriented task graph. Instead of one gate, the system used several small recursive gates over related actions: route to tool, repair or abstain, and commit or veto. This made the target closer to a real local workflow, where the model must preserve valid repairs, avoid false commits, and decide when to abstain.

Key artifact:

```text
C:\projects\VDP\docs\metta_trm_amalgam.md
```

Core contribution:

- Represented the workflow as typed task nodes rather than a single classifier.
- Trained one small recursive organelle per task node.
- Compared modular gate behavior against a monolithic baseline.
- Built a component registry from decomposed heads.

This phase introduced the repair-preservation metric as a central signal. That matters because raw accuracy can look good while a model silently damages the repair workflow. The useful steering target became: improve or preserve repair behavior without increasing false commits or false rejects.

Claim boundary:

The repair phase shows that a task graph can be decomposed into auditable TRM organelles and that VPD can identify head components associated with workflow decisions. It does not yet show that components can be moved between organelles.

Paper role:

This phase should appear as the bridge from a single router to workflow-level repair optimization. It motivates why repair preservation needs to be measured separately from accuracy.

### Phase 3: Organelle Transfer System

The third phase tests whether decomposed components can act as portable runtime control motifs. The Organelle Arena trains multiple related TRM gates, decomposes their heads, maps compatible decision labels, and applies source components to target organelles as runtime grafts.

Primary run artifact so far:

```text
D:\Research_Engine\runs\metta_organelle_arena_full_20260531T180830Z\arena_summary.json
```

Current stricter overnight replication run:

```text
D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z
```

The replication run completed with empty stderr:

```text
accepted_count: 16
graft_record_count: 1305171
organelle_count: 186
stable_graft_count: 110731
status: completed
```

Core contribution:

- Defines source and target organelles over Amalgam and Intellect-3-Logic instantiation-matrix tasks.
- Decomposes output heads into SVD and class-row components.
- Applies runtime additive grafts through shared decision labels.
- Separates true accepted grafts from neutral stable grafts.
- Requires accepted grafts to preserve guardrails, produce positive movement, and beat random controls.

The strongest completed evidence is the strict full arena run:

```text
accepted_count: 23
graft_record_count: 389746
organelle_count: 93
stable_graft_count: 54295
```

Representative accepted pattern:

```text
source: amalgam:route_to_tool
target: instantiation_matrix:typed_gate_graph:repair_step
shared_decision: abstain
effect: improved accuracy and repair/target preservation without false-commit or false-reject regression
```

Representative six-seed replication patterns:

```text
source: instantiation_matrix:facts_only:signature_route
target: instantiation_matrix:facts_only:repair_step
shared_decision: abstain
effect: +0.005181 accuracy, +0.04 repair preservation, no false-commit or false-reject regression

source: instantiation_matrix:task_dag_organelle:format_commit
target: instantiation_matrix:task_dag_organelle:repair_step
shared_decision: repair
effect: +0.005181 accuracy, +0.04 repair preservation, no false-commit or false-reject regression

source: amalgam:commit_veto
target: instantiation_matrix:typed_gate_graph:repair_step
shared_decision: repair
effect: +0.005181 accuracy, +0.04 repair preservation, no false-commit or false-reject regression
```

Claim boundary:

The organelle phase supports the paper's narrow thesis: VPD can identify a small number of portable runtime control motifs between related TRM gates. It remains a runtime-graft result, not a checkpoint-edit result.

Paper role:

This phase is the main result. It tests whether decomposed components act like transferable control motifs rather than merely local explanations of a single gate.

### Phase 4: Eval-Aligned Edit Gate

The fourth phase asks a stricter question than the Organelle Arena. A runtime graft can improve a proxy gate metric and still fail to help the downstream task that motivated the edit. The eval-aligned phase therefore takes candidate VPD edits from the arena, aligns them to pinned Intellect-3 failure clusters, and accepts an edit only if it clears an eval-facing target estimate, holdout/guardrail check, and random-control threshold.

Primary artifact:

```text
D:\Research_Engine\runs\trm_eval_aligned_edits_intellect3_logic_20260602
```

Run summary:

```text
target_eval: intellect_3_logic
candidate_path: D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\graft_sweeps.jsonl
eval_run: D:\Research_Engine\runs\trm_eval_feedback_loop_smoke_20260601_v3
candidate_count: 7764
cluster_count: 3
result_count: 12
accepted_eval_aligned_count: 0
status: completed
```

The selected Intellect-3-Logic clusters were:

```text
router_confusion_logic_law
router_confusion_logic_math
format_commit_failure
```

The first eval-aligned pass found candidates that were close enough to inspect, but none were accepted for the paper-grade eval showcase. A representative rejected candidate aligned to the `router_confusion_logic_law` cluster:

```text
candidate: instantiation_matrix:task_dag_organelle:signature_route:head_row:route_plain
target: instantiation_matrix:facts_only:signature_route
proxy_gain: +0.020833 accuracy_delta
alignment: 1.0
estimated eval delta: +0.006944
guardrail pass: true
selected action: reject_or_collect_more_evidence
```

Core contribution:

- Separates proxy-positive VPD edits from downstream eval-positive edits.
- Converts the feedback loop into a falsifiable gate rather than a showcase-only demo.
- Produces structured decision traces for accepted and rejected edits without recording hidden chain-of-thought.
- Establishes that the current VPD control motifs are not yet sufficient to claim downstream Intellect-3-Logic improvement.

Claim boundary:

This phase does not show an eval-improving VPD edit yet. It shows the stricter acceptance process working: proxy improvements are allowed to fail at the downstream gate. If this remains true under stronger candidate generation, that negative result is publishable because it defines the boundary between local TRM control motifs and real task-level generalization.

Paper role:

This phase is the bridge from a promising mechanistic result to a credible learning/editing claim. It prevents the paper from overclaiming. It also gives the next experiment a concrete target: produce one eval-aligned accepted edit, or show that current VPD proxy motifs systematically fail to transfer to Intellect-3 downstream tasks.

## Claim Ledger

| Claim | Evidence needed | Current evidence | Paper stance |
| --- | --- | --- | --- |
| VPD can expose causal components in tiny TRM gates. | Ablation changes gate behavior more than random controls. | Router and repair-gate component tests. | Supported narrowly. |
| Repair optimization needs workflow metrics beyond accuracy. | Cases where repair preservation or false-commit metrics matter independently. | Amalgam and arena metrics include repair preservation and guardrails. | Supported as methodology. |
| VPD components can transfer between related TRM organelles. | Runtime graft improves target behavior, preserves guardrails, and beats random controls. | Strict full arena run found 23 accepted grafts. | Provisional main result. |
| Stable neutral grafts are successful steering. | Positive target movement. | Many stable grafts had no positive signal. | Reject; report separately. |
| Runtime grafts prove checkpoint-level editing. | Persistent edited checkpoint with retained behavior. | Not attempted yet. | Not claimed. |
| Results generalize to unrelated task families. | Cross-domain replication beyond current MeTTa/Intellect-3-Logic families. | Not available. | Not claimed. |
| Proxy-positive VPD edits improve downstream eval behavior. | Pinned eval before/after improvement under guardrails. | First Intellect-3-Logic eval-aligned run found 12 candidates and 0 accepted. | Not yet supported; active boundary result. |
| A zero-accept eval-alignment run is useful evidence. | Clear separation between proxy and eval metrics. | Eval-aligned gate rejected proxy candidates while preserving traceable rationale. | Supported as methodology / negative result. |

## Method

Each experiment follows the same basic loop:

1. Build deterministic task rows from typed workflow traces or synthetic MeTTa-style task graphs.
2. Train small TRM gates with explicit action vocabularies.
3. Decompose gate heads into candidate components.
4. Score causal effects through ablation or runtime grafting.
5. Compare against random rank-one and label-shuffled controls.
6. Accept only edits that improve a target metric, preserve guardrails, and beat matched controls.

The important design choice is that the state remains external and measurable. The TRMs are not asked to absorb an entire workflow into a free-form hidden state. Instead, they operate over explicit gate families, target decisions, and split-specific metrics.

## Metrics

Primary metrics:

- Accuracy on gate decisions.
- False-commit rate.
- False-reject rate.
- Repair-preservation rate.
- Target-preservation rate.
- Positive-signal rate.
- Accepted VPD graft count.
- Stable-but-neutral graft count.
- Random-control positive rate.

Acceptance criteria for strict grafts:

- `component_source == "vdp_or_class_row"`
- `control_family == "vdp_graft"`
- guardrail pass
- positive signal
- beats random control

Neutral safe edits are tracked separately as stable grafts and should not be counted as successful steering.

## Results Ledger

These are working result slots. Values should be updated only from artifact summaries or reproducible analysis scripts.

| Result family | Artifact | Status | Notes |
| --- | --- | --- | --- |
| Router ablation | `metta_keygate_v1` outputs | needs table extraction | Use for causal benchmark setup. |
| Repair graph ablation | `metta_trm_amalgam_v1` outputs | needs table extraction | Use for repair-preservation motivation. |
| Early loose feature steering | `trm_feature_steering_overnight_20260531T002834Z` | exploratory only | Over-counted neutral and random-derived accepted edits. |
| Strict full arena | `metta_organelle_arena_full_20260531T180830Z` | completed | 23 accepted, 389746 records, 93 organelles, 54295 stable grafts. |
| Six-seed paper arena | `metta_organelle_arena_paper_20260601T024147Z` | completed | 16 accepted, 1305171 records, 186 organelles, 110731 stable grafts. |
| Feedback-loop paper run | `trm_feedback_loop_paper_fixed_20260601T115330Z` | completed | 53 failure modes, 48 interventions, 4 rounds; replay and mixed policies beat random, direct VPD edits did not move selected metrics. |
| VPD-editable feedback run | `trm_feedback_loop_vpd_editable_arena_20260601T155701Z` | completed | 12 editable failures from arena graft evidence; LLM edit proxy beat replay on total gain under guardrails. |
| Full-sweep VPD-editable run | `trm_feedback_loop_full_sweep_tiebreak_20260601T162528Z` | completed | Parsed 2GB graft sweep into 7764 useful arena candidates; LLM edit proxy beat replay on accepted metric gain. |
| Filtered full-sweep run | `trm_feedback_loop_full_sweep_filtered_20260601T163437Z` | completed | Matched-random filtration collapses replay/random to zero filtered gain; VPD policies retain filtered accepted gain. |
| VPD edit showcase | `vpd_edit_showcase` | completed | Dashboard-ready before/delta/after examples; separates accepted proxy edit, borderline edit, and no-edit decision. |
| Eval-aligned Intellect-3-Logic gate | `trm_eval_aligned_edits_intellect3_logic_20260602` | completed | 7764 candidates, 3 clusters, 12 aligned candidate rows, 0 accepted eval-aligned edits. |

## Current Thesis

VDP is useful for TRM feature steering when the model family is small, the action space is explicit, and the workflow has measurable guardrails. In that setting, decomposition does not merely describe weights; it yields candidate control motifs that can be stress-tested through ablation, runtime grafting, and random-control comparisons.

The broader architecture is a generalization-to-specialization loop. The larger controller supplies general goals and evaluation pressure; the TRMs supply cheap, auditable specialized decisions; VPD supplies a way to inspect and edit the specialization boundary.

The strongest paper claim should remain:

```text
VPD identifies a small set of portable TRM control motifs that improve hard-slice behavior under guardrails and outperform matched random controls.
```

Claims to avoid for now:

- VPD provides general language-model steering.
- Runtime grafts prove permanent model editing.
- Stable neutral grafts are successful feature steering.
- Results transfer to unrelated task families without additional evidence.
- The present harness is already full reinforcement learning.
- Supervised TRMs alone solve goal-directed adaptation without an outer feedback loop.

## Experiments Needed For Paper

### Replication

The six-seed replication run increased seeds and random-control pressure:

```text
targets: amalgam,instantiation_matrix
seeds: 20260601-20260606
epochs: 140
hidden_dim: 24
random_controls: 8
hard_slice: true
```

Observed result:

```text
status: completed
stderr: empty
accepted VPD grafts: 16
stable neutral grafts: 110731
graft records: 1305171
```

Accepted graft seed distribution from the summary:

```text
20260601: 2
20260602: 2
20260603: 1
20260605: 7
```

Interpretation:

The stricter replication preserved a nonzero accepted-graft signal under more random-control pressure, but the accepted grafts were not evenly distributed across all six seeds. The result strengthens the narrow portability claim while keeping recurrence as an open robustness issue.

Success criteria status:

- stderr empty: met
- accepted VPD grafts remain nonzero: met
- accepted grafts recur across seeds or gate families: partially met
- stable grafts remain separated from accepted grafts: met
- random-control positives are reported, not hidden: pending histogram/table extraction

### Feedback-Loop Specialization

The first outer-loop paper run tested whether an LLM-style controller could select failure modes and compare intervention policies over existing TRM candidate sweeps:

```text
run: D:\Research_Engine\runs\trm_feedback_loop_paper_fixed_20260601T115330Z
targets: keygate,amalgam,instantiation_matrix
seeds: 20260601-20260604
rounds: 4
failures_per_round: 3
mcp_contract_probe: true
status: completed
stderr: empty
```

Observed policy comparison:

```text
vpd_greedy:      12 interventions, 0 accepted,  total_metric_gain 0.0
failure_replay:  12 interventions, 12 accepted, total_metric_gain 4.2
random_control:  12 interventions, 0 accepted,  total_metric_gain 0.0
mixed_controller:12 interventions, 12 accepted, total_metric_gain 4.2
```

Interpretation:

This run supports the feedback-loop framing but not yet the stronger claim that direct VPD edits solve the selected failures. The selected failures were mainly preservation gaps where replay-style specialization had a synthetic gain model, while direct VPD candidates did not move the selected metric. That is useful evidence: the next loop should either select VPD-editable failures or use VPD to choose which replay slices to add.

Known bad artifact:

```text
D:\Research_Engine\runs\trm_feedback_loop_paper_20260601T115019Z
```

The earlier run had a metric-delta overwrite bug that zeroed replay gains. Treat it as a rejected scoring artifact, not as paper evidence.

### VPD-Editable Failure Selection

The follow-up run seeded the feedback loop with accepted Organelle Arena grafts and derived failure cards directly from positive VPD evidence:

```text
run: D:\Research_Engine\runs\trm_feedback_loop_vpd_editable_arena_20260601T155701Z
arena_candidates: D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z
failure_selection: editable_first
editable_failures: 12
llm_decision_packets: 12
status: completed
stderr: empty
```

Observed policy comparison:

```text
vpd_greedy:       5 interventions,  5 accepted,  total_metric_gain 0.200000
failure_replay:  12 interventions, 12 accepted,  total_metric_gain 0.131439
mixed_controller: 4 interventions,  4 accepted,  total_metric_gain 0.160000
llm_edit_proxy:  12 interventions, 12 accepted,  total_metric_gain 0.209439
```

Interpretation:

This is the first run that matches the intended mode: the controller works on failures known to be VPD-editable, and the compact LLM-style edit proxy selects VPD edits when they have observed guardrail-passing gain. It supports the next paper claim more directly than the replay-driven run: VPD evidence can shape the outer-loop failure queue and intervention choice.

Limitation:

The run used arena summary grafts rather than the full arena sweep, so random-control rows were not available in the feedback runner. The source arena acceptance already required beating random controls, but the feedback-loop table should still be rerun from full graft sweeps before final paper figures.

### Full-Sweep VPD-Editable Feedback

The full-sweep run streamed the 2GB `graft_sweeps.jsonl` artifact and retained only guardrail-passing positive VPD grafts plus positive random controls:

```text
run: D:\Research_Engine\runs\trm_feedback_loop_full_sweep_tiebreak_20260601T162528Z
arena_candidates: D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\graft_sweeps.jsonl
arena_candidate_count: 7764
editable_failures: 35
llm_decision_packets: 20
status: completed
```

Observed policy comparison:

```text
vpd_greedy:       total_gain 0.660724, accepted_gain 0.320000, accepted 8/20
failure_replay:  total_gain 0.231252, accepted_gain 0.231252, accepted 20/20
random_control:  total_gain 0.660724, accepted_gain 0.000000, accepted 0/20
mixed_controller:total_gain 0.634724, accepted_gain 0.214000, accepted 6/20
llm_edit_proxy:  total_gain 0.309252, accepted_gain 0.309252, accepted 20/20
```

Interpretation:

This run is the current best evidence for the VPD-editable feedback-loop mode. Random controls can match VPD on raw scalar movement, which means total metric gain alone is not enough. The useful distinction is accepted metric gain: VPD-selected edits pass the richer guardrail and acceptance filter, while random controls do not. The LLM-style proxy uses that evidence to prefer accepted VPD edits on ties and beats replay on accepted metric gain.

Superseded artifact:

```text
D:\Research_Engine\runs\trm_feedback_loop_full_sweep_20260601T161732Z
```

That run parsed the full sweep correctly but used an over-conservative proxy that fell back to replay on VPD/random metric ties. Use the tie-break run above for paper figures.

### Matched-Random Filtration

The filtered full-sweep run applies the bench filtration method to the 2GB graft sweep:

```text
run: D:\Research_Engine\runs\trm_feedback_loop_full_sweep_filtered_20260601T163437Z
arena_candidate_count: 7764
editable_failures: 35
interventions: 100
status: completed
```

Filtration criteria:

```text
guardrail pass
positive task signal
matched random P95/best gain threshold
matched random P95 efficiency threshold
matched random P95 specificity threshold
accepted VPD source
```

Filtered policy comparison:

```text
vpd_greedy:       filtered_accept_count 5, filtered_metric_gain 0.200000
failure_replay:  filtered_accept_count 0, filtered_metric_gain 0.000000
random_control:  filtered_accept_count 0, filtered_metric_gain 0.000000
mixed_controller:filtered_accept_count 1, filtered_metric_gain 0.040000
llm_edit_proxy:  filtered_accept_count 1, filtered_metric_gain 0.040000
```

Interpretation:

This filtration isolates the bench from random baseline movement. Random controls matched VPD on raw scalar gain in the unfiltered table, but collapse under accepted gain, efficiency, specificity, and matched-random thresholds. Replay also collapses because it is a training-slice fallback, not an accepted VPD edit. The filtered table therefore supports the claim that the VPD signal is not merely random metric movement, although the current LLM proxy remains conservative and leaves more filtered gain to deterministic VPD-greedy than desired.

### Eval-Aligned VPD Edit Gate

The eval-aligned run asks whether the current proxy-positive VPD candidates transfer to pinned Intellect-3-Logic downstream failures:

```text
run: D:\Research_Engine\runs\trm_eval_aligned_edits_intellect3_logic_20260602
source eval run: D:\Research_Engine\runs\trm_eval_feedback_loop_smoke_20260601_v3
source candidates: D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\graft_sweeps.jsonl
target_eval: intellect_3_logic
candidate_count: 7764
cluster_count: 3
result_count: 12
accepted_eval_aligned_count: 0
```

The harness considered aligned candidates for router-confusion clusters and format/commit failure clusters. The strongest visible candidates had nonzero proxy movement and clean guardrails, but did not satisfy the stricter eval-aligned acceptance criteria. The dashboard now reports this explicitly:

```text
proxy accepted runtime VPD edits: 16
filtered proxy accepts: 7
eval-aligned accepts on Intellect-3-Logic: 0
```

Interpretation:

This result means the project is past the easy demonstration stage. The current evidence is enough to say that VPD finds portable proxy control motifs in related TRM organelles. It is not yet enough to say those motifs directly hill-climb downstream Intellect-3-Logic performance. That distinction is valuable. It turns the next section of the paper into a real experimental fork:

- If future runs find eval-aligned accepted edits, the paper can claim the loop is beginning to bridge mechanistic proxy control and downstream task improvement.
- If stronger candidate generation still yields zero eval-aligned accepts, the paper can claim a negative boundary: VPD proxy editability does not automatically imply task-level steering.

This is why the current status is “over halfway” but not finished. The proxy/control-motif hill has been climbed. The eval-alignment hill may become a cliff, and that cliff would still define the publishable limit of the method.

### Edit Showcase and Decision Traces

A dashboard-ready edit showcase now exists:

```text
D:\Research_Engine\runs\vpd_edit_showcase
```

Generated artifacts:

```text
edit_showcase_manifest.json
edit_examples.jsonl
decision_traces.jsonl
```

The showcase includes three example classes:

- Filtered proxy-accepted VPD edit.
- Borderline or rejected proxy edit.
- No-edit fallback decision.

The visual panel shows before/delta/after color maps and structured decision traces. The trace policy is intentionally limited to audit rationale:

```text
observed failure
candidate evidence
random-control threshold
guardrail checks
selected action
reason for accept/reject
```

It does not record hidden chain-of-thought. This is the right format for a paper figure because it shows the mechanics of the harness without making unsupported claims about internal reasoning.

### Ablation Table

The paper needs a table separating:

- Router ablations.
- Repair graph component ablations.
- Organelle runtime grafts.
- Random rank-one controls.
- Label-shuffled controls.

Draft table columns:

| Phase | Model/gate family | Intervention | Control | Target metric | Delta | Guardrail effect | Accepted? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Router | post-repair keygate | VPD ablation | random rank-one | false-commit / accuracy | TBD | TBD | TBD |
| Repair optimization | Amalgam task graph | VPD ablation | random rank-one | repair preservation | TBD | TBD | TBD |
| Organelle transfer | Amalgam to Matrix | runtime graft | random + shuffled labels | repair preservation / accuracy | from arena | no regression required | strict accepted only |

### Histograms

Required figures:

- Accepted VPD graft delta histogram.
- Random-control delta histogram.
- Stable neutral graft histogram.
- Transfer heatmap by source and target gate family.
- Accepted graft recurrence by seed.

Figure checklist:

| Figure | Source artifact | Purpose | Status |
| --- | --- | --- | --- |
| Router component ablation bars | keygate scores | Show first causal component test. | pending extraction |
| Repair-preservation tradeoff | Amalgam scores | Show why accuracy is insufficient. | pending extraction |
| Accepted VPD vs random deltas | arena `graft_sweeps.jsonl` | Main control comparison. | pending six-seed run |
| Transfer heatmap | arena `transfer_matrix.csv` / SVG | Show source-target structure. | available for strict full run |
| Accepted recurrence by seed | arena summary + sweeps | Show robustness. | pending six-seed run |

Six-seed run figures now available:

```text
D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\figures\control_histogram.svg
D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\figures\decision_resonance.svg
D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\figures\transfer_heatmap.svg
```

### Negative Results

The paper should explicitly include:

- Loose early acceptance over-counted neutral grafts.
- Random controls can produce positive movement on some hard slices.
- Strict acceptance reduces yield substantially.
- Current results are strongest around repair-step and candidate-verification gates.
- The first feedback-loop run was replay-driven; direct VPD edits did not improve the selected failure metrics.
- VPD-editable selection worked only after seeding from Organelle Arena graft evidence; the single-model scale-edit sweep had no positive matrix VPD candidates.
- Full-sweep random controls matched VPD on raw metric gain, so accepted metric gain is the better reporting metric for the feedback-loop table.
- Matched-random filtration is necessary: without it, random controls can look competitive on raw gain.
- The first eval-aligned Intellect-3-Logic pass produced 12 candidate rows and zero accepted downstream edits.
- Proxy-positive control motifs do not yet imply downstream task improvement.

Negative-result handling:

Do not bury these as caveats. They are part of the methodological contribution: the harness became useful only after neutral stable grafts, random-derived candidates, and loose acceptance were separated from true accepted VPD grafts.

## Paper Structure

1. Introduction: TRMs as supervised specialists and the need for goal-directed specialization.
2. System framing: outer controller, feedback loop, VPD edits, and TRM organelles.
3. Background: VPD, TRMs, typed MeTTa workflows, feature steering.
4. Phase 1 Router: post-repair keygate and false-commit boundary.
5. Phase 2 Repair Optimization: task graph organelles and repair preservation.
6. Phase 3 Organelle Transfer: runtime grafts and portability tests.
7. Phase 4 Eval Alignment: downstream Intellect-3 gate and zero-accept boundary result.
8. Results: accepted grafts, controls, hard-slice behavior, recurrence, eval-aligned rejection.
9. Limitations: runtime-only, small models, deterministic eval estimates, synthetic/typed workflow rows, not full RL yet.
10. Discussion: MCP/TRM context curation and future checkpoint-level steering.

## Open Questions

- Do accepted grafts recur under the expanded six-seed run?
- Are repair-step gains a real transferable motif or a consequence of the hard-slice construction?
- Can a graft selected in the arena improve a downstream MCP retrieval or context-curation harness?
- Can the same component family be promoted from runtime graft to checkpoint edit without losing guardrails?
- Can candidate generation be targeted enough to produce one eval-aligned Intellect-3-Logic accepted edit?
- If eval-aligned accepts remain zero, is that a limitation of VPD, the organelle proxy task, or the current deterministic alignment estimate?
- What minimum task diversity is needed before the result stops being an artifact of shared labels?
- What is the right outer-loop policy for choosing between runtime grafting, checkpoint editing, and supervised retraining?
- Can an LLM-managed controller reliably turn repeated failures into better TRM training slices without drifting away from guardrails?

## Live Artifact Index

Router:

```text
C:\projects\VDP\docs\metta_keygate_vdp.md
```

Repair optimization:

```text
C:\projects\VDP\docs\metta_trm_amalgam.md
```

Organelle arena code:

```text
C:\projects\VDP\param_decomp\experiments\metta_organelle_arena.py
C:\projects\VDP\scripts\run_metta_organelle_arena.py
```

Feature steering code:

```text
C:\projects\VDP\param_decomp\experiments\trm_feature_steering.py
C:\projects\VDP\scripts\run_trm_feature_steering_paper.py
```

Completed strict full arena run:

```text
D:\Research_Engine\runs\metta_organelle_arena_full_20260531T180830Z
```

Completed paper replication run:

```text
D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z
```

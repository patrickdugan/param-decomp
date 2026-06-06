# VPD-Abetted Feature Steering in Tiny Recursive Models

Working draft, started 2026-06-01.

## Abstract

We study whether Vector/Variational/Virtual Parameter Decomposition (VPD) can expose reusable control features inside small task-recursive models (TRMs). The motivating use case is not broad language-model steering, but local workflow control: routing, verifier-aware repair, context curation, and transfer of compact decision motifs between related micro-models. We treat each TRM as a small organelle: a learned gate with explicit task state, a narrow action vocabulary, and a measurable safety or correctness boundary.

The larger thesis is that TRMs may be powerful when trained well, but isolated TRM training is still ordinary supervised learning. The system becomes reinforcement-like only when TRMs are placed inside a feedback loop: a larger controller, such as an LLM-managed harness, proposes goals, evaluates failures, selects edits or new training slices, and specializes the TRMs toward better downstream behavior. In this view, VPD is the editing and attribution layer that could let the outer architecture turn broad generalization pressure into targeted specialization.

The current evidence supports a bounded positive claim. VPD can identify a small set of runtime-editable components that behave like portable control motifs across related TRM gates. These components sometimes improve hard-slice behavior while preserving guardrails and outperforming matched random controls. The feedback-loop experiments also show why strict controls matter: proxy-positive edits, activation-local proxies, and stacked route-rule searches can all fail when compared against broad fixed-label or matched random controls. The evidence does not show broad-based reinforcement learning, permanent checkpoint-level skill transfer, or reliable transfer outside the current MeTTa-derived task families.

## Central Framing

TRMs are supervised specialists by default. A router, verifier, repair gate, or context-curation gate can be trained from labeled traces, but that only gives a static model of prior examples. The research target here is the next layer up: a goal-directed loop that observes where those specialists fail, uses VPD to identify editable control features, and then either patches, grafts, or retrains the relevant organelles.

The intended pipeline is:

```text
general controller -> task/goal selection -> TRM execution -> scored outcome
  -> failure analysis -> VPD component selection -> edit/graft/retrain
  -> specialized TRM -> new evaluation
```

This resembles reinforcement learning at the system level because behavior is improved through feedback over outcomes. The current implementation is not a full RL algorithm: there is no learned policy over edits and no long-horizon reward optimizer. The paper therefore describes it as a reinforcement-style or goal-directed specialization loop, with VPD providing the mechanism for interpretable intervention.

The key claim is not that a tiny TRM magically generalizes. The key claim is that a larger architecture can manage a population of tiny TRMs, use explicit feedback to discover where each one should specialize, and use VPD to make that specialization less blind than ordinary retraining. This paper closes the present phase at that bounded claim: the broader generalized editing methodology remains a research direction rather than a demonstrated result.

### Nomenclature

This draft uses **Tesseract** as the name of our local training and benchmarking harness: the router, adapter banks, serialized benchmark runner, receipts, and data-root conventions used to evaluate task-specialist lanes. It is not introduced as a public benchmark name. The downstream tasks themselves are Prime Intellect-style environments, especially `intellect_3_logic`, `intellect_3_math`, and related normalized trajectory/eval rows. When we report "Intellect-3-Logic" results, the benchmark substrate is the Prime Intellect environment; when we report a "Tesseract scorer bridge," the term refers only to the local harness that reruns those Prime Intellect environment samples through routed adapter lanes.

## Development Phases

The project has three separable phases. The paper preserves that order because each phase tightened the experimental claim.

| Phase | System | Main question | Primary metric | Current status |
| --- | --- | --- | --- | --- |
| 1 | Router gate | Can typed workflow traces produce a causal VPD steering benchmark? | false-commit rate, decision accuracy | completed as narrow keygate benchmark |
| 2 | Repair optimization | Can a workflow be decomposed into auditable recursive repair gates? | repair preservation, false-commit avoidance | completed as Amalgam task graph |
| 3 | Organelle transfer | Can VPD components transfer between related TRM organelles? | accepted graft count, guardrail pass, beats-random rate | strict replication completed |
| 4 | Eval-aligned editing | Do proxy-positive VPD edits improve pinned downstream Intellect eval failures? | eval-aligned accept count, target eval delta, holdout/guardrail delta | strict cliff plus targeted-probe rappel completed |

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
| Proxy-positive VPD edits improve downstream eval behavior. | Pinned eval before/after improvement under guardrails. | Strict six-round hill climb found 0 non-targeted accepted eval edits; targeted retargeting found 36 probe accepts before controls, then 0 accepts with targeted random controls. | Not yet supported as full downstream claim; active boundary/result fork. |
| A zero-accept eval-alignment run is useful evidence. | Clear separation between proxy and eval metrics. | Eval-aligned gate and strict hill climb rejected proxy candidates while preserving traceable rationale. | Supported as methodology / negative result. |
| Targeted retargeting can rappel up the eval cliff. | Same-gate VPD evidence creates plausible target-cluster probes with provenance and guardrails. | Targeted format-commit run found 36 accepted targeted probes before controls; strict targeted controls collapsed accepted count to 0. | Useful falsification scaffold; not a positive result yet. |
| Stateful edits bootstrap TRM ability across rounds. | At least two accepted edits, cumulative score gain, zero damage, and each edit beating fixed-label controls. | Cached ARC microcycle accepted one `D over A` edit, then stopped because the residual `B over D` edit lost to a broad fixed-label control. | Not yet supported; first-edit gain plus stacked-bootstrap boundary. |
| Broad failure-family search improves the feedback signal. | Multi-family sweep that distinguishes first-edit, stacked-edit, and control-blocked regions. | ARC sweep scored 15 families; 3 `D->A` families produced first-edit gains, 0 families produced stacked gains. | Supported as methodology; not yet ability bootstrap. |

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
| Strict eval hill climb | `trm_eval_hill_climb_intellect3_logic_full_20260602` | completed | 6 rounds, 1450 frontier rows, 24 near misses, 0 accepted eval-aligned edits; router candidates failed matched-random gate. |
| Targeted format-commit rappel | `trm_eval_hill_climb_intellect3_logic_targeted_final_20260602` | completed | 6 rounds, 2458 frontier rows, 36 accepted targeted probes, 0 non-targeted accepts; all accepts lacked matched random controls. |
| Targeted rappel with controls | `trm_eval_hill_climb_intellect3_logic_targeted_controls_strict_20260602` | completed | 6 rounds, 2506 frontier rows, 1056 targeted probes, 976 targeted random controls, 0 accepted eval-aligned edits. |
| Logic-efficacy feature-map miner | `trm_logic_efficacy_feature_map_20260602` | completed | 9 vectorization hypotheses, 0 positive eval-margin components, 0 accepted components under strict controls. |
| ARC Challenge reward-loop probe | `trm_choice_rl_feedback_arc_challenge_deduped_20260604` | completed | 48 deduped samples, 48 candidate policies, 21 accepted policies; best top-margin rule improved cached pass rate from 0.708333 to 0.791667. |
| ARC Challenge fresh-seed rerun | `trm_choice_rl_feedback_arc_challenge_3seed_deduped_20260604` | completed | Fresh seed 101 preserved positive movement; three-card deduped aggregate has 56 unique samples and top-margin ties fixed-D reward while touching far fewer rows. |
| ARC Challenge memetic rule search | `trm_choice_memetic_arc_challenge_3seed_20260604` | completed | Four generations over 56 deduped samples selected `top_margin:max_0_25:penalty_0_25`, preserving +0.053571 delta with zero damages and 0.160714 touch rate. |
| Gain-function policy trainer | `trm_gain_function_memetic_arc_bootstrap_20260604` | completed | Trained `vpd_edit_policy_arc_bootstrap_v1` from ARC route-rule evidence; prefers selective top-margin suppression and downranks broad fixed-label priors. |
| Gain-function fresh-card validation | `trm_gain_function_memetic_arc_4seed_20260604` | completed | Seed 151 kept the trained tight top-margin policy positive; four-seed aggregate keeps it best with +0.048387 delta, zero damages, and 0.16129 touch rate. |
| Gain-policy VPD bridge | `trm_gain_policy_vpd_bridge_arc_20260604` | completed | Converts the trained selective top-margin policy into a claimable logit-hook analogue, matched broad-prior controls, and a non-claimable VPD feature-search plan. |
| Gain-policy hook scoring | `trm_gain_policy_hook_score_arc_4seed_20260604` | completed | Selective hook remains positive on cached four-seed ARC scores, but ties the best broad-prior control; promotion to VPD feature search is blocked until it beats controls. |
| Rescue-family analysis | `trm_gain_policy_rescue_families_arc_4seed_20260604` | completed | Splits hook rescues into shared versus isolated families; all three hook rescues are shared with controls, producing the next edit contract instead of a premature VPD claim. |
| Family split scoring | `trm_gain_policy_family_split_score_arc_4seed_20260604` | completed | Scores the hook against controls on family, margin, and held-out score-file splits; finds five lower-touch specificity ties but zero promotion-ready splits. |
| Conditional policy mining | `trm_gain_policy_condition_miner_arc_4seed_20260604` | completed | Mines stricter cached predicates and finds a higher-scoring `top_D runner_A margin<=1.0` condition with +0.064516 delta, 4/0 rescues/damages, and 0.064516 touch rate. |
| Conditional policy cross-validation | `trm_gain_policy_condition_cv_arc_4seed_20260604` | completed | Leave-one-score-file-out validation rediscovers the same `top_D runner_A margin<=1.0` condition in every fold; all held-out folds accept, one beats fixed controls, and two tie controls with lower touch. |
| Feature-search packet | `trm_gain_policy_feature_search_packet_arc_20260604` | completed | Converts the validated `D over A` controller predicate into positive/negative contrast sets and an activation-local VPD feature-search contract. |
| Activation contrast probe requests | `trm_gain_policy_activation_contrast_arc_20260604` | completed | Emits eight module activation probe requests for the `D over A` contrast packet; later activation capture and feature-map bridge consume this handoff. |
| Activation feature-map bridge | `trm_gain_policy_activation_feature_map_arc_20260604` | completed | Joins the contrast packet with the probe-request run and emits a probe-only activation feature map plus eight runtime edit trial requests. |
| Activation-gated runtime proxy scoring | `trm_gain_policy_runtime_edit_score_arc_20260604` | completed | Scores ranked activation-map trials over the nine captured ARC contrast rows; best trial is positive but loses to fixed `D` suppression and fails the touch-rate promotion gate. |
| Edit bootstrap microcycle | `trm_edit_bootstrap_microcycle_arc_20260604` | completed boundary | Stateful cached route-rule search accepts one `top_D runner_A margin<=1.0` edit, improving score from 0.709677 to 0.774194, then rejects the second residual edit because fixed `B` control has higher reward. |
| Multi-family bootstrap sweep | `trm_multi_family_bootstrap_sweep_arc_20260604` | completed boundary | Scores 15 failure families with the microcycle as inner gate; 3 first-edit families, 0 stacked-bootstrap families, best family `D->A:A:medium_1_0` with +0.076923 slice delta. |
| Paper roundout artifacts | `vpd_trm_paper_roundout_20260604` | completed | Emits the final experiment manifest, compact metric table, and summary SVG for the conservative paper track while preserving activation-local runtime edits as open work. |

## Current Thesis

VDP is useful for TRM feature steering when the model family is small, the action space is explicit, and the workflow has measurable guardrails. In that setting, decomposition does not merely describe weights; it yields candidate control motifs that can be stress-tested through ablation, runtime grafting, and random-control comparisons.

The broader architecture is a generalization-to-specialization loop. The larger controller supplies general goals and evaluation pressure; the TRMs supply cheap, auditable specialized decisions; VPD supplies a way to inspect and edit the specialization boundary.

The strongest paper claim:

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

## Frozen Evidence Set

The paper phase treats the following artifacts as the evidence freeze rather than as a staging area for more open-ended searches:

```text
D:\Research_Engine\runs\vpd_trm_paper_roundout_20260604
```

Roundout contents:

```text
paper_claim_ledger.csv
paper_experiment_manifest.csv
paper_metric_table.csv
paper_policy_comparison.csv
figures/roundout_summary.svg
figures/filtered_feedback_policy_gain.svg
figures/eval_alignment_collapse.svg
```

Current roundout summary:

```text
manifest rows: 10
metric rows: 37
policy rows: 5
claim rows: 6
included main-claim runs: 3
excluded runs: 1
```

Paper posture:

- Use the organelle transfer, filtered feedback-loop, and ARC condition-validation rows as the positive empirical backbone.
- Use Intellect-3 eval alignment, activation-local runtime scoring, and stacked bootstrap searches as boundary evidence.
- Do not add new exploratory runs unless a reproducibility hole is discovered.
- Treat broad RL-style TRM editing as future work motivated by these results, not as the paper's demonstrated contribution.

## Completed Experiment Evidence

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

Acceptance status:

- stderr empty: met
- accepted VPD grafts remain nonzero: met
- accepted grafts recur across seeds or gate families: partially met
- stable grafts remain separated from accepted grafts: met
- random-control positives are reported, not hidden: met by roundout claim ledger and control figures

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

### Eval-Aligned Hill Climb and Targeted Rappel

The first full hill-climb run kept the acceptance gate strict and did not add targeted retargeting:

```text
run: D:\Research_Engine\runs\trm_eval_hill_climb_intellect3_logic_full_20260602
rounds: 6
source_candidate_count: 7764
frontier_row_count: 1450
near_miss_count: 24
accepted_eval_aligned_count: 0
max_packet_tokens: 953
```

This is the clearest cliff result so far. The router-confusion candidates reached plausible deterministic eval deltas, but the matched random controls reached the same or better values. The best frontier stayed at `0.023148` estimated eval delta, and strict acceptance stayed at zero. This is not a harness failure; it is evidence that router/signature-route proxy movement is not yet enough to claim downstream Intellect-3-Logic steering.

To rappel up the cliff rather than merely report it, the next run added a targeted candidate generator. When a cluster has too little direct frontier, the generator retargets same-gate VPD evidence into the missing Intellect cluster while preserving provenance:

```text
targeted gates: format_commit, candidate_verify
variant type: targeted_gate_retarget
acceptance: runtime only, deterministic estimate, no checkpoint mutation
provenance: parent recipe, source gate, retargeted_from, targeted_probe flag
```

The targeted run produced the first accepted eval-aligned rows:

```text
run: D:\Research_Engine\runs\trm_eval_hill_climb_intellect3_logic_targeted_final_20260602
rounds: 6
source_candidate_count: 7764
frontier_row_count: 2458
targeted_probe_count: 1008
accepted_eval_aligned_count: 36
accepted_targeted_probe_count: 36
accepted_non_targeted_count: 0
accepted_without_matched_random_count: 36
max_packet_tokens: 1212
```

Interpretation:

- The cliff is real for non-targeted router transfer under matched random controls.
- The targeted rappel finds format-commit candidates, but all accepted rows are targeted probes.
- The accepted targeted probes clear guardrail and deterministic eval-alignment checks, but they do not yet clear a matched-random comparison because no matched random controls exist for that format-commit cluster.
- Therefore this is a productive ascent step, not yet the final paper-grade positive result.

The follow-up experiment generated matched random controls for the targeted format-commit/candidate-verify probes:

```text
run: D:\Research_Engine\runs\trm_eval_hill_climb_intellect3_logic_targeted_controls_strict_20260602
rounds: 6
source_candidate_count: 7764
frontier_row_count: 2506
targeted_probe_count: 1056
targeted_random_control_count: 976
accepted_eval_aligned_count: 0
accepted_targeted_probe_count: 0
accepted_without_matched_random_count: 0
max_packet_tokens: 1193
```

The targeted controls collapse the apparent positive result. The first targeted run was still useful because it showed how to climb from a missing candidate surface into plausible format-commit probes. The controlled run is more important for the paper: once same-gate random controls are retargeted into the same missing cluster and the gate requires superiority over matched-random P95 on eval delta and efficiency, no targeted probe remains accepted.

This sharpens the boundary claim. Current VPD evidence can find reusable gate motifs and can generate plausible retargeted probes, but it has not yet shown robust downstream Intellect-3-Logic hill climbing under controlled comparison. That is a publishable cliff if it holds under a live scorer or a richer candidate generator.

### Feature-Map Status

The current project has two different kinds of feature maps, and the paper does not conflate them.

The HRM-text feature-map path is mechanically real. It produced chunked low-rank SVD refinements over 16 HRM-text chunks and writes coarse/refined feature-map JSON artifacts. Those maps show that the VPD editing stack can rank salient modules, split them into lower-rank slices, and produce measurable logit deltas. They are not, however, organelle TRM maps and they are not grounded in Intellect-3-Logic eval improvement.

The organelle TRM evidence is stronger for proxy control than for eval-grounded logical efficacy. The Organelle Arena provides head-row and head-SVD components, transfer matrices, accepted proxy grafts, matched random controls, and decision-resonance artifacts. That is enough to rank candidate vectorization families. It is not yet enough to claim a trained semantic feature map for logical efficacy.

Current vectorization hypotheses:

- `head_svd:*` over `format_commit` and `repair_step`: strongest proxy family for repair/format preservation, but not yet an eval-proven logic edit.
- `head_row:veto`, `head_row:repair`, `head_row:commit`: semantically aligned with verifier behavior and worth testing for format/commit failures.
- `signature_route:head_row:route_plain`: tempting for router confusion, but currently a weak candidate because router near misses fail matched random eval gates.
- Low-rank SVD slices of organelle heads: likely better than whole-component grafts because whole head/row edits appear too coarse under eval controls.

The next artifact should therefore be an eval-grounded organelle feature-map miner, not a reuse of the HRM-text feature map. Its output should be treated as a hypothesis map:

```text
logic_efficacy_feature_map.json
ranking = eval_delta - matched_random_p95
filters = guardrail pass, specificity, recurrence, cluster alignment
claim = candidate vectorizations to test, not proven controlled edit gains
```

The first mined artifact is:

```text
run: D:\Research_Engine\runs\trm_logic_efficacy_feature_map_20260602
source: D:\Research_Engine\runs\trm_eval_hill_climb_intellect3_logic_targeted_controls_strict_20260602
component_record_count: 9
feature_map_entry_count: 9
accepted_component_count: 0
positive_eval_margin_count: 0
```

Top hypotheses are `signature_route:head_row:route_plain` for router confusion and `format_commit` components such as `head_svd:0001`, `head_row:repair`, and `head_row:veto` for format-commit failures. None has positive margin over matched random controls. The feature map is therefore a ranked hypothesis map, not an edit plan that should be applied as a claimed gain.

### Edit-Discovery Protocol

The next scaffold turns the negative-result boundary into a closed-loop edit-discovery protocol with four separated stages: eval problem isolation, feature correlation, bandit policy state, and controlled hill-climb validation. The exported protocol note is:

```text
C:\projects\VDP\papers\VPD_TRM_Feature_Steering\vpd_trm_hill_climb_protocol.md
```

The first artifact-driven protocol run is:

```text
run: D:\Research_Engine\runs\trm_edit_discovery_protocol_intellect3_logic_20260602
eval_run: D:\Research_Engine\runs\trm_eval_feedback_loop_smoke_20260601_v3
hill_climb_run: D:\Research_Engine\runs\trm_eval_hill_climb_intellect3_logic_targeted_controls_strict_20260602
feature_map_run: D:\Research_Engine\runs\trm_logic_efficacy_feature_map_20260602
problem_card_count: 7
feature_scan_count: 9
feature_family_count: 4
edit_action_trial_count: 9
accepted_controlled_edit_count: 0
policy_packet_estimate: 725
```

This run emits `eval_problem_cards.jsonl`, `feature_correlation_scan.jsonl`, `feature_family_summary.jsonl`, `edit_action_trials.jsonl`, `reward_history.jsonl`, `rl_feature_policy_state.json`, and `self_model.json`. The top recommended trials are low-rank refinements over `format_commit` features: `head_svd:0000`, `head_row:repair`, `head_row:veto`, `head_svd:0002`, and `head_svd:0001`. Router-oriented `signature_route` rows are recorded as failed or abstain families because their mean eval margin stays below matched random controls.

This is the first version of the "editable failures" loop in paper form. It does not mutate weights and does not claim a positive downstream edit. Its contribution is procedural: failures are now cards, feature hypotheses are ranked separately from edits, the policy state is compact enough for an 8K-context controller, and the next probes are machine-readable.

The first focused follow-up tried low-rank blend refinements over the protocol-selected format-commit features:

```text
run: D:\Research_Engine\runs\trm_format_commit_refinement_intellect3_logic_20260602
source_protocol: D:\Research_Engine\runs\trm_edit_discovery_protocol_intellect3_logic_20260602
source_frontier: D:\Research_Engine\runs\trm_eval_hill_climb_intellect3_logic_targeted_controls_strict_20260602
selected_components: head_svd:0000, head_row:repair, head_row:veto, head_svd:0002, head_svd:0001
base_row_count: 34
blend_count: 64
strict_gate_pass_estimate_count: 0
prompt_packet_estimate: 1140
```

This run is another useful negative result. Every generated blend cleared the eval-delta margin against its matched random P95, but every blend failed efficiency and specificity. The best blend, `head_svd:0001 + head_svd:0002`, had `target_eval_delta_margin = 0.011366`, but `efficiency_margin = -0.392342` and `specificity_margin = -0.000002`. This suggests that naive combination can increase the estimated target movement while making the edit too broad or too costly relative to matched random controls. The next positive route is therefore not "add more features"; it is either live scoring of the most efficient single-feature near misses, or a refinement method that explicitly optimizes efficiency and specificity rather than only target delta.

The RL signal should come from live scoring, not from the offline feature map. The live-scoring scaffold now separates those roles:

```text
offline role: choose compact candidate probes and matched-control thresholds
live role: score runtime edit on pinned eval/holdout cards and emit reward
reward: live_eval_delta - matched_random_p95_delta + efficiency/specificity margins - guardrail/holdout penalties
```

The first live-RL queue is:

```text
run: D:\Research_Engine\runs\trm_live_rl_signal_intellect3_logic_20260602
source_refinement: D:\Research_Engine\runs\trm_format_commit_refinement_intellect3_logic_20260602
request_count: 12
scored_count: 0
pending_count: 12
accepted_live_edit_count: 0
prompt_packet_estimate: 4371
```

No live gain is claimed here because no real scorer rows have been written yet. A separate mock-positive run verifies that the reward path can accept scored candidates, but that run is harness-only and is excluded from the result claim. The next experimental step is to bind these `live_score_requests.jsonl` records to the actual Intellect-3-Logic scorer, then feed `live_score_results.jsonl` back into the policy state.

The first scorer bridge into the local Tesseract harness is now prepared:

```text
run: D:\Research_Engine\runs\trm_tesseract_live_score_bridge_intellect3_logic_20260602
live_run: D:\Research_Engine\runs\trm_live_rl_signal_intellect3_logic_20260602
request_count: 12
resolved_envs: intellect_3_logic = 1
unresolved_sample_count: 0
sample_receipt: tesseract_live_sample_receipt.json
bench_output: tesseract_live_bench_output.json
runtime_hook_status: template_without_concrete_hooks
```

The bridge recovers the pinned failed Prime Intellect environment sample as `intellect_3_logic_330`, the Mathador prompt with trace `e0587481-c4ac-4559-ae1b-324739bb1570`, and emits a local Tesseract `comprehensive_bench.py` command for the 2B routed adapter lane. This is still not an accepted live edit, because the current local harness can rerun the routed adapter on the Prime Intellect sample but cannot yet apply a VPD runtime candidate during generation. Running it now would produce an unedited calibration score, not a VPD intervention result. The immediate engineering blocker is therefore the runtime hook that applies a candidate feature edit inside the scorer call.

The local harness now has a runtime-hook contract:

```text
benchmark args: --vpd-live-request, --vpd-runtime-hook
supported concrete hook: module_path + mode=scale_output + scale
per-sample output: routed_*_vpd_hook_status
claimability: false unless concrete hooks are applied and manifest source is vpd_runtime_hook
```

The regenerated bridge command includes both hook arguments and writes:

```text
vpd_live_request.json
vpd_runtime_hook_template.json
```

The current template is intentionally non-claimable:

```text
candidate_id: format_commit_refine:0304:head_svd:0001+head_svd:0002
abstract_component_ids: head_svd:0001, head_svd:0002
hooks: []
claimable: false
runtime_hook_status: template_without_concrete_hooks
```

This turns the remaining blocker into a precise mapping problem: map the abstract organelle VPD feature IDs (`head_svd:*`, `head_row:*`) into concrete Hugging Face module paths and output-scaling hooks for the routed adapter model. Until that mapping exists, live scorer runs remain calibration runs rather than accepted VPD edits.

The first abstract-to-concrete mapping pass now emits candidate hook manifests from the routed adapter metadata:

```text
adapter: D:\Research_Engine\tesseract_persistent\data\models\adapters\2B\2026-03-12-overnight\intellect_3_logic
base model: D:\Research_Engine\models\Qwen3.5\Qwen3.5-2B-Base-HF
adapter target_modules: o_proj, out_proj, down_proj, plus other LoRA projections
hook_candidate_manifest_count: 24
hook_candidate_manifest_index: hook_candidate_manifest_index.jsonl
hook_candidate_benchmark_commands: hook_candidate_benchmark_commands.jsonl
```

The first generated candidate maps `head_svd:0001` to:

```text
module_path: base_model.model.model.layers.23.self_attn.o_proj
mode: scale_output
scale: 1.1
verification_status: unverified_module_path
claimable: false
```

This is progress, but not yet evidence. These hook candidates let us run one concrete routed-model path probe at a time. They do not yet prove that the organelle feature and the Hugging Face module are the same causal object. A candidate can become claimable only after the runtime hook resolves, the mapping is reviewed, and live scorer rows clear the matched-random reward gates.

The first resolved hook probe cleared the runtime blocker. The local 2B routed adapter accepted a concrete hook at:

```text
module_path: base_model.model.model.language_model.layers.23.self_attn.o_proj
mode: scale_output
scale: 1.1
hook_status: applied
hook_count: 1
failed_hooks: 0
```

The corresponding Intellect-3-Logic one-sample run did not improve the task score, but it established that the scorer can now apply an edit during generation rather than merely replaying an unedited adapter. A follow-up simpler-eval probe used `wordle` because it has compact actions and dense near-miss structure. Unmasked Wordle saturated at 4/4 because several normalized trajectory prompts leak the target guess. The harness therefore added a non-destructive `--mask-wordle-target` mode that masks explicit mentions of the target guess in the prompt while preserving the same sampled records and target actions.

Masked Wordle produced a non-saturated baseline:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603
samples: 4
baseline_instruct_sr: 0.25
routed_2b_sr: 0.50
```

A one-load masked Wordle hook sweep then tested resolvable runtime hooks over late attention/MLP projections:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\sweep_small_late_strong
examples: 2
candidate_count: 24
resolvable_candidate_count: 24
baseline_score: 0.50
accepted_live_edit_count: 0
best_delta: 0.0
```

The result is a useful negative control rather than a success claim. Extreme output scaling did perturb generations, including destructive changes, so the hook path is not inert. However, coarse layer-output scaling did not discover an exact-score-improving edit. The next candidate generator should therefore move from blanket module scaling to failure-correlated feature selection: identify features active on the failing masked Wordle case but not on preserved cases, then steer those vectors or LoRA subcomponents instead of scaling whole layer outputs.

That failure-correlated pass has now been attempted. The first ranked sweep captured per-module mean activation magnitudes on the masked Wordle baseline and selected modules by failed-vs-passed contrast:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\sweep_activation_top4
examples: 4
selected_module_count: 4
candidate_count: 20
baseline_score: 0.50
accepted_live_edit_count: 0
top selected modules: layers 11, 19, 7, 15 self_attn.q_proj
```

This run showed that failure-correlated module selection can identify a coherent family, q-projection modules with slightly lower failed-case activation magnitude, but whole-module scaling of those ranked modules still did not produce a score gain.

A second pass used mean activation vectors rather than scalar activation magnitude. It captured pass-minus-fail vectors, ranked modules by vector contrast, and applied additive activation-direction hooks:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\direction_top3_strong
examples: 4
selected_module_count: 3
candidate_count: 18
baseline_score: 0.50
accepted_live_edit_count: 0
top selected modules:
  layer 23 mlp.gate_proj
  layer 23 self_attn.q_proj
  layer 19 self_attn.q_proj
```

Large additive alphas moved behavior, including malformed or degraded guesses, but did not improve exact Wordle accuracy. This narrows the next protocol change: the live edit path is operational, and failure-correlated feature ranking is measurable, but mean module-level activation edits are still too coarse. The next credible step is either token-position-specific steering at the decision token or direct LoRA subcomponent edits, especially `lora_A`/`lora_B` rows for the ranked q-projection and gate-projection modules.

A first direct LoRA subcomponent diagnostic was then added. The run targeted the highest-contrast module from the direction sweep, layer 23 `mlp.gate_proj`, and scaled only its `lora_B.default` output branch:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\lora_B_gate23_diag
examples: 2
module_path: base_model.model.model.language_model.layers.23.mlp.gate_proj.lora_B.default
scales: 0.0, 2.0, 4.0
baseline_score: 0.50
accepted_live_edit_count: 0
predictions: [house], [crane] unchanged across tested scales
```

This result suggests that simply scaling the adapter output branch at this MLP gate is not enough to move the decision on the failing masked Wordle sample. It also explains why broad LoRA sweeps are not currently efficient on CPU: each candidate still requires autoregressive generation, and small branch-level edits may be inert unless selected at the correct token position or applied as a weight-space row/vector edit. The follow-up therefore avoided more blind LoRA scale sweeps and instrumented token-level constrained steering over legal Wordle guesses.

The first constrained token-level probe produced a small positive diagnostic. Instead of free-form generation, the scorer evaluated a compact candidate action set and applied a generic penalty to the overused default action `[crane]`:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\constrained_anti_crane_4sample
examples: 4
candidate actions: [crane], [house], [flame], [grant], [slate]
baseline constrained score: 0.50
best policy: penalty 8.0 against [crane]
best constrained score: 0.75
accepted policy count: 2
```

Per-sample behavior:

```text
baseline: [house], [crane], [crane], [crane]
targets:  [house], [flame], [grant], [crane]
penalty:  [house], [flame], [grant], [flame]
```

This is the first positive live-scored hill-climb signal in the Wordle scaffold, but it is not yet a VPD weight-edit claim. The intervention is an action-level constrained decode policy derived from the observed over-selection failure mode. It improves the two stuck non-`[crane]` failures while sacrificing the true `[crane]` case, giving a net +0.25 on the four-sample card. The result is useful because it shows that the failure is not opaque: the correct alternatives already have relatively high likelihood, and a simple anti-default token policy can expose them. The next paper-grade step is to distill this policy into a model-internal edit, for example by steering only when the final-action distribution is dominated by `[crane]` and nearby legal candidates have compatible Wordle constraints.

The first extrapolation check added a simple context gate: apply the anti-`[crane]` penalty only after the initial Wordle step. This preserves legitimate opening `[crane]` guesses while suppressing repeated defaulting later in the trajectory. On the original four-sample card, this improved the constrained score from `0.50` to `1.00`:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\constrained_post_initial_4sample
mode: post_initial
best policy: penalty 8.0 against [crane]
baseline score: 0.50
best score: 1.00
best delta: +0.50
control penalties accepted: 0
```

An 8-sample validation preserved the same pattern:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\constrained_post_initial_8sample
examples: 8
candidate actions: 7
baseline score: 0.50
best target policy: post_initial penalty 8.0 against [crane]
best target score: 1.00
best target delta: +0.50
accepted target policy count: 2
accepted control policy count: 0
```

This makes the pattern more than a one-off on the original pinned card. The extrapolatable hypothesis is now: **after an initial Wordle opener, the routed model over-defaults to `[crane]`; a gated anti-default policy can recover nearby legal alternatives without suppressing legitimate opener use.** The claim remains narrow because the action set is constrained and small, but the matched control result is encouraging: penalizing `[house]`, `[flame]`, `[grant]`, or `[slate]` did not beat baseline on the same 8-sample card.

The policy was distilled into a reusable score-file artifact:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\policy_distill_post_initial_8sample
input score file: constrained_post_initial_8sample\wordle_constrained_sample_scores.jsonl
distilled policy: wordle_post_initial_anti_crane
policy type: constrained_decode_penalty
gate: post_initial
penalized action: [crane]
penalty: 8.0
baseline score: 0.50
best target score: 1.00
best target delta: +0.50
accepted target policy count: 45
accepted control policy count: 0
policy card: wordle_post_initial_anti_crane_policy.json
```

This gives the next stage a concrete object to distill into model internals. Rather than asking "can any VPD edit help?", the internal-edit target is now specific: reproduce the constrained policy `post_initial -> penalize [crane] by about 8 log-prob units` while preserving initial-step `[crane]` behavior. Candidate mechanisms include a final-action logit hook, a small auxiliary TRM route rule, or a LoRA row/vector edit targeted at the final answer-token distribution.

The policy card was then exported into a compact TRM-style route rule:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\policy_card_apply_post_initial_8sample
route rule: wordle_trm_route_rule.json
rule_id: wordle_post_initial_anti_crane
rule_type: post_initial_action_penalty
condition: env_id=wordle, step > 0, candidate action [crane] present
action: penalize [crane] by 8.0 log-prob units
baseline score: 0.50
policy score: 1.00
delta: +0.50
```

This route rule is the first compact artifact suitable for the TRM/gym side of the loop. It is not a learned weight edit, but it is a distilled decision rule with explicit trigger, action, reward, and claim boundary. That makes it a candidate teacher label for a small route TRM, or a target behavior for a later VPD/logit-hook internalization experiment.

The validation harness now supports cached score-card replay and matched-control evaluation without reloading the model:

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\policy_validation_cached_4plus8
cards: constrained_post_initial_4sample, constrained_post_initial_8sample
aggregate samples: 12
accepted cards: 2 / 2
mean target delta: +0.50
mean best-control delta: 0.0
all cards accepted: true
```

The current validation is still small because fresh constrained score-card generation is CPU-bound. The harness therefore separates expensive score generation from cheap policy validation: future overnight runs should generate additional `wordle_constrained_sample_scores.jsonl` cards, then reuse the cached validator to test the route rule and controls instantly.

The first larger cached validation completed on June 4, 2026 after a small-GPU retry. The initial overnight scorer failed because it materialized a full `float32` log-softmax tensor on the 4 GB RTX 3050 Laptop GPU. The scorer was patched to evaluate candidate actions in batch size 1 and gather only the target-token log-probabilities. That preserved the constrained scoring semantics while keeping the live scorer inside the available VRAM.

```text
run: D:\Research_Engine\runs\trm_wordle_vpd_hook_probe_20260603\overnight_validation_32x2_20260604_retry_smallgpu
score cards: 4, 8, 32(seed 23), 32(seed 37)
aggregate samples: 76
accepted cards: 4 / 4
all cards accepted: true
mean target delta: +0.359375
mean best-control delta: +0.007812
policy card: wordle_post_initial_anti_crane_policy.json
validation summary: policy_validation_4_8_32_32\wordle_policy_validation_summary.json
```

The two new 32-sample cards independently preserved positive target movement:

```text
seed 23: baseline 0.34375 -> policy 0.53125, target delta +0.1875, best-control delta +0.03125
seed 37: baseline 0.28125 -> policy 0.53125, target delta +0.25, best-control delta 0.0
```

This is the strongest current evidence that the Wordle route rule is an extrapolatable failure-mode intervention rather than a one-off rescue. The result still should not be described as a VPD weight edit. It is a live-scored constrained decode policy with a compact TRM route-rule representation. The paper-grade claim is that the feedback loop has isolated a reproducible, measurable behavioral attractor and a gateable intervention target: post-initial over-defaulting to `[crane]`.

After this result, a local constrained-env audit checked whether another normalized Tesseract environment could support the same short-candidate protocol without additional candidate extraction:

```text
run: D:\Research_Engine\runs\trm_constrained_env_audit_20260604_v3
env count: 49
short constrained-decode suitable envs: 0
```

This is an important boundary for the method. Several envs have fixed labels (`arc_easy`, `arc_challenge`, MMLU subsets), but their normalized prompts do not expose answer choices, so a constrained label scorer would be under-specified. Other envs expose replay-like response candidates but the actions are long free-form strings, not compact action tokens. The next replication step is therefore not to force a second Wordle-shaped policy onto those envs. It is to add a candidate-extraction stage that reconstructs explicit action alternatives from task metadata or original datasets, then reruns the same score-card, matched-control, and route-rule validation protocol.

That candidate-extraction stage now exists for local ARC and MMLU choice environments. It rehydrates normalized trajectory rows from the raw Parquet files under `D:\Research_Engine\prime_envs`, reconstructs prompts with explicit choices, and emits compact candidate-card rows keyed by normalized `trajectory_id`:

```text
run: D:\Research_Engine\runs\trm_choice_candidate_cards_20260604
envs: arc_easy, arc_challenge, mmlu_formal_logic
candidate cards: 192
matched target rows: 192
mismatches: 0
card file: choice_candidate_cards.jsonl
```

A first small ARC Easy live score then exercised the generic constrained-choice scorer:

```text
run: D:\Research_Engine\runs\trm_choice_constrained_arc_easy_8_20260604
env: arc_easy
examples: 8
candidate action count: 4
baseline constrained score: 0.875
best policy: penalize B by 1.0
best score: 1.0
best delta: +0.125
```

This is a pilot signal, not a validated second-environment result. It repaired one near miss, `arc_easy_37`, where baseline chose `B` and the target was `A`; the small `B` penalty flipped that sample without breaking the other seven examples. The next validation step is to run additional ARC Easy seeds and matched action controls exactly as Wordle did before making an extrapolation claim.

The next ARC rung used `arc_challenge` as the local "medium" choice environment. The 8-example pilot again found a small action-penalty gain:

```text
run: D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_8_20260604
env: arc_challenge
examples: 8
baseline constrained score: 0.625
best policy: penalize B by 1.0
best score: 0.75
best delta: +0.125
```

Two 32-example cards preserved smaller positive movement:

```text
run: D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed23_20260604
baseline: 0.625
best policy: penalize D by 1.0
best score: 0.6875
best delta: +0.0625

run: D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed37_20260604
baseline: 0.78125
best policy: penalize B by 1.0
best score: 0.84375
best delta: +0.0625
```

This is better than a single-card fluke, but weaker than the Wordle result because the best penalized action is not stable across 32-example seeds. The useful interpretation is that ARC Challenge has local over-selection near misses that the constrained scorer can expose, but it does not yet show one clean reusable route rule. The next ARC experiment should validate a family-level policy such as "small penalty on the current overconfident top choice when the second choice is close" rather than a fixed anti-`B` or anti-`D` rule.

That family-level probe now exists as a tight cached reward loop. It loads cached constrained-choice score files, deduplicates overlapping `env_id:trajectory_id` rows, scores fixed-action and top-margin policies, and writes MCP-style state resources for the next live run. The reward is immediate rather than retrospective: `delta + 0.01 * (rescues - damages)`, where rescues are baseline misses repaired by the policy and damages are baseline hits broken by the policy.

```text
run: D:\Research_Engine\runs\trm_choice_rl_feedback_arc_challenge_deduped_20260604
input score cards: arc_challenge_32_seed23, arc_challenge_32_seed37
raw samples: 64
deduped samples: 48
candidates scored: 48
accepted candidates: 21
best policy: top_margin:max_0_5:penalty_0_5
baseline pass rate: 0.708333
policy pass rate: 0.791667
best delta: +0.083333
best reward: +0.123333
rescues/damages: 6 / 2
prompt packet estimate: 265 tokens
```

This is the first ARC Challenge result shaped like an RL-acute teaching loop rather than a manual analysis pass. The loop does not wait for a broad post-hoc report; it converts each cached score card into policy reward, updates `rl_policy_state.json`, writes `reward_history.jsonl` and `replay_candidates.jsonl`, and emits a compact `prompt_packet.txt` that tells the next harness step what to try. The current next action is clear and falsifiable: run one fresh-seed ARC Challenge live score card with the top-margin policy, then accept it only if the positive delta survives deduped scoring and matched fixed-action controls.

The fresh-seed rerun completed on a new 32-example ARC Challenge card:

```text
score run: D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed101_20260604
env: arc_challenge
examples: 32
baseline constrained score: 0.71875
best fixed-label policy: penalize D by 1.0
best fixed-label score: 0.8125
best fixed-label delta: +0.09375
```

When the same score card is fed through the RL feedback loop, the cached top-margin policy remains accepted:

```text
reward run: D:\Research_Engine\runs\trm_choice_rl_feedback_arc_challenge_seed101_20260604
best policy: fixed:D:penalty_1_0
best reward: +0.12375
best delta: +0.09375
top-margin policy: top_margin:max_0_5:penalty_0_5
top-margin accepted: true
top-margin score: 0.78125
top-margin delta: +0.0625
top-margin reward: +0.0825
top-margin rescues/damages: 4 / 2
top-margin touched samples: 7 / 32
```

The three-card deduped aggregate now has 56 unique ARC Challenge samples:

```text
run: D:\Research_Engine\runs\trm_choice_rl_feedback_arc_challenge_3seed_deduped_20260604
raw samples: 96
deduped samples: 56
baseline pass rate: 0.714286
best fixed-label policy: fixed:D:penalty_1_0
best fixed-label pass rate: 0.767857
best fixed-label delta: +0.053571
top-margin policy: top_margin:max_0_5:penalty_0_5
top-margin pass rate: 0.767857
top-margin delta: +0.053571
top-margin reward: +0.083571
top-margin rescues/damages: 6 / 3
top-margin touched samples: 11 / 56
```

This strengthens the family-rule interpretation. Fixed `D` and the top-margin rule tie on aggregate reward, but the top-margin policy touches far fewer samples (`11/56` versus `55/56` for fixed `D`). For a route-rule or TRM-teaching loop, that selectivity matters: it is closer to a causal failure-mode intervention and less like a global answer-label prior. The next acceptance gate should therefore include a selectivity term or tie-breaker, not only raw reward.

The first memetic version of this search adds exactly that pressure. It treats candidate route rules as a population, locally refines each family over margin/penalty variants, then selects elites using a fitness that penalizes broad interventions:

```text
fitness = reward - touch_penalty * touch_rate - complexity_penalty * policy_complexity
```

The run:

```text
run: D:\Research_Engine\runs\trm_choice_memetic_arc_challenge_3seed_20260604
input score cards: arc_challenge_32_seed23, arc_challenge_32_seed37, arc_challenge_32_seed101
raw samples: 96
deduped samples: 56
generations: 4
elite count: 8
touch penalty: 0.02
complexity penalty: 0.002
best elite: top_margin:max_0_25:penalty_0_25
baseline pass rate: 0.714286
elite pass rate: 0.767857
delta: +0.053571
reward: +0.083571
fitness: +0.076357
rescues/damages: 3 / 0
touch rate: 0.160714
prompt packet estimate: 285 tokens
```

This is a better teaching signal than the raw fixed-label winner. The previous aggregate top-margin policy (`max_margin=0.5`, `penalty=0.5`) tied fixed `D` on reward but still caused three damages. The memetic local-refinement pass tightened the rule to `max_margin=0.25`, `penalty=0.25`, preserving the same score delta while reducing damage to zero and touching only nine of fifty-six samples. That is the local maximum we want the meta-skill to prefer: not the largest global label shove, but the most selective rule that repairs a recurring failure mode.

The next layer converts that search result into a controller-trainable VPD edit-policy state. The trainer does not mutate model weights. It trains the gain-function policy over cached route-rule evidence, emits checkpoints and training events, and records the learned preference as a compact MCP resource:

```text
run: D:\Research_Engine\runs\trm_gain_function_memetic_arc_bootstrap_20260604
trained object: vpd_edit_policy
input score cards: arc_challenge_32_seed23, arc_challenge_32_seed37, arc_challenge_32_seed101
raw samples: 96
deduped samples: 56
generations completed: 4
ram cap: 8192 MB
peak traced RAM: 1.424 MB
checkpoint count: 4
best candidate: gain:top_margin:max_0_25:penalty_0_25
edit family: selective_top_margin_suppression
preferred families: selective_top_margin_suppression
downranked families: broad_fixed_label_prior
fitness: +0.076357
reward: +0.083571
score delta: +0.053571
rescues/damages: 3 / 0
touch rate: 0.160714
prompt packet estimate: 274 tokens
```

This is the first concrete version of the “controlled training regime” for the meta-skill. The gain function has been externalized into `vpd_edit_policy_state.json`: score delta and rescues are rewarded, damages and touch rate are penalized, and broad fixed-label priors are downranked when a selective rule reaches the same reward surface. The claim boundary is still narrow. This is controller-policy training from route-rule evidence, not a VPD weight-edit result. Its purpose is to teach the next harness step which edit families deserve live scorer budget before promoting them to VPD-hook or feature-map search.

The first fresh-card validation of that trained gain policy used a new 32-example ARC Challenge card:

```text
score run: D:\Research_Engine\runs\trm_choice_constrained_arc_challenge_32_seed151_20260604
env: arc_challenge
examples: 32
baseline constrained score: 0.6875
best fixed-label policy: penalize D by 1.0
best fixed-label score: 0.75
best fixed-label delta: +0.0625
```

On this fresh card alone, a broader top-margin variant won the local search, but the trained tight policy remained positive and damage-free:

```text
run: D:\Research_Engine\runs\trm_gain_function_memetic_arc_seed151_20260604
best seed-local policy: top_margin:max_1_0:penalty_1_0
best seed-local fitness: +0.1335
best seed-local delta: +0.125
trained tight policy: top_margin:max_0_25:penalty_0_25
trained tight fitness: +0.036
trained tight reward: +0.04125
trained tight delta: +0.03125
trained tight rescues/damages: 1 / 0
trained tight touch rate: 0.0625
```

After folding seed 151 into the aggregate, the trained tight policy remained the best controller-policy candidate:

```text
run: D:\Research_Engine\runs\trm_gain_function_memetic_arc_4seed_20260604
raw samples: 128
deduped samples: 62
best policy: top_margin:max_0_25:penalty_0_25
fitness: +0.071161
reward: +0.078387
score delta: +0.048387
rescues/damages: 3 / 0
touch rate: 0.16129
preferred family: selective_top_margin_suppression
downranked family: broad_fixed_label_prior
```

This is the first evidence that the gain function itself is beginning to generalize: the seed-local optimum can shift, but the lower-touch trained policy stays positive on a fresh card and remains the best four-seed aggregate choice. That was enough to promote the family to the bridge stage, but not enough to claim a VPD hook result. The bridge maps `selective_top_margin_suppression` to a concrete VPD/logit-hook candidate and compares it against matched broad-prior and random-feature controls.

That bridge now exists as a claim-safe artifact generator:

```text
run: D:\Research_Engine\runs\trm_gain_policy_vpd_bridge_arc_20260604
source policy: D:\Research_Engine\runs\trm_gain_function_memetic_arc_4seed_20260604\vpd_edit_policy_state.json
policy id: vpd_edit_policy_arc_bootstrap_v1
logit-hook candidate: logit_hook:top_margin:max_0_25:penalty_0_25
matched controls: 3
prompt packet estimate: 196 tokens
```

The bridge emits three artifact classes:

```text
logit_hook_candidate.json
matched_controls.jsonl
vpd_search_plan.json
```

The logit-hook candidate is the direct analogue of the trained route rule:

```text
trigger: candidate_set_required, final_choice_action_scores, top_minus_runner_up_max_margin <= 0.25
action: penalize current_top_action by 0.25
expected behavior: suppress only close-call overselected top choices and preserve wider-margin predictions
claim boundary: logit-level route-rule analogue; not a VPD weight edit
```

The matched controls are broad fixed-label priors against `B` and `D` with the same penalty, plus a no-edit control. The VPD search plan is deliberately non-claimable. It names the next mechanisms to try, `final_choice_logit_hook`, `decision-token_activation_suppression`, and `adapter_lora_row_vector_edit`, but it requires fresh scoring and feature/module mapping before any VPD edit claim can be made. This is the right next boundary: first prove the logit-hook analogue remains positive against matched controls, then search for the smallest VPD feature or adapter-row edit that reproduces it.

The first bridge-score run gives a useful constraint rather than a clean promotion:

```text
run: D:\Research_Engine\runs\trm_gain_policy_hook_score_arc_4seed_20260604
samples: 62 deduped from 128 cached four-seed ARC score rows
logit hook: logit_hook:top_margin:max_0_25:penalty_0_25
baseline score: 0.709677
hook score: 0.758065
hook delta: +0.048387
rescues/damages: 3/0
touch rate: 0.16129
best control: control_fixed_label:B:penalty_0_25
best control delta: +0.048387
promotion_ready: false
prompt packet estimate: 182 tokens
```

This result sharpens the claim boundary. The selective top-margin hook is accepted as a route-rule/logit-hook analogue because it improves the cached score with zero damage and low touch rate. It is not isolated as a VPD-search promotion because a broad `B` prior ties its delta, even though that broad control is less specific. The follow-up family split therefore treats it as a useful controller prior, not as a feature-local causal edit.

That split now exists:

```text
run: D:\Research_Engine\runs\trm_gain_policy_rescue_families_arc_4seed_20260604
samples: 62
families: 24
hook rescues/damages: 3/0
isolated/shared hook rescues: 0/3
promotion_ready: false
prompt packet estimate: 155 tokens
```

Top rescue families:

```text
B->D:close_0_25 | samples 2 | hook rescues 2 | isolated 0 | shared 2
B->C:close_0_25 | samples 1 | hook rescues 1 | isolated 0 | shared 1
```

This is a meaningful negative control result. The hook's gains are real in the cached scorer, but every rescue is shared with a broad `B` prior. The next edit contract therefore becomes:

```text
objective: beat broad-prior controls by rescuing close-margin failures with lower touch and no damage
positive set: samples with hook_rescue=true and margin_bucket close_0_25
negative set: same baseline label where broad controls rescue or already-correct close calls
candidate feature: activation/module row active on close-call wrong-top-choice states, inactive on broad fixed-label prior states
required next condition: isolated_hook_rescue_count > 0 or hook delta exceeds best broad-prior control on held-out family split
```

This is the first clean form of the continuous-learning loop: score a candidate, compare it against controls, decompose the residual failure/gain families, emit a new edit contract, and only then search for a narrower VPD/TRM edit. The value is the edit-policy iteration, not the first hook.

The next iteration scorer operationalizes the contract over held-out and family slices:

```text
run: D:\Research_Engine\runs\trm_gain_policy_family_split_score_arc_4seed_20260604
splits scored: 22
promotion-ready splits: 0
touch-specificity ties: 5
best split: heldout_score_file:trm_choice_constrained_arc_challenge_32_seed23_20260604
prompt packet estimate: 151 tokens
```

Representative tied split:

```text
split: trm_choice_constrained_arc_challenge_32_seed23_20260604
samples: 32
hook delta/reward: +0.09375 / 0.12375
hook rescues/damages: 3/0
hook touch rate: 0.21875
best control: control_fixed_label:B:penalty_0_25
control delta/reward: +0.09375 / 0.12375
control touch rate: 0.96875
promotion_ready: false
```

This is exactly the kind of intermediate signal the continuous-learning loop needs. The current hook is not better than the broad prior in reward, but it is much more selective on every tied held-out score-file split. That means the controller should not promote the hook as a VPD edit, but it should preserve it as a compression of the broad prior into a lower-touch condition. The next edit search should look for an activation-local predicate that keeps the lower touch rate and breaks the reward tie.

The next controller iteration mined stricter cached predicates:

```text
run: D:\Research_Engine\runs\trm_gain_policy_condition_miner_arc_4seed_20260604
samples: 62
candidates scored: 375
accepted candidates: 45
best condition: cond_top_margin:top_D:runner_A:bucket_any:max_1_0:penalty_1_0
baseline score: 0.709677
condition score: 0.774194
delta: +0.064516
reward: 0.104516
efficiency reward: 0.09229
rescues/damages: 4/0
touch rate: 0.064516
prompt packet estimate: 156 tokens
```

This is the first local hill-climb step beyond the initial hook. The original selective hook improved the cached score to 0.758065 with 3 rescues and 0.16129 touch rate. The mined condition improves the cached score to 0.774194 with 4 rescues and 0.064516 touch rate by targeting cases where the current top action is `D`, the runner-up is `A`, the margin is at most 1.0, and the penalty is 1.0. In plain terms, the loop discovered a different failure family: instead of only compressing broad `B` suppression, it found a low-touch `D over A` correction.

The claim boundary still matters. This is a cached controller predicate, not a VPD feature edit. The completed follow-up validates this condition on held-out score files and maps the positive set to activation-local candidates:

```text
positive set: top_action D, runner_up A, margin <= 1.0, baseline miss corrected to A
negative set: correct D predictions, D-over-A wide-margin cases, and D-over-A misses not corrected by the condition
feature target: module/component activity that separates wrong D-over-A commitment from valid D answers
promotion gate: condition reward beats fixed-label controls on held-out split and feature-local edit reproduces the gain
```

Leave-one-score-file-out validation now supports the controller predicate:

```text
run: D:\Research_Engine\runs\trm_gain_policy_condition_cv_arc_4seed_20260604
folds: 4
held-out accepted folds: 4
beats-control folds: 1
tie-lower-touch folds: 2
best fold: trm_choice_constrained_arc_challenge_32_seed151_20260604
prompt packet estimate: 169 tokens
```

In every fold, mining on the other three score files selected:

```text
cond_top_margin:top_D:runner_A:bucket_any:max_1_0:penalty_1_0
```

Best held-out fold:

```text
heldout: seed151
heldout delta/reward: +0.09375 / 0.12375
rescues/damages: 3/0
touch rate: 0.09375
best fixed control: fixed:D:penalty_1_0
control delta/reward: +0.0625 / 0.0825
control touch rate: 0.96875
beats_control: true
```

This materially improves the evidence. The loop is no longer only finding a combined-cache optimum; it rediscovers the same condition under leave-one-score-file-out training, accepts on all held-out folds, beats fixed controls on one fold, and ties controls with much lower touch on two more. The remaining limitation is still important: this is cached score validation, not activation-level VPD editing. The next promotion step should generate feature candidates for the `D over A` positive set and test whether a feature-local edit can reproduce the held-out controller predicate.

The feature-search packet now exists:

```text
run: D:\Research_Engine\runs\trm_gain_policy_feature_search_packet_arc_20260604
condition: cond_top_margin:top_D:runner_A:bucket_any:max_1_0:penalty_1_0
samples: 62
positive rescues: 4
negative contrast rows: 5
contrast rows emitted: 9
prompt packet estimate: 218 tokens
```

The positive set is the `D over A` rescue set:

```text
arc_challenge_34 | D -> A | margin 0.875 | seed23
arc_challenge_49 | D -> A | margin 0.375 | seed23
arc_challenge_19 | D -> A | margin 0.25  | seed37
arc_challenge_39 | D -> A | margin 0.875 | seed101
```

The negative set contains same-pair `D over A` cases where the condition should not fire, mostly wide-margin correct `D` cases plus one wide-margin unresolved miss. This gives a clean activation contrast request:

```text
rank modules/components by positive-vs-negative activation contrast
test runtime edits that suppress wrong top-action D only on positive-like states
promotion gate: reproduce held-out condition reward, beat fixed-label controls on at least one held-out score file, damage_count == 0, touch_rate <= controller predicate touch_rate
```

This is the first point where the loop is ready to leave cached logit predicates and ask VPD for a mechanistic candidate. The immediate next run should not search the whole model blindly; it should rank candidate modules/components against this nine-row contrast packet and only then try runtime component edits.

The activation contrast harness is now staged:

```text
run: D:\Research_Engine\runs\trm_gain_policy_activation_contrast_arc_20260604
condition: cond_top_margin:top_D:runner_A:bucket_any:max_1_0:penalty_1_0
positive/negative samples: 4/5
module probe requests: 8
ranked modules: 0
status: probe_requests_ready
prompt packet estimate: 137 tokens
```

The eight probe requests cover late and mid-layer attention/output and MLP/down-projection modules:

```text
layers 23,19,15,11 x {self_attn.o_proj, mlp.down_proj}
```

Each request carries the same four positive rescue sample ids and five negative same-pair sample ids, with the metric `mean_abs_activation`. This is not yet evidence for a VPD edit. It is the ready-to-run activation capture plan. Once activation stats are available, the ranker will sort modules by absolute positive-vs-negative contrast and the next harness should test top runtime edits against fixed-label controls.

The feature-map bridge is now staged too:

```text
run: D:\Research_Engine\runs\trm_gain_policy_activation_feature_map_arc_20260604
status: probe_only
contrast run: D:\Research_Engine\runs\trm_gain_policy_feature_search_packet_arc_20260604
probe run: D:\Research_Engine\runs\trm_gain_policy_activation_contrast_arc_20260604
feature map entries: 8
edit trial requests: 8
activation capture requests: 8
prompt packet estimate: 174 tokens
```

The emitted feature map is intentionally neutral:

```text
source: activation_probe_request
scale: 1.0
confidence: 0.0
notes: probe_request=<id>; claim_boundary=probe_only
```

The accompanying `activation_edit_trials.jsonl` file gives the next runtime test contract without claiming a runtime gain. That is the right shape for now: one packet defines the contrast, one packet defines the probes, and one packet defines the feature-map/edit-trial handoff. When activation stats arrive, the same bridge can be rerun in ranked mode and the edit trials can be reprioritized by actual contrast instead of probe order.

The bridge now also emits `activation_capture_requests.jsonl` and `activation_capture_manifest.json`, which makes the next runtime step explicit: capture `mean_abs_activation` for the probe modules over the positive and negative sample sets, then feed those stats back into the ranked bridge mode.

That capture step has now run:

```text
run: D:\Research_Engine\runs\trm_gain_policy_activation_capture_arc_20260604
sample_count: 9
module_count: 8
activation_row_count: 72
status: completed
```

Feeding the resulting `activation_stats.jsonl` back into the bridge produced a ranked map. The strongest contrast was still small, but it was now real captured data rather than probe order:

```text
base_model.model.model.language_model.layers.23.mlp.down_proj
positive_mean: 0.393123
negative_mean: 0.384413
contrast: +0.00871
```

The remaining top rows were similarly low-magnitude and the bridge still stops short of a runtime-edit claim. The important change is that the loop now has all three stages wired: contrast packet, activation capture, and ranked feature-map handoff.

The ranked map was then scored as an activation-gated runtime proxy:

```text
run: D:\Research_Engine\runs\trm_gain_policy_runtime_edit_score_arc_20260604
samples: 9
trials: 8
best trial: activation_edit_trial:0001:base_model.model.model.language_model.layers.19.self_attn.o_proj
best trial delta/reward: +0.333333 / 0.363333
best trial touch rate: 0.333333
best control: control_fixed_label:D:penalty_1_0
best control delta/reward: +0.444444 / 0.484444
promotion_ready: false
```

This is a useful negative boundary. The activation-gated trial can rescue some captured contrast rows, but the broad fixed-label control still scores better and the touch rate exceeds the original controller predicate. The result should be reported as an implemented positive-track harness plus a failed promotion, not as a VPD runtime edit win.

The stateful edit-bootstrap microcycle now tests the more important learning question: after an accepted edit changes the current state, can the loop find another accepted residual edit and produce cumulative ability gain?

```text
run: D:\Research_Engine\runs\trm_edit_bootstrap_microcycle_arc_20260604
samples: 62
baseline score: 0.709677
final score: 0.774194
accepted edits: 1
stacked bootstrap success: false
stop reason: fixed_control_not_beaten
prompt packet estimate: 173 tokens
```

Round 1 promoted the same validated controller predicate:

```text
cond_top_margin:top_D:runner_A:bucket_any:max_1_0:penalty_1_0
delta/reward: +0.064517 / 0.104517
rescues/damages: 4/0
touch rate: 0.064516
best fixed control: fixed:B:penalty_0_25
control delta/reward: +0.048388 / 0.078388
```

After applying that edit as state, round 2 found a plausible residual condition:

```text
cond_top_margin:top_B:runner_D:bucket_any:max_0_25:penalty_0_25
delta/reward: +0.032258 / 0.052258
rescues/damages: 2/0
touch rate: 0.064516
best fixed control: fixed:B:penalty_0_25
control delta/reward: +0.048387 / 0.078387
```

The second residual condition is positive and low-touch, but it does not beat the broad fixed-label control. The microcycle therefore records a first-edit gain, not an ability-bootstrap win. This is the clearest current answer to the "paydirt" question: the harness can iterate statefully, but the evidence has not yet shown stacked accepted edits under the control gate.

The broader multi-family sweep tests whether that failure is an artifact of zooming too closely into one residual path:

```text
run: D:\Research_Engine\runs\trm_multi_family_bootstrap_sweep_arc_20260604
samples: 62
families scored: 15
first-edit families: 3
stacked-bootstrap families: 0
control-blocked families: 3
best family: D->A:A:medium_1_0
best cumulative delta: +0.076923
prompt packet estimate: 123 tokens
```

This widened view is useful. It shows the gain region is not arbitrary: the successful first-edit families are all `D->A` with runner-up `A` and close or medium margins. The blocked regions are mostly `B`-top families, where local conditional edits either tie or lose to broad fixed-label controls. The resulting `gain_policy_training_rows.jsonl` file gives the next controller a small supervised/RL-style training table over family features:

```text
label: first_edit_gain | D->A:A:medium_1_0 | accepted_edit_count 1 | delta +0.076923
label: first_edit_gain | D->A:A:close_0_25 | accepted_edit_count 1 | delta +0.04
label: first_edit_gain | D->A:A:close_0_5 | accepted_edit_count 1 | delta +0.04
label: blocked | B->D:D:close_0_25 | fixed_control_not_beaten
label: blocked | B->C:C:close_0_25 | fixed_control_not_beaten
```

This answers the zoom question: the single-family loop was too narrow to learn a general edit policy, but the wider sweep still does not produce stacked ability bootstrap. Its value is that it identifies which family features are worth sending into activation-local VPD mapping and which should be treated as broad-prior artifacts.

The conservative paper track now has deterministic roundout artifacts:

```text
run: D:\Research_Engine\runs\vpd_trm_paper_roundout_20260604
manifest rows: 10
metric rows: 37
policy rows: 5
claim rows: 6
included main-claim runs: 3
excluded runs: 1
```

This manifest separates `claim_support`, `boundary_result`, `open_positive_track`, and `excluded` artifacts so the paper figures can freeze around supported claims while the activation-local runtime edit work remains visibly open.

The roundout run now emits the paper-facing assets:

```text
paper_experiment_manifest.csv
paper_metric_table.csv
paper_policy_comparison.csv
paper_claim_ledger.csv
figures/roundout_summary.svg
figures/filtered_feedback_policy_gain.svg
figures/eval_alignment_collapse.svg
```

Use these as the current figure/table source of truth. The claim ledger is especially important because it explicitly marks the activation-local ARC runtime edit as `not_supported_open_track` rather than allowing the positive trial delta to be over-read.

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
| Router component ablation bars | keygate scores | Show first causal component test. | supplemental, not in frozen roundout |
| Repair-preservation tradeoff | Amalgam scores | Show why accuracy is insufficient. | supplemental, not in frozen roundout |
| Accepted VPD vs random deltas | arena `graft_sweeps.jsonl` | Main control comparison. | covered by six-seed control histogram |
| Transfer heatmap | arena `transfer_matrix.csv` / SVG | Show source-target structure. | available in six-seed figures |
| Accepted recurrence by seed | arena summary + sweeps | Show robustness. | summarized by six-seed arena artifact |

Six-seed run figures now available:

```text
D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\figures\control_histogram.svg
D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\figures\decision_resonance.svg
D:\Research_Engine\runs\metta_organelle_arena_paper_20260601T024147Z\figures\transfer_heatmap.svg
```

### Negative Results

The paper explicitly includes:

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
- The strict six-round eval hill climb found 24 near misses and zero non-targeted accepted eval edits.
- Targeted format-commit retargeting produced 36 accepted probes, but all were targeted lineage and none had matched random controls.
- After adding 976 targeted random controls, the targeted format-commit accepts collapsed from 36 to 0 under the strict matched-random gate.
- Activation-gated runtime proxy scoring produced a positive trial delta, but it lost to broad fixed-label control and failed promotion.
- The stateful edit-bootstrap microcycle accepted one cached route-rule edit, then stopped when the second residual edit lost to fixed-label control.
- The multi-family sweep found three first-edit `D->A` gain families, but zero stacked-bootstrap families.

Negative-result handling:

Do not bury these as caveats. They are part of the methodological contribution: the harness became useful only after neutral stable grafts, random-derived candidates, broad-prior artifacts, and loose acceptance were separated from true accepted VPD grafts.

## Paper Structure

1. Introduction: TRMs as supervised specialists and the need for goal-directed specialization.
2. System framing: outer controller, feedback loop, VPD edits, and TRM organelles.
3. Background: VPD, TRMs, typed MeTTa workflows, feature steering.
4. Phase 1 Router: post-repair keygate and false-commit boundary.
5. Phase 2 Repair Optimization: task graph organelles and repair preservation.
6. Phase 3 Organelle Transfer: runtime grafts and portability tests.
7. Phase 4 Eval Alignment: downstream Intellect-3 gate, strict cliff result, and targeted rappel.
8. Results: accepted grafts, controls, hard-slice behavior, filtered feedback, ARC gain-region mapping, and eval-aligned rejection.
9. Limitations: runtime-only, small models, deterministic eval estimates, synthetic/typed workflow rows, no broad RL/editing breakthrough.
10. Discussion: why VPD/TRM feedback loops are a promising research field despite the failed stacked-bootstrap and eval-transfer gates.

## Post-Freeze Direction: TinyLoRA Swarms

The next research direction is to replace ad hoc edit units with a formal tinyLoRA organism:

```text
tinyLoRA organism =
  target module
  low-rank rank/alpha/scale
  task-family trigger
  adapter seed
  mutation lineage
  scorer fitness
  guardrail and control status
```

The first cached scaffold exists as a proxy-only run:

```text
run: D:\Research_Engine\runs\trm_tinylora_swarm_arc_20260606
generations: 4
organisms scored: 96
accepted proxy organisms: 46
best organism: tiny_lora:g1:0001
best module: base_model.model.model.language_model.layers.19.self_attn.o_proj
rank/scale: 1 / 1.0
delta/control margin/fitness: +0.064517 / +0.052259 / 0.132614
```

This does not change the frozen evidence claim because no adapter weights were trained or merged. Its value is structural: it turns the next phase into a memetic search over reversible tinyLoRA organisms rather than a sequence of one-off route rules, hooks, and replay slices. Any real tinyLoRA run must use hard resource caps, checkpointing, chunked scoring, and PID-owned cleanup before it can become paper evidence.

The real-training handoff is also staged, still without executing training:

```text
run: D:\Research_Engine\runs\trm_tinylora_training_handoff_arc_20260606
candidates: 8
source swarm: D:\Research_Engine\runs\trm_tinylora_swarm_arc_20260606
caps: 2048 MB RAM, 50% CPU, 50 MB/s IO
checkpoint cadence: generation_or_120s
wrapper: run_tinylora_jobobject.ps1
```

The top candidate is a rank-1 adapter organism targeting:

```text
base_model.model.model.language_model.layers.19.self_attn.o_proj
trigger: D over A, max margin 1.0
proxy delta/control margin/fitness: +0.064517 / +0.052259 / 0.132614
```

This handoff makes the next phase decision-complete: train one candidate at a time inside the generated Windows Job Object wrapper, checkpoint every generation or 120 seconds, log aborts as valid outcomes, and accept no adapter unless it beats fixed-label and random tinyLoRA controls under live scoring.

## Open Questions

- How much recurrence should be required beyond the current six-seed accepted-graft signal?
- Are repair-step gains a real transferable motif or partly a consequence of the hard-slice construction?
- Can a graft selected in the arena improve a downstream MCP retrieval or context-curation harness?
- Can the same component family be promoted from runtime graft to checkpoint edit without losing guardrails?
- Can targeted format-commit and candidate-verify probes beat matched random controls under a live scorer or richer candidate generator?
- Can non-targeted candidate generation produce an eval-aligned Intellect-3-Logic accepted edit, or is the router cliff a real transfer boundary?
- Since targeted accepts collapsed under matched random controls, is that a limitation of VPD, the organelle proxy task, or the deterministic alignment estimate?
- What minimum task diversity is needed before the result stops being an artifact of shared labels?
- What is the right outer-loop policy for choosing between runtime grafting, checkpoint editing, and supervised retraining?
- Can an LLM-managed controller reliably turn repeated failures into better TRM training slices without drifting away from guardrails?
- Can the `D->A` ARC gain region be promoted from cached route rule to activation-local VPD edit without losing to broad fixed-label controls?
- Can a hard-capped tinyLoRA swarm turn the `D->A` gain region into a real adapter-delta population that beats route-rule and random tinyLoRA controls?

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

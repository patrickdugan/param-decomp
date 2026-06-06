# TinyLoRA Swarm Formalization

Current status: the tinyLoRA swarm has not achieved a real model-edit gain yet.

The current scaffold achieved a structural result:

- It converted edit search into formal organisms with target module, rank, alpha, scale, trigger family, lineage, mutation operator, proxy fitness, and control gate.
- It re-found the known `D->A` gain region in cached ARC score space.
- It produced a ranked handoff for real training under hard caps.

No LoRA adapter weights have been trained, merged, or live-scored. The correct claim is:

```text
tinyLoRA swarm scaffold found promising proxy organisms;
it has not yet demonstrated a real tinyLoRA edit.
```

## Organism Definition

Let a tinyLoRA organism be:

```text
o = (m, r, alpha, s, tau, theta, a)
```

where:

- `m` is the target module.
- `r` is the LoRA rank.
- `alpha` is the LoRA alpha.
- `s` is the runtime scale.
- `tau` is a trigger predicate over task state.
- `theta` is the adapter parameter pair `(A, B)`.
- `a` is ancestry and mutation metadata.

A LoRA edit modifies a module weight:

```text
W'_m = W_m + s * (alpha / r) * B A
```

with:

```text
A in R^{r x d_in}
B in R^{d_out x r}
```

The trigger-gated version is:

```text
h' = W_m h + 1[tau(x)=1] * s * (alpha / r) * B A h
```

## Search Space

The tinyLoRA search space is:

```text
Omega = M x R x A x S x T x Theta
```

where:

- `M` is the set of target modules.
- `R` is the allowed rank set.
- `A` is the alpha set.
- `S` is the scale set.
- `T` is the trigger-predicate family.
- `Theta` is the low-rank adapter parameter space.

## Fitness

A general fitness objective is:

```text
F(o) =
  Delta_task(o)
  - lambda_g * max(0, -Delta_guardrail(o))
  - lambda_c * max(0, Delta_control(o) - Delta_task(o))
  - lambda_t * touch_rate(o)
  - lambda_r * rank(o)
  - lambda_d * damage(o)
```

The cached scaffold currently uses:

```text
F(o) =
  reward(o) - reward(best_control)
  + beta * rescues(o)
  - gamma * damages(o)
  - lambda * touch_rate(o)
  + eta / rank(o)
```

## Memetic Loop

The population loop is:

```text
P_0 ~ seed(VPD features, failure families)

for generation g:
  score all o in P_g
  E_g = select_elites(P_g, F)
  C_g = mutate(E_g) union crossover(E_g)
  P_{g+1} = E_g union C_g
```

Mutation operators:

```text
module_shift:      m -> m'
rank_shift:        r -> r'
scale_shift:       s -> s'
trigger_shift:     tau -> tau'
adapter_mutation:  theta -> theta + epsilon
crossover:         (m_i, tau_i, theta_i) + (m_j, tau_j, theta_j)
```

## Research Question

The theoretical move is to stop treating edits as isolated points and instead treat them as a population over a structured manifold:

```text
editable manifold ~= low-rank adapter deltas conditioned on task-family triggers
```

The clean experiment is:

```text
Does VPD-guided initialization produce better search trajectories through Omega
than random tinyLoRA initialization or broad route-rule controls?
```

Or, operationally:

```text
Does the swarm find a Pareto-improving frontier of tiny, reversible,
triggered edits under strict controls?
```

## Current Artifact

Cached proxy run:

```text
D:\Research_Engine\runs\trm_tinylora_swarm_arc_20260606
```

Training handoff:

```text
D:\Research_Engine\runs\trm_tinylora_training_handoff_arc_20260606
```

Auto-research manager loop:

```text
D:\Research_Engine\runs\trm_tinylora_auto_research_arc_20260606
cycles: 2
trainer mode: dry_run
best proxy fitness: 0.132614
accepted live edits: 0
next agent packet: next_agent_packet.txt
```

The manager loop is now:

```text
for cycle k:
  P_k = cached_tinyLoRA_swarm(score_cards, VPD_failure_families)
  H_k = hard_cap_training_handoff(top(P_k), random_controls)
  R_k = trainer_bridge(H_k)
  S_{k+1} = update_research_state(S_k, R_k)
  emit_next_agent_packet(S_{k+1})
```

In dry-run mode this proves orchestration, not model improvement. A paper-valid
hill climb requires `accepted_live_edits > 0`, where a live edit is accepted only
after a trained adapter beats fixed-label controls and random tinyLoRA controls
while preserving guardrails.

Top proxy organism:

```text
organism: tiny_lora:g1:0001
target module: base_model.model.model.language_model.layers.19.self_attn.o_proj
trigger: D over A, max margin 1.0
rank / scale: 1 / 1.0
proxy delta / control margin / fitness: +0.064517 / +0.052259 / 0.132614
```

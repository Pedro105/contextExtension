# Decisions

A running log of non-obvious choices, the alternatives considered, and why.
Written as decisions are made, not reconstructed afterwards. Feeds the README's
rationale section and the report's method section.

---

## D1 — Research question framing

**Decision.** The question is _"does measured serving cost follow the analytical
prediction, and does capability gained justify cost paid?"_ — not _"how much does
concurrency drop when context is extended?"_

**Alternative rejected.** The second framing. It was the original one.

**Why.** KV cache is linear in context length. If memory binds at both ends of a
comparison, the concurrency ratio _is_ the context ratio — exactly 8× for a 4k→32k
extension, on every model and every GPU. That is arithmetic, not a finding. The
project would have spent four weeks measuring one line of algebra.

Reframing makes the arithmetic a _prediction_ rather than a result. The finding
lives in the deviations: scheduler admission caps, prefill contention, and
effective-versus-claimed context.

---

## D2 — SmolLM2-1.7B as the training subject

**Decision.** SmolLM2-1.7B is the model that gets extended.

**Alternatives rejected.** Qwen2.5-1.5B (`max_position_embeddings` 32768 — already
long, nothing to extend). Llama-3.2-1B/3B (131072 — same problem, worse).

**Why.**

- Native context of 8192 (pretrained at 2048, extended to 8192). Short enough that
  extending it is a genuine intervention rather than a no-op.
- Full MHA — 32 KV heads, 192 KiB/token. Seven times Qwen's cache per token. The
  memory effect binds early and cleanly instead of hiding behind the scheduler cap
  at both ends of the sweep.
- 1.7B parameters: full fine-tuning is feasible on 4× A100 40G.

The large KV cache initially looked like a liability. It is the opposite — it is
what makes the effect under study measurable on affordable hardware.

---

## D3 — Full fine-tuning rather than LoRA

**Decision.** All weights update. No parameter-efficient methods.

**Why.** The distributed half of this project measures gradient all-reduce cost
against compute. LoRA reduces the all-reduce volume to near-zero, which would make
the scaling efficiency measurement trivially good and scientifically empty. The
communication volume _is_ the thing being measured.

Cost: higher memory (~27 GB of weights, gradients, and optimizer states before a
single activation), which rules out 24 GB cards for the real runs.

---

## D4 — Model roster and the role of each

**Decision.** Four models, two experiments, strictly separated.

| Model          | Experiment | Role                                                   |
| -------------- | ---------- | ------------------------------------------------------ |
| SmolLM2-1.7B   | 1 and 2    | The subject. Control (base @ 8k) and all treatments    |
| Qwen2.5-1.5B   | 2 only     | Aggressive GQA (2 KV heads). Tests the head-count term |
| Ministral 3 3B | 2 only     | Professionally YaRN-extended (16k → 256k, factor 16)   |
| Gemma 3        | 2 only     | Interleaved SWA. Tests the sublinear branch            |

**Alternative rejected.** Comparing extended SmolLM2 against Ministral as
"post-hoc-extended vs natively-long-pretrained."

**Why rejected.** Confounded beyond repair. The two models differ in parameter
count, pretraining corpus, tokenizer, and architecture. Any observed difference is
uninterpretable. Experiment 1 is strictly within-model.

---

## D5 — Separating the cost axis from the capability axis

**Decision.** Cost is measured on all models at all context lengths. Capability is
measured only within a model's trained range, and compared only within the SmolLM2
arms.

**Why.** KV cache is allocated whether or not the output is coherent. Serving
SmolLM2-base forced to 32k yields valid cost numbers alongside garbage generations.
That is not a defect — it isolates _the cost of context_ from _the capability of
context_, which is exactly the pairing under study.

---

## D6 — Implementing PI, NTK-aware, and YaRN by hand

**Decision.** Write the RoPE scaling methods directly rather than using
`transformers`' built-in `rope_scaling` config option.

**Why.** Roughly 60 lines for the per-dimension ramp and attention temperature
term. Doing it by hand is the difference between citing the method and
understanding it, and the positional mechanics are the deep learning content of
this project. Also required in order to ablate the temperature term separately,
which a config flag does not expose.

---

## D7 — Hand-written `torch.distributed` loop

**Decision.** Implement data-parallel training directly on `torch.distributed`
collectives rather than using `DistributedDataParallel` or HF `Trainer`.

**Why.** At 32k sequence length, micro-batch size is forced to 1, which makes
gradient accumulation mandatory. A naive implementation all-reduces after every
backward pass, paying N× the communication for no benefit across N accumulation
steps. The correct implementation suppresses the all-reduce on intermediate steps.
PyTorch exposes this as `no_sync()`; implementing the mechanism directly is how
the cost of getting it wrong becomes concrete.

`DistributedDataParallel` and FSDP2 are retained as comparison points in the
scaling benchmark, not as the implementation.

---

## D8 — Including Gemma 3 to exercise the SWA branch

**Decision.** Add Gemma 3 as a fourth cost-model test case.

**Alternative considered.** Delete the sliding-window branch of the cost model and
scope it explicitly to full attention.

**Why.** Gemma 3 interleaves 5 local attention layers (1024-token window) per 1
global layer, giving roughly a 5× KV cache reduction at 32k. Without it, the SWA
branch is untested code carrying an unsupported claim.

Including it also upgrades the report's central claim from _"long context is
expensive"_ to _"long context is expensive **under full attention**, and here is
the architecture that avoids most of it."_ That is an architectural argument rather
than a cost observation.

**Note for the cost model.** Interleaved SWA does not make KV growth constant. Only
uniform SWA does. With 5:1 interleaving the growth stays linear with roughly
one-sixth the slope, because global layers keep scaling. Getting this wrong is an
easy silent error.

---

## D9 — Build order: measurement apparatus before training

**Decision.** Serving benchmark harness and evaluation harness are built and
validated before any training happens.

**Why.**

- Training carries unbounded variance (a method that does not converge, a data bug
  found late); infrastructure carries bounded variance. Scheduling unbounded
  variance upstream of bounded variance guarantees the overrun lands on the
  deliverable that matters most.
- The base model at native context is a required arm of the final results and needs
  no training, so the infrastructure phase produces real results rather than
  scaffolding.
- Without a validated evaluation harness, a failed training run is
  indistinguishable from a broken evaluation.

---

## D10 — Predictions committed in advance

**Decision.** Predictions are written into the report's introduction before
measurement begins.

1. KV-limited concurrency scales exactly inversely with context length (arithmetic,
   the baseline).
2. Measured throughput ratios come in _below_ the arithmetic, because at short
   context the system is scheduler-bound (`--max-num-seqs`, default 256) rather
   than memory-bound. The naive model therefore overstates the cost of extension.
3. Time-to-first-token degrades worse than linearly — prefill attention is O(n²).
   The cost of long context appears primarily in latency, not throughput.
4. Effective context lags claimed context substantially.
5. Extension degrades short-context performance; mitigated by mixing sequence
   lengths in training data.

**Why.** "We predicted X, observed Y, and here is why they differ" is categorically
more credible than presenting observations as though they were expected all along.
Any of these being wrong is more interesting than all of them being right.

---

## D11 — Scope: measurement study, not method contribution

**Decision.** State plainly in the introduction that this implements published
methods (PI, NTK-aware, YaRN) and measures their consequences. The contribution is
the joint cost–capability measurement and the validated cost model.

**Why.** Calibration reads better than overclaiming, and the claim survives
scrutiny. A reviewer who finds an overclaim stops trusting everything else.

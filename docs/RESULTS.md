# Full Results

Complete tables for S2R-Merge. See the [README](../README.md) for the method description
and headline numbers.

All experiments share the same initial agent pool (15 role-based agents) and role
descriptions. Multi-agent baselines use identical maximum communication rounds and
communication order: $T=1$ for MMLU, $T=2$ elsewhere. Sampling settings and inference
parameters are identical across methods. Reported values average at least three
independent runs.

---

## 1. Main results

| Method | Edge DR. | Node DR. | Node ME. | MMLU | GSM8K | AQuA | MultiArith | SVAMP | HumanEval | **Avg.** |
|---|:-:|:-:|:-:|---|---|---|---|---|---|---|
| **Llama-3-8B-Instruct** ||||||||||||
| Vanilla | ✗ | ✗ | ✗ | 58.8 | 63.2 | 43.7 | 88.3 | 77.3 | 85.7 | 69.52 |
| CoT | ✗ | ✗ | ✗ | 57.3 | 65.3 | 43.2 | 90.0 | 79.7 | 86.8 | 70.38 |
| MAS (round=1) | ✗ | ✗ | ✗ | 61.5 | 67.2 | 51.7 | 90.4 | 80.6 | 85.2 | 72.77 |
| MAS (round=T) | ✗ | ✗ | ✗ | 62.6 | 66.8 | 52.1 | 91.1 | 80.2 | 85.1 | 72.98 |
| AgentPrune | ✓ | ✗ | ✗ | 59.3 | 64.5 | 50.7 | 93.8 | 81.4 | 83.4 | 72.18 |
| AgentDropout | ✓ | ✓ | ✗ | 62.0 | 65.3 | 55.9 | 93.4 | 83.2 | 90.5 | 73.38 |
| **S2R-Merge (ours)** | ✓ | ✗ | ✓ | **63.4** | **69.9** | **56.2** | **94.4** | **83.7** | **95.7** | **77.22** |
| **GPT-4** ||||||||||||
| Vanilla | ✗ | ✗ | ✗ | 71.2 | 92.4 | 79.2 | 98.8 | 92.0 | 86.5 | 86.68 |
| CoT | ✗ | ✗ | ✗ | 77.3 | 90.8 | 81.8 | 97.8 | 91.8 | 86.3 | 87.65 |
| MAS (round=1) | ✗ | ✗ | ✗ | 78.2 | 93.1 | 80.7 | 97.2 | 92.3 | 86.9 | 88.07 |
| MAS (round=T) | ✗ | ✗ | ✗ | 78.1 | 94.0 | 82.0 | 98.4 | 92.3 | 86.7 | 88.58 |
| AgentPrune | ✓ | ✗ | ✗ | 78.0 | 94.5 | 82.6 | 98.2 | 92.2 | 87.5 | 88.83 |
| AgentDropout | ✓ | ✓ | ✗ | 78.9 | 95.2 | 83.3 | 98.3 | 91.4 | 99.2 | 91.05 |
| **S2R-Merge (ours)** | ✓ | ✗ | ✓ | **80.3** | **95.9** | **85.4** | **100.0** | **92.6** | **100.0** | **92.37** |

Baselines: Vanilla and CoT are single-agent; MAS (round=1 / round=T) are multi-agent
without structural optimization; AgentPrune and AgentDropout reduce communication cost via
edge / node pruning.

---

## 2. Token usage

| Method | MMLU | GSM8K | AQuA | MultiArith | SVAMP | HumanEval | **Avg.** |
|---|---|---|---|---|---|---|---|
| | P / C | P / C | P / C | P / C | P / C | P / C | P / C |
| Vanilla | 206K / 207K | 110K / 408K | 25K / 117K | 11K / 31K | 19K / 54K | 25K / 15K | 66K / 139K |
| CoT | 605K / 195K | 740K / 553K | 248K / 117K | 243K / 68K | 180K / 54K | 141K / 99K | 359K / 181K |
| MAS (round=1) | 18.4M / 1.5M | 70.1M / 5.6M | 12.7M / 1.5M | 13M / 1.5M | 9.1M / 1M | 1.1M / 123K | 20.7M / 1.9M |
| MAS (round=T) | 63.3M / 7.6M | 131M / 11.9M | 29.6M / 3.1M | 13.6M / 1.2M | 21.8M / 1.8M | 7.4M / 627K | 44.4M / 4.3M |
| AgentPrune | 18.4M / 1.5M | 174M / 10.8M | 34.4M / 3M | 34.7M / 2.8M | 24.2M / 2M | 3.3M / 382K | 48.2M / 3.4M |
| AgentDropout | 8.5M / 1.1M | 80.4M / 7.9M | 15.7M / 2.5M | 15.5M / 2.4M | 10.1M / 1.4M | 1.8M / 205K | 22M / 2.6M |
| **S2R-Merge (ours)** | **3.1M / 649K** | **11.2M / 2.8M** | **3.9M / 1.4M** | **1.1M / 265K** | **1.8M / 370K** | **460K / 209K** | **3.6M / 938K** |

P = prompt tokens, C = completion tokens.

---

## 3. Router ablation

Performance and token usage with and without the router, pool fixed at $n=15$ (GPT-4).

| | Method | MMLU | GSM8K | AQuA | MultiArith | SVAMP | HumanEval | **Avg.** | **Avg. Ptok.** | **Avg. Ctok.** |
|---|---|---|---|---|---|---|---|---|---|---|
| **w/ Router** | AgentPrune | 78.8 | 93.0 | 82.6 | 91.6 | 89.6 | 85.7 | 86.88 | 34.5M | 6.7M |
| | AgentDropout | 79.0 | 92.0 | 84.6 | 100.0 | 91.2 | 99.0 | 90.97 | 15.9M | 4.9M |
| | **Ours** | **80.3** | **95.9** | **85.4** | **100.0** | **92.6** | **100.0** | **92.37** | **3.6M** | **938K** |
| **w/o Router** | AgentPrune | 77.8 | 94.5 | 82.6 | 97.6 | 92.2 | 87.5 | 88.79 | 48.2M | 3.4M |
| | AgentDropout | 78.9 | 95.2 | 83.3 | 98.3 | 91.4 | 99.2 | 91.05 | **22M** | **2.5M** |
| | **Ours** | **80.5** | **96.1** | **85.2** | **98.9** | **93.0** | **100.0** | **92.28** | 43.3M | 3.4M |

The router and merge-based optimization play complementary roles. The router narrows the
initial collaboration scope from query semantics; merging compresses the selected structure
while preserving task-relevant agent information.

---

## 4. Merge target selection criterion

Three selection strategies under identical conditions — same absorption-based merge
procedure, differing only in how the target is chosen. Run without the router on a fixed
pool of 5 agents to isolate the criterion.

| Method | MMLU | GSM8K | HumanEval | **Avg.** |
|---|---|---|---|---|
| **$\Delta W(c)$ (ours)** | **80.3** | **93.6** | **100.0** | **91.3** |
| Degree-based | 79.8 | 93.1 | 99.3 | 90.7 |
| Random | 80.1 | 93.2 | 99.3 | 90.9 |

Degree-based and random selection perform comparably to each other, suggesting simple
connectivity-based or unstructured rules do not reliably identify merge targets that
preserve task performance.

---

## 5. Prompt-level fusion strategy

Against a Concatenation baseline that appends the absorbed node's prompt without
summarization.

| Method | MMLU | GSM8K | HumanEval | **Avg.** | Fusion Ptok. | Fusion Ctok. |
|---|---|---|---|---|---|---|
| Concatenation | 80.0 | 91.4 | 99.3 | 90.2 | 0 | 0 |
| **Prompt-level Fusion (ours)** | **80.6** | **91.9** | **100.0** | **90.8** | 331 | 91 |

Fusion token counts are per merge operation, averaged. The overhead is marginal relative
to the savings from merge-based structural optimization.

---

## 6. Robustness across initial communication topologies

Without the router, full 15-agent pool.

| Graph | Method | MMLU | GSM8K | AQuA | MultiArith | SVAMP | HumanEval | **Avg.** |
|---|---|---|---|---|---|---|---|---|
| **FullConnected** | AgentPrune | 78.0 | 93.7 | 82.6 | 97.6 | 92.2 | 87.5 | 88.60 |
| | AgentDropout | 78.9 | 95.2 | 83.3 | 98.3 | 91.4 | 99.2 | 91.05 |
| | **Ours** | **80.5** | **96.1** | **85.2** | **98.9** | **93.0** | **100.0** | **92.28** |
| **Layered** | AgentPrune | 77.9 | 91.9 | 85.0 | 97.6 | 86.7 | 83.8 | 87.15 |
| | AgentDropout | 79.6 | 92.0 | 85.3 | **100.0** | 92.5 | **100.0** | 91.56 |
| | **Ours** | **80.8** | **94.2** | **85.6** | **100.0** | **92.6** | **100.0** | **92.20** |
| **Random** | AgentPrune | 79.2 | 90.7 | 84.6 | 97.8 | 88.3 | 90.6 | 88.53 |
| | AgentDropout | 80.0 | 91.0 | 84.7 | **100.0** | **96.4** | 96.3 | 91.40 |
| | **Ours** | **80.2** | **94.9** | **86.4** | **100.0** | 94.3 | 99.3 | **92.52** |

Maximum variation across the three topologies is 0.32%p for ours, against a 1.45%p swing
for AgentPrune (88.60 → 87.15). Where the optimal communication topology is unknown a
priori, absorption-based merging preserves reasoning performance without topology-specific
tuning.

---

## 7. Merge iteration selection

GPT-4. $n$ = agents remaining after routing, i.e. the initial state of merging. Ptok. and
Ctok. are totals under each merge-iteration setting.

| Merge Iter. | | MMLU (n=4) | GSM8K (n=4) | MultiArith (n=4) | SVAMP (n=4) | AQuA (n=6) | HumanEval (n=4) |
|---|---|---|---|---|---|---|---|
| **1** | Accuracy | 77.3 | 93.9 | 100 | 92.3 | 85.4 | 100 |
| | Ptok. | 4.6M | 11.2M | 1.7M | 2.8M | 3.9M | 644.4K |
| | Ctok. | 825.4K | 2.8M | 343.2K | 518.9K | 1.4M | 220.9K |
| **2** | Accuracy | 76.5 | 91.2 | 100 | 92.0 | 84.3 | 100 |
| | Ptok. | 3.1M | 7.9M | 1.1M | 1.8M | 3.5M | 460.6K |
| | Ctok. | 649.3K | 2.2M | 265.2K | 370.3K | 1.2M | 209.2K |
| **3** | Accuracy | — | — | — | — | 83.5 | — |
| | Ptok. | — | — | — | — | 2.2M | — |
| | Ctok. | — | — | — | — | 1.1M | — |
| **4** | Accuracy | — | — | — | — | 83.1 | — |
| | Ptok. | — | — | — | — | 1.5M | — |
| | Ctok. | — | — | — | — | 846.1K | — |
| **Final Merge Iter.** | | **2** | **1** | **2** | **2** | **1** | **1** |

Additional merge iterations reduce token consumption by shrinking the communication graph,
but excessive merging degrades task performance once role-specific information is
over-compressed. The final depth per benchmark is the smallest that yields substantial
token reduction without a large accuracy drop — different tasks require different levels of
collaboration and structural compression.

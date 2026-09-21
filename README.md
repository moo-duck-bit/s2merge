# S2R-Merge

<img width="1406" height="529" alt="figure1_3" src="https://github.com/user-attachments/assets/6eb4f78f-cfaf-44a7-80ea-b30cc42bc8be" />

Route, then merge — an efficiency framework for LLM multi-agent collaboration.

A two-stage router picks a query-relevant agent subset instead of running the full pool,
then structural optimization **merges** each redundant agent into its strongest neighbour
rather than deleting it, so the communication graph shrinks without losing the role.

**Best average accuracy on both backbones, at a fraction of the tokens.**

| | Avg. accuracy | Avg. prompt tokens |
|---|---|---|
| MAS (no optimization) | 88.58 | 44.4M |
| AgentPrune | 88.83 | 48.2M |
| AgentDropout | 91.05 | 22M |
| **S2R-Merge** | **92.37** | **3.6M** |

*GPT-4, averaged over six benchmarks. 12.3× fewer prompt tokens than unoptimized MAS.*

---

## How it works

**Router** — selects agents from query semantics, not a fixed Top-K.
[`s2rmerge/router/`](s2rmerge/router)

1. **Stage-0** · An LLM writes a structured summary $t$ of the query (never an answer). One
   mixed embedding $\mathbf{v} = \alpha \cdot \mathrm{Embed}(q) + (1-\alpha) \cdot \mathrm{Embed}(t)$
   is built once and shared by every later stage.
2. **Stage-1** · Score domain *blocks* against $\mathbf{v}$, softmax, then measure normalized
   entropy. Higher uncertainty raises the coverage threshold, so ambiguous queries recruit
   wider scopes. Blocks are taken by cumulative probability mass.
3. **Stage-2** · Same mechanism over the *roles* inside the selected blocks only —
   restricting candidates keeps role-level entropy from picking up coarse domain ambiguity.

**Node merge** — the selected agents form a communication graph whose edge weights are
learned by policy gradient (LLM outputs are non-differentiable).
[`s2rmerge/merge/`](s2rmerge/merge), [`s2rmerge/mas/graph.py`](s2rmerge/mas/graph.py)

4. **Target** · A transfer-efficiency score

   $$\Delta W(c) = \frac{\sum w_{in}}{\mathrm{Var}(w_{in}) + \epsilon} - \frac{\sum w_{out}}{\mathrm{Var}(w_{out}) + \epsilon}$$

   contrasts each node's stable incoming signal against its stable outgoing one. A node
   whose incoming signal dominates absorbs information without passing it on — a structural
   bottleneck. Exactly one node is merged per step; no ratio or threshold to tune.
5. **Partner** · The connected node with the strongest learned interaction,
   $S(c^{*},j) = \max(\tilde{A}_{j,c^{*}}, \tilde{A}_{c^{*},j})$.
6. **Absorb** · An LLM folds the target's role into the partner's prompt (~331 prompt / ~91
   completion tokens per merge); the target and all its edges are removed.
7. **Prune** · Merging invalidates the learned weights, so the graph is re-optimized and the
   lowest-weight edges are dropped.

Confining all of this to the router-selected subgraph cuts learnable edge parameters from
$O(N^2)$ over the full pool to $O(k^2)$, $k \ll N$.

## Results

Six benchmarks, 15-agent initial pool, averaged over ≥3 runs. Full tables and ablations in
**[docs/RESULTS.md](docs/RESULTS.md)**.

| Method | Edge DR. | Node DR. | Node ME. | MMLU | GSM8K | AQuA | MultiArith | SVAMP | HumanEval | **Avg.** |
|---|:-:|:-:|:-:|---|---|---|---|---|---|---|
| **Llama-3-8B-Instruct** ||||||||||||
| Vanilla | ✗ | ✗ | ✗ | 58.8 | 63.2 | 43.7 | 88.3 | 77.3 | 85.7 | 69.52 |
| CoT | ✗ | ✗ | ✗ | 57.3 | 65.3 | 43.2 | 90.0 | 79.7 | 86.8 | 70.38 |
| MAS (round=T) | ✗ | ✗ | ✗ | 62.6 | 66.8 | 52.1 | 91.1 | 80.2 | 85.1 | 72.98 |
| AgentPrune | ✓ | ✗ | ✗ | 59.3 | 64.5 | 50.7 | 93.8 | 81.4 | 83.4 | 72.18 |
| AgentDropout | ✓ | ✓ | ✗ | 62.0 | 65.3 | 55.9 | 93.4 | 83.2 | 90.5 | 73.38 |
| **S2R-Merge** | ✓ | ✗ | ✓ | **63.4** | **69.9** | **56.2** | **94.4** | **83.7** | **95.7** | **77.22** |
| **GPT-4** ||||||||||||
| Vanilla | ✗ | ✗ | ✗ | 71.2 | 92.4 | 79.2 | 98.8 | 92.0 | 86.5 | 86.68 |
| CoT | ✗ | ✗ | ✗ | 77.3 | 90.8 | 81.8 | 97.8 | 91.8 | 86.3 | 87.65 |
| MAS (round=T) | ✗ | ✗ | ✗ | 78.1 | 94.0 | 82.0 | 98.4 | 92.3 | 86.7 | 88.58 |
| AgentPrune | ✓ | ✗ | ✗ | 78.0 | 94.5 | 82.6 | 98.2 | 92.2 | 87.5 | 88.83 |
| AgentDropout | ✓ | ✓ | ✗ | 78.9 | 95.2 | 83.3 | 98.3 | 91.4 | 99.2 | 91.05 |
| **S2R-Merge** | ✓ | ✗ | ✓ | **80.3** | **95.9** | **85.4** | **100.0** | **92.6** | **100.0** | **92.37** |

*DR. = Dropout, ME. = Merge.*

Two effects separate cleanly. **Merge drives accuracy**: with no router and the full
15-agent pool it still scores 92.28 vs 91.05 (AgentDropout). **The router drives
efficiency**: adding it moves accuracy 92.28 → 92.37 while prompt tokens fall 43.3M → 3.6M.

## Quickstart

```bash
pip install -r requirements.txt
cp template.env .env                    # set LLAMA_MODEL_ID / API keys
python scripts/download_datasets.py     # all six benchmarks into data/
```

Llama runs against a local vLLM server; `LLAMA_MODEL_ID` must match the `--model` string
vLLM was launched with.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 6789 --max-model-len 8192
```

```bash
python -m s2rmerge gsm8k --llm Meta-Llama-3.1-8B-Instruct
```

Each benchmark carries its reported communication rounds and merge depth, so the command
above reproduces the main-table setting with no extra flags. One entry point covers every
ablation in the paper:

| | |
|---|---|
| `python -m s2rmerge gsm8k --no-router` | w/o Router ablation (RESULTS §3) |
| `python -m s2rmerge gsm8k --no-fusion` | concatenation instead of fusion (§5) |
| `python -m s2rmerge gsm8k --no-router --topology Layered` | topology robustness (§6) |
| `python -m s2rmerge aqua --merge-iterations 3` | override the merge depth (§7) |

Each run writes `results/<benchmark>_<timestamp>/` with `report.json` — the score, the
merges performed, and prompt/completion tokens split by pipeline stage — plus the full
agent transcript. `python -m s2rmerge --help` lists the rest.

## Layout

```
s2rmerge/
  cli.py            one entry point for every benchmark and ablation
  pipeline.py       route -> optimize -> merge -> prune -> evaluate
  accounting.py     per-stage token and cost ledger
  answers.py        parsing final answers out of free-form responses
  topology.py       initial communication topologies
  router/           stage0 (representation), stage1 (blocks), stage2 (roles)
  merge/            criterion (deltaW, partner), fusion (prompt-level), operator
  mas/              communication graph, agents, prompts, LLM backends
  benchmarks/       the six datasets, their agent setup and scoring
configs/            per-benchmark router config: alpha, lambda, temperatures,
                    coverage bounds, block and role definitions
prototypes/         cached block/role prototype embeddings (generated on first run)
tests/              merge criterion, routing rules, accounting, answer parsing
```

Roles carry the same names in `configs/*.yaml` and in the matching prompt set, and the
pipeline checks that correspondence at startup.

## Implementation notes

- **Routing is per query.** Queries that land on the same agent set share one graph,
  which is what makes policy-gradient optimization over a fixed node set well defined.
  In practice a benchmark yields a small number of distinct sets; `--max-agent-groups`
  bounds it if a run produces more than you want to pay for.
- **Agent pools.** The math benchmarks and MMLU route over 15 role-based agents;
  HumanEval routes over 13. Each pool is listed in its prompt set and its config.
- **HumanEval splits.** HumanEval ships a single split, so graph optimization reuses the
  evaluation set, as in the AgentDropout code this builds on. The run logs a note when it
  does.
- **Token accounting.** Every LLM call is attributed to the stage that made it — router,
  optimization, fusion or inference — through a context variable, so the split survives the
  concurrent fan-out of a graph run and needs no manual bookkeeping at call sites.
- **Merge iterations.** `--merge-iterations n` performs `n` absorptions, each followed by
  re-optimization of the restructured graph. The per-benchmark defaults are the depths in
  [docs/RESULTS.md](docs/RESULTS.md) §7, declared on each `Benchmark` subclass.

## Acknowledgements

The communication-graph MAS in `s2rmerge/mas/` builds on
[AgentDropout](https://arxiv.org/abs/2503.18891) (ACL 2025) and, before it, AgentPrune.
S2R-Merge replaces node *dropout* with absorption-based node *merge* and prepends the
uncertainty-aware router.

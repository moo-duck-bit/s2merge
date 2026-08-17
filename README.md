# S2R-Merge

**Semantic Routing + Structural Merging for efficient multi-agent collaboration.**

An uncertainty-aware two-stage router selects a query-relevant agent subset, then
merge-based structural optimization consolidates redundant agents *into* their strongest
neighbour instead of deleting them — preserving role information while shrinking the
communication graph.

On six reasoning and code-generation benchmarks, S2R-Merge reaches the best average
accuracy with both Llama-3-8B-Instruct (**77.22%**) and GPT-4 (**92.37%**), while using
**3.6M prompt tokens** on average — a **12.3× reduction** over unoptimized multi-agent
collaboration and **6.1× less** than AgentDropout.

---

## Motivation

LLM multi-agent systems earn their accuracy from many specialised agents talking to each
other, and that is exactly where their token cost comes from. The standard remedy is to
prune: drop the least useful agent node, drop the least useful edge. Pruning is cheap but
lossy — whatever the dropped agent knew leaves the system with it.

S2R-Merge changes two things:

1. **Route before you build.** Rather than treating agent selection as a fixed-cardinality
   decision, the router estimates the *semantic coverage* a query requires and derives the
   collaboration scale from it. Fewer agents enter the graph in the first place, and
   structural optimization is confined to the selected subgraph — reducing learnable edge
   parameters from $O(N^2)$ over the full pool to $O(k^2)$ with $k \ll N$.
2. **Merge instead of drop.** When optimization identifies a structurally harmful agent,
   an LLM folds its role description into its strongest neighbour and its edges are
   rewired. The graph loses a node; the team does not lose the role.

## Method

### Stage-0 — Representation generation

A query's surface form often under-determines the expertise it needs. We prompt an LLM to
produce a *structured summary* $t$ (domain hint, key concepts, required strategy) — never
an answer — and treat it as a weak prior. A single mixed representation is built once:

$$\mathbf{v} = \alpha \cdot \mathrm{Embed}(q) + (1-\alpha) \cdot \mathrm{Embed}(t), \qquad \alpha = 0.5$$

$\mathbf{v}$ is the **only** semantic representation shared across all routing stages;
Stage-1 and Stage-2 introduce no intermediate representations. Every downstream difference
therefore comes from the *selection policy*, not from a change of representation — which
keeps routing interpretable and stable without any training.

### Stage-1 — Adaptive soft block routing

Each block is a high-level domain with a prototype $\mathbf{p}_B$, obtained by embedding a
YAML-defined block description with the same encoder used in Stage-0 (so block semantics
are fixed and reproducible at inference time). Relevance combines continuous similarity
with coarse domain guidance:

$$s_B = \cos(\mathbf{v}, \mathbf{p}_B) + \lambda \cdot \pi(h, B)$$

where $\pi(h, B)$ is a predefined prior derived from the LLM's domain hint $h$.

Scores become a distribution via temperature-controlled softmax. Rather than reading
individual scores, we model the *shape* of that distribution — it reflects the query's
inherent ambiguity. Routing uncertainty is measured as normalized entropy $\tilde{u}_B$
and drives the coverage threshold:

$$\rho_B(\tilde{u}_B) = \rho_{\min} + (\rho_{\max}-\rho_{\min}) \cdot \sigma\big(\kappa(\tilde{u}_B - \tau)\big)$$

Blocks are taken in descending probability until cumulative mass exceeds $\rho_B$.
Confident queries yield compact scopes, ambiguous ones broader coverage — overcoming the
rigidity of fixed Top-$K$ routing.

### Stage-2 — Adaptive role routing

Stage-2 picks concrete agents *within* the Stage-1 scope, using the same machinery
(scoring → softmax → entropy → cumulative-mass selection) at finer granularity.

The separation matters. Block-level and role-level selection are different decision
granularities; fusing them into one stage forces semantically unrelated roles into the
same competition space, flattening the distribution and obscuring fine-grained role
preferences — especially when only a narrow set of specialists is needed. Entropy would
then reflect coarse *domain* ambiguity rather than *role* ambiguity, recruiting broader
teams than necessary. Restricting candidates to the selected blocks removes that spurious
competition.

The router thus proceeds causally — uncertainty → coverage scope → agent cardinality —
rather than deciding team size directly.

### Node merge

**Communication graph.** Agents are nodes $v \in V$; a message from agent $i$ to $j$ is a
directed edge with learnable weight $\tilde{A}_{ij} \in (0,1)$ parameterizing its inclusion
probability. Each iteration samples $M$ graphs, evaluates task performance
$\mu(\mathcal{G}_m)$, and — since LLM outputs are non-differentiable — updates by policy
gradient:

$$\tilde{A} \leftarrow \tilde{A} + \eta \cdot \frac{1}{M}\sum_{m=1}^{M}\mu(\mathcal{G}_m)\,\nabla \log p(\mathcal{G}_m)$$

**Merge target selection.** Rather than asking which agent contributes least in isolation,
we ask which agent acts as a *structural bottleneck* in information propagation. For each
node $c$ we compute a transition-efficiency score $\Delta W(c)$ that contrasts the strength
of its incoming edge weights $w_{in}$ against its outgoing edge weights $w_{out}$, as
learned by the policy-gradient optimization.

A node whose incoming signal dominates its outgoing signal absorbs information from the
conversation without passing a comparably strong signal onward — it weakens or disrupts
transmission, lowering the chance of a correct answer. That node is selected as the merge
target $c^{*}$.

Unlike node-dropout methods that remove several nodes per a predefined ratio, exactly
**one** node is merged per step. There is no threshold or ratio hyperparameter to tune,
and the collaboration structure changes minimally.

**Merge partner selection.** Candidates are all directly connected nodes in either
direction, $\mathcal{N}(c^{*}) = \{ j \mid \tilde{A}_{j,c^{*}} > 0 \ \text{or}\ \tilde{A}_{c^{*},j} > 0 \}$,
consistent with $\Delta W$ being computed over both directions. Scoring by strongest
learned interaction,

$$S(c^{*}, j) = \max\big(\tilde{A}_{j,c^{*}},\, \tilde{A}_{c^{*},j}\big), \qquad j^{*} = \arg\max_{j \in \mathcal{N}(c^{*})} S(c^{*}, j)$$

Ties break by (1) preferring an incoming edge $j \rightarrow c^{*}$, then (2) falling back
to execution order — selecting the agent that most directly shaped $c^{*}$'s reasoning in
the chain-of-thought procedure.

**Absorption-based merge.** No new node is created: $j^{*} \leftarrow c^{*}$. The target is
removed with all its edges, the partner is retained, and — to avoid semantic loss — an LLM
summarizes $c^{*}$'s role description and reasoning context and appends it to $j^{*}$'s
prompt. Nodes and edges shrink structurally; the absorbed agent survives at the prompt
level. This *Prompt-level Fusion* costs only ~331 prompt / ~91 completion tokens per merge
and beats naive concatenation (see ablations).

**Post-merge edge dropout.** Merging changes the topology, invalidating the learned edge
weights, and the neighbourhood around $j^{*}$ can remain over-connected. We therefore
reinitialize the adjacency matrix, rerun policy-gradient learning on the restructured
graph, and prune:

$$\mathcal{E}_{\mathrm{keep}} = \operatorname{TopK}_{\lceil (1-\beta)|E| \rceil}(E; \tilde{A}), \qquad \mathcal{E}_{\mathrm{drop}} = E \setminus \mathcal{E}_{\mathrm{keep}}$$

The pruning rule follows AgentDropout, but is applied after merge-induced restructuring
rather than after simple node removal. It is a complementary post-processing step; node
merging remains the core operation.

---

## Results

Six benchmarks spanning general reasoning (MMLU), mathematical reasoning (GSM8K, AQuA,
MultiArith, SVAMP) and code generation (HumanEval). Accuracy or Pass@1, averaged over at
least three independent runs. Initial pool: 15 role-based agents. Rounds: $T=1$ for MMLU,
$T=2$ elsewhere.

### Main results

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

*DR. = Dropout, ME. = Merge.*

With Llama-3-8B-Instruct, S2R-Merge leads on every benchmark — notably +5.2%p over
AgentDropout on HumanEval and +4.6%p on GSM8K. With GPT-4 it reaches 100% on both
MultiArith and HumanEval. The gain holds across a small open model and a large commercial
one, whereas AgentDropout's margin varies with backbone capacity.

### Token efficiency

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

*P = prompt tokens, C = completion tokens.*

The saving is largest on multi-step reasoning: GSM8K drops from 80.4M (AgentDropout) and
174M (AgentPrune) to 11.2M prompt tokens; MultiArith from 15.5M / 34.7M to 1.1M. Routing
and structural compression pay off most where tasks demand deeper reasoning and stronger
preservation of intermediate information.

### Router × merge are complementary

Fixing the pool at $n=15$ and toggling the router (GPT-4):

| | Method | Avg. Acc. | Avg. Ptok. | Avg. Ctok. |
|---|---|---|---|---|
| **w/ Router** | AgentPrune | 86.88 | 34.5M | 6.7M |
| | AgentDropout | 90.97 | 15.9M | 4.9M |
| | **Ours** | **92.37** | **3.6M** | **938K** |
| **w/o Router** | AgentPrune | 88.79 | 48.2M | 3.4M |
| | AgentDropout | 91.05 | 22M | 2.5M |
| | **Ours** | **92.28** | 43.3M | 3.4M |

Two separable effects:

- **Merge carries the accuracy.** Even with no router and the full 15-agent pool, ours
  reaches 92.28 vs 91.05 (AgentDropout) and 88.79 (AgentPrune). Merging structurally
  inefficient agents into related ones preserves more useful information than deleting
  nodes or edges.
- **The router carries the efficiency.** Adding it moves accuracy 92.28 → 92.37 while
  prompt tokens fall 43.3M → 3.6M (**12.0×**) and completion tokens 3.4M → 938K
  (**3.6×**). At equal accuracy, ours uses 9.6× fewer prompt tokens than AgentPrune and
  4.4× fewer than AgentDropout under the router.

### Further ablations

Full tables in [`docs/RESULTS.md`](docs/RESULTS.md).

- **Merge target criterion** — $\Delta W(c)$ 91.3 avg vs degree-based 90.7 and random 90.9.
  Connectivity-based and unstructured rules do not reliably find targets whose removal
  preserves performance.
- **Prompt-level fusion** — 90.8 avg vs 90.2 for plain concatenation, at ~331 prompt / ~91
  completion tokens per merge, negligible against the overall savings.
- **Topology robustness** — 92.28 / 92.20 / 92.52 on fully-connected / layered / random
  initial graphs: a 0.32%p spread. AgentPrune swings 88.60 → 87.15 across the same
  settings. Absorption-based merging needs no topology-specific tuning.
- **Merge iterations** — chosen per dataset by the accuracy–token trade-off; more merging
  keeps cutting tokens but eventually over-compresses role-specific information.

---

## Implementation

```
Router/                        Two-stage uncertainty-aware agent router
  src/stage0.py                  structured summary + mixed embedding v
  src/stage1.py                  block scoring, prior π(h,B), adaptive ρ_B
  src/stage2.py                  role scoring within selected blocks
  src/llm_client.py              vLLM / OpenAI client, router-side token accounting
  config/*.yaml                  α, λ, temperature, ρ_min/ρ_max/τ/κ; block & role defs
  data/prototypes/*.npy          block and role prototype embeddings

AgentDropout/                  Communication-graph MAS (see Acknowledgements)
  graph/graph_merge.py           merge-aware graph
  utils/globals.py               token counters
  prompt/*_merge.py              merge-capable prompt sets

experiments/
  run_<dataset>_with_merge.py    ΔW scoring, partner selection, absorption merge

run_experiment_<dataset>.py      end-to-end route → merge → prune → evaluate
```

Blocks and their roles are declared in the router config (15 roles total for the math
configs), so the agent pool is data, not code.

### Token accounting

Routing is not free — Stage-0 calls an LLM. Reporting a single total would hide that, so
tokens are tracked in two separate global counters:

- **`T_sunk`** — spent by the router selecting the team; paid once, before any reasoning.
  `SunkPromptTokens` / `SunkCompletionTokens` / `SunkCost` in
  [`AgentDropout/utils/globals.py`](AgentDropout/utils/globals.py), incremented from
  [`Router/src/llm_client.py`](Router/src/llm_client.py).
- **`T_inference`** — spent merging and answering, computed as `Total − T_sunk`.

Per-agent transcripts are logged to `result/<run>/logs/agent_conversations.txt` without
touching either counter, so conversations can be inspected without corrupting the cost
measurement.

## Setup

```bash
pip install -r requirements.txt
cp template.env .env    # then edit
```

Llama experiments run against a local vLLM server; GPT-4 experiments go through the
OpenAI API.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 6789 --max-model-len 8192
```

`LLAMA_MODEL_ID` in `.env` must match the `--model` string vLLM was launched with — the
OpenAI-compatible endpoint keys models by exactly that string. Verify with
`python check_vllm_server.py`.

Datasets:

```bash
python convert_to_jsonl.py                  # GSM8K -> datasets/gsm8k/
python dataset_download.py --dataset all    # MultiArith, SVAMP, AQuA, MMLU, HumanEval
```

## Running

```bash
python run_experiment_gsm8k.py \
    --llm_name "Meta-Llama-3.1-8B-Instruct" \
    --num_iterations 10 \
    --merge_iterations 5 \
    --batch_size 20
```

Useful flags: `--test_samples` (cap the eval set), `--merge_iterations` (merge depth),
`--pruning_rate` ($\beta$, edge dropout), `--num_rounds` (must be ≥ 2), `--lr`,
`--router_config`. Each run writes `result/Router-NodeMerge_<DATASET>_<timestamp>/` with
accuracy, the `T_sunk` / `T_inference` split, and the full agent transcript.

## Acknowledgements

The communication-graph MAS in `AgentDropout/` builds on
[AgentDropout](https://arxiv.org/abs/2503.18891) (ACL 2025) and, before it, AgentPrune,
which introduced learnable edge weights and policy-gradient topology optimization for
token-efficient multi-agent collaboration. S2R-Merge replaces node *dropout* with
absorption-based node *merge* and prepends the uncertainty-aware router.

# 2-Stage Soft-Gated, Uncertainty-Aware Multi-Agent Router

A sophisticated routing system for multi-agent problem-solving that adaptively selects relevant agents based on question semantics and uncertainty estimation.

## Overview

The Router implements a two-stage selection process:

### **Stage-0: Representation Generation**
- Generates structured summary using LLM (zero-shot, no reasoning)
- Creates mixed embedding: `v = α·Embed(q) + (1-α)·Embed(t)`
- Refines semantic signals without making judgments

### **Stage-1: Adaptive Soft Block Routing**
- Computes block relevance using cosine similarity
- Generates probability distribution with temperature-controlled softmax
- Measures uncertainty using normalized entropy
- Adapts coverage threshold based on uncertainty (more uncertain → wider scope)
- Selects blocks using cumulative probability mass (**NO Top-K!**)

### **Stage-2: Adaptive Role Routing**
- Constructs candidate role set from selected blocks
- Computes role relevance and probability distribution
- Measures role-level uncertainty
- Adapts participation threshold (fewer/more agents based on uncertainty)
- Selects final agent set A₀ using cumulative probability mass

## Architecture

```
Router/
├── config/
│   └── mmlu_config.yaml        # Configuration and hyperparameters (MMLU dataset)
├── data/
│   └── prototypes/             # Prototype embeddings
│       ├── block_prototypes.npy
│       └── role_prototypes.npy
├── src/
│   ├── __init__.py
│   ├── router.py               # Main Router class
│   ├── stage0.py               # Representation generation
│   ├── stage1.py               # Block routing
│   └── utils.py                # Utility functions
├── tests/
│   └── test_router.py          # Test suite
├── mmlu/                       # MMLU dataset
└── requirements.txt
```

## Installation

```bash
# Install dependencies
conda activate router
pip install -r requirements.txt

# Or install specific packages
pip install numpy pyyaml sentence-transformers torch
```

## Quick Start

```python
from src.router import Router

# Initialize router
router = Router(config_path="config/mmlu_config.yaml")

# Route a question
question = "What is the derivative of x^2 + 3x + 5?"
result = router.route(question)

# Access results with probabilities (cumulative coverage-based)
print(f"Selected Blocks ({result['num_blocks']}):")
for block_id, prob in result['block_probabilities'].items():
    print(f"  • {block_id} → p_B = {prob:.4f}")

print(f"\nSelected Agents ({result['num_agents']}):")
for role_id, prob in result['role_probabilities'].items():
    print(f"  • {role_id} → p_r = {prob:.4f}")

# Uncertainty and coverage info
print(f"\nUncertainty:")
print(f"  Block: {result['uncertainty']['block_uncertainty']:.4f}")
print(f"  Role:  {result['uncertainty']['role_uncertainty']:.4f}")
```

## Configuration

Key hyperparameters in `config/mmlu_config.yaml`:

### Stage-0
- `alpha`: Mixing weight for embeddings (default: 0.5)
- `embedding_model`: Sentence transformer model

### Stage-1 (Block Routing)
- `temperature_block`: Softmax temperature (default: 0.3)
- `rho_min`: Minimum coverage threshold (default: 0.60)
- `rho_max`: Maximum coverage threshold (default: 0.95)
- `tau`: Uncertainty threshold for adaptation (default: 0.5)
- `beta`: Sigmoid steepness (default: 8.0)

### Stage-2 (Role Routing)
- `temperature_role`: Softmax temperature (default: 0.3)
- `rho_min_role`: Minimum coverage threshold (default: 0.50)
- `rho_max_role`: Maximum coverage threshold (default: 0.90)
- `tau_role`: Uncertainty threshold for adaptation (default: 0.5)
- `beta_role`: Sigmoid steepness (default: 8.0)

## Testing

```bash
# Run test suite
cd Router
python tests/test_router.py
```

The test suite includes:
- Basic routing functionality
- Uncertainty variation analysis
- Coverage threshold adaptation

## Blocks and Roles

### Predefined Blocks
1. **MathLogic**: Mathematics, Logic, Statistics
2. **CS_Eng_Physics**: Computer Science, Engineering, Physics
3. **Bio_Med**: Biology, Medicine, Health Sciences
4. **Econ_Law_Social**: Economics, Law, Social Sciences
5. **Humanities**: History, Philosophy, Literature
6. **General_Meta**: General Knowledge, Meta-reasoning

### Roles per Block
Each block contains 3 specialized roles (18 roles total).

## Key Features

### Uncertainty-Aware Adaptation
- Low uncertainty → narrow scope (single-domain)
- High uncertainty → wide scope (cross-domain)
- Smooth transition via sigmoid function

### NO Top-K Selection
- Uses cumulative probability mass instead of fixed K
- Adaptive number of blocks based on uncertainty
- More principled than arbitrary thresholds

### Temperature-Controlled Softmax
- Adjustable distribution sharpness
- Lower temperature → more confident selection
- Higher temperature → more exploratory

### Prototype-Based Similarity
- Fast cosine similarity computation
- Pre-computed prototype embeddings
- Scalable to large agent sets

## Example Outputs

### Low Uncertainty (Single-Domain)
```
Question: "Calculate the integral of x^2"

Stage-1 (Blocks):
  Block Uncertainty: 0.12
  Adaptive Coverage Threshold: 0.61
  Actual Coverage: 0.65

  Selected Blocks (1):
    • MathLogic → p_B = 0.6520

Stage-2 (Roles):
  Role Uncertainty: 0.24
  Adaptive Coverage Threshold: 0.52
  Actual Coverage: 0.55

  Selected Agents (A_0): 2 agents
    1. Mathematician      → p_r = 0.3850  (핵심 역할)
    2. Statistician       → p_r = 0.1650  (보조 역할)

  Key: p_r represents each agent's relevance/contribution weight
```

### High Uncertainty (Cross-Domain)
```
Question: "Discuss AI ethics in healthcare from medical and philosophical perspectives"

Stage-1 (Blocks):
  Block Uncertainty: 0.78
  Adaptive Coverage Threshold: 0.92
  Actual Coverage: 0.95

  Selected Blocks (3):
    • Bio_Med         → p_B = 0.4200
    • Humanities      → p_B = 0.3500
    • General_Meta    → p_B = 0.1800

Stage-2 (Roles):
  Role Uncertainty: 0.65
  Adaptive Coverage Threshold: 0.82
  Actual Coverage: 0.86

  Selected Agents (A_0): 7 agents
    1. Doctor                 → p_r = 0.1850
    2. Philosopher            → p_r = 0.1650
    3. Biologist              → p_r = 0.1450
    4. Historian              → p_r = 0.1250
    5. Generalist             → p_r = 0.1100
    6. Critic                 → p_r = 0.0950
    7. Common_Sense_Reasoner  → p_r = 0.0750

  High uncertainty → More diverse agents needed for comprehensive coverage
```

## Research Notes

### Adaptive Coverage Formula
```
ρ_B(ũ_B) = ρ_min + (ρ_max - ρ_min) · σ(β(ũ_B - τ))
```

where:
- `ũ_B`: Normalized entropy (uncertainty)
- `ρ_min, ρ_max`: Coverage bounds
- `τ`: Uncertainty threshold
- `β`: Transition steepness
- `σ`: Sigmoid function

### Cumulative Selection Algorithm
```python
1. Sort blocks by probability (descending)
2. Accumulate probability mass
3. Select prefix until mass ≥ threshold
4. Return selected blocks
```

## Development

### Adding New Blocks/Roles
1. Edit `config/mmlu_config.yaml`
2. Add block definition and role mapping
3. Add prototype descriptions
4. Regenerate prototypes (will auto-generate on next run)

### Custom Embedding Models
Replace `embedding_model` in config with any sentence-transformers model:
- `all-mpnet-base-v2` (default, balanced)
- `all-MiniLM-L6-v2` (faster, smaller)
- `multi-qa-mpnet-base-dot-v1` (Q&A optimized)

### LLM Integration
The Stage-0 client lives in `src/llm_client.py` and is selected by `stage0.llm_model` in
the router config:

- a model name containing `llama` / `meta-llama` routes to `LlamaClient`, which talks to a
  local vLLM server over its OpenAI-compatible endpoint;
- any other name routes to the OpenAI client, which reads its key from
  `config/api_keys.yaml` or the `OPENAI_API_KEY` environment variable;
- with neither available it falls back to `MockLLMClient`, which returns a stub structured
  summary so the routing stages can be exercised without a live backend.

## TODOs

- [x] Implement Stage-2 (Role Selection)
- [x] Add real LLM client integration **NEW: OpenAI API integrated!**
- [ ] Add caching for embeddings
- [ ] Implement batch processing optimization
- [ ] Add visualization tools
- [ ] Create Jupyter notebook examples
- [ ] Add performance benchmarks
- [ ] MMLU evaluation pipeline
- [ ] Compare with baseline routing methods

## Citation

If you use this router in your research, please cite:

```bibtex
@misc{router2024,
  title={2-Stage Soft-Gated, Uncertainty-Aware Multi-Agent Router},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

**Status**: Full 2-Stage Router Complete!
**Version**: 1.0.0
**Last Updated**: 2024-01-14

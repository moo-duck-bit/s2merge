# Stage-1: Adaptive Soft Block Routing 작동 방식

## 개요

Stage-1은 질문을 받아서 **어떤 Block(도메인)을 활성화할지** 결정하는 단계입니다.
"어디를 봐야 할까?" (Where to look?)를 결정하는 **coarse-grained filtering** 단계입니다.

---

## 입력 (Input)

- **v**: Stage-0에서 생성된 질문 표현 벡터 (768차원)
- **domain_hint**: Stage-0의 LLM이 생성한 도메인 힌트 (예: "mathematics", "biology")

---

## Stage-1 처리 과정 (5단계)

### Step 1: Block Relevance Score 계산

**수식**: `s_B = cos(v, p_B) + λ * π(h, B)`

1. **Embedding Similarity**: 질문 벡터 `v`와 각 Block prototype `p_B`의 cosine similarity 계산
   ```
   cos(v, p_B) = (v · p_B) / (||v|| * ||p_B||)
   ```

2. **Domain Prior Injection** (Hybrid Scoring):
   - LLM이 생성한 `domain_hint`를 기반으로 Block에 prior boost 추가
   - 예: `domain_hint = "mathematics"` → `MathLogic` block에 +0.3 boost
   - `λ = 0.5`로 prior의 강도 조절

**예시**:
```
Block: MathLogic
  - Embedding score: 0.65
  - Domain prior (mathematics → MathLogic): 0.3
  - λ = 0.5
  - 최종 score: 0.65 + 0.5 * 0.3 = 0.80
```

---

### Step 2: Block Probability Distribution 계산

**수식**: `p_B = exp(s_B / T_B) / Σ exp(s_B' / T_B)`

- **Softmax with Temperature (T = 0.1)** 적용
- 낮은 temperature → 더 sharp한 확률 분포 (확실한 block이 더 두드러짐)
- 모든 block의 확률 합 = 1.0

**예시**:
```
Block scores: {MathLogic: 0.80, CS_Eng_Physics: 0.45, ...}
↓ Softmax (T=0.1)
Block probabilities: {MathLogic: 0.60, CS_Eng_Physics: 0.15, ...}
```

---

### Step 3: Uncertainty 계산

**수식**: `ũ_B = -Σ p_B log(p_B) / log(|B|)`

- **Normalized Entropy** 사용
- 확률 분포가 고르면 → 높은 uncertainty (0에 가까울수록 확실)
- 확률이 한 곳에 집중되면 → 낮은 uncertainty (1에 가까울수록 불확실)

**예시**:
- 모든 block이 비슷한 확률 → `ũ_B ≈ 1.0` (매우 불확실)
- 한 block이 0.9, 나머지 0.025 → `ũ_B ≈ 0.2` (확실)

---

### Step 4: Adaptive Coverage Threshold 계산

**수식**: `ρ_B(ũ_B) = ρ_min + (ρ_max - ρ_min) * σ(β(ũ_B - τ))`

- **Sigmoid 함수**를 사용하여 uncertainty에 따라 threshold 조절
- **ρ_min = 0.30**: 확실할 때 최소 coverage (적은 block 선택)
- **ρ_max = 0.80**: 불확실할 때 최대 coverage (많은 block 선택)
- **τ = 0.5**: Uncertainty threshold (이 지점에서 adaptation 시작)
- **β = 8.0**: Sigmoid steepness (전환 속도)

**예시**:
```
ũ_B = 0.2 (확실) → ρ_B = 0.30 (적은 block 선택)
ũ_B = 0.8 (불확실) → ρ_B = 0.80 (많은 block 선택)
ũ_B = 0.5 (중간) → ρ_B ≈ 0.55 (중간)
```

---

### Step 5: Block 선택 (Cumulative Probability Mass)

**원칙**: **NO Top-K!** (고정된 K개 선택 안 함)

**방법**: 확률이 높은 순으로 block을 선택하면서, **누적 확률이 threshold를 넘을 때까지** 선택

**알고리즘**:
1. Block들을 확률 내림차순으로 정렬
2. 확률이 높은 것부터 하나씩 선택
3. 선택한 block들의 확률 합계 계산
4. 누적 확률이 `ρ_B`를 넘으면 중단

**예시**:
```
Block probabilities (정렬):
  MathLogic: 0.60
  CS_Eng_Physics: 0.25
  Humanities: 0.10
  Bio_Med: 0.05

ρ_B = 0.30 (threshold)

선택 과정:
  1. MathLogic 선택 → 누적: 0.60 ≥ 0.30 → 중단
  → 최종: [MathLogic] (1개)

ρ_B = 0.80 (threshold)

선택 과정:
  1. MathLogic 선택 → 누적: 0.60 < 0.80 → 계속
  2. CS_Eng_Physics 선택 → 누적: 0.85 ≥ 0.80 → 중단
  → 최종: [MathLogic, CS_Eng_Physics] (2개)
```

---

## 출력 (Output)

- **selected_blocks**: 선택된 Block ID 리스트
- **debug_info**:
  - block_scores: 각 block의 원본 점수
  - block_probs: 각 block의 확률
  - uncertainty: 계산된 불확실성
  - coverage_threshold: 적용된 threshold
  - selection_info: 선택 과정 정보

---

## 핵심 특징

### 1. **Hybrid Scoring**
- Embedding similarity + Domain prior
- LLM의 domain hint를 활용하여 embedding만으로는 부족한 정보 보완

### 2. **Uncertainty-Aware**
- 불확실할수록 더 많은 block 선택 (안전하게)
- 확실할수록 적은 block 선택 (효율적으로)

### 3. **NO Top-K**
- 고정된 K개가 아닌, 확률 기반 동적 선택
- 문제의 특성에 따라 선택 개수가 자동 조절

### 4. **Adaptive Threshold**
- Uncertainty에 따라 threshold가 자동 조절
- 단순한 문제: 낮은 threshold → 적은 block
- 복잡한 문제: 높은 threshold → 많은 block

---

## 예시 시나리오

### 시나리오 1: 단순한 수학 문제
```
질문: "What is 2+2?"
domain_hint: "mathematics"

Step 1: MathLogic block score = 0.85 (높음)
Step 2: MathLogic probability = 0.90 (매우 높음)
Step 3: Uncertainty = 0.15 (낮음, 확실)
Step 4: Threshold = 0.30 (낮음)
Step 5: MathLogic만 선택 (누적 0.90 ≥ 0.30)

→ 결과: [MathLogic] (1개 block)
```

### 시나리오 2: 복잡한 융합 문제
```
질문: "How does quantum computing affect cryptography?"
domain_hint: "computer_science"

Step 1: 여러 block이 비슷한 score
Step 2: 확률이 고르게 분산
Step 3: Uncertainty = 0.85 (높음, 불확실)
Step 4: Threshold = 0.75 (높음)
Step 5: 여러 block 선택 (누적 0.75까지)

→ 결과: [CS_Eng_Physics, MathLogic, ...] (3-4개 block)
```

---

## 하이퍼파라미터

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `temperature_block` | 0.1 | Softmax temperature (낮을수록 sharp) |
| `lambda_prior` | 0.5 | Domain prior 강도 |
| `rho_min` | 0.30 | 최소 coverage (확실할 때) |
| `rho_max` | 0.80 | 최대 coverage (불확실할 때) |
| `tau` | 0.5 | Uncertainty threshold |
| `beta` | 8.0 | Sigmoid steepness |

---

## 왜 이렇게 설계했나?

1. **Coarse-grained Filter**: Stage-1은 "어디를 봐야 할까?"만 결정
   - 너무 공격적으로 필터링하면 → 후속 단계에서 최적화 여지 부족
   - 적절한 "optimization room" 확보 필요

2. **Uncertainty 활용**: 불확실할수록 더 넓게 선택
   - 놓치는 것 방지
   - Stage-2에서 추가 정제 가능

3. **Domain Prior**: LLM의 도메인 힌트 활용
   - Embedding만으로는 부족한 정보 보완
   - 정확도 향상 (40% → 86%)

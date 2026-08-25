# TopoTrace 연구 계획서

> **가제:** *Did the Shape Forget? Oracle-Calibrated Topological Auditing of Machine Unlearning*  
> **프로젝트명:** TopoTrace  
> **핵심 질문:** Machine unlearning 이후 삭제 데이터가 모델 내부 표현에 남긴 topology가, 해당 데이터를 처음부터 학습하지 않은 exact-retrained model의 topology와 통계적으로 구별되지 않는가?

---

## 1. 연구 개요

### 1.1 문제 정의

전체 학습 데이터셋을 다음과 같이 나눈다.

\[
D = D_r \cup D_f,\qquad D_r \cap D_f = \varnothing
\]

- \(D_r\): retain set
- \(D_f\): forget set
- \(M_O\): 전체 데이터 \(D\)로 학습한 original model
- \(M_U\): \(M_O\)에 machine unlearning을 적용한 unlearned model
- \(M_R\): \(D_f\)를 제외한 \(D_r\)만으로 처음부터 학습한 exact-retrained model

본 연구에서는 단순히 \(D_f\)의 topology가 “사라졌는가”를 묻지 않는다. 대신 동일한 audit probe set \(S\)에 대해 다음 조건을 평가한다.

\[
\mathcal{T}(M_U,S) \approx \mathcal{T}(M_R,S)
\]

여기서 \(\mathcal{T}\)는 특정 layer의 representation point cloud에서 계산한 persistent-topology fingerprint이다.

### 1.2 핵심 가설

- **H1 — Residual-topology hypothesis:** 일부 approximate unlearning method는 output-level metric을 통과하더라도 internal representation에 deletion-induced topology를 남긴다.
- **H2 — Layer-localization hypothesis:** 잔존 topology의 크기는 layer에 따라 다르며, 중간 layer 또는 penultimate layer에서 가장 크게 나타난다.
- **H3 — Complementarity hypothesis:** TopoTrace는 accuracy, membership inference, pointwise representation metric이 놓치는 불완전한 forgetting을 검출한다.
- **H4 — Operational-relevance hypothesis:** topological residual이 큰 모델은 exact-retrained model보다 forget set을 빠르게 relearn하거나 topology-based distinguisher에 더 쉽게 구별된다.
- **H5 — Targeted-deletion hypothesis:** topology-support sample을 의도적으로 삭제하는 조건에서 random deletion보다 더 명확한 unlearning failure mode가 관찰된다.

---

## 2. 연구 질문

### RQ1
Approximate machine-unlearning method는 삭제 데이터가 모델 내부 표현에 유도한 topology를 제거하는가?

### RQ2
Unlearned model의 topological fingerprint는 여러 exact-retrained model로 구성된 oracle distribution과 통계적으로 구별되는가?

### RQ3
Topological audit은 output-level metric, membership-inference metric, pointwise representation metric이 놓치는 residual information을 검출하는가?

### RQ4
Residual topology는 relearning speed 또는 unlearned-vs-retrained distinguishability와 연결되는가?

### RQ5
어떤 layer와 homology dimension이 deletion-induced topology에 가장 민감한가?

### RQ6
Relative persistent homology가 forget-set-only persistent homology보다 높은 검정력과 안정성을 제공하는가?

---

## 3. 연구 범위

### 3.1 1차 논문에서 반드시 포함할 범위

- Synthetic topology benchmark
- MNIST 또는 FashionMNIST
- CIFAR-10
- Small CNN 및 ResNet-18
- No-op, Exact Retraining, Fine-tuning, 대표 unlearning method 2개 이상
- \(H_0\), \(H_1\)
- Layerwise persistent-homology analysis
- Retrain–retrain oracle calibration
- Conventional unlearning metric과 TopoTrace 비교
- Relearning 또는 topology-based distinguisher 중 최소 하나

### 3.2 초기 논문에서 제외하거나 부록으로 둘 범위

- \(H_2\) 이상의 고차 homology
- 대규모 language model
- Web-scale pretrained model
- Certified unlearning 자체의 제안
- 의료·생체 민감 데이터
- 지나치게 많은 데이터셋과 unlearning method

### 3.3 성공 후 확장 범위

- CIFAR-100 subclass deletion
- Tiny ImageNet
- User-level 또는 identity-level deletion
- Vision Transformer
- Graph neural network 또는 multimodal model
- Topological regularizer를 이용한 새로운 unlearning algorithm

---

## 4. 핵심 기여 목표

1. **Oracle-calibrated topological auditing**  
   단일 retrained model이 아니라 여러 exact-retrained seed에서 얻은 topology distribution을 oracle로 사용한다.

2. **Topological Imprint Gate**  
   원래 삭제가 측정 가능한 topology 변화를 일으키는 조건에서만 residual을 해석한다.

3. **Topological Residual Ratio**  
   approximate unlearning이 original topology에서 exact-retrained topology로 얼마나 이동했는지 정량화한다.

4. **Progress–Artifact decomposition**  
   retraining 방향으로의 유효한 변화와, retraining으로 설명되지 않는 파괴적 topology 변화를 분리한다.

5. **Layerwise Topological Forgetting Profile**  
   topology 잔존 위치를 layer와 homology dimension별로 분석한다.

6. **Topology-targeted deletion benchmark**  
   persistent feature를 지지하는 sample을 삭제 대상으로 선택하여 기존 평가에서 드러나지 않는 failure mode를 유도한다.

7. **Operational validation**  
   residual topology가 실제 distinguisher 성능 또는 relearning speed와 연결됨을 검증한다.

---

## 5. 전체 연구 파이프라인

```text
Dataset construction
        │
        ├── Retain set Dr
        ├── Forget set Df
        └── Audit probe set S
        │
        ▼
Model preparation
        │
        ├── Original models MO
        ├── Unlearned models MU
        └── Exact-retrained models MR
        │
        ▼
Layerwise representation extraction
        │
        ▼
Distance matrix construction
        │
        ├── Forget-set topology
        ├── Local relative topology
        └── Augmentation-orbit topology
        │
        ▼
Persistent homology
        │
        ├── H0
        └── H1
        │
        ▼
Topological fingerprint
        │
        ├── Persistence diagram
        ├── Persistence image
        ├── Persistence landscape
        └── Betti curve
        │
        ▼
Oracle calibration and statistical testing
        │
        ├── Retrain–retrain null
        ├── Original–retrain imprint
        └── Unlearned–retrain residual
        │
        ▼
Comparison with conventional metrics
        │
        ▼
Operational validation
```

---

## 6. 데이터셋 및 삭제 시나리오

### 6.1 Synthetic benchmark

| Benchmark | Ground-truth topology | Forget-set 구성 | 예상 변화 |
|---|---|---|---|
| Satellite Components | 다중 \(H_0\) component | 작은 satellite cluster | component 수 또는 merge scale 변화 |
| Bridge Points | 두 cluster를 연결하는 구조 | bridge sample | connectivity 변화 |
| Ring / Annulus | 강한 \(H_1\) loop | loop-support sample | loop persistence 감소 |
| Figure-Eight | 두 개의 \(H_1\) loop | 한쪽 loop support | 특정 persistent bar 제거 |
| Topological Canary | 예측과 독립적인 구조 | canary support | output 변화 없이 topology 변화 |

### 6.2 실제 데이터셋

| 단계 | 데이터셋 | 목적 |
|---|---|---|
| Pilot | MNIST 또는 FashionMNIST | 빠른 debugging 및 metric sanity check |
| Main 1 | CIFAR-10 | 표준 image-classification setting |
| Main 2 | CIFAR-100 | class/subclass 구조 및 semantic deletion |
| Extension | SVHN 또는 Tiny ImageNet | 일반화 검증 |

### 6.3 삭제 시나리오

| ID | 삭제 방식 | 비율 | 목적 |
|---|---|---:|---|
| D1 | Uniform random deletion | 1%, 5%, 10% | 표준 setting |
| D2 | Class deletion | 특정 class 전체 또는 일부 | 고신호 semantic deletion |
| D3 | Subclass deletion | CIFAR-100 subclass | 동일 superclass 내부 세밀한 삭제 |
| D4 | Cluster deletion | representation cluster 단위 | \(H_0\) 변화 검증 |
| D5 | Topology-targeted deletion | 5% | persistent feature support 제거 |
| D6 | Matched-random deletion | D5와 동일 크기·밀도·class 구성 | D5의 공정한 대조군 |

---

## 7. 모델 및 unlearning baseline

### 7.1 모델

| 단계 | 모델 | 비고 |
|---|---|---|
| Synthetic | 3-layer MLP | 해석과 시각화 용이 |
| Pilot | Small CNN | 빠른 반복 실험 |
| Main | ResNet-18 | 주 결과 |
| Extension | ViT-Tiny 또는 ConvNeXt-Tiny | architecture generalization |

### 7.2 Baseline

| 범주 | 방법 | 역할 |
|---|---|---|
| Negative control | No-op | 삭제를 전혀 하지 않은 경우 |
| Gold standard | Exact Retraining | 이상적인 empirical reference |
| Naive | Fine-tuning on \(D_r\) | 단순한 approximate baseline |
| Naive | Gradient Ascent / NegGrad | forget loss를 직접 증가 |
| Established | SCRUB | distillation-based baseline 후보 |
| Established | SSD | parameter-level suppression baseline 후보 |
| Destructive control | Random reinitialization 또는 strong noise | utility를 파괴하는 가짜 forgetting |

최소 실행 구성:

```text
No-op
Exact Retraining
Fine-tuning
Gradient Ascent 또는 NegGrad
SCRUB
SSD
```

---

## 8. Representation 추출

### 8.1 대상 layer

ResNet-18 기준:

```text
L0: stem output
L1: layer1 output
L2: layer2 output
L3: layer3 output
L4: layer4 output
L5: penultimate embedding
L6: logits
```

### 8.2 정규화

각 embedding \(h^\ell_\theta(x)\)를 다음과 같이 정규화한다.

\[
z_i =
\frac{h^\ell_\theta(x_i)}
{\|h^\ell_\theta(x_i)\|_2+\epsilon}
\]

### 8.3 거리 함수

주 분석:

\[
d_{ij}
=
\sqrt{2-2\langle z_i,z_j\rangle}
\]

보조 분석:

- Euclidean distance
- Cosine distance
- Correlation distance

### 8.4 Probe set 고정 원칙

- 동일 실험 조건에서는 모든 모델에 정확히 같은 probe sample을 사용한다.
- forget, retain-neighborhood, test probe를 별도로 저장한다.
- data augmentation seed를 고정한다.
- topology 계산에 사용한 sample ID를 결과 파일에 기록한다.

---

## 9. Topology 정의

### 9.1 Forget-set-only topology

\[
Z^\ell_\theta(D_f)
=
\{h^\ell_\theta(x):x\in D_f\}
\]

장점:

- 구현이 간단함
- baseline으로 적합함

한계:

- 작은 forget set에서 불안정함
- retain set과의 관계를 반영하지 못함

### 9.2 Local relative topology

삭제 데이터 주변의 retain sample을 선택한다.

\[
N_q(D_f)
=
q\text{-nearest retain neighbors of }D_f
\]

다음 pair를 구성한다.

\[
\left(
Z^\ell_\theta(N_q(D_f)\cup D_f),
Z^\ell_\theta(N_q(D_f))
\right)
\]

목표:

- forget set이 주변 retain manifold에 추가한 topology를 측정
- 전체 retain set 계산량을 줄임
- random density 차이의 영향을 완화

### 9.3 Augmentation-orbit topology

개별 forget sample \(x\)에 대해 augmentation set을 생성한다.

\[
\mathcal{O}_\theta(x)
=
\{h_\theta(a_j(x))\}_{j=1}^{B}
\]

활용:

- single-example deletion
- user-level deletion
- local decision geometry 분석

초기 논문에서는 secondary analysis로 사용한다.

---

## 10. Persistent-homology 계산

### 10.1 주 설정

- Filtration: Vietoris–Rips
- Homology dimensions: \(H_0, H_1\)
- Coefficients: \(\mathbb{Z}_2\)
- Input: precomputed distance matrix
- Max edge length: 데이터별 거리 분위수 기반 결정
- Point count: 조건별 동일하게 유지

### 10.2 Topological representation

| 표현 | 용도 |
|---|---|
| Persistence Diagram | 정성적 분석 및 diagram distance |
| Persistence Image | 주 통계 분석 |
| Persistence Landscape | sensitivity analysis |
| Betti Curve | 해석 가능한 시각화 |
| Persistence Entropy | 보조 요약 통계 |

### 10.3 계산 안정성 점검

- audit subset bootstrap
- point-count sensitivity
- max-edge sensitivity
- persistence threshold sensitivity
- distance-function sensitivity
- random-seed sensitivity

---

## 11. 제안 metric

### 11.1 Oracle 내부 변동

여러 retrained model의 topology vector \(v_{R_t}\)를 사용한다.

\[
D_{RR}
=
\operatorname{median}_{t\neq t'}
\|v_{R_t}-v_{R_{t'}}\|_2
\]

### 11.2 Original–Retrain 차이

\[
D_{OR}
=
\operatorname{median}_{s,t}
\|v_{O_s}-v_{R_t}\|_2
\]

### 11.3 Topological Imprint

\[
I_{\text{topo}}
=
D_{OR}-D_{RR}
\]

해석:

- \(I_{\text{topo}}\le 0\): 삭제가 검출 가능한 topology 변화를 만들었다고 보기 어려움
- \(I_{\text{topo}}>0\): original topology가 retrain oracle variation을 넘어섬

### 11.4 Unlearned–Retrain 차이

\[
D_{UR}
=
\operatorname{median}_{s,t}
\|v_{U_s}-v_{R_t}\|_2
\]

### 11.5 Topological Residual Ratio

\[
\operatorname{TRR}
=
\frac{D_{UR}-D_{RR}}
{D_{OR}-D_{RR}+\epsilon}
\]

예상 anchor:

| 모델 | 기대 TRR |
|---|---:|
| Exact Retrain | 약 0 |
| No-op | 약 1 |
| 부분 unlearning | 0과 1 사이 |
| Destructive unlearning | 1보다 크거나 불안정 |

### 11.6 Topological Progress

\[
\alpha
=
\frac{
\langle v_U-v_O,\bar v_R-v_O\rangle
}{
\|\bar v_R-v_O\|_2^2+\epsilon
}
\]

### 11.7 Topological Artifact

\[
\eta
=
\frac{
\|(v_U-v_O)-\alpha(\bar v_R-v_O)\|_2
}{
\|\bar v_R-v_O\|_2+\epsilon
}
\]

해석:

- \(\alpha\approx1,\eta\approx0\): retraining 방향으로 정확히 이동
- \(\alpha\approx0\): topology가 original 부근에 잔존
- \(\eta\)가 큼: retraining으로 설명되지 않는 파괴적 변화
- \(\alpha>1\): overshooting 가능성

### 11.8 Layerwise Profile

\[
\operatorname{TRR}_{\ell,k},
\quad
\alpha_{\ell,k},
\quad
\eta_{\ell,k}
\]

출력:

```text
Layer × Homology dimension × Unlearning method
```

---

## 12. 기존 unlearning metric

### 12.1 Utility

- Retain accuracy
- Test accuracy
- Forget accuracy
- Expected Calibration Error
- Class-wise accuracy
- Retain loss
- Forget loss

### 12.2 Privacy 및 forgetting

- Membership inference attack
- Per-example likelihood-ratio 기반 attack
- Forget-vs-retrain loss distribution
- Parameter distance
- Prediction KL divergence
- Retrain-model distinguishability

### 12.3 Representation-level 비교

- Cosine similarity
- Nearest-neighbor overlap
- CKA
- Linear-probe performance
- Layerwise representation distance

### 12.4 TopoTrace의 추가 가치 판단

다음 조건을 찾는 것이 핵심이다.

```text
Conventional metric: 통과
Pointwise representation metric: 통과 또는 약한 차이
TopoTrace: exact retrain과 유의하게 구별됨
```

---

## 13. Topology-targeted deletion

### 13.1 Selector model 분리

평가 모델과 다른 독립 encoder 또는 별도 seed의 selector model을 사용한다.

목적:

- 평가 모델 topology를 보고 deletion set을 고르는 adaptive bias 방지
- targeted deletion의 재현성 확보

### 13.2 Sample score 예시

Persistent cycle \(c_j\)의 persistence를 사용한다.

\[
s_i
=
\sum_j
\mathbf{1}[i\in c_j]
(d_j-b_j)
\]

상위 score sample을 topology-targeted forget set으로 선택한다.

### 13.3 Matched control

Targeted deletion과 다음 조건을 맞춘 random control을 구성한다.

- 삭제 sample 수
- class 분포
- training loss 분포
- confidence 분포
- local density
- nearest-neighbor distance
- augmentation difficulty

### 13.4 검증 질문

- Targeted deletion에서 \(I_{\text{topo}}\)가 더 큰가?
- 같은 삭제 비율에서 unlearning method 간 차이가 더 명확해지는가?
- Relative PH가 forget-set-only PH보다 높은 검정력을 보이는가?

---

## 14. 통계 분석 계획

### 14.1 독립 표본

독립 표본 단위는 **model training seed**이다.

- 모델 seed: 독립 표본
- audit subset bootstrap: seed 내부 변동
- point cloud resampling: seed 내부 변동
- 동일 모델에서 얻은 여러 layer: 반복 측정

### 14.2 Seed 수

| 단계 | 권장 seed |
|---|---:|
| Debugging | 2–3 |
| Pilot | 5 |
| Main result | 10 이상 |
| 강한 permutation resolution 필요 조건 | 20 고려 |

### 14.3 Primary endpoint

사전에 다음을 주 endpoint로 고정한다.

```text
Dataset: CIFAR-10
Deletion: topology-targeted 5%
Model: ResNet-18
Layer: penultimate
Homology: H1
Metric: Persistence-image TRR
Comparison: approximate unlearning vs exact-retrained oracle
```

### 14.4 검정

- Two-sample permutation test
- Bootstrap confidence interval
- Mixed-effects model
- Spearman correlation
- Partial correlation 또는 regression
- Multiple-comparison correction: Benjamini–Hochberg FDR

### 14.5 Equivalence 관점

단순히 유의한 차이가 없다는 이유로 equivalence를 주장하지 않는다.

가능하면 retrain–retrain variation을 이용해 equivalence margin \(\delta\)를 정의한다.

\[
\delta
=
q_{0.95}
\left(
\|v_{R_t}-v_{R_{t'}}\|_2
\right)
\]

판정 예시:

- \(D_{UR}<\delta\): oracle-equivalent region
- \(D_{UR}\ge\delta\): oracle distribution 바깥
- confidence interval이 경계와 겹침: inconclusive

---

## 15. Operational validation

### 15.1 Topology-based distinguisher

목표:

\[
M_U \quad\text{vs.}\quad M_R
\]

입력 feature 후보:

- Persistence image
- Betti curve
- Layerwise TRR vector
- Layerwise persistence entropy
- Multi-layer concatenated topology vector

평가 지표:

- ROC-AUC
- Balanced accuracy
- TPR at 1% 또는 5% FPR
- Cross-dataset generalization

주의:

- train/test model seed를 분리한다.
- topology selector와 classifier 학습 seed를 분리한다.
- 너무 복잡한 classifier보다 logistic regression을 우선 사용한다.

### 15.2 Relearning experiment

절차:

1. \(M_U\)와 \(M_R\)를 준비한다.
2. 동일한 forget subset으로 동일 optimizer와 learning rate를 사용해 재학습한다.
3. 매 step 또는 epoch마다 forget accuracy/loss를 기록한다.
4. 회복 곡선의 AUC와 threshold 도달 step을 계산한다.

주 분석:

\[
\operatorname{corr}
(
\operatorname{TRR},
\text{relearning speed}
)
\]

보조 분석:

- conventional metric을 통제한 partial correlation
- layer별 topology 중 어떤 layer가 relearning을 가장 잘 예측하는지 비교

---

## 16. Ablation study

| Ablation | 질문 |
|---|---|
| \(H_0\) vs \(H_1\) | 어떤 homology dimension이 더 민감한가? |
| Forget-only vs Relative PH | 상대 topology가 추가 정보를 주는가? |
| Single retrain vs Retrain distribution | oracle distribution calibration이 필요한가? |
| Penultimate only vs Layerwise | 잔존 topology 위치를 놓치는가? |
| Random vs Topology-targeted deletion | signal strength가 달라지는가? |
| Persistence image vs Landscape | vectorization 선택에 민감한가? |
| Cosine/chordal vs Euclidean | 거리 함수에 민감한가? |
| Fixed point count vs variable point count | sample-size artifact가 있는가? |
| Matched-random control 제거 | targeted deletion 결과가 density confound인가? |
| Pretrained vs from-scratch | 사전학습 정보가 결과를 오염시키는가? |

---

## 17. 재현성 원칙

### 17.1 모든 결과에 저장할 정보

- Git commit hash
- Dataset version
- Train/validation/test indices
- Forget/retain sample IDs
- Model seed
- Data-loader seed
- Unlearning seed
- Audit subset seed
- Topology parameters
- Library versions
- Hardware 정보
- Config file path

### 17.2 결과 파일 규격 예시

```text
results/
├── dataset/
│   └── model/
│       └── deletion_scenario/
│           └── method/
│               └── seed_000/
│                   ├── config.yaml
│                   ├── metrics.json
│                   ├── model.pt
│                   ├── embeddings/
│                   ├── persistence/
│                   ├── figures/
│                   └── logs/
```

### 17.3 Figure replay

모든 figure는 다음 입력으로 재생성 가능해야 한다.

```text
figure config + result file + random seed
```

---

## 18. 저장소 구조 제안

```text
topotrace/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── unlearning/
│   ├── topology/
│   └── experiments/
├── src/
│   └── topotrace/
│       ├── data/
│       │   ├── datasets.py
│       │   ├── splits.py
│       │   └── deletions.py
│       ├── models/
│       │   ├── mlp.py
│       │   ├── cnn.py
│       │   └── resnet.py
│       ├── training/
│       │   ├── trainer.py
│       │   └── retraining.py
│       ├── unlearning/
│       │   ├── noop.py
│       │   ├── finetune.py
│       │   ├── neggrad.py
│       │   ├── scrub.py
│       │   └── ssd.py
│       ├── representations/
│       │   ├── hooks.py
│       │   ├── extract.py
│       │   └── distances.py
│       ├── topology/
│       │   ├── filtrations.py
│       │   ├── persistence.py
│       │   ├── relative.py
│       │   ├── vectorize.py
│       │   └── metrics.py
│       ├── attacks/
│       │   ├── membership.py
│       │   ├── topology_distinguisher.py
│       │   └── relearning.py
│       ├── statistics/
│       │   ├── bootstrap.py
│       │   ├── permutation.py
│       │   ├── equivalence.py
│       │   └── correction.py
│       └── visualization/
│           ├── diagrams.py
│           ├── profiles.py
│           └── tables.py
├── scripts/
│   ├── train_original.py
│   ├── train_retrained.py
│   ├── run_unlearning.py
│   ├── extract_embeddings.py
│   ├── compute_topology.py
│   ├── run_statistics.py
│   └── reproduce_figures.py
├── tests/
├── notebooks/
├── results/
└── paper/
```


## 21. 실험 행렬

### 21.1 최소 실험 행렬

| Dataset | Model | Deletion | Method | Seeds | Layers | Homology |
|---|---|---|---|---:|---|---|
| Synthetic Ring | MLP | Loop support | No-op | 5 | all | \(H_0,H_1\) |
| Synthetic Ring | MLP | Loop support | Exact Retrain | 5 | all | \(H_0,H_1\) |
| Synthetic Ring | MLP | Loop support | Fine-tuning | 5 | all | \(H_0,H_1\) |
| MNIST | Small CNN | Random 5% | 전체 baseline | 5 | all | \(H_0,H_1\) |
| CIFAR-10 | ResNet-18 | Random 5% | 전체 baseline | 10 | selected layers | \(H_0,H_1\) |
| CIFAR-10 | ResNet-18 | Targeted 5% | 전체 baseline | 10 | selected layers | \(H_0,H_1\) |
| CIFAR-10 | ResNet-18 | Matched random 5% | 전체 baseline | 10 | selected layers | \(H_0,H_1\) |

### 21.2 우선순위

```text
P0: Synthetic sanity check
P0: CIFAR-10 random 5%
P0: CIFAR-10 targeted 5%
P0: Exact retrain distribution
P0: TRR and oracle-equivalence test
P1: Relearning or topology distinguisher
P1: CIFAR-100 subclass deletion
P2: ViT extension
P2: Augmentation-orbit topology
```

---

## 22. 예상 Figure

### Figure 1 — Framework overview

```text
Original → Unlearning / Exact Retraining → Layerwise embeddings
→ Persistent topology → Oracle-calibrated audit
```

### Figure 2 — Synthetic sanity check

- Original
- Exact retrain
- No-op
- Approximate unlearning
- persistence diagram/barcode 비교

### Figure 3 — Topological Imprint Gate

- \(D_{RR}\)
- \(D_{OR}\)
- 조건별 detectable imprint 여부

### Figure 4 — Layerwise Topological Forgetting Profile

- x축: layer
- y축: TRR
- line: unlearning method
- panel: \(H_0/H_1\)

### Figure 5 — Progress–Artifact plane

- x축: \(\alpha\)
- y축: \(\eta\)
- method별 위치

### Figure 6 — Conventional metric disagreement

- accuracy/MIA/representation/TopoTrace 통과 여부 heatmap

### Figure 7 — Random vs Topology-targeted deletion

- targeted 조건에서 imprint와 residual 증가 여부

### Figure 8 — Operational validation

- topology distinguisher ROC
- 또는 TRR vs relearning speed scatter plot

---

## 23. 예상 Table

### Table 1 — Dataset, model, deletion setting

연구 전반의 실험 조건 요약.

### Table 2 — Main unlearning performance

| Method | Retain Acc. | Forget Metric | MIA | Representation | TRR | \(\alpha\) | \(\eta\) |

### Table 3 — Oracle-equivalence test

| Method | Layer | Homology | Distance | CI | Adjusted \(p\) | Decision |

### Table 4 — Ablation

Forget-only/relative PH, \(H_0/H_1\), vectorization, distance 선택 비교.

### Table 5 — Operational relevance

Residual topology와 distinguisher 또는 relearning 성능 관계.

---

## 24. 위험 요소 및 대응

| 위험 | 영향 | 조기 신호 | 대응 |
|---|---|---|---|
| Random deletion에서 topology imprint가 없음 | 핵심 분석 불가 | \(D_{OR}\approx D_{RR}\) | Imprint Gate 적용, targeted/class deletion 사용 |
| PH가 density만 감지 | 해석 오류 | point count·density 변화에 metric 민감 | matched control, 동일 point 수, local relative PH |
| Seed variation이 너무 큼 | 통계력 저하 | \(D_{RR}\)가 매우 큼 | training protocol 고정, seed 증가, layer 선별 |
| 계산량 과다 | 일정 지연 | PH 또는 retraining 병목 | point subsampling, selected layers, staged experiment |
| Exact retrain 하나만 사용 | 잘못된 기준 | seed별 결과가 크게 다름 | 최소 5–10 retrain seeds |
| Topological residual의 의미가 불명확 | 주장 약화 | conventional metric과 아무 관계 없음 | distinguisher 또는 relearning 실험 추가 |
| Targeted deletion selection bias | 결과 신뢰도 저하 | selector/evaluator 동일 모델 | selector model 및 seed 분리 |
| Pretraining contamination | 삭제 의미 왜곡 | retrain에도 signature 유지 | from-scratch를 주 실험으로 사용 |
| Destructive method가 좋은 점수 | metric failure | utility 급락인데 TRR만 낮음 | \(\eta\), utility constraint, Pareto analysis |
| Point-cloud bootstrap 오용 | 과장된 유의성 | 비현실적으로 작은 p-value | model seed만 독립 표본으로 처리 |

---

## 25. Go/No-Go 기준

### Gate 1 — Metric validity

다음 anchor가 synthetic benchmark에서 성립해야 한다.

- [ ] Exact Retrain의 TRR이 0 부근
- [ ] No-op의 TRR이 1 부근
- [ ] destructive baseline의 \(\eta\)가 높음
- [ ] ground-truth \(H_0/H_1\) 변화가 검출됨

**실패 시:** 실제 데이터 실험으로 넘어가지 않고 metric 및 filtration을 수정한다.

### Gate 2 — Detectable imprint

실제 데이터의 적어도 한 deletion setting에서 다음이 성립해야 한다.

\[
D_{OR} > D_{RR}
\]

- [ ] confidence interval 기준으로 imprint가 검출됨
- [ ] 여러 seed에서 방향이 일관됨

**실패 시:** class/subclass 또는 topology-targeted deletion으로 전환한다.

### Gate 3 — Added value

적어도 한 approximate method에서 다음 중 하나가 나타나야 한다.

- [ ] conventional metric 통과, TopoTrace 실패
- [ ] pointwise metric 통과, TopoTrace 실패
- [ ] topology residual이 relearning speed를 유의하게 예측
- [ ] topology distinguisher가 chance보다 유의하게 높음

**실패 시:** 논문 주장을 topology benchmark 또는 negative result 중심으로 재구성할지 판단한다.

### Gate 4 — Generalization

- [ ] 두 개 이상의 dataset 또는 architecture에서 핵심 경향 재현
- [ ] targeted deletion 결과가 matched-random보다 강함
- [ ] 주 결과가 topology hyperparameter에 과도하게 의존하지 않음

---

## 26. 최소 성공 기준

다음 세 조건이 만족되면 최소 논문화 가능성을 검토한다.

1. Original과 exact retrain 사이에 oracle variation을 넘어서는 topological imprint가 존재한다.
2. 적어도 하나의 approximate unlearning method가 exact retrain과 topology상 구별된다.
3. 해당 결과가 여러 model seed에서 재현된다.

---

## 27. 강한 논문 기준

다음 중 두 개 이상을 달성하는 것을 목표로 한다.

- TopoTrace가 MIA와 pointwise representation metric이 놓치는 failure case를 검출
- Relative PH가 forget-set-only PH보다 높은 검정력 제공
- Topological residual이 relearning speed를 유의하게 예측
- Topology-based distinguisher가 unlearned와 exact-retrained model을 구별
- Layerwise analysis가 output suppression과 internal deletion의 차이를 보여줌
- Topology-targeted benchmark가 method ranking을 바꿈
- Progress–Artifact decomposition이 destructive unlearning을 분리

---

## 28. 논문 주장 강도 가이드

### 허용 가능한 주장

> 선택한 dataset, probe set, layer 및 topological statistic 하에서 일부 approximate unlearning model은 exact-retrained model distribution과 구별되었다.

> Topological residual은 기존 output-level metric과 상보적인 failure signal을 제공했다.

> Topology-targeted deletion은 random deletion보다 unlearning method 간 차이를 더 명확하게 드러냈다.

### 피해야 할 주장

> Topology가 같으므로 데이터가 완전히 삭제되었다.

> Persistent homology만으로 privacy deletion을 증명했다.

> 특정 representation의 topology가 남았으므로 원본 데이터가 직접 복원 가능하다.

> TopoTrace가 certified unlearning을 대체한다.

TopoTrace는 **empirical falsification and auditing framework**로 위치시킨다.

---

## 29. 논문 목차 초안

```text
1. Introduction
2. Background and Problem Formulation
   2.1 Machine Unlearning
   2.2 Persistent Homology
   2.3 Oracle-Calibrated Verification
3. TopoTrace
   3.1 Topological Fingerprints
   3.2 Topological Imprint Gate
   3.3 Residual Ratio
   3.4 Progress–Artifact Decomposition
   3.5 Layerwise Audit
4. Topology-Targeted Deletion Benchmark
5. Experimental Setup
6. Results
   6.1 Synthetic Validation
   6.2 Standard Deletion
   6.3 Targeted Deletion
   6.4 Comparison with Existing Metrics
   6.5 Operational Validation
7. Ablations and Sensitivity Analysis
8. Limitations
9. Conclusion
```

---

## 30. 즉시 시작할 작업

### 오늘

- [ ] GitHub repository 생성
- [ ] 본 계획서를 `docs/research_plan.md`에 저장
- [ ] `research_spec.md` 생성
- [ ] Synthetic ring dataset 구현
- [ ] MLP original/retrain 학습 코드 작성
- [ ] embedding hook 구현

### 첫 번째 기술 목표

다음 결과를 한 장의 figure로 만든다.

```text
Synthetic ring
├── Original model persistence diagram
├── Exact-retrained model persistence diagram
├── No-op model persistence diagram
└── Fine-tuned model persistence diagram
```

### 첫 번째 판정 목표

```text
Exact Retrain: TRR ≈ 0
No-op: TRR ≈ 1
Fine-tuning: 0 < TRR < 1 또는 residual 검출
```

이 결과가 확인되면 CIFAR-10으로 확장한다.

---

## 31. 최종 연구 한 문장

> **TopoTrace evaluates whether the multi-scale topology induced by forgotten data in an unlearned model becomes statistically indistinguishable from the topology of models that never observed those data.**

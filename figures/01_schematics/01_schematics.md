# 01 Schematics — Architecture, Method & Mathematical Mechanics

This document provides high-depth, publication-grade architectural diagrams and mathematical formulations for the closed-form streaming forecasting engine ($S^2\text{-RLS}$).

---

## Master System Architecture (`full_system.png` / `full_system.tex`)

Integrated master schematic displaying the complete two-stage architecture: Multi-scale Feature Extraction, $S^2$-LRU Complex Memory Projection, Offline Warm-Start Closed-Form Solve, and Real-Time Online Woodbury RLS Adaptation.

![Master System Architecture](file:///C:/Projects/Application%20of%20Core%20Slover/figures/01_schematics/full_system.png)

```mermaid
flowchart TD
    subgraph Stage1["Stage 1: Multi-Scale Feature & LRU Memory Extraction"]
        direction LR
        RAW["Raw Stream Window<br/><b>x[t-95 ... t]</b>"]
        FEAT["Multi-Scale Feature Map<br/>Stats + Lags + FFT + Cyclic"]
        LRU["Complex S2-LRU Bank<br/><b>h_{t,d} = λ_d h_{t-1,d} + γ_d x_t</b>"]
        PHI["Augmented Context Vector<br/><b>φ_S2(t) ∈ ℝ^(F+2D)</b>"]

        RAW --> FEAT --> LRU --> PHI
    end

    subgraph Stage2["Stage 2: Closed-Form Warm Start & Online Woodbury Engine"]
        direction TB
        subgraph OfflineSolve["Offline Warm-Start Ridge Solve"]
            ACC["Covariance Accumulation<br/><b>A = Σ φφ^T, B = Σ φx^T</b>"]
            SOLVE["Exact Direct Solve<br/><b>W_0 = (A + λ I)^(-1) B</b><br/><i>O(D³) One-Shot Solve</i>"]
            W0["Initial Weights<br/><b>W_0 ∈ ℝ^((F+2D) × H)</b>"]

            ACC --> SOLVE --> W0
        end

        subgraph OnlineRLS["Real-Time Streaming Woodbury RLS"]
            IN["Streaming Input<br/><b>x_t ∈ ℝ</b>"]
            PRED["Real-Time Forecast<br/><b>x̂[t+r] = W_t^T φ_t</b>"]
            ERR["Innovation Error<br/><b>e_t = x_t - x̂_t</b>"]
            WOODBURY["Woodbury RLS Update<br/><b>A_t^(-1) Rank-1 Step</b>"]
            WT["Updated Weights<br/><b>W_t ∈ ℝ^((F+2D) × H)</b>"]

            IN --> PRED --> ERR --> WOODBURY --> WT
        end
    end

    PHI ==> ACC
    W0 -. "Seed Weights W_0" .-> PRED
```

---

## 1. Closed-Form Streaming Forecasting Architecture ($S^2\text{-RLS}$)

The system operates across two distinct phases: **Phase 1: Warm-Start Initialization** and **Phase 2: Real-Time Woodbury RLS Streaming**.

![Closed-Form Streaming Pipeline](file:///C:/Projects/Application%20of%20Core%20Slover/figures/01_schematics/pipeline.png)

---

## 2. Mathematical Mechanics of Online Woodbury RLS Update

Demonstrates the exact rank-1 update of the inverse covariance matrix $\mathbf{A}_t^{-1}$ using the Woodbury matrix identity, eliminating $O(F^3)$ matrix inversions at streaming runtime.

![Woodbury RLS Update](file:///C:/Projects/Application%20of%20Core%20Slover/figures/01_schematics/rls_woodbury_update.png)

---

## 3. $S^2$ Linear Recurrent Unit ($S^2$-LRU) Context Bank Architecture

Integrates a parallel bank of complex diagonal recurrent units ($S^2$-LRU) to capture infinite-horizon temporal dynamics and multi-scale periodicities.

![S2-LRU Context Bank](file:///C:/Projects/Application%20of%20Core%20Slover/figures/01_schematics/lru_bank.png)

---

## 4. Causal Multi-Scale Feature Vector Decomposition ($F = 112$)

Details the sub-vector structure of the zero-lookahead feature vector $\phi(t) \in \mathbb{R}^{112}$. Raw inputs and output vectors are rendered in white fill / black stroke, with transformation modules highlighted in golden amber.

![Feature Map Decomposition](file:///C:/Projects/Application%20of%20Core%20Slover/figures/01_schematics/feature_map.png)

---

## 5. Strict Zero-Data-Leakage Evaluation Protocol Timeline

Demonstrates strict causal boundaries ensuring zero lookahead bias or data contamination.

![Protocol Timeline](file:///C:/Projects/Application%20of%20Core%20Slover/figures/01_schematics/protocol_timeline.png)

---

## 6. Unified Closed-Form Engine Across LTSF & Scientific Chaos

Illustrates how a single algorithmic core engine solves both real-world forecasting and complex non-linear physics.

![Chaos & LTSF Core Engine](file:///C:/Projects/Application%20of%20Core%20Slover/figures/01_schematics/chaos_core.png)

---

## 7. Iterative Backpropagation vs. Exact One-Shot Closed-Form Solving

Side-by-side comparison between gradient descent deep learning and our algebraic closed-form solver.

![Closed Form vs Backprop](file:///C:/Projects/Application%20of%20Core%20Slover/figures/01_schematics/closed_form_vs_backprop.png)

---

### Summary Comparison Table

| Metric / Property | Conventional Deep Learning (SGD / Backprop) | Ours: Closed-Form $S^2\text{-RLS}$ Engine |
| :--- | :---: | :---: |
| **Training Paradigm** | Iterative Gradient Descent ($K = 100+$ Epochs) | **Exact One-Shot Algebraic Solve ($O(D^3)$)** |
| **Streaming Adaptation** | Retraining / Fine-tuning ($O(K \cdot N \cdot |\Theta|)$) | **Real-Time Woodbury RLS Update ($O(D^2)$)** |
| **Convergence Guarantee** | Stochastic, subject to local minima & gradient vanishing | **Deterministic Global Minimum Solution** |
| **VRAM Memory Peak** | High (stores computation graph & gradients) | **Ultra-Low ($O(D^2)$ covariance matrix size)** |
| **Hyperparameters** | Learning rate, momentum, weight decay, epoch count | **Single Ridge Regularization parameter $\lambda$** |

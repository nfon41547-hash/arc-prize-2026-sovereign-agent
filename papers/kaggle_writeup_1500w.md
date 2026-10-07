# Sovereign Autonomous Hyper-Cortex: High-Throughput Test-Time Reasoning, Topological Invariance, and Grounded Multi-Agent Search for ARC-AGI-3

**Team:** bkk  
**Track:** ARC Prize 2026 — Paper Track  
**Submission Match:** ARC-AGI-3 Prediction Track (`bang1850/arc-agi-3-starter-kernel-v32-profile-3`)  
**Open-Source Repository:** [github.com/nfon41547-hash/arc-prize-2026-sovereign-agent](https://github.com/nfon41547-hash/arc-prize-2026-sovereign-agent)  
**Hardware Profile:** NVIDIA RTX Pro 6000 (Blackwell 96GB)  

---

## 1. Executive Summary & Verified Results
Solving interactive, multi-stage environments in ARC-AGI-3 requires two complementary capabilities: **extreme test-time inference throughput** and **mathematically grounded abstract visual reasoning**. Pure game-bot heuristics (e.g., greedy spatial centroid clustering) degrade generalization on combinatorial ARC tasks, while uncalibrated LLM exploration suffers from context bloat and fatal state revisitation.

We introduce the **Sovereign Autonomous Hyper-Cortex**, an open-source architecture that bridges high-throughput speculative serving with non-greedy topological abstraction:
- **Verified Benchmark Performance:** On the 25 public ARC-AGI-3 test environments, our system achieves a **Mean Score of 37.91**, securing **100% full solves on 6 complex environments** (`ar25-0c556536`, `cd82-fb555c5d`, `lp85-305b61c3`, `sb26-7fbdac44`, and others) and solving deep multi-level progressions (up to level 8/8).
- **Generation Speed:** Sustained generation throughput of **390–470 tokens/second** using SGLang with `fp8_e4m3` KV-cache, FlashInfer attention kernels, and Next-N MTP Speculative Decoding on an INT4 quantized foundation model.
- **Zero-Waste Execution:** 100% deterministic test coverage across 180 unit tests with strict Kolmogorov Minimum Description Length (MDL) gating.

---

## 2. Theoretical Framework: Why the Architecture Works

```text
               ┌────────────────────────────────────────────────────────┐
               │              Input State Observation S_t               │
               └───────────────────────────┬────────────────────────────┘
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                ▼                                                     ▼
    ┌───────────────────────┐                             ┌───────────────────────┐
    │  D4-Canonical Recall  │ (Exact Invariant Match)     │   Active Kinematics   │
    │  Trajectory in 0 Tok  │                             │  Player/Goal Grounding│
    └───────────────────────┘                             └───────────┬───────────┘
                                                                      │
                                                                      ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │               Multi-Hypothesis Beam & Agent-Q MCTS Search                       │
    │  Q_critique = tanh( 2.5 * Delta_MDL + 2.0 * Delta_D4 + 1.0 * Delta_Entropy )     │
    │  Pruning: Loop Oscillation (-0.85), Wall Collision (-0.75), Dead Click (-0.70)  │
    └─────────────────────────────────────────┬───────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │          High-Throughput SGLang Engine (390-470 tok/s on RTX Pro 6000)         │
    │  Model: Qwen3.8-Flash-Next W4A16 AutoRound + Albucino MTP Draft (NEXTN)         │
    └─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Non-Greedy Topological Invariance vs. Game-Bot Biases
A critical flaw in standard reinforcement learning applied to ARC is assuming physical spatial proximity correlates with task progress. In ARC, goals often demand global symmetry completion, parity flipping, or discrete topological containment. We replace spatial distance heuristics with a **Topological Information Value Function**:

1. **Kolmogorov Minimum Description Length (MDL) Gain:**  
   $$\Delta\text{MDL}(S_t, S_{t+1}) = \text{MDL}(S_t) - \text{MDL}(S_{t+1})$$  
   Where $\text{MDL}(S)$ measures 2D run-length complexity and discrete block periodicity. An action is rewarded if and only if it simplifies programmatic state regularity.

2. **$D_4$ Dihedral Symmetry Group Equivariance:**  
   $$\text{Sym}_{D_4}(S) = \frac{1}{5}\left[ \mu(S == S^{\text{flip\_h}}) + \mu(S == S^{\text{flip\_v}}) + \mu(S == S^T) + \mu(S == S^{\text{diag2}}) + \mu(S == \text{rot}_{180}(S)) \right]$$  
   Actions that restore broken visual symmetries receive direct algebraic confirmation.

3. **Active Kinematic Grounding (Zero Hardcoded Entity Guesses):**  
   Instead of hardcoding "preferred player colors", the system analyzes empirical state transitions $(S_t, a_t, S_{t+1})$. If a connected component translates by exactly $\Delta(a_t)$ under directional actions $a \in \{1, 2, 3, 4\}$, it is inductively registered as the true controlled agent with 100% confidence.

---

## 3. High-Throughput Hardware & Serving Architecture

To explore multi-step decision trees within Kaggle's 9-hour operational envelope, the inference stack was engineered for maximum compute efficiency:

- **Foundation Model:** `Qwen3.8-Flash-Next` quantized to INT4 via Intel AutoRound ($W4A16$).
- **Draft Model:** `albucino-qwen3-8-flash-next-drafter` MTP (Multi-Token Prediction) running NEXTN speculative decoding.
- **SGLang Engine Optimization:**
  - Memory Headroom: `--mem-fraction-static 0.93` prevents Radix Cache thrashing during multi-hour runs.
  - Latency Smoothing: `--chunked-prefill-size 4096` ensures prompt prefill never blocks the event loop for $>5\text{s}$.
  - Graph Optimization: `--cuda-graph-bs-decode 1 2 4` minimizes kernel launch overhead on Ada/Blackwell SM architecture.
- **Client Resilience:** Enforced a $45.0\text{s}$ minimum HTTP timeout floor, eliminating artificial `gave_up` failures on late-game states.

---

## 4. Multi-Agent Reasoning & Search Methodology

### 4.1 Grounded MPC Multi-Agent Debate & Belief Books
Open-ended multi-agent deliberation often degenerates into circular wandering under tight token budgets. Our framework implements **Grounded MPC Debate**:
- Communicative speech acts are restricted to four typed operators: `PROPOSE`, `CRITIQUE`, `REVISE`, and `VOTE`.
- Each subagent maintains an explicit Theory-of-Mind (ToM) belief state $B_i(s)$ tracking color-object affordances.
- Consensus terminates early upon reaching supermajority quorum ($\theta \ge 0.60$), saving up to 70% of reasoning tokens.

### 4.2 Agent-Instruct Autonomous Steering & AGent Unanswerability Filter
Inspired by Crispino et al. (2023) and Tran et al. (2023):
- An external orchestrator decomposes tasks into 4 discrete cognitive phases (Invariant Perception $\to$ Topological Classification $\to$ Causal Hypothesis $\to$ Physical Affordance).
- Epistemic confidence divergence $V(h) = c_a \alpha^{n_a} - c_u \beta^{n_u}$ prunes unanswerable / hallucinatory hypotheses before physical actions are dispatched.

### 4.3 D4-Canonical Invariant Indexing (Agent-KB)
When a level is solved, its trajectory is indexed under its 8-fold dihedral invariant canonical key $\text{hash}(D_4(S))$. If any symmetrical variant is encountered in subsequent games or levels, the exact trajectory is replayed in **0 tokens and $<0.1\text{ms}$**, preserving compute for genuinely novel puzzles.

---

## 5. Empirical Results & Detailed Ablation

### 5.1 Public ARC-AGI-3 Benchmark Breakdown (Verified 25 Games)
| Environment ID | Score (%) | Levels Solved | Actions Taken | Generated Tokens | Primary Mechanism |
|---|---|---|---|---|---|
| `ar25-0c556536` | **100.00** | 8.0 / 8 | 329 | 160,047 | Active Kinematics + BFS |
| `cd82-fb555c5d` | **100.00** | 6.0 / 6 | 126 | 91,032 | D4 Invariant Completion |
| `lp85-305b61c3` | **100.00** | 8.0 / 8 | 104 | 132,645 | Topological Loop Fill |
| `sb26-7fbdac44` | **100.00** | 8.0 / 8 | 132 | 32,648 | Agent-Instruct Decomposition |
| `tr87-cd924810` | **66.18** | 5.0 / 6 | 347 | 177,340 | Grounded MPC Debate |
| `ft09-0d8bbf25` | **47.62** | 4.0 / 6 | 88 | 120,920 | MDL Complexity Reduction |
| `sc25-635fd71a` | **47.62** | 4.0 / 6 | 166 | 141,143 | Multi-Hypothesis Beam |
| `re86-8af5384d` | **41.67** | 5.0 / 8 | 399 | 172,008 | Reversible Probe Search |
| **Full 25-Game Mean** | **37.91** | **Overall** | **Avg: 128 act** | **Throughput: 420 t/s** | **Sovereign Hyper-Cortex** |

### 5.2 Ablation Study of Core Components
| Configuration | Mean Score | Zero-Score Games | Avg Turn Latency |
|---|---|---|---|
| Baseline LLM (No Search / Naive Centroid) | 12.31 | 14 / 25 | 1.84s |
| + SGLang Speculative Drafter (NEXTN) | 22.40 | 9 / 25 | **0.32s** |
| + Anti-Oscillation & Wall-Hit Pruning | 29.85 | 6 / 25 | 0.35s |
| + MDL & D4 Dihedral Value Function | 34.60 | 4 / 25 | 0.38s |
| **+ Active Kinematics & 45s Timeout Floor (Full)** | **37.91** | **2 / 25** | **0.36s** |

---

## 6. Limitations & Failure Analysis
1. **Long-Tail Non-Kinematic Puzzles:** Environments with hidden continuous state machines (e.g. `g50t-5849a774`) require deeper micro-experiments before player identities emerge.
2. **Late-Run Memory Fragmentation:** SGLang instances running over 3+ hours require proactive Radix Cache compaction to prevent latency degradation.

---

## 7. Conclusion & Roadmap to ARC AGI
By replacing fragile spatial heuristics with **information-theoretic topological invariants** and decoupling fast speculative inference from formal multi-agent debate, the Sovereign Hyper-Cortex proves that test-time compute can be efficiently converted into genuine abstract generalization. All code, prompts, configs, and test harnesses are open-sourced under the MIT license to accelerate community progress toward AGI.

---
**Word Count:** 1,280 words (Compliant with $\le 1,500$ word constraint).

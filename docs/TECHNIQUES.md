# ARC-AGI-3 SOTA Techniques — Complete Reference

## 16 Techniques Wired into the Decision Chain

### 1. SovereignSkillsMatrix (`skills.py`)
**Type**: Grandmaster cognitive skills  
**Skills**: 11 total
- SpatialRaycastNavigation — multi-directional raycasts for walls/corridors
- TopologicalTunneling — 1-pixel bottlenecks/choke-points
- TetrisGravityPhysics — falling-block dynamics
- ChromaticParityTransformation — chess-like parity lattices
- InvariantPatternExtrapolation — horizontal periodicity continuation
- GeodesicTopologicalCentroidTargeter — object center-of-mass clicks
- DihedralGroupD4SymmetryCompleter — mirror-asymmetry repair
- BoundaryEnclosedVoidInfill — enclosed-cavity clicks
- DynamicOnlineEvolvedSkills — FullAuto runtime rune learner
- GoalAcquisitionRarestProbe — rarest-color salient probe
- LatticeVerticalContinuation — vertical periodicity continuation

### 2. AbstractionSkillRegistry (`abstraction_skill_registry.py`)
**Type**: Game archetype prior (advisory)  
**Role**: Read-only, provenance-carrying abstraction lookup. Never fires actions directly.

### 3. arc_color_transform (`arc_color_transform.py`)
**Type**: Color pattern detection  
**Capabilities**:
- Rare color centroid detection
- Diagonal color swap detection
- Color gradient completion

### 4. arc_object_tracker (`arc_object_tracker.py`)
**Type**: Multi-object tracking  
**Capabilities**:
- Connected component labeling (4-connectivity)
- Smallest object detection
- Movement lead prediction
- Containment detection

### 5. arc_pattern_completion (`arc_pattern_completion.py`)
**Type**: Pattern completion  
**Capabilities**:
- Row/column gap fill
- Horizontal/vertical/diagonal symmetry repair

### 6. AlgebraicPlanningEngine (APE) (`algebraic_planning_engine.py`)
**Type**: Exact morphism discovery  
**Core idea**: ARC tasks live in the Category Grid. Discovers morphism f: TrainIn → TrainOut by:
1. Enumerating candidate morphisms (bounded word length, A* search)
2. Verifying against ALL training pairs (exact match)
3. Proving generalization via algebraic invariants
4. Self-upgrading: verified morphisms add new laws + heuristics

### 7. CausalChainReasoner (`causal_chain_reasoner.py`)
**Type**: Causal chain confidence  
**Core idea**: Models thinking as CAUSE → EFFECT lattice over abstract grid transforms.
- Chain prior: PRODUCT of per-step confidences × empirical transition agreement
- Multiplicative self-compounding: verified transitions raise edge weights exponentially
- Bounded k-step chains, log-domain sums

### 8. FusionSupremacyCore (`fusion_supremacy.py`)
**Type**: Top-5 techniques fused  
**Fused techniques**:
1. DeadSignatureLedger — dead type suppression
2. ObjectHashTracker — position-independent object identity
3. ContainmentTopology — nested cavity detection
4. Fusion click proposal — smallest non-dead interactive component

### 9. SpectralTopologicalReasoning (DSTS) (`spectral_topological_reasoning.py`)
**Type**: D4-canonical spectral-topological signature  
**Invariants**:
1. Per-color Betti-b0 + Euler characteristic
2. Fiedler value lambda2 (algebraic connectivity)
3. Sorted 16x16 cross-color adjacency spectrum

**Properties**: D4-invariant, interior-shift stable, strictly separates baseline-blind pairs

### 10. RealtimeAbstractCortex (`realtime_abstract_cortex.py`)
**Type**: Pure-abstract invariants + empirical self-upgrade  
**Levels**:
- L0: D4 canonical hash, Betti-b0, entropy, RLE bound, centroid, bbox, shift/color-perm votes
- L1: One-shot rule induction (identity/rot90/flip/transpose/shift/color_perm/toggle/fill_rect)
- L2: Empirical self-upgrade (hypothesis ledger, fatal shield, stagnation breaker, RHAE economy)

### 11. WorldModelSimulator (`world_model_simulator.py`)
**Type**: Multi-entity dynamics simulation  
**Physics**:
- Multi-block chain cascades (Sokoban)
- Portal/wormhole warp
- Symmetry/reflection propagation
- Switch/pressure plate barrier activation
- Centroid click affordances + color inversion

### 12. MCTSPlanner (`mcts_planner.py`)
**Type**: MCTS with Zobrist hashing  
**Features**:
- Terminal-horizon planning
- Dynamic horizon expansion (up to 40 steps)
- Geodesic wavefront tunneling
- Transposition graph caching + deadlock detection
- Macro-action enqueue for RHAE efficiency

### 13. ObjectSegmentation (`object_segmentation.py`)
**Type**: Exact object segmentation  
**Features**:
- Topological containment + adjacency
- Translation-invariant hash
- Exact nesting children
- Cross-frame matching

### 14. HypothesisLedger (`hypothesis_ledger.py`)
**Type**: Propose → predict → observe → confirm/refute  
**Features**:
- Laplace-smoothed confirm rate
- Shadow-mode telemetry for denied tiers
- Bounded (64 games, 64 tags)

### 15. FluxSearch (Xi-FLUX) (`flux_search.py`)
**Type**: Reversible information-flux search  
**Features**:
- Virtual branches via world-model surrogate
- Boundary principle: PRUNE when U_a < L_best - delta
- Pareto annihilation for dominated actions
- Stop certificate: LCB(a*) > max UCB + margin

### 16. TurnMemo (`turn_memo.py`)
**Type**: Per-turn shared memo  
**Role**: Memoizes segment(grid) by (grid-shape, fingerprint) so every tier reads the same object graph.

## Design Principles

- **Fail-open**: Any exception → empty list (other tiers take over)
- **Intrinsic only**: No game-ID-keyed manuals, no memorization
- **Bounded**: Max 12 proposals per turn, O(n) per turn
- **Novelty gate**: Only fire on NEW states
- **Veto**: Never override legal+safe actions
- **Calibration**: Per-technique Laplace smoothing

## Integration

All 16 techniques feed into `skill_orchestrator.evaluate_skills()` which returns a prioritized proposal list. The TAAF hook (`taaf_stepenv_hook.py`) calls this every turn and substitutes the highest-confidence proposal that passes the gate+veto law.

# Melvin System Rules and Definitions

This document extracts all rules, definitions, constants, and configuration parameters from `melvin.c` and `melvin.h` that define how the Melvin system operates.

## System Philosophy

**Core Principle**: Indirect rules that directly affect performance
- The system contains NO explicit intelligence - only mechanical physics rules
- Intelligence emerges from how the graph responds to these rules under pressure
- Rules are minimal physics laws that the graph self-organizes around

**Indirect vs Direct Rules**:
- **Indirect**: Don't say "delete useless nodes" → weights decay to zero (indirect pruning)
- **Indirect**: Don't say "create patterns" → similarity creates pressure (indirect formation)
- **Indirect**: Don't say "avoid chaos" → size penalty creates selectivity (indirect constraint)
- **Direct**: Learning rule (Δw = η * a * err) → directly reduces prediction error
- **Direct**: Simplicity score (penalize size, reward compression) → directly minimizes energy
- **Direct**: Pattern matching → directly enables compression (less storage, same function)

**Energy Loop**: A complete circle of cause and effect
- Energy flows: Input → Process → Output → Input → ...
- Creates internal cause-and-effect cycles where the system can "think" by having its own outputs become inputs

## Core Definitions

### Node Kinds
- `NODE_KIND_BLANK` (0): Placeholder for variables/data in patterns
- `NODE_KIND_DATA` (1): Data nodes containing byte values
- `NODE_KIND_PATTERN_ROOT` (2): Root node of a pattern structure
- `NODE_KIND_CONTROL` (3): Control nodes that execute MC functions
- `NODE_KIND_TAG` (4): Tag nodes for categorization
- `NODE_KIND_META` (5): Meta nodes for system channels (FBIN, FBOUT, REWD)

### Edge Flags
- `EDGE_FLAG_ACTIVE` (1 << 0): Edge is active
- `EDGE_FLAG_SEQ` (1 << 1): Sequence edge (temporal ordering)
- `EDGE_FLAG_BIND` (1 << 2): Binding edge (connects related nodes)
- `EDGE_FLAG_CONTROL` (1 << 3): Control edge (affects execution)
- `EDGE_FLAG_ROLE` (1 << 4): Role edge (defines node role in pattern)
- `EDGE_FLAG_REL` (1 << 5): Relation edge (defines relationship)
- `EDGE_FLAG_CHAN` (1 << 6): Channel edge (connects to I/O channels)
- `EDGE_FLAG_PATTERN` (1 << 7): Pattern edge (part of pattern structure)
- `EDGE_FLAG_MODULE_BYTES` (1 << 8): Module bytes edge

### Channel Types
- `CH_TEXT` (1): Text input/output channel
- `CH_SENSOR` (2): Sensor input channel
- `CH_MOTOR` (3): Motor output channel
- `CH_VISION` (4): Vision input channel
- `CH_REWARD` (4): Reward channel (reuses CH_VISION value)

### Role Flags (for pattern edges)
- `ROLE_SEQ_FIRST` (1), `ROLE_SEQ_SECOND` (2), `ROLE_SEQ_THIRD` (3), `ROLE_SEQ_FOURTH` (4)
- `ROLE_COND` (5), `ROLE_THEN` (6), `ROLE_ELSE` (7), `ROLE_BODY` (8)
- `ROLE_INPUT` (9), `ROLE_OUTPUT` (10), `ROLE_CONTROL` (11)
- `ROLE_SUGGEST` (12), `ROLE_BLANK` (13), `ROLE_SLOT` (14)
- `ROLE_LHS` (15), `ROLE_RHS` (16), `ROLE_OP` (17)

### System Magic Numbers
- `MELVIN_MAGIC`: 0x4D4C564E ("MLVN")
- `MELVIN_VERSION`: 2

### Meta Node Values (hex)
- `0x4642494E` ("FBIN"): Feedback input channel
- `0x46424F55` ("FBOUT"): Feedback output channel
- `0x52455744` ("REWD"): Reward channel

## Constants and Limits

### Initial Capacities
- `INITIAL_NODE_CAPACITY`: 65536 nodes
- `INITIAL_EDGE_CAPACITY`: 262144 edges
- Growth: Graph grows by 50% when capacity is exceeded

### MC Function Limits
- `MAX_MC_FUNCS`: 256 maximum MC functions

### Buffer Limits
- Active node buffer: Grows dynamically, starts at 10000
- Path buffer capacity: 1000 paths
- Pattern search: Up to 5000 nodes per tick (round-robin sampling)
- Path length: Maximum 8 nodes per path

### Threshold Constants
- Similarity threshold for pattern formation: 0.5 (50% similarity)
- Pattern match threshold: 0.6 (60% match score)
- Active node threshold (pattern matching): 0.2 activation
- Pattern activation threshold: 0.3 activation

## Graph-Configurable Parameters

These parameters are stored in the BrainHeader and can be modified by the graph via MC functions. Default values are set on initialization:

### Default Values
- `edge_activation_threshold`: 0.3 (activation threshold for edge creation)
- `mc_execution_threshold`: 0.3 (activation threshold for MC node execution)
- `decay_factor`: 0.99 (activation decay factor per tick)
- `edge_creation_score_threshold`: 0.15 (minimum score to create edge)
- `learning_rate`: 0.01 (weight update learning rate)
- `alpha_blend`: 0.3 (activation blending factor)

### MC Functions for Parameter Updates
- `mc_set_edge_threshold`: Updates edge activation threshold (0.0-1.0 range)
- `mc_set_mc_threshold`: Updates MC execution threshold (0.0-1.0 range)
- `mc_set_decay_factor`: Updates decay factor (0.0-1.0 range)
- `mc_set_edge_score_threshold`: Updates edge creation score threshold (0.0-1.0 range)
- `mc_set_learning_rate`: Updates learning rate (0.0-1.0 range)
- `mc_set_alpha_blend`: Updates alpha blend factor (0.0-1.0 range)

## Learning Rules

### Weight Update Rule
```
Δw = η * a_src * err_dst
```
Where:
- `η` (eta): Learning rate (default: 0.001 in update_edges, 0.01 configurable)
- `a_src`: Source node activation
- `err_dst`: Destination node prediction error

**Natural Pruning**: Edges with zero weight naturally have no effect (effectively pruned)
- Weights clamp to range [-W_MAX, W_MAX] where W_MAX = 10.0

### Eligibility Trace
```
elig = λ * elig + a_src
```
Where `λ` (lambda) = 0.9 (eligibility trace decay)

### Unused Edge Decay
- If edge source activation < 0.01 and usage_count == 0:
  - Weight decays: `w *= 0.99` (very slow decay, can recover if needed)

## Edge Creation Rules

### Co-Activation Condition
Edges are created between nodes when:
1. Both nodes are co-activating (activation > edge_activation_threshold)
2. Edge doesn't already exist (checked via adjacency list)
3. Creation score meets threshold (see below)

### Initial Edge Weight
- Weight = 0.1 × src_activation × dst_activation
- Initial flags: `EDGE_FLAG_BIND` (binds nodes together)

### Edge Creation Score Calculation
```
creation_score = co_activation
               + error_pressure × 0.3
               + pattern_bonus
               + reliability_bonus
               + simplicity_bonus
               + complexity_penalty
```

Where:
- `co_activation` = src_activation × dst_activation
- `error_pressure` = (src_error + dst_error) × 0.5
  - Error = 1.0 - reliability
- `pattern_bonus`:
  - Both nodes in pattern: +0.3
  - One node in pattern: +0.15
- `reliability_bonus` = (src_reliability + dst_reliability) × 0.5 × 0.2
- `simplicity_bonus` = g_simplicity_pressure × 0.1
- `complexity_penalty`: See Complexity Rules section

### Dynamic Threshold
Base threshold: `edge_creation_score_threshold` (default: 0.15)

Threshold multipliers based on complexity:
- Edge density > 100 AND pattern ratio < 0.01: ×2.0 (much more selective)
- Edge density > 50 AND pattern ratio < 0.05: ×1.5 (more selective)
- Capacity usage > 0.9: ×1.8 (very selective near capacity)
- Capacity usage > 0.7: ×1.3 (selective when high capacity)
- High prediction error pressure (< -0.2): ×0.7 (more permissive)
- High simplicity pressure (> 0.1): ×1.3 (more selective)

Edge is created if: `creation_score >= threshold`

### Sequence Edge Creation
- New nodes automatically create bidirectional sequence edges to previous node
- Forward edge: previous → current (weight 0.5, EDGE_FLAG_SEQ)
- Backward edge: current → previous (weight 0.3, EDGE_FLAG_SEQ)

## Node Activation Rules

### New Node Initialization
- Activation: 1.0 (strong activation pulse for co-activation)
- Bias: 0.1 (default prediction potential)
- Decay: 0.01 (slow decay, persists across ticks)

### Activation Prediction
```
predicted_a[i] = sigmoid(sum(weight × src_activation) + bias)
```
Clamped to [0, 1] range

### Activation Update (apply_environment)
1. If activation > 0.9: Preserve strong activation (new node pulse)
2. If activation < 0.01: Use prediction directly
3. Otherwise: Blend with prediction using alpha:
   ```
   a = alpha × predicted + (1 - alpha) × current_a
   ```
   Where alpha = `alpha_blend` (default: 0.3)

### Activation Decay
Called AFTER edge creation to preserve co-activation window:
```
a *= (1 - node_decay × decay_factor)
```
Where:
- `node_decay`: Node-specific decay (default: 0.01 if not set)
- `decay_factor`: Global decay factor (default: 0.99)

Clamped to [0, 1] range

## Pattern Formation Rules

### Pattern Induction Conditions
Pattern formation is triggered when:
1. Force formation (edge density > 100 AND pattern ratio < 0.01) OR
2. Force formation (edge density > 50 AND pattern ratio < 0.05) OR
3. Large graph (nodes > 100) with pattern ratio < 0.1 OR
4. High prediction error pressure (< -0.001) OR
5. High simplicity pressure (> 0.0001)

Skipped if: Very small graph (nodes < 100) AND no pressure AND not forcing

### Path Similarity Detection
- Paths are sequences of up to 8 connected nodes
- Similarity threshold: 0.5 (50% common nodes)
- Pattern created when two paths have similarity >= threshold

### Pattern Structure
- Pattern root: `NODE_KIND_PATTERN_ROOT`
- Invariants: Common nodes (matched in both paths) → concrete edges
- Variants: Different nodes → BLANK placeholder edges
- Edges flagged with `EDGE_FLAG_PATTERN | EDGE_FLAG_BIND`
- Sequence positions flagged with `EDGE_FLAG_SEQ`

### Pattern Capacity Limits
- When capacity > 95%: Max 2 edges per pattern
- Otherwise: Max 8 edges per pattern
- Pattern creation skipped if capacity > 90% and can't grow

## Pattern Matching Rules

### Match Score Calculation
1. Find concrete anchors in pattern (must match exactly)
2. Find matching concrete nodes in local context (active nodes)
3. Grow match through constraint edges (SEQ, ROLE, REL)
4. Bind BLANK nodes to candidate concrete nodes
5. Score = (concrete_matches + blank_bindings) / match_count

### Match Threshold
- Pattern activates if match_score >= 0.6 (60%)

### Pattern Activation
- When pattern matches: activation = max(current_activation, match_score)
- Pattern edges strengthened: `weight += 0.01 × match_score`
- Max weight: 10.0

### Pattern Decay
- If pattern doesn't match: `activation *= 0.95` (5% decay per tick)

## Complexity and Bloat Rules

### Edge Density Calculation
```
edge_density = num_edges / num_nodes
```

### Pattern Ratio Calculation
```
pattern_ratio = num_patterns / num_nodes
```

### Complexity Penalty (Edge Creation)
- Edge density > 100 AND pattern ratio < 0.01:
  - Penalty = -0.5 × (edge_density / 1000.0)
- Edge density > 50 AND pattern ratio < 0.05:
  - Penalty = -0.2 × (edge_density / 500.0)
- Edge density > 20 AND pattern ratio < 0.1:
  - Penalty = -0.1

### Capacity Penalty (Edge Creation)
- Capacity usage > 0.9:
  - Penalty -= 0.3 × (capacity_usage - 0.9) × 10.0
- Capacity usage > 0.7:
  - Penalty -= 0.1 × (capacity_usage - 0.7) × 5.0

### Bloat Calculation (Simplicity Metrics)
Base bloat = 1.0 (no penalty)

- Edge density > 100 AND pattern_count < nodes × 0.01:
  - Bloat = edge_density / 10.0 (severe penalty)
- Edge density > 50 AND pattern_count < nodes × 0.05:
  - Bloat = edge_density / 20.0 (moderate penalty)
- Edge density > 20 AND pattern_count < nodes × 0.1:
  - Bloat = edge_density / 50.0 (small penalty)

Minimum bloat: 0.1 (prevent division by zero)

### Complexity Penalty (Reward System)
- Edge density > 100 AND pattern ratio < 0.01:
  - Reward -= 0.3 × (edge_density / 1000.0)
- Edge density > 50 AND pattern ratio < 0.05:
  - Reward -= 0.15 × (edge_density / 500.0)
- Edge density > 20 AND pattern ratio < 0.1:
  - Reward -= 0.05

### Capacity Penalty (Reward System)
- Capacity usage > 0.95: Reward -= 0.2 (severe)
- Capacity usage > 0.9: Reward -= 0.1 (moderate)
- Capacity usage > 0.8: Reward -= 0.05 (small)

## Simplicity Metrics and Objective Function

### Unified Global Efficiency Rule
```
Efficiency = (Output Production × Output Quality) / Bloat
```

### Output Production
- Pattern usage rate: ×10.0
- Episodic compression: ×5.0
- CPU/GPU success rate: ×15.0 (strong reward)
- CPU/GPU failure rate: -×10.0 (penalty)
- GPU operations: +5.0 bonus
- Storage operations: +3.0 bonus
- No outputs: -10.0 (heavy penalty)

### Output Quality
```
output_quality = 1.0 - min(avg_error, 1.0)
```
Where `avg_error = pred_error_total / num_nodes`

- If quality < 0.1: quality = -5.0 (penalty for inaccurate outputs)

### Simplicity Score
```
simplicity_score = efficiency + compression × 2.0 + size_penalty
```

Where:
- `efficiency` = (output_production × output_quality) / bloat
- `compression` = episodic_compression (pattern usage)
- `size_penalty` = total_size × -1e-7 (very small)

### Reward Signal
```
reward = simplicity_score × 0.01
```
Clamped to [-10.0, 10.0] range

## Reliability Rules

### Reliability Update
```
reliability = 0.99 × reliability + 0.01 × reliability_update
```

Where:
```
reliability_update = 1.0 - min(1.0, abs(error))
```

Clamped to [0, 1] range

### Success/Failure Tracking
- Error < 0.1: success_count++
- Error >= 0.1: failure_count++

## Feedback Loop Rules

### Feedback Input Channel (FBIN)
- Collects system outputs from previous tick
- Decay: 0.9 (moderate decay, feedback persists but fades)
- Connected to feedback output channel (FBOUT)

### Feedback Output Channel (FBOUT)
- Collects system activity:
  1. Pattern roots with activation > 0.3: ×0.3 contribution
  2. MC nodes with success_count > 0 and activation > 0.3: ×0.2 contribution
  3. Reward node: ×0.2 contribution
- Decay: 0.95 (slow decay, output persists)
- Output normalized and fed back to FBIN

### Reward Channel (REWD)
- Stores simplicity pressure as activation
- Decay: 0.95 (slow decay, reward persists)
- Connected to high-error nodes (error > 0.3) to drive learning
- Connection weight: error × 0.5

## Global Pressure Variables

### Simplicity Pressure
- Set by reward injection
- Positive = reward, Negative = penalty
- Used by edge creation and pattern induction
- Range: [-10.0, 10.0] (from reward signal)

### Prediction Error Pressure
- Computed from average prediction error
- High error = high negative pressure (pressure to reduce it)
- Range: [-1.0, 1.0] (normalized)
- Used by edge creation selectivity

## GPU Acceleration Rules

### GPU Usage Conditions
- Propagation: Graph has > 10,000 edges
- Error computation: Graph has > 10,000 edges
- Weight updates: Graph has > 50,000 edges

### GPU Functions (if available)
- `mc_gpu_propagate`: GPU-accelerated activation propagation
- `mc_gpu_compute_error`: GPU-accelerated error computation
- `mc_gpu_update_edges`: GPU-accelerated weight updates

Note: CPU fallback always available for correctness

## Tick Sequence

The main tick function executes in this order:

1. **Initialize**: Reset simplicity metrics for tick
2. **Input**: `ingest_input()` - External stimuli + internal feedback
3. **Propagate**: `propagate_predictions()` - Energy flows through network
4. **Activate**: `apply_environment()` - Predictions become actual activations
5. **Measure Error**: `compute_error()` - Compare prediction vs reality
6. **Learn**: 
   - `update_edges()` - Create/strengthen connections
   - `update_nodes_from_error()` - Update node reliability
7. **Decay**: `decay_activations()` - Energy slowly fades
8. **Act**: `run_mc_nodes()` - Execute machine code functions
9. **Discover**: `induce_patterns()` - Find patterns in repeated structures
10. **Recognize**: `match_patterns()` - Match patterns against current state
11. **Evaluate**: 
    - `sm_measure_complexity()` - Measure graph complexity
    - `sm_measure_patterns()` - Measure pattern compression
    - `sm_compute_objective()` - Compute simplicity score
12. **Reward**: `melvin_send_intrinsic_reward()` - Inject reward signal
13. **Output**: `emit_output()` - Collect system outputs (feeds back next tick)
14. **Log**: Debug information
15. **Increment**: tick++

## Special Rules

### Natural Pruning (Indirect)
- No explicit deletion
- Useless edges → weight decays to zero → no effect
- Useless nodes → never activate → never selected
- Size penalty creates pressure to avoid creating useless structures

### Sequence Edge Creation
- Every new node automatically creates bidirectional sequence edges to previous node
- Enables temporal learning and pattern formation

### Node Deduplication
- `find_or_create_node()` searches for existing node by hash and kind
- Prevents duplicates - nodes should be reused, not duplicated

### Graph Growth
- Grows dynamically when capacity is exceeded
- Growth factor: 50% increase or minimum needed, whichever is larger
- No hard limits - limited only by disk space

### MC Node Execution
- MC nodes execute when activation >= `mc_execution_threshold` (default: 0.3)
- Execution tracked as CPU/GPU communication for metrics
- Success/failure counts tracked for reliability

### Pattern Effect Execution
- Patterns can execute effects via BLANK → OUTPUT_CHANNEL connections
- Effect strength encoded in edge weight
- Effects modify channel activation or bias based on channel type

## Error Handling and Guards

### NaN/Infinity Guards
- All calculations check for NaN and infinity
- Invalid values set to 0.0
- Prevents calculation explosions

### Buffer Initialization
- New buffer memory initialized to zero
- Prevents garbage/NaN values in calculations

### Capacity Bounds Checking
- All node/edge accesses check bounds
- Graph growth checked before operations
- Graceful handling of capacity limits


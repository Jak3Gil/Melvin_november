# GPU Math Learning Paths

## Overview

The system can **learn to route to GPU EXEC nodes** for math operations through universal laws, without manual wiring.

## Two Levels of Learning

### 1. **Paths TO GPU EXEC Nodes** (LEARNED ✅)

The system discovers GPU EXEC nodes through:

- **Curiosity Law**: Connects cold GPU EXEC nodes to hot data regions
- **Free-Energy Laws**: GPU is cheaper (lower cost multiplier), so FE drops when using GPU
- **Edge Formation**: Co-activation and FE-drop bonding create edges to GPU EXEC

**Example:**
```
Data nodes (A, B) → [curiosity + FE laws] → GPU ADD EXEC node
```

No manual wiring needed - the system learns these paths automatically!

### 2. **GPU vs CPU Choice** (PARAM-CONTROLLED)

The decision to use GPU vs CPU is controlled by param nodes:
- `NODE_ID_PARAM_EXEC_GPU_ENABLED` - Enable GPU path (activation > 0.5)
- `NODE_ID_PARAM_EXEC_GPU_COST_MULTIPLIER` - GPU cost (default: 0.5x, making GPU cheaper)

This is a **tunable parameter**, not a hardcoded decision. The system can learn to prefer GPU by:
- Lower FE when using GPU (due to lower cost)
- Higher stability for GPU EXEC nodes (due to better efficiency)

## GPU Math Operations

### Real Arithmetic on GPU

The GPU can perform actual math operations:

1. **ADD**: `a + b` (scalar addition)
2. **MULTIPLY**: `a * b` (scalar multiplication)  
3. **SUBTRACT**: `a - b` (scalar subtraction)

These are implemented as CUDA kernels that run on the GPU, not just activation sums.

### How It Works

1. **EXEC node created** with GPU math operation type
2. **System learns paths** to GPU EXEC via curiosity + FE laws
3. **When EXEC triggers**, dispatch chooses GPU path (if enabled)
4. **GPU kernel executes** actual math operation
5. **Result returned** as energy, injected back into graph

## Learning Process

### Initial State
- GPU EXEC nodes: **Cold** (no incoming edges)
- Data nodes: **Hot** (high traffic from math queries)
- No connections between them

### After Learning
- **Curiosity** connects data nodes → GPU EXEC nodes
- **FE laws** strengthen edges (GPU is cheaper, so FE drops)
- **Co-activation** creates edges when data and GPU EXEC fire together
- **Result**: System routes math queries to GPU EXEC automatically!

## Example Flow

```
1. Math query arrives: "A=50, B=30, compute A+B"
2. Data nodes A and B activate
3. Curiosity law: "A and B are hot, GPU ADD is cold - connect them!"
4. Edge forms: A → GPU ADD EXEC
5. Edge forms: B → GPU ADD EXEC
6. Activation flows to GPU ADD EXEC
7. GPU ADD EXEC triggers (activation > threshold)
8. GPU kernel executes: 50 + 30 = 80
9. Result (80) injected as energy
10. Energy activates result node R
11. FE drops (GPU was cheaper than CPU)
12. Edge weights strengthen (FE-drop bonding)
13. System learns: "GPU ADD is efficient for this!"
```

## Key Insight

**The system doesn't need to be told which paths to use.**

It discovers them through:
- **Curiosity**: Explores unused capacity (cold GPU EXEC nodes)
- **Free-Energy**: Prefers lower-cost paths (GPU is cheaper)
- **Stability**: Reinforces efficient paths (GPU reduces FE)

All through **universal laws** - no special cases for GPU!

## Files

- `melvin_gpu_math.cu` - GPU math kernels (ADD, MULTIPLY, SUBTRACT)
- `test_gpu_math_learning.c` - Test that demonstrates path learning
- `melvin.c` - Dispatch logic (chooses GPU vs CPU based on params)

## Status

✅ **GPU can do real math** (not just activation sums)  
✅ **Paths to GPU EXEC are learned** (via curiosity + FE laws)  
✅ **No manual wiring needed** (system discovers paths automatically)  
✅ **GPU choice is param-controlled** (tunable, not hardcoded)

The system learns to use GPU for math operations through the same universal laws that govern all edge formation!


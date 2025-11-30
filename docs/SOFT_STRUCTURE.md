# Soft Structure Architecture

## Philosophy: Guiding Without Constraining

The soft structure provides **semantic scaffolding** that guides the graph's behavior while preserving its emergent, self-organizing properties. Unlike hard constraints, soft structure can be modified, overridden, or extended by the graph through UEL physics.

## Core Principles

1. **Structure guides, doesn't constrain**: Prebuilt layers are suggestions, not rules
2. **Graph can override**: UEL physics can reshape everything
3. **Emergence still possible**: Graph can create patterns beyond the scaffold
4. **Functional from day 1**: Prebuilt structure ensures basic I/O works
5. **Self-improving**: Graph learns to optimize the scaffold itself

## Port Range Semantics

Port ranges provide **suggested semantic regions** for different types of nodes. The graph can repurpose these or create new ports beyond these ranges.

### Standard Port Ranges

```
0-99:     INPUT PORTS
  - 0-9:   Text input ports (primary input channel)
  - 10-19: Sensor data ports (secondary input channels)
  - 20-29: Query/command ports (structured input)
  - 30-39: Feedback input ports (external feedback signals)
  - 40-99: Reserved for future input types

100-199:  OUTPUT PORTS
  - 100-109: Text output ports (primary output channel)
  - 110-119: Action output ports (motor/action commands)
  - 120-129: Prediction output ports (future predictions)
  - 130-139: Confidence output ports (uncertainty signals)
  - 140-199: Reserved for future output types

200-255:  CONTROL/MEMORY PORTS
  - 200-209: Working memory ports (short-term state)
  - 210-219: Long-term memory ports (persistent patterns)
  - 220-229: Attention ports (focus/selection)
  - 230-239: Meta-control ports (self-monitoring)
  - 240-255: Temporal ports (now, recent, future anchors)
```

**Key**: These are **suggestions**. Nodes beyond 255 can represent emergent concepts. The graph can repurpose any port if it finds better patterns.

## Node Propensity Hints

Each node has **soft type hints** that suggest its role, but these can be modified through learning:

### Input Propensity (0.0 - 1.0)
- **High** (0.7-1.0): Node tends to receive external input
- **Medium** (0.3-0.7): Node receives mixed input (internal + external)
- **Low** (0.0-0.3): Node primarily processes internal signals

**Initialization**: Nodes in input port ranges (0-99) start with high input_propensity (0.8)

### Output Propensity (0.0 - 1.0)
- **High** (0.7-1.0): Node tends to produce external output
- **Medium** (0.3-0.7): Node produces mixed output (internal + external)
- **Low** (0.0-0.3): Node primarily processes internal signals

**Initialization**: Nodes in output port ranges (100-199) start with high output_propensity (0.8)

### Memory Propensity (0.0 - 1.0)
- **High** (0.7-1.0): Node retains state over long periods (slow decay)
- **Medium** (0.3-0.7): Node retains state over medium periods
- **Low** (0.0-0.3): Node has fast decay (transient activation)

**Initialization**: Nodes in memory port ranges (200-255) start with high memory_propensity (0.8)

**Key**: These propensities are **learnable**. UEL physics can modify them based on actual usage patterns.

## Initial Edge Structure

The system creates **weak initial edges** that suggest data flow, but the graph can strengthen, weaken, or rewire them:

### Suggested Initial Connections

1. **Input → Working Memory** (ports 0-99 → 200-209)
   - Weak initial weights (0.1)
   - Graph decides which connections to strengthen

2. **Working Memory → Output** (ports 200-209 → 100-199)
   - Weak initial weights (0.1)
   - Graph learns optimal routing

3. **Output → Feedback** (ports 100-199 → 30-39)
   - Weak initial weights (0.05)
   - Graph learns to correlate outputs with feedback

4. **Memory → Memory** (ports 200-255 → 200-255)
   - Sparse connections (10% of possible)
   - Graph can create more as needed

**Key**: All initial edges are **weak suggestions**. Graph decides what to keep through UEL physics.

## Temporal Anchors

Special nodes that provide temporal scaffolding (graph can create its own temporal patterns):

- **Node 240 ("now")**: Always activated with current input (temporal anchor)
- **Node 241 ("recent")**: Tracks recent activations (sliding window)
- **Node 242 ("memory")**: Long-term patterns (slow decay)
- **Node 243 ("future")**: Prediction anchor (graph can use for anticipation)

**Key**: These are **anchors**, not constraints. Graph can create its own temporal nodes.

## Feedback Channels

Feedback nodes that the graph learns to interpret:

- **Node 30 ("positive_feedback")**: Activated when system performs well
- **Node 31 ("negative_feedback")**: Activated when system fails
- **Node 32 ("uncertainty")**: Activated when output is ambiguous
- **Node 33 ("curiosity_signal")**: Activated to encourage exploration

**Key**: Feedback is **information**, not hard constraints. Graph interprets and learns from it.

## Data Protocol Hints

### Text Input/Output
- Feed tokens sequentially through text input ports (0-9)
- Graph learns token sequences and patterns
- Output nodes (100-109) activate to produce text

### Structured Data
- Map fields to specific port ranges
- Graph learns field relationships
- Output through structured output ports

### Time Series
- Feed with temporal markers (use temporal anchors)
- Graph learns temporal patterns
- Can predict future values

## Initialization on Bootup

When a new brain file is created:

1. **Initialize port ranges**: Set propensity hints for nodes 0-255
2. **Create weak initial edges**: Suggest input→memory→output flow
3. **Initialize temporal anchors**: Set up now/recent/memory/future nodes
4. **Initialize feedback channels**: Set up feedback nodes
5. **Initialize data nodes**: Nodes 0-255 represent byte values (existing)

When an existing brain file is opened:

1. **Preserve learned structure**: Don't overwrite existing propensities
2. **Add missing structure**: Only initialize if not already present
3. **Respect graph evolution**: Graph may have repurposed ports

## Emergence Preservation

The soft structure **enables** emergence by:

1. **Providing starting points**: Graph has structure to build on
2. **Allowing modification**: All hints are learnable
3. **Permitting extension**: Graph can create nodes/ports beyond ranges
4. **Encouraging discovery**: Graph can find better patterns than scaffold

The graph should feel like it's **discovering** the prebuilt structure rather than being **constrained** by it.

## Implementation Notes

- Propensity values are stored as floats in node metadata (if space allows) or computed on-the-fly
- Port ranges are logical (not enforced in code)
- Initial edges are created with weak weights (0.05-0.1)
- All structure can be modified through UEL physics
- Graph can create emergent patterns beyond the scaffold


# How to Build Output Patterns in the Graph

## Overview

Output patterns connect **internal graph state** (pattern activations) to **OUTPUT nodes**. All intelligence is encoded in the graph structure - C code is just hardware.

## Architecture

```
Input Data → Patterns Match → Pattern Activation → Output Patterns (Edges) → OUTPUT Nodes → Output Bytes
```

## Step-by-Step Guide

### 1. Create Input Patterns

Define patterns that match your input:

```graph
> pattern "ab" { atom 0: byte 'a'; atom 1: byte 'b'; }
Created pattern "ab" (id: 9223372036854775808, 2 atoms)
```

### 2. Feed Data and Learn

Feed training data and let patterns learn:

```graph
> feed "abab"
Fed 4 bytes as DATA nodes

> learn episodes=5
Running 5 learning episodes...
  Episode 0: error=0.0000
  ...
```

Patterns will:
- Bind to matching data nodes
- Improve quality `q` based on consistency
- Build edges to data nodes they match

### 3. Create Output Patterns

Connect patterns to output nodes using `output_pattern`:

```graph
> output_pattern "ab_to_x" when pattern_id=9223372036854775808 output byte 'X'
Created output pattern "ab_to_x": pattern 9223372036854775808 -> output 'X' (node 4611686018427387904)
```

**What this does:**
- Creates an OUTPUT node for byte 'X' (if it doesn't exist)
- Creates an edge: `pattern → OUTPUT node` with weight 1.0
- When the pattern activates, it propagates activation to the output

### 4. Generate Output

Feed new input and generate output:

```graph
> feed "ab"
Fed 2 bytes as DATA nodes

> output
Output: X
```

**What happens:**
1. Patterns are activated based on match scores with recent data
2. Activation propagates through edges to OUTPUT nodes
3. OUTPUT nodes with activation > threshold emit their bytes

## Complete Example

```graph
# Step 1: Create pattern
> pattern "hello" { atom 0: byte 'h'; atom 1: byte 'e'; atom 2: byte 'l'; atom 3: byte 'l'; atom 4: byte 'o'; }
Created pattern "hello" (id: 9223372036854775808, 5 atoms)

# Step 2: Train
> feed "hello world"
Fed 11 bytes as DATA nodes
> learn episodes=5
Running 5 learning episodes...

# Step 3: Create output pattern
> output_pattern "hello_response" when pattern_id=9223372036854775808 output byte 'H'
Created output pattern "hello_response": pattern 9223372036854775808 -> output 'H'

# Step 4: Use it
> feed "hello"
Fed 5 bytes as DATA nodes
> output
Output: H
```

## How It Works (Graph-Native)

### Pattern Activation

When `output` is called:
1. `dsl_activate_matching_patterns()` checks all patterns against recent data
2. For each pattern, computes `pattern_match_score()` at various anchor positions
3. Sets pattern activation: `pattern.a = match_score * pattern.q`

### Activation Propagation

`graph_propagate()` spreads activation:
- Patterns with high `a` send activation to connected nodes
- Edges have weights `w` that scale the activation
- OUTPUT nodes accumulate activation from incoming edges

### Output Selection

OUTPUT nodes with `activation > threshold` emit their bytes.

## Key Principles

1. **All intelligence in graph** - No hardcoded "if input then output" in C
2. **Pattern-driven** - Outputs determined by which patterns are active
3. **Learnable** - Pattern qualities improve with experience
4. **Composable** - Multiple patterns can contribute to outputs
5. **Graph-native** - Outputs emerge from graph structure and dynamics

## Advanced: Multiple Outputs

You can create multiple output patterns for the same input pattern:

```graph
> pattern "yes" { atom 0: byte 'y'; atom 1: byte 'e'; atom 2: byte 's'; }
> output_pattern "yes1" when pattern_id=X output byte 'Y'
> output_pattern "yes2" when pattern_id=X output byte 'E'
> output_pattern "yes3" when pattern_id=X output byte 'S'

# When "yes" matches, outputs "YES"
```

## DSL Commands

- `pattern "name" { ... }` - Create input pattern
- `output_pattern "name" when pattern_id=X output byte 'Y'` - Connect pattern to output
- `feed "data"` - Feed input data
- `learn [episodes=N]` - Train patterns
- `output [threshold=X]` - Generate output from graph state
- `show outputs` - List all OUTPUT nodes and their activations

## Implementation Details

### OUTPUT Node Type

Added `NODE_OUTPUT` to `NodeKind`:
- Similar to `NODE_DATA` but represents outputs
- Has payload (byte value)
- Activation `a` determines if it fires

### Output Pattern Creation

`output_pattern` command:
1. Finds or creates OUTPUT node for the byte
2. Creates edge: `pattern_id → output_node_id` with weight 1.0
3. This edge is the "output pattern" - it's just a graph edge!

### Pattern Activation

`dsl_activate_matching_patterns()`:
- Checks patterns against recent data nodes
- Uses `pattern_match_score()` to compute match quality
- Sets `pattern.a = match_score * pattern.q`

### Output Generation

`output` command:
1. Activates matching patterns
2. Propagates activation (3 steps)
3. Collects OUTPUT nodes with `a > threshold`
4. Emits their byte values

## Next Steps

The system can be extended to support:
- Pattern activation rules (when to activate patterns)
- Output selection rules (which outputs to prefer)
- Multi-pattern outputs (combine multiple patterns)
- Output sequences (patterns that output multiple bytes)
- Weighted outputs (adjust edge weights for different outputs)

All of this stays in the graph - C remains pure hardware.


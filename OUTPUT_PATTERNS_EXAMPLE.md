# Output Patterns - How to Build Outputs in the Graph

## Concept

Output patterns connect **internal graph state** (pattern activations) to **OUTPUT nodes**. The graph decides what to output based on which patterns are active and their bindings to output nodes.

## Architecture

```
Input Data → Patterns Match → Pattern Activation → Output Patterns → OUTPUT Nodes → Output Bytes
```

All intelligence is in the graph:
- Patterns match input
- Pattern quality/activation determines strength
- Output patterns (edges) connect patterns to outputs
- Graph propagation activates outputs
- No hardcoded "if input then output" in C

## Step-by-Step Example

### 1. Create Input Patterns

```graph
> pattern "ab" { atom 0: byte 'a'; atom 1: byte 'b'; }
Created pattern "ab" (id: 9223372036854775808, 2 atoms)

> pattern "cd" { atom 0: byte 'c'; atom 1: byte 'd'; }
Created pattern "cd" (id: 9223372036854775809, 2 atoms)
```

### 2. Feed Data and Learn

```graph
> feed "abab"
Fed 4 bytes as DATA nodes

> learn episodes=10
Running 10 learning episodes...
  Episode 0: error=0.0000
  ...
```

### 3. Create Output Patterns

```graph
# When pattern "ab" is active, output 'X'
> output_pattern "ab_to_x" when pattern_id=9223372036854775808 output byte 'X'
Created output pattern "ab_to_x": pattern 9223372036854775808 -> output 'X'

# When pattern "cd" is active, output 'Y'
> output_pattern "cd_to_y" when pattern_id=9223372036854775809 output byte 'Y'
Created output pattern "cd_to_y": pattern 9223372036854775809 -> output 'Y'
```

### 4. Activate Patterns and Generate Output

```graph
# Feed input that matches patterns
> feed "ab"
Fed 2 bytes as DATA nodes

# Activate patterns (they match, so they get high activation)
# Then propagate to outputs
> output
Output: X
```

## How It Works

### Pattern Activation

When a pattern matches input:
1. Pattern's quality `q` increases (from learning)
2. Pattern activation `a` can be set based on match score
3. Pattern propagates activation to connected nodes

### Output Pattern Edges

Output patterns are **edges** from patterns to OUTPUT nodes:
- `pattern_id → output_node_id` with weight `w`
- When pattern is active, it sends activation to output
- Output node's activation = sum of incoming pattern activations × edge weights

### Output Generation

The `output` command:
1. Propagates activation through the graph
2. Collects OUTPUT nodes with activation > threshold
3. Emits their byte values

## Complete Example Session

```graph
> pattern "hello" { atom 0: byte 'h'; atom 1: byte 'e'; atom 2: byte 'l'; atom 3: byte 'l'; atom 4: byte 'o'; }
Created pattern "hello" (id: 9223372036854775808, 5 atoms)

> feed "hello world"
Fed 11 bytes as DATA nodes

> learn episodes=5
Running 5 learning episodes...
  Episode 0: error=0.0000
  ...

> show patterns
=== Patterns ===
Pattern id=9223372036854775808, q=0.9463

> output_pattern "hello_response" when pattern_id=9223372036854775808 output byte 'H'
Created output pattern "hello_response": pattern 9223372036854775808 -> output 'H'

> output_pattern "hello_response2" when pattern_id=9223372036854775808 output byte 'i'
Created output pattern "hello_response2": pattern 9223372036854775808 -> output 'i'

> feed "hello"
Fed 5 bytes as DATA nodes

> output
Output: Hi
```

## Advanced: Pattern-Based Output Selection

You can create multiple output patterns for the same input pattern, and the graph will activate all of them:

```graph
> pattern "yes" { atom 0: byte 'y'; atom 1: byte 'e'; atom 2: byte 's'; }
> output_pattern "yes1" when pattern_id=X output byte 'Y'
> output_pattern "yes2" when pattern_id=X output byte 'E'
> output_pattern "yes3" when pattern_id=X output byte 'S'

# When "yes" matches, outputs "YES"
```

## Key Principles

1. **All intelligence in graph** - No hardcoded output logic in C
2. **Pattern-driven** - Outputs determined by pattern activations
3. **Learnable** - Pattern qualities improve with experience
4. **Composable** - Multiple patterns can contribute to outputs
5. **Graph-native** - Outputs emerge from graph structure and dynamics

## Next Steps

The DSL can be extended to support:
- Pattern activation rules (when to activate patterns)
- Output selection rules (which outputs to prefer)
- Multi-pattern outputs (combine multiple patterns)
- Output sequences (patterns that output multiple bytes)

All of this stays in the graph - C remains pure hardware.


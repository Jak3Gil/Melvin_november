# Graph DSL Design - Programming the Graph as Data

## Concept

Instead of hardcoding graph intelligence in C, we create a **Domain-Specific Language (DSL)** that lets us:
- Write graph structures as code/data
- Feed this code into an interpreter
- The interpreter builds the graph structures
- Then we just feed raw data and let the graph learn

## Architecture

```
┌─────────────────┐
│  DSL Terminal   │  ← You write graph code here
│  (Interactive)  │
└────────┬────────┘
         │
         │ DSL code
         ▼
┌─────────────────┐
│  DSL Interpreter│  ← Parses DSL, calls graph API
│  (melvin_dsl.c) │
└────────┬────────┘
         │
         │ graph_add_pattern()
         │ graph_add_edge()
         │ etc.
         ▼
┌─────────────────┐
│  Graph Core     │  ← Pure hardware (melvin.c)
│  (melvin.c)     │
└─────────────────┘
```

## DSL Syntax Design

### Pattern Definition
```graph
// Define a pattern
pattern "ab" {
    atom 0: byte 'a'
    atom 1: byte 'b'
    initial_q: 0.5
}

// Define a pattern with wildcards
pattern "a*b" {
    atom 0: byte 'a'
    atom 1: blank
    atom 2: byte 'b'
    initial_q: 0.3
}

// Pattern from data (induction)
pattern from_data {
    scan: bigrams
    min_occurrences: 2
    initial_q: 0.4
}
```

### Learning Rules
```graph
// Define how patterns learn
rule pattern_quality {
    on: self_consistency_error
    update: q += lr * (1.0 - error)
    lr: 0.2
    clamp: [0.0, 1.0]
}

// Define edge weight updates
rule edge_weight {
    on: reconstruction_error
    update: w += -lr * src_activation * error
    lr: 0.1
}
```

### Pattern Application Rules
```graph
// When to apply patterns
rule pattern_match {
    threshold: 0.8
    selection: greedy_consistent
    bind_strength: 1.0
}
```

### Graph Structure
```graph
// Create node types
node_type VALUE {
    fields: [v: float]
    update_rule: value_propagation
}

// Create edges
edge pattern_to_data {
    from: PATTERN
    to: DATA
    initial_weight: 0.0
    learning_rule: edge_weight
}
```

## Example: Complete Graph Program

```graph
// graph_program.gdsl

// Define patterns to discover
patterns {
    // Explicit patterns
    pattern "ab" {
        atom 0: byte 'a'
        atom 1: byte 'b'
        initial_q: 0.5
    }
    
    // Induced patterns
    pattern from_data {
        scan: bigrams
        min_occurrences: 2
        initial_q: 0.4
    }
    
    pattern from_data {
        scan: trigrams
        min_occurrences: 2
        initial_q: 0.3
    }
}

// Learning configuration
learning {
    episodes: 10
    match_threshold: 0.8
    lr_q: 0.2
    lr_w: 0.1
}

// Output rules
output {
    format: json
    include: [patterns, bindings, compression]
}
```

## Terminal Interface

```bash
$ melvin-dsl
Melvin Graph DSL Terminal
> 

# Load a graph program
> load program.gdsl

# Or write inline
> pattern "hello" { atom 0: byte 'h'; atom 1: byte 'e'; ... }

# Feed data
> feed "hello world"

# Run learning
> learn episodes=10

# Query graph state
> show patterns
> show bindings pattern_id=123
> show stats

# Save graph
> save graph.bin
```

## Implementation Plan

### Phase 1: Basic DSL Parser
- Simple pattern definition syntax
- Parse and call `graph_add_pattern()`
- Terminal REPL

### Phase 2: Pattern Induction
- `pattern from_data { scan: bigrams }` → automatically creates patterns
- Moves intelligence from C to DSL

### Phase 3: Learning Rules
- Define learning rules in DSL
- Interpreter applies rules during learning

### Phase 4: Full Graph Programming
- Node types, edge types, update rules all in DSL
- C becomes pure hardware substrate

## Benefits

1. **Separation of Concerns**
   - C = hardware (frozen)
   - DSL = intelligence (mutable)
   - Easy to experiment with different graph structures

2. **Rapid Iteration**
   - Change graph logic without recompiling C
   - Test different pattern sets quickly
   - Version control graph programs

3. **Graph as Data**
   - Store graph programs as files
   - Share graph configurations
   - Reproduce experiments

4. **Intelligence in Graph**
   - All smarts encoded as graph structures
   - C never makes decisions
   - Pure graph-native intelligence


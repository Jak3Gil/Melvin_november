# Plan for Teaching Graph Outputs

## Current State

### What Exists:
1. **OUTPUT nodes** - Created via `graph_add_output_byte()`
2. **Pattern→OUTPUT edges** - Created in `melvin_learn_cli.c` (lines 637-644)
3. **Error feedback** - When no outputs, strengthens pattern→OUTPUT edges (lines 767-798)
4. **Output emission** - `graph_emit_output()` exists but **not called** in main loop
5. **Local learning rule** - `local_update_pattern_to_output()` exists

### What's Missing:
1. **`graph_emit_output()` not called** - Outputs are collected but never emitted
2. **Output creation is pattern-based** - Creates OUTPUT from pattern's first atom (may not be correct)
3. **No graph-native output learning** - Still in C code, not driven by LEARNER nodes
4. **No input→output examples** - System doesn't learn "when I see X, emit Y"

## The Plan: Graph-Native Output Learning

### Phase 1: Enable Output Emission (Immediate)
**Goal:** Actually emit bytes when OUTPUT nodes activate

1. **Call `graph_emit_output()` in main loop**
   - After propagation and `graph_run_local_rules()`
   - Before error feedback
   - Emit to stdout or collect in JSON

2. **Update JSON output**
   - Include emitted bytes in `graph_output` field
   - Show which OUTPUT nodes activated

### Phase 2: Graph-Native Output Learning (Core)
**Goal:** Move output learning into the graph, driven by LEARNER nodes

1. **Create OUTPUT nodes from training examples**
   - When training with input→output pairs, create OUTPUT nodes for target outputs
   - Store in EPISODE nodes: `(input_sequence, output_sequence)`

2. **Use LEARNER nodes for output learning**
   - LEARNER nodes connected to EPISODE nodes with input→output pairs
   - When LEARNER active, create pattern→OUTPUT edges based on:
     - Patterns that match the input
     - OUTPUT nodes that match the target output

3. **Error feedback via ERROR nodes**
   - When no output produced → activate ERROR node
   - ERROR node triggers learning rule to strengthen pattern→OUTPUT connections
   - All graph-native, no C logic

### Phase 3: Pattern→Output Mapping (Advanced)
**Goal:** Learn which patterns should produce which outputs

1. **Create OUTPUT nodes for common outputs**
   - Via VALUE nodes or training data
   - Or let graph create them as needed

2. **Learn pattern→output associations**
   - When pattern matches input AND output is known:
     - Strengthen edge from pattern to OUTPUT node
   - Use Hebbian learning: co-activation → stronger connection

3. **Use APPLICATION nodes for output tracking**
   - APPLICATION nodes can track: pattern matched + output produced
   - Learn from successful applications

## Implementation Steps

### Step 1: Enable Output Emission (5 min)
```c
// In melvin_learn_cli.c, after graph_run_local_rules():
graph_emit_output(g, g_sys.max_output_bytes_per_tick, STDOUT_FILENO);
```

### Step 2: Create OUTPUT Learning Rule (30 min)
```c
// In graph_run_local_rules(), for LEARNER nodes:
// If LEARNER connected to EPISODE with input→output pair:
//   1. Find patterns matching input
//   2. Find/create OUTPUT nodes for target output
//   3. Create/strengthen pattern→OUTPUT edges
```

### Step 3: Error Feedback via ERROR Nodes (30 min)
```c
// When no outputs produced:
//   1. Activate ERROR node
//   2. ERROR node triggers learning rule
//   3. Strengthen pattern→OUTPUT edges for active patterns
```

### Step 4: Training with Examples (1 hour)
- Create training format: `input:output` pairs
- Create EPISODE nodes with both input and output
- Connect LEARNER nodes to these EPISODEs
- Let graph learn the mapping

## Example Training Data Format

```
abc:xyz
hello:world
123:456
```

This teaches:
- When pattern matching "abc" activates → emit "xyz"
- When pattern matching "hello" activates → emit "world"
- etc.

## Success Criteria

1. ✓ `graph_emit_output()` called and bytes emitted
2. ✓ OUTPUT nodes created from training examples
3. ✓ Pattern→OUTPUT edges learned automatically
4. ✓ Error feedback strengthens connections
5. ✓ System produces outputs after training

## Next Steps

1. **Immediate:** Call `graph_emit_output()` in main loop
2. **Short-term:** Add OUTPUT learning to `graph_run_local_rules()`
3. **Medium-term:** Support input→output training examples
4. **Long-term:** Fully graph-native output learning via LEARNER nodes


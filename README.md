# Melvin Graph-Based AGI System

A graph-based artificial general intelligence system where **the graph is the mind** and **C is frozen hardware**. All intelligence, learning, and decision-making lives in the graph structure itself, while C provides only the minimal substrate for memory management, adjacency lists, and basic graph operations.

## Core Architecture: C = Hardware, Graph = Intelligence

### Hardware Layer (C) - Only 3 Node Types

The C core can **only** create these three fundamental node types:

1. **NODE_DATA** - Raw world events / bytes / codes
   - Created by: `graph_add_data_byte(Graph *g, uint8_t b)`
   - `id` = global position in stream (0, 1, 2, ...)
   - Store byte payload in graph blob
   - Created with activation `a = 1.0` (recent events are active)

2. **NODE_BLANK** - Universal wildcard
   - Created by: `graph_add_blank_node(Graph *g)`
   - Exactly one node in the graph
   - Fixed ID: `UINT64_MAX`
   - Used implicitly in patterns for "any value" matching

3. **NODE_PATTERN** - Templates over relative positions
   - Created by: `graph_add_pattern(Graph *g, const PatternAtom *atoms, size_t num_atoms, float initial_q)`
   - `id` comes from separate counter (starts at `1ULL << 63`)
   - Payload contains array of `PatternAtom` structures
   - Each atom specifies: offset from anchor, mode (CONST_BYTE or BLANK), and value
   - Have quality score `q` for self-consistency

### Generic Hardware Function

For all other node types, C provides a **generic hardware function** with no cognitive decisions:

```c
Node *graph_create_node(Graph *g, NodeKind kind, uint64_t id, const void *payload, size_t payload_len);
```

This is pure hardware - it allocates memory and registers the node. **The graph decides what to create; C just executes.**

### Software Layer (Graph) - Graph-Native Node Types

All other node types are **created by the graph itself**, not by C functions:

- **NODE_OUTPUT** - Output bytes/actions (created by patterns when they should produce output)
- **NODE_ERROR** - Error feedback (created by patterns when failures occur)
- **NODE_EPISODE** - Learning episodes/spans (created by patterns to mark learning windows)
- **NODE_APPLICATION** - Pattern matches/applications (created when patterns match)
- **NODE_VALUE** - Cognitive parameters (thresholds, learning rates, etc.)
- **NODE_LEARNER** - Triggers learning when active
- **NODE_MAINTENANCE** - Triggers pruning/decay/compression when active

**Key Principle:** Patterns and graph rules decide to create these nodes by calling `graph_create_node()`. C never makes cognitive decisions about when or what to create.

## Graph-Native Intelligence

### Learning

Learning is **graph-driven**, not hard-coded in C:

- **LEARNER nodes** trigger pattern induction when active
- **EPISODE nodes** mark learning windows
- **APPLICATION nodes** represent pattern matches
- **VALUE nodes** store learning parameters (min/max pattern length, learning rates, etc.)
- `graph_run_local_rules()` finds active LEARNER nodes and executes local learning rules

### Maintenance

Maintenance (pruning, decay, compression) is **graph-driven**:

- **MAINTENANCE nodes** trigger pruning when active
- **VALUE nodes** store maintenance parameters (min pattern usage, decay rates, work budgets)
- `graph_run_local_rules()` finds active MAINTENANCE nodes and executes local maintenance rules
- All thresholds and rates come from VALUE nodes, not hard-coded constants

### Output Generation

Output generation is **graph-native**:

- **OUTPUT nodes** represent output bytes/actions
- Patterns create OUTPUT nodes when they should produce output
- `graph_emit_output()` collects active OUTPUT nodes and emits top-k bytes
- Patterns learn to create OUTPUT nodes through graph-native learning rules

### Pattern-on-Pattern Learning

Patterns can be built from other patterns, not just raw data:

- `build_symbol_sequence_from_episode_node()` builds sequences of DATA and PATTERN node IDs
- `graph_create_pattern_from_sequences()` creates patterns from symbol sequences
- This enables patterns to build on top of other patterns, creating hierarchical abstractions

## Core Operations

### Graph Management (Hardware)

- `graph_create()` - Allocate and initialize graph with capacity limits
- `graph_destroy()` - Free all resources
- `graph_add_data_byte()` - Create new DATA node (hardware)
- `graph_add_blank_node()` - Create new BLANK node (hardware)
- `graph_add_pattern()` - Create new PATTERN node (hardware)
- `graph_create_node()` - Generic node creation (hardware, no decisions)
- `graph_add_edge()` - Create edge between any two nodes
- `graph_remove_edge()` - Remove edge by index

### Graph-Native Operations

- `graph_run_local_rules()` - Execute graph-native learning and maintenance rules
- `graph_emit_output()` - Collect and emit active OUTPUT nodes
- `graph_get_value()` - Get cognitive parameter from VALUE node
- `graph_propagate()` - Generic message-passing physics (activation flows along edges)

### Pattern Matching

- `pattern_match_score()` - Score how well a pattern matches at a given anchor position
  - Returns value in [0, 1] based on fraction of atoms that match
  - Handles CONST_BYTE (exact match) and BLANK (always matches) modes

### Pattern Creation

- `graph_create_pattern_from_sequences()` - Create pattern from two symbol sequences
- `build_symbol_sequence_from_episode_node()` - Build symbol sequence from EPISODE node
- Patterns can include BLANK atoms for generalization

## Edge Structure

Universal edge type used for all connections:
- `DATA → DATA` (sequence, association, long-range links)
- `PATTERN → DATA` (pattern binding / influence)
- `PATTERN → PATTERN` (patterns built from patterns)
- `PATTERN → OUTPUT` (patterns produce outputs)
- Any connection involving `BLANK`

Edges have:
- Weight `w` representing influence/strength
- Bidirectional adjacency lists (`first_out_edge`, `first_in_edge`, `next_out_edge`, `next_in_edge`)
- O(1) node lookups via ID→index hash table

## Building

```bash
make
```

This creates:
- `melvin_learn_cli` - Main learning/runtime CLI
- `melvin_dsl` - Graph DSL interpreter
- `test_graph_driven_learning` - Test graph-native learning
- `test_graph_self_maintenance` - Test graph-native maintenance

## Running

### Basic Learning

```bash
echo "abc" | ./melvin_learn_cli --load /dev/null --save /dev/null
```

### Persistent Runtime

```bash
./melvin_learn_cli --load graph.bin --save graph.bin
# Then feed input via stdin
```

### Graph DSL

```bash
./melvin_dsl < commands.dsl
```

## Design Principles

1. **C is frozen hardware** - Only memory management, adjacency lists, ID→index maps, mmap persistence, ports, and the main tick loop
2. **Graph is the mind** - All learning, credit assignment, decisions, and policies must be encoded as graph structures and local rules
3. **No cognitive features in C** - C must not implement task-specific logic, reasoning, planning, or output policies
4. **Graph-native intelligence** - Learning, maintenance, and output generation are all driven by graph nodes (LEARNER, MAINTENANCE, OUTPUT, etc.)
5. **Local rules only** - All graph operations must be local (adjacency-based), no global scans in hot paths
6. **Pattern-on-pattern** - Patterns can be built from other patterns, enabling hierarchical abstractions

## Code Structure

- `src/melvin.h` - Public API, structs, enums
- `src/melvin.c` - Core implementation (hardware layer)
- `src/melvin_learn_cli.c` - CLI for learning/runtime
- `src/melvin_dsl.c` - Graph DSL interpreter
- `src/legacy_learning.c` - Legacy global learning (training-only, guarded by `training_enabled`)
- `Makefile` - Build configuration

## Performance

- **Snapshot load**: ~3-10 ms for large graphs (1M+ edges)
- **Per-task processing**: ~1-3 ms (runtime mode, training disabled)
- **Adjacency-based traversal**: O(degree) instead of O(E)
- **O(1) node lookups**: Via ID→index hash table
- **No global scans in hot paths**: All operations are local via adjacency lists

## Notes

- The system runs as a **persistent process** - graph loads once at startup, then processes a stream of inputs
- Training vs runtime modes: `SystemConfig.training_enabled` controls whether heavy learning runs
- Graph persistence uses a **snapshot format** for near-instant loading
- All cognitive parameters live in VALUE nodes, not hard-coded constants
- The graph can create its own nodes (OUTPUT, ERROR, EPISODE, etc.) via patterns and rules

The substrate is designed so that **all intelligence lives in the graph**, while C provides only the minimal mechanical operations needed to keep the graph running.

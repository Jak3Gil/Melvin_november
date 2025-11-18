# Graph Coding Guide - Consistent Structure

## File Organization Template

```
melvin.c structure:
===================

1. INCLUDES & FORWARDS
   - Standard includes
   - Forward declarations
   
2. HARDWARE LAYER (frozen - only fix bugs/performance)
   - Hash table helpers (ID→index mapping)
   - Memory allocation helpers
   - Adjacency list operations
   - Graph create/destroy
   - Node/Edge creation (mechanical)
   - Save/Load persistence
   
3. GRAPH INTELLIGENCE LAYER (where all smarts live)
   - Pattern matching rules
   - Pattern creation/induction
   - Learning rules (local updates)
   - Activation/propagation dynamics
   - Pattern selection/competition
   - Reward/value propagation
   - Output generation
   
4. OBSERVABILITY (reporting only)
   - Stats printing
   - Debug dumps
   - Profiling helpers
```

## Graph Intelligence Function Patterns

### Pattern 1: Local Node/Edge Update Rules
```c
// LOCAL RULE: Update based on node's own state + neighbors
// Operates on single node/edge, uses adjacency lists
void graph_update_node_X(Graph *g, Node *node) {
    // 1. Read node's own state
    float current_value = node->some_field;
    
    // 2. Traverse neighbors via adjacency
    uint32_t eid = node->first_in_edge;  // or first_out_edge
    float sum = 0.0f;
    while (eid != UINT32_MAX) {
        Edge *e = &g->edges[eid];
        Node *neighbor = graph_find_node_by_id(g, e->src);  // or dst
        sum += neighbor->some_field * e->w;
        eid = e->next_in_edge;  // or next_out_edge
    }
    
    // 3. Apply local rule (simple math, no global decisions)
    node->some_field = current_value * 0.9f + sum * 0.1f;
}
```

### Pattern 2: Pattern Matching (Graph-Native)
```c
// PATTERN RULE: Match pattern against graph state
// Uses pattern atoms to query DATA nodes
float pattern_match_score(const Graph *g, const Node *pattern, uint64_t anchor_id) {
    // 1. Read pattern from blob
    const PatternAtom *atoms = (const PatternAtom *)(g->blob + pattern->payload_offset);
    
    // 2. For each atom, check if DATA node matches
    size_t matches = 0;
    for (size_t i = 0; i < num_atoms; i++) {
        uint64_t target_id = anchor_id + atoms[i].delta;
        Node *target = graph_find_node_by_id(g, target_id);  // O(1) lookup
        if (target && /* check match */) {
            matches++;
        }
    }
    
    // 3. Return score (local computation)
    return (float)matches / (float)num_atoms;
}
```

### Pattern 3: Learning Rule (Local Update)
```c
// LEARNING RULE: Update pattern quality based on error
// Operates on single pattern, uses local error signal
void graph_update_pattern_from_error(Graph *g, Node *pattern, float error, float lr) {
    // 1. Compute target quality from error
    float target_q = 1.0f - error;  // perfect = 1.0, worst = 0.0
    
    // 2. Update pattern quality (local rule)
    pattern->q += lr * (target_q - pattern->q);
    
    // 3. Clamp to valid range
    if (pattern->q < 0.0f) pattern->q = 0.0f;
    if (pattern->q > 1.0f) pattern->q = 1.0f;
}
```

### Pattern 4: Pattern Application (Graph Traversal)
```c
// PATTERN APPLICATION: Apply pattern to create bindings
// Uses adjacency lists to find/create edges
void graph_apply_pattern(Graph *g, Node *pattern, uint64_t anchor_id) {
    const PatternAtom *atoms = /* read from blob */;
    
    // For each atom, bind pattern to DATA node
    for (size_t i = 0; i < num_atoms; i++) {
        uint64_t data_id = anchor_id + atoms[i].delta;
        
        // Find or create edge (pattern → data)
        Edge *e = graph_find_edge(g, pattern->id, data_id);
        if (!e) {
            e = graph_add_edge(g, pattern->id, data_id, 0.0f);
        }
        
        // Strengthen binding (local rule)
        e->w += 0.1f;  // or use error-based update
    }
}
```

## Consistent Naming Conventions

### Function Prefixes
- `graph_` - Graph-level operations
- `pattern_` - Pattern-specific operations
- `explanation_` - Explanation/compression operations
- `static` - Internal helpers (not in header)

### Graph Intelligence Functions (must be local)
- `graph_update_*` - Local update rules
- `pattern_*` - Pattern matching/application
- `graph_*_from_*` - Learning from signals (error, reward)
- `graph_apply_*` - Apply patterns to create bindings

### Hardware Functions (frozen)
- `graph_create/destroy` - Memory management
- `graph_add_*` - Mechanical node/edge creation
- `graph_find_*` - Lookup operations
- `graph_rebuild_*` - Maintenance operations

## Coding Rules for Graph Intelligence

### ✅ DO: Local Rules
```c
// Update based on local state + neighbors
void graph_update_node_activation(Graph *g, Node *node) {
    // Traverse adjacency, compute locally
    float sum = 0.0f;
    uint32_t eid = node->first_in_edge;
    while (eid != UINT32_MAX) {
        Edge *e = &g->edges[eid];
        Node *src = graph_find_node_by_id(g, e->src);
        sum += src->a * e->w;
        eid = e->next_in_edge;
    }
    node->a = tanh(sum);  // Simple local rule
}
```

### ✅ DO: Pattern-Based Logic
```c
// Use patterns to encode behavior
void graph_select_output(Graph *g, Node *context) {
    // Find patterns that match context
    // Let pattern qualities determine selection
    // No hardcoded "if context == X" logic
}
```

### ❌ DON'T: Global Decisions in C
```c
// BAD: Hardcoded decision logic
void graph_handle_input(Graph *g, const char *input) {
    if (strcmp(input, "hello") == 0) {
        // Do something specific
    }
}

// GOOD: Let graph patterns handle it
void graph_feed_input(Graph *g, const char *input) {
    // Add as DATA nodes
    // Let existing patterns match
    // Patterns determine behavior via their bindings
}
```

## Section Organization in melvin.c

```c
// ============================================================================
// SECTION 1: HARDWARE LAYER (Memory, Adjacency, ID Mapping)
// ============================================================================
// - Hash table helpers
// - Allocation helpers  
// - graph_create/destroy
// - graph_add_node/edge (mechanical)
// - graph_find_node_by_id (O(1) lookup)
// - graph_rebuild_adjacency
// - Save/Load

// ============================================================================
// SECTION 2: ACTIVATION & PROPAGATION (Graph Dynamics)
// ============================================================================
// - graph_propagate
// - Local activation rules
// - Edge weight updates

// ============================================================================
// SECTION 3: PATTERN MATCHING (Graph Intelligence)
// ============================================================================
// - pattern_match_score
// - pattern_reconstruct_segment
// - Pattern application logic

// ============================================================================
// SECTION 4: LEARNING RULES (Graph Intelligence)
// ============================================================================
// - graph_update_pattern_quality
// - graph_update_edge_from_error
// - Self-consistency episodes
// - Pattern binding

// ============================================================================
// SECTION 5: EXPLANATION & COMPRESSION (Graph Intelligence)
// ============================================================================
// - graph_build_explanation_*
// - graph_reconstruct_from_explanation
// - explanation_select_*

// ============================================================================
// SECTION 6: OBSERVABILITY (Reporting Only)
// ============================================================================
// - graph_print_stats
// - graph_debug_print_*
// - Profiling helpers
```

## Adding New Graph Intelligence

When adding new behavior:

1. **Define it as a local rule** operating on nodes/edges
2. **Use adjacency lists** to traverse neighbors
3. **Store state in nodes/edges** (not global variables)
4. **Let patterns encode behavior** (not if/else in C)
5. **Use simple math** for updates (no complex logic)

Example: Adding "value" propagation
```c
// Add to Node struct (in melvin.h):
// float v;  // value field

// Add local update rule (in melvin.c):
void graph_update_node_value(Graph *g, Node *node) {
    // Read incoming edges
    float value_sum = 0.0f;
    uint32_t eid = node->first_in_edge;
    while (eid != UINT32_MAX) {
        Edge *e = &g->edges[eid];
        Node *src = graph_find_node_by_id(g, e->src);
        value_sum += src->v * e->w;
        eid = e->next_in_edge;
    }
    
    // Local update rule
    node->v = node->v * 0.9f + value_sum * 0.1f;
}

// Call from graph_propagate or separate tick
```

## Key Principles

1. **All intelligence = graph structures + local rules**
2. **C only provides: memory, adjacency, lookup, ticking**
3. **No global decisions in C** - patterns make decisions
4. **Local updates only** - each node/edge updates from neighbors
5. **Use adjacency lists** - never scan all nodes/edges in hot paths


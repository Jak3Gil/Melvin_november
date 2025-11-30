# Edge Pointer Consistency Solution

## Problem Analysis

### Current Issues:
1. **Line 144**: Uses `allocated_nodes` (calculated heuristically) - could be wrong
2. **Line 198**: Uses `g->header->num_nodes` - CORRECT
3. **Line 2102**: Uses `g.header->num_nodes` - CORRECT  
4. **Line 2122**: Uses `g.header->num_nodes` - CORRECT

### Root Cause:
- Inconsistent edge pointer calculation
- `update_pointers_after_mmap()` uses heuristic calculation instead of direct `num_nodes`
- File layout is: `[Header] [Nodes: num_nodes] [Edges: num_edges] [Unused space]`
- Edges MUST start right after actual nodes (num_nodes), not allocated space

## Solution

### 1. Create Helper Function
Add a helper function that always calculates edge pointer correctly:

```c
// Calculate edge pointer - ALWAYS uses num_nodes (actual count)
static inline Edge* calculate_edges_pointer(Brain *g) {
    size_t header_size = sizeof(BrainHeader);
    size_t node_size = sizeof(Node);
    return (Edge*)((uint8_t*)g->nodes + g->header->num_nodes * node_size);
}
```

### 2. Use Helper Everywhere
Replace all edge pointer calculations with the helper:

- Line 144: `g->edges = calculate_edges_pointer(g);`
- Line 198: `g->edges = calculate_edges_pointer(g);`
- Line 2102: `g.edges = calculate_edges_pointer(&g);`
- Line 2122: `g.edges = calculate_edges_pointer(&g);`

### 3. Add Validation
Add validation to ensure edge pointer is correct:

```c
// Validate edge pointer is correct
static void validate_edge_pointer(Brain *g) {
    size_t header_size = sizeof(BrainHeader);
    size_t node_size = sizeof(Node);
    Edge *expected = (Edge*)((uint8_t*)g->nodes + g->header->num_nodes * node_size);
    if (g->edges != expected) {
        fprintf(stderr, "ERROR: Edge pointer mismatch! Expected %p, got %p\n", 
                expected, g->edges);
        g->edges = expected; // Fix it
    }
}
```

### 4. File Layout Guarantee
Ensure file layout is always:
```
[Header: 256 bytes]
[Nodes: num_nodes × 44 bytes]  <- Actual nodes
[Edges: num_edges × 48 bytes]  <- Edges start HERE
[Unused allocated space]       <- Extra space for growth
```

## Implementation Steps

1. Add helper function after `update_pointers_after_mmap()`
2. Replace all edge pointer calculations
3. Add validation calls after pointer updates
4. Test with existing corrupted files


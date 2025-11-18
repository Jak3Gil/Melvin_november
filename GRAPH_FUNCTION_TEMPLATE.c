// ============================================================================
// GRAPH INTELLIGENCE FUNCTION TEMPLATE
// ============================================================================
// Use this template when adding new graph-native intelligence
// Copy and modify for your specific use case

// ----------------------------------------------------------------------------
// TEMPLATE 1: Local Node Update Rule
// ----------------------------------------------------------------------------
// Updates a single node based on its neighbors (via adjacency lists)
// This is the fundamental building block of graph intelligence

void graph_update_node_X(Graph *g, Node *node) {
    if (!g || !node) return;
    
    // Step 1: Read node's current state
    float current_value = node->some_field;
    
    // Step 2: Traverse incoming edges (or outgoing, depending on your rule)
    float neighbor_sum = 0.0f;
    uint32_t eid = node->first_in_edge;  // or first_out_edge
    
    while (eid != UINT32_MAX) {
        if (eid >= g->num_edges) break;  // Safety check
        
        Edge *e = &g->edges[eid];
        
        // Get neighbor node (O(1) lookup)
        uint32_t neighbor_idx = graph_id_map_lookup(g, e->src);  // or e->dst
        if (neighbor_idx != UINT32_MAX && neighbor_idx < g->num_nodes) {
            Node *neighbor = &g->nodes[neighbor_idx];
            
            // Accumulate neighbor contribution
            neighbor_sum += neighbor->some_field * e->w;
        }
        
        eid = e->next_in_edge;  // or next_out_edge
    }
    
    // Step 3: Apply local update rule (simple math only)
    // NO if/else logic here - just numeric computation
    node->some_field = current_value * 0.9f + neighbor_sum * 0.1f;
    
    // Step 4: Optional clamping
    if (node->some_field < 0.0f) node->some_field = 0.0f;
    if (node->some_field > 1.0f) node->some_field = 1.0f;
}

// ----------------------------------------------------------------------------
// TEMPLATE 2: Pattern-Based Rule
// ----------------------------------------------------------------------------
// Uses patterns to encode behavior (not hardcoded if/else)

float graph_compute_pattern_X(const Graph *g, Node *pattern, uint64_t context_id) {
    if (!g || !pattern || pattern->kind != NODE_PATTERN) return 0.0f;
    
    // Step 1: Read pattern atoms from blob
    size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
    if (num_atoms == 0) return 0.0f;
    
    const PatternAtom *atoms = (const PatternAtom *)(g->blob + pattern->payload_offset);
    
    // Step 2: Check pattern against graph state
    size_t matches = 0;
    for (size_t i = 0; i < num_atoms; i++) {
        uint64_t target_id = context_id + (int64_t)atoms[i].delta;
        
        // O(1) lookup
        uint32_t target_idx = graph_id_map_lookup(g, target_id);
        if (target_idx == UINT32_MAX) continue;
        
        Node *target = &g->nodes[target_idx];
        
        // Check match condition (pattern-specific)
        if (atoms[i].mode == 0) {  // CONST_BYTE
            if (target->kind == NODE_DATA && target->payload_len > 0) {
                uint8_t byte_val = g->blob[target->payload_offset];
                if (byte_val == atoms[i].value) {
                    matches++;
                }
            }
        }
    }
    
    // Step 3: Return score (local computation)
    return (float)matches / (float)num_atoms;
}

// ----------------------------------------------------------------------------
// TEMPLATE 3: Learning Rule (Local Update from Error/Reward)
// ----------------------------------------------------------------------------
// Updates graph state based on error signal (local rule only)

void graph_update_from_error(Graph *g, Node *node, float error, float lr) {
    if (!g || !node) return;
    
    // Step 1: Compute target value from error
    float target = 1.0f - error;  // perfect = 1.0, worst = 0.0
    
    // Step 2: Update node field (local rule)
    node->q += lr * (target - node->q);
    
    // Step 3: Clamp to valid range
    if (node->q < 0.0f) node->q = 0.0f;
    if (node->q > 1.0f) node->q = 1.0f;
    
    // Step 4: Optionally update incoming edges
    uint32_t eid = node->first_in_edge;
    while (eid != UINT32_MAX) {
        if (eid >= g->num_edges) break;
        
        Edge *e = &g->edges[eid];
        uint32_t src_idx = graph_id_map_lookup(g, e->src);
        if (src_idx != UINT32_MAX && src_idx < g->num_nodes) {
            Node *src = &g->nodes[src_idx];
            
            // Local edge update rule
            e->w += -lr * src->a * error;
            
            // Optional clamping
            if (e->w < -1.0f) e->w = -1.0f;
            if (e->w > 1.0f) e->w = 1.0f;
        }
        
        eid = e->next_in_edge;
    }
}

// ----------------------------------------------------------------------------
// TEMPLATE 4: Pattern Application (Create Bindings)
// ----------------------------------------------------------------------------
// Applies a pattern to create/strengthen edges in the graph

void graph_apply_pattern_binding(Graph *g, Node *pattern, uint64_t anchor_id, float strength) {
    if (!g || !pattern || pattern->kind != NODE_PATTERN) return;
    
    // Step 1: Read pattern atoms
    size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
    if (num_atoms == 0) return;
    
    const PatternAtom *atoms = (const PatternAtom *)(g->blob + pattern->payload_offset);
    
    // Step 2: For each atom, create/strengthen edge
    for (size_t i = 0; i < num_atoms; i++) {
        if (atoms[i].mode != 0) continue;  // Only CONST_BYTE atoms create bindings
        
        uint64_t data_id = anchor_id + (int64_t)atoms[i].delta;
        
        // Find or create edge (pattern → data)
        Edge *e = graph_find_edge(g, pattern->id, data_id);
        if (!e) {
            e = graph_add_edge(g, pattern->id, data_id, 0.0f);
            if (!e) continue;
        }
        
        // Strengthen binding (local rule)
        e->w += strength;
        
        // Optional clamping
        if (e->w > 1.0f) e->w = 1.0f;
    }
}

// ============================================================================
// USAGE NOTES
// ============================================================================
//
// 1. All functions operate on graph structures (nodes, edges, patterns)
// 2. Use adjacency lists (first_in_edge, first_out_edge) for traversal
// 3. Use graph_find_node_by_id (O(1)) for lookups
// 4. Keep rules LOCAL - each node/edge updates from neighbors only
// 5. NO global decisions - patterns encode behavior
// 6. Simple math only - no complex if/else logic
//
// To add new graph intelligence:
// 1. Copy relevant template
// 2. Modify for your specific use case
// 3. Add to appropriate section in melvin.c
// 4. Call from graph_propagate or learning loops
//


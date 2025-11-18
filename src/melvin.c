#include "melvin.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>  // for fsync

// Profiling helper: get current time in seconds (monotonic clock)
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Global system configuration (default: runtime mode, no learning)
// Cognitive parameters (learning rates, thresholds, etc.) are now in VALUE nodes
SystemConfig g_sys = {
    .training_enabled = 0,  // Default: fast inference, no heavy learning
    .max_output_bytes_per_tick = 4  // Emit up to 4 bytes per tick
};

// Forward declarations
static Edge *graph_find_edge(Graph *g, uint64_t src, uint64_t dst);

// Hash table helpers for ID→index mapping
static uint32_t hash_id(uint64_t id, uint32_t table_size) {
    // Simple multiplicative hash
    return (uint32_t)(id ^ (id >> 32)) & (table_size - 1);
}

static void graph_ensure_id_map_capacity(Graph *g) {
    // Grow hash table when load factor > 0.75
    if (g->id_map_size == 0 || g->id_map_count * 4 > g->id_map_size * 3) {
        uint32_t new_size = g->id_map_size == 0 ? 16 : g->id_map_size * 2;
        
        IdMapEntry *old_map = g->id_to_index;
        uint32_t old_size = g->id_map_size;
        
        g->id_to_index = calloc(new_size, sizeof(IdMapEntry));
        if (!g->id_to_index) {
            // Allocation failed - keep old map
            g->id_to_index = old_map;
            return;
        }
        
        // Initialize all slots as empty
        for (uint32_t i = 0; i < new_size; i++) {
            g->id_to_index[i].id = UINT64_MAX;
        }
        
        g->id_map_size = new_size;
        g->id_map_count = 0;
        
        // Rehash all existing entries
        if (old_map) {
            for (uint32_t i = 0; i < old_size; i++) {
                if (old_map[i].id != UINT64_MAX) {
                    uint32_t idx = hash_id(old_map[i].id, new_size);
                    // Linear probing
                    while (g->id_to_index[idx].id != UINT64_MAX) {
                        idx = (idx + 1) & (new_size - 1);
                    }
                    g->id_to_index[idx] = old_map[i];
                    g->id_map_count++;
                }
            }
            free(old_map);
        }
    }
}

static void graph_id_map_insert(Graph *g, uint64_t id, uint32_t index) {
    if (g->id_map_size == 0) {
        graph_ensure_id_map_capacity(g);
    }
    
    uint32_t pos = hash_id(id, g->id_map_size);
    uint32_t start_pos = pos;
    uint32_t probes = 0;
    uint32_t max_probes = g->id_map_size;  // Safety limit
    
    // Linear probing to find empty slot or existing entry
    while (g->id_to_index[pos].id != UINT64_MAX && g->id_to_index[pos].id != id) {
        pos = (pos + 1) & (g->id_map_size - 1);
        probes++;
        if (probes >= max_probes) {
            // Table is full or corrupted - force a resize
            graph_ensure_id_map_capacity(g);
            // Retry from beginning
            pos = hash_id(id, g->id_map_size);
            start_pos = pos;
            probes = 0;
            max_probes = g->id_map_size;
        }
    }
    
    if (g->id_to_index[pos].id == UINT64_MAX) {
        g->id_map_count++;
    }
    
    g->id_to_index[pos].id = id;
    g->id_to_index[pos].index = index;
    
    // Check if we need to grow
    graph_ensure_id_map_capacity(g);
}

static uint32_t graph_id_map_lookup(const Graph *g, uint64_t id) {
    if (!g || g->id_map_size == 0) return UINT32_MAX;
    
    uint32_t pos = hash_id(id, g->id_map_size);
    uint32_t start_pos = pos;
    uint32_t probes = 0;
    uint32_t max_probes = g->id_map_size;  // Safety limit
    
    // Linear probing
    while (g->id_to_index[pos].id != UINT64_MAX) {
        if (g->id_to_index[pos].id == id) {
            return g->id_to_index[pos].index;
        }
        pos = (pos + 1) & (g->id_map_size - 1);
        probes++;
        if (pos == start_pos || probes >= max_probes) {
            // Wrapped around or too many probes - not found
            break;
        }
    }
    
    return UINT32_MAX;
}

// Helper: allocate space in the blob
static uint32_t blob_alloc(Graph *g, size_t len) {
    if (g->blob_used + len > g->blob_cap) {
        // Grow blob capacity
        size_t new_cap = g->blob_cap * 2;
        if (new_cap < g->blob_used + len) {
            new_cap = g->blob_used + len;
        }
        uint8_t *new_blob = realloc(g->blob, new_cap);
        if (!new_blob) return UINT32_MAX;
        g->blob = new_blob;
        g->blob_cap = new_cap;
    }
    uint32_t offset = g->blob_used;
    g->blob_used += len;
    return offset;
}

// Helper: grow nodes array if needed
static int ensure_node_capacity(Graph *g) {
    if (g->num_nodes >= g->nodes_cap) {
        size_t new_cap = g->nodes_cap * 2;
        if (new_cap == 0) new_cap = 16;
        Node *new_nodes = realloc(g->nodes, new_cap * sizeof(Node));
        if (!new_nodes) return 0;
        g->nodes = new_nodes;
        g->nodes_cap = new_cap;
    }
    return 1;
}

// Helper: grow edges array if needed
static int ensure_edge_capacity(Graph *g) {
    if (g->num_edges >= g->edges_cap) {
        size_t new_cap = g->edges_cap * 2;
        if (new_cap == 0) new_cap = 16;
        Edge *new_edges = realloc(g->edges, new_cap * sizeof(Edge));
        if (!new_edges) return 0;
        g->edges = new_edges;
        g->edges_cap = new_cap;
    }
    return 1;
}

// Helper: create a new node slot
static Node *graph_new_node(Graph *g) {
    if (!ensure_node_capacity(g)) return NULL;
    uint32_t node_index = (uint32_t)g->num_nodes;
    Node *node = &g->nodes[node_index];
    memset(node, 0, sizeof(Node));
    node->first_out_edge = UINT32_MAX;  // Initialize adjacency lists
    node->first_in_edge  = UINT32_MAX;
    g->num_nodes++;
    return node;
}

// Helper: register a node in the ID→index mapping (call after setting node->id)
static void graph_register_node_id(Graph *g, Node *node) {
    if (!g || !node) return;
    uint32_t node_index = (uint32_t)(node - g->nodes);
    if (node_index < g->num_nodes) {
        graph_id_map_insert(g, node->id, node_index);
    }
}

Graph *graph_create(size_t node_cap, size_t edge_cap, size_t blob_cap) {
    Graph *g = calloc(1, sizeof(Graph));
    if (!g) return NULL;
    
    g->nodes_cap = node_cap > 0 ? node_cap : 16;
    g->edges_cap = edge_cap > 0 ? edge_cap : 16;
    g->blob_cap = blob_cap > 0 ? blob_cap : 1024;
    
    g->nodes = calloc(g->nodes_cap, sizeof(Node));
    g->edges = calloc(g->edges_cap, sizeof(Edge));
    g->blob = calloc(g->blob_cap, sizeof(uint8_t));
    
    if (!g->nodes || !g->edges || !g->blob) {
        graph_destroy(g);
        return NULL;
    }
    
    // Initialize ID→index mapping (hash table)
    g->id_to_index = NULL;
    g->id_map_size = 0;
    g->id_map_count = 0;
    graph_ensure_id_map_capacity(g);
    
    // Initialize all nodes' adjacency lists (calloc zeros to 0, but we need UINT32_MAX)
    for (size_t i = 0; i < g->nodes_cap; i++) {
        g->nodes[i].first_out_edge = UINT32_MAX;
        g->nodes[i].first_in_edge  = UINT32_MAX;
    }
    // Initialize all edges' next pointers
    for (size_t i = 0; i < g->edges_cap; i++) {
        g->edges[i].next_out_edge = UINT32_MAX;
        g->edges[i].next_in_edge  = UINT32_MAX;
    }
    
    g->num_nodes = 0;
    g->num_edges = 0;
    g->blob_used = 0;
    g->next_data_pos = 0;
    g->next_pattern_id = (1ULL << 63);
    g->blank_id = UINT64_MAX;
    
    // Create the single BLANK node
    Node *blank = graph_add_blank_node(g);
    if (!blank) {
        graph_destroy(g);
        return NULL;
    }
    
    return g;
}

void graph_destroy(Graph *g) {
    if (!g) return;
    free(g->nodes);
    free(g->edges);
    free(g->blob);
    free(g->id_to_index);  // Free ID→index hash table
    free(g);
}

Node *graph_add_blank_node(Graph *g) {
    if (!g) return NULL;
    
    // Check if BLANK node already exists
    Node *existing = graph_find_node_by_id(g, g->blank_id);
    if (existing) return existing;
    
    Node *node = graph_new_node(g);
    if (!node) return NULL;
    
    node->id = g->blank_id;
    node->kind = NODE_BLANK;
    node->payload_len = 0;
    node->payload_offset = 0;
    node->a = 0.0f;
    node->q = 0.0f;
    node->flags = 0;
    
    graph_register_node_id(g, node);
    return node;
}

Node *graph_add_data_byte(Graph *g, uint8_t b) {
    if (!g) return NULL;
    
    Node *node = graph_new_node(g);
    if (!node) return NULL;
    
    uint32_t offset = blob_alloc(g, 1);
    if (offset == UINT32_MAX) {
        g->num_nodes--; // rollback
        return NULL;
    }
    
    g->blob[offset] = b;
    
    node->id = g->next_data_pos++;
    node->kind = NODE_DATA;
    node->payload_offset = offset;
    node->payload_len = 1;
    node->a = 1.0f;  // recent event is active
    node->q = 0.0f;
    node->flags = 0;
    
    graph_register_node_id(g, node);
    return node;
}

// Generic hardware function: creates a node (no cognitive decisions)
// This is pure hardware - the graph decides what to create, C just executes
Node *graph_create_node(Graph *g, NodeKind kind, uint64_t id, const void *payload, size_t payload_len) {
    if (!g) return NULL;
    
    Node *node = graph_new_node(g);
    if (!node) return NULL;
    
    // Allocate payload if provided
    uint32_t offset = 0;
    if (payload && payload_len > 0) {
        offset = blob_alloc(g, payload_len);
        if (offset == UINT32_MAX) {
            g->num_nodes--; // rollback
            return NULL;
        }
        memcpy(g->blob + offset, payload, payload_len);
    }
    
    node->id = id;
    node->kind = kind;
    node->payload_offset = offset;
    node->payload_len = payload_len;
    node->a = 0.0f;
    node->q = 0.0f;
    node->flags = 0;
    
    graph_register_node_id(g, node);
    return node;
}

// DEPRECATED: Use graph-native creation instead (patterns/rules create OUTPUT nodes)
// Kept as static for migration only
static Node *graph_add_output_byte_internal(Graph *g, uint8_t b) {
    if (!g) return NULL;
    
    static uint64_t next_output_id = (1ULL << 62);
    uint8_t payload = b;
    return graph_create_node(g, NODE_OUTPUT, next_output_id++, &payload, 1);
}

Node *graph_add_pattern(Graph *g,
                        const PatternAtom *atoms,
                        size_t num_atoms,
                        float initial_q) {
    if (!g || !atoms || num_atoms == 0) return NULL;
    
    Node *node = graph_new_node(g);
    if (!node) return NULL;
    
    size_t payload_size = num_atoms * sizeof(PatternAtom);
    uint32_t offset = blob_alloc(g, payload_size);
    if (offset == UINT32_MAX) {
        g->num_nodes--; // rollback
        return NULL;
    }
    
    memcpy(g->blob + offset, atoms, payload_size);
    
    node->id = g->next_pattern_id++;
    node->kind = NODE_PATTERN;
    node->payload_offset = offset;
    node->payload_len = payload_size;
    node->a = 0.0f;
    node->q = initial_q;
    node->flags = 0;
    
    graph_register_node_id(g, node);
    return node;
}

// DEPRECATED: Use graph-native creation instead (patterns/rules create EPISODE nodes)
// Kept as static for migration only
static Node *graph_add_episode_internal(Graph *g, uint64_t start_id, uint64_t end_id) {
    if (!g) return NULL;
    
    uint64_t payload[2] = {start_id, end_id};
    static uint64_t next_episode_id = (1ULL << 60);
    Node *node = graph_create_node(g, NODE_EPISODE, next_episode_id++, payload, sizeof(payload));
    if (node) {
        node->a = 1.0f;  // Episodes start active
    }
    return node;
}

// DEPRECATED: Use graph-native creation instead (patterns/rules create APPLICATION nodes)
// Kept as static for migration only
static Node *graph_add_application_internal(Graph *g, uint64_t pattern_id, uint64_t anchor_id, float score) {
    if (!g) return NULL;
    
    typedef struct {
        uint64_t pattern_id;
        uint64_t anchor_id;
        float score;
    } AppPayload;
    
    AppPayload payload = {pattern_id, anchor_id, score};
    static uint64_t next_app_id = (1ULL << 59);
    Node *node = graph_create_node(g, NODE_APPLICATION, next_app_id++, &payload, sizeof(payload));
    if (node) {
        node->a = score;  // Activation stores match score
        // Create edge: APPLICATION -> PATTERN
        graph_add_edge(g, node->id, pattern_id, score);
    }
    return node;
}

// DEPRECATED: Use graph-native creation instead (patterns/rules create LEARNER nodes)
// Kept as static for migration only
static Node *graph_add_learner_internal(Graph *g) {
    if (!g) return NULL;
    
    static uint64_t next_learner_id = (1ULL << 58);
    return graph_create_node(g, NODE_LEARNER, next_learner_id++, NULL, 0);
}

// DEPRECATED: Use graph-native creation instead (patterns/rules create MAINTENANCE nodes)
// Kept as static for migration only
static Node *graph_add_maintenance_internal(Graph *g) {
    if (!g) return NULL;
    
    static uint64_t next_maintenance_id = (1ULL << 57);
    return graph_create_node(g, NODE_MAINTENANCE, next_maintenance_id++, NULL, 0);
}

// Convert Explanation to graph nodes (EPISODE and APPLICATION)
void explanation_to_graph(Graph *g, const Explanation *exp, uint64_t episode_node_id) {
    if (!g || !exp) return;
    
    Node *episode_node = graph_find_node_by_id(g, episode_node_id);
    if (!episode_node || episode_node->kind != NODE_EPISODE) return;
    
    // Create APPLICATION node for each pattern application
    for (size_t i = 0; i < exp->count; i++) {
        uint64_t pattern_id = exp->apps[i].pattern_id;
        uint64_t anchor_id = exp->apps[i].anchor_id;
        
        // Get match score (default to 1.0 if not available)
        float score = 1.0f;
        Node *pattern = graph_find_node_by_id(g, pattern_id);
        if (pattern && pattern->kind == NODE_PATTERN) {
            score = pattern_match_score(g, pattern, anchor_id);
        }
        
        Node *app_node = graph_add_application_internal(g, pattern_id, anchor_id, score);
        if (app_node) {
            // Link APPLICATION -> EPISODE
            graph_add_edge(g, app_node->id, episode_node_id, 1.0f);
        }
    }
}

Edge *graph_add_edge(Graph *g, uint64_t src, uint64_t dst, float w) {
    if (!g) return NULL;
    
    if (!ensure_edge_capacity(g)) return NULL;
    
    uint32_t eid = g->num_edges++;
    Edge *edge = &g->edges[eid];
    edge->src = src;
    edge->dst = dst;
    edge->w = w;
    edge->next_out_edge = UINT32_MAX;  // Initialize adjacency links
    edge->next_in_edge  = UINT32_MAX;
    
    // Link into source node's outgoing adjacency list
    Node *src_node = graph_find_node_by_id(g, src);
    if (src_node) {
        edge->next_out_edge = src_node->first_out_edge;
        src_node->first_out_edge = eid;
    }
    
    // Link into destination node's incoming adjacency list
    Node *dst_node = graph_find_node_by_id(g, dst);
    if (dst_node) {
        edge->next_in_edge = dst_node->first_in_edge;
        dst_node->first_in_edge = eid;
    }
    
    return edge;
}

Node *graph_find_node_by_id(const Graph *g, uint64_t id) {
    if (!g) return NULL;
    
    // O(1) lookup using hash table
    uint32_t index = graph_id_map_lookup(g, id);
    if (index == UINT32_MAX || index >= g->num_nodes) {
        return NULL;
    }
    
    // Verify the node at this index actually has the requested ID
    // (safety check in case of hash collision or corruption)
    if (g->nodes[index].id == id) {
        return &g->nodes[index];
    }
    
    // Hash collision or corruption - fall back to linear search (should be rare)
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].id == id) {
            return &g->nodes[i];
        }
    }
    return NULL;
}

void graph_propagate(Graph *g, size_t steps) {
    if (!g || steps == 0) return;
    
    float *next_a = calloc(g->num_nodes, sizeof(float));
    if (!next_a) return;
    
    for (size_t s = 0; s < steps; s++) {
        // zero next activations
        for (uint64_t i = 0; i < g->num_nodes; i++) {
            next_a[i] = 0.0f;
        }
        
        // push activation along edges using adjacency lists (local, neuron-like)
        // Iterate over nodes and traverse their outgoing edges
        for (uint64_t i = 0; i < g->num_nodes; i++) {
            Node *src = &g->nodes[i];
            if (src->a == 0.0f) continue;  // Skip inactive nodes
            
            // Traverse outgoing edges from this node
            // Lightweight safety check: detect obviously invalid edge indices
            uint32_t eid = src->first_out_edge;
            uint32_t visited_count = 0;
            uint32_t max_visit = 1024;  // Reasonable limit per node (debug safety only)
            
            while (eid != UINT32_MAX) {
                // Safety check: invalid edge index
                if (eid >= g->num_edges) {
                    #ifdef DEBUG
                    fprintf(stderr, "ERROR: Invalid edge index %u >= %llu in adjacency list for node %llu\n",
                            eid, (unsigned long long)g->num_edges, (unsigned long long)src->id);
                    #endif
                    break;
                }
                
                // Debug-only cycle detection (should not trigger in normal operation)
                if (visited_count++ > max_visit) {
                    #ifdef DEBUG
                    fprintf(stderr, "ERROR: Possible cycle in adjacency list for node %llu (visited %u edges)\n",
                            (unsigned long long)src->id, visited_count);
                    #endif
                    break;
                }
                
                Edge *edge = &g->edges[eid];
                
                // O(1) lookup for destination node
                uint32_t dst_idx = graph_id_map_lookup(g, edge->dst);
                if (dst_idx != UINT32_MAX && dst_idx < g->num_nodes) {
                    // simple linear influence
                    float contrib = src->a * edge->w;
                    next_a[dst_idx] += contrib;
                }
                
                eid = edge->next_out_edge;
            }
        }
        
        // apply nonlinearity / decay
        for (uint64_t i = 0; i < g->num_nodes; i++) {
            float x = next_a[i];
            
            // simple squashing + leak:
            // a_new = tanh(x) with small decay
            float a_new = x;
            if (a_new > 5.0f) a_new = 5.0f;
            if (a_new < -5.0f) a_new = -5.0f;
            
            // Normalize to [-1,1]
            a_new = a_new / 5.0f;
            
            // Decay factor
            a_new *= 0.95f;
            
            g->nodes[i].a = a_new;
        }
    }
    
    free(next_a);
}

float pattern_match_score(const Graph *g, const Node *pattern, uint64_t anchor_id) {
    if (!g || !pattern || pattern->kind != NODE_PATTERN) return 0.0f;
    
    size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
    if (num_atoms == 0) return 0.0f;
    
    const PatternAtom *atoms = (const PatternAtom *)(g->blob + pattern->payload_offset);
    
    size_t matches = 0;
    
    for (size_t i = 0; i < num_atoms; i++) {
        const PatternAtom *atom = &atoms[i];
        int64_t target_id = (int64_t)anchor_id + (int64_t)atom->delta;
        
        // Check if target_id is valid (non-negative for DATA nodes)
        if (target_id < 0) {
            continue; // partial mismatch
        }
        
        Node *target = graph_find_node_by_id(g, (uint64_t)target_id);
        if (!target || target->kind != NODE_DATA) {
            continue; // no DATA node at that position
        }
        
        if (atom->mode == 0) { // CONST_BYTE
            if (target->payload_len > 0) {
                uint8_t byte_val = g->blob[target->payload_offset];
                if (byte_val == atom->value) {
                    matches++;
                }
            }
        } else if (atom->mode == 1) { // BLANK
            matches++; // always accept
        }
    }
    
    return (float)matches / (float)num_atoms;
}

void graph_update_pattern_quality(Graph *g,
                                  Node *pattern,
                                  float delta_quality) {
    if (!g || !pattern || pattern->kind != NODE_PATTERN) return;
    
    pattern->q += delta_quality;
    
    // Clamp to [0.0, 1.0]
    if (pattern->q < 0.0f) pattern->q = 0.0f;
    if (pattern->q > 1.0f) pattern->q = 1.0f;
}

void graph_update_edge_from_error(Graph *g,
                                  uint64_t src_id,
                                  uint64_t dst_id,
                                  float error,
                                  float lr) {
    if (!g) return;
    
    // Find the edge using adjacency list (local, O(out_degree(src)) instead of O(E))
    Edge *edge = graph_find_edge(g, src_id, dst_id);
    if (edge) {
        Node *src = graph_find_node_by_id(g, src_id);
        if (src) {
            // Simple local rule: w += -lr * src_activation * error_at_dst
            edge->w += -lr * src->a * error;
        }
    }
}

void graph_print_stats(const Graph *g) {
    if (!g) return;
    
    uint64_t data_count = 0;
    uint64_t pattern_count = 0;
    uint64_t blank_count = 0;
    
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        switch (g->nodes[i].kind) {
            case NODE_DATA:
                data_count++;
                break;
            case NODE_PATTERN:
                pattern_count++;
                break;
            case NODE_BLANK:
                blank_count++;
                break;
        }
    }
    
    printf("Graph stats:\n");
    printf("  Nodes: %llu (DATA: %llu, PATTERN: %llu, BLANK: %llu)\n",
           (unsigned long long)g->num_nodes,
           (unsigned long long)data_count,
           (unsigned long long)pattern_count,
           (unsigned long long)blank_count);
    printf("  Edges: %llu\n", (unsigned long long)g->num_edges);
    printf("  Blob used: %llu / %llu bytes\n",
           (unsigned long long)g->blob_used,
           (unsigned long long)g->blob_cap);
    printf("  Next DATA pos: %llu\n", (unsigned long long)g->next_data_pos);
    printf("  Next PATTERN id: %llu\n", (unsigned long long)g->next_pattern_id);
}

// Log core stats for debugging (lightweight, no full graph scan)
void graph_log_core_stats(Graph *g) {
    if (!g) return;
    
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        if (g->nodes[i].kind == NODE_PATTERN) {
            pattern_count++;
        }
    }
    
    fprintf(stderr,
        "[core_stats] nodes=%llu edges=%llu patterns=%llu training=%d\n",
        (unsigned long long)g->num_nodes,
        (unsigned long long)g->num_edges,
        (unsigned long long)pattern_count,
        g_sys.training_enabled ? 1 : 0);
}

// Collect active OUTPUT nodes and emit top-k bytes (local, no global scans)
void graph_emit_output(Graph *g, size_t max_bytes, int fd) {
    if (!g || max_bytes == 0) return;
    
    // Collect OUTPUT nodes with activation > threshold
    typedef struct {
        float activation;
        uint8_t byte_value;
        Node *node;
    } OutputCandidate;
    
    OutputCandidate candidates[256];  // Max 256 possible bytes
    size_t num_candidates = 0;
    const float threshold = 0.1f;  // Minimum activation to emit
    
    // Scan OUTPUT nodes (O(N) but only OUTPUT nodes, typically small)
    for (uint64_t i = 0; i < g->num_nodes && num_candidates < 256; i++) {
        Node *node = &g->nodes[i];
        if (node->kind != NODE_OUTPUT) continue;
        if (node->a < threshold) continue;
        if (node->payload_len == 0) continue;
        
        uint8_t byte_val = g->blob[node->payload_offset];
        candidates[num_candidates].activation = node->a;
        candidates[num_candidates].byte_value = byte_val;
        candidates[num_candidates].node = node;
        num_candidates++;
    }
    
    // Sort by activation (simple insertion sort for small n)
    for (size_t i = 1; i < num_candidates; i++) {
        size_t j = i;
        while (j > 0 && candidates[j-1].activation < candidates[j].activation) {
            OutputCandidate tmp = candidates[j];
            candidates[j] = candidates[j-1];
            candidates[j-1] = tmp;
            j--;
        }
    }
    
    // Emit top-k bytes
    size_t to_emit = (num_candidates < max_bytes) ? num_candidates : max_bytes;
    for (size_t i = 0; i < to_emit; i++) {
        write(fd, &candidates[i].byte_value, 1);
    }
}

// Get value from VALUE node (graph-native parameter lookup)
float graph_get_value(const Graph *g, const char *value_name, float default_val) {
    if (!g || !value_name) return default_val;
    
    // Search for VALUE node with matching name in payload
    // For now, use a simple approach: search by payload content
    // TODO: Add proper name indexing if needed
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        Node *node = &g->nodes[i];
        if (node->kind != NODE_VALUE) continue;
        if (node->payload_len == 0) continue;
        
        // Check if payload matches value_name (simple string compare)
        const char *payload_str = (const char *)(g->blob + node->payload_offset);
        size_t name_len = strlen(value_name);
        if (node->payload_len >= name_len + 1 + sizeof(float)) {
            if (strncmp(payload_str, value_name, name_len) == 0 && payload_str[name_len] == '\0') {
                // Found matching VALUE node, extract float value
                float *val_ptr = (float *)(g->blob + node->payload_offset + name_len + 1);
                return *val_ptr;
            }
        }
    }
    
    return default_val;
}

// Create or find VALUE node
// DEPRECATED: Use graph-native creation instead (patterns/rules create VALUE nodes)
// Kept as static for migration only
static Node *graph_get_or_create_value_internal(Graph *g, const char *value_name, float default_val) {
    if (!g || !value_name) return NULL;
    
    // Search for existing VALUE node
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        Node *node = &g->nodes[i];
        if (node->kind != NODE_VALUE) continue;
        if (node->payload_len == 0) continue;
        
        const char *payload_str = (const char *)(g->blob + node->payload_offset);
        size_t name_len = strlen(value_name);
        if (node->payload_len >= name_len + 1 + sizeof(float)) {
            if (strncmp(payload_str, value_name, name_len) == 0 && payload_str[name_len] == '\0') {
                return node;  // Found existing
            }
        }
    }
    
    // Create new VALUE node using generic hardware function
    size_t name_len = strlen(value_name);
    size_t payload_size = name_len + 1 + sizeof(float);
    char *payload = malloc(payload_size);
    if (!payload) return NULL;
    
    memcpy(payload, value_name, name_len + 1);
    float *val_ptr = (float *)(payload + name_len + 1);
    *val_ptr = default_val;
    
    static uint64_t next_value_id = (1ULL << 61);
    Node *node = graph_create_node(g, NODE_VALUE, next_value_id++, payload, payload_size);
    free(payload);
    
    if (node) {
        node->a = default_val;  // Activation stores the value
    }
    return node;
}

// Local learning rule: strengthen pattern->OUTPUT edges when both are active
void local_update_pattern_to_output(Graph *g, Node *pattern_node, Node *output_node) {
    if (!g || !pattern_node || !output_node) return;
    if (pattern_node->kind != NODE_PATTERN) return;
    if (output_node->kind != NODE_OUTPUT) return;
    
    float activation_threshold = graph_get_value(g, "output_activation_threshold", 0.1f);
    if (pattern_node->a < activation_threshold || output_node->a < activation_threshold) {
        return;  // Both must be active
    }
    
    // Find or create edge
    Edge *edge = graph_find_edge(g, pattern_node->id, output_node->id);
    if (!edge) {
        // Create edge with initial weight
        edge = graph_add_edge(g, pattern_node->id, output_node->id, 0.5f);
        if (!edge) return;
    }
    
    // Strengthen edge weight (get learning rate from VALUE node)
    float learning_rate = graph_get_value(g, "output_learning_rate", 0.1f);
    float max_weight = graph_get_value(g, "max_output_weight", 2.0f);
    
    edge->w += learning_rate;
    if (edge->w > max_weight) {
        edge->w = max_weight;
    }
}

// Helper: get pattern span length (how many DATA positions it covers)
static size_t get_pattern_span_length(const Graph *g, const Node *pattern) {
    if (!g || !pattern || pattern->kind != NODE_PATTERN) return 0;
    
    size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
    if (num_atoms == 0) return 0;
    
    const PatternAtom *atoms = (const PatternAtom *)(g->blob + pattern->payload_offset);
    int16_t max_delta = 0;
    for (size_t i = 0; i < num_atoms; i++) {
        if (atoms[i].delta > max_delta) max_delta = atoms[i].delta;
    }
    
    return (size_t)(max_delta + 1);
}

// Build symbol sequence from episode using greedy cover with existing patterns
// Local: only uses matches in the Explanation, no global scans
void build_symbol_sequence_from_episode(Graph *g,
                                        const Explanation *exp,
                                        uint64_t start_id,
                                        uint64_t end_id,
                                        uint64_t *out_seq,
                                        uint16_t *out_len,
                                        uint16_t max_len) {
    if (!g || !exp || !out_seq || !out_len || max_len == 0) {
        if (out_len) *out_len = 0;
        return;
    }
    
    if (start_id > end_id) {
        *out_len = 0;
        return;
    }
    
    // Track which positions are covered
    size_t span_len = (size_t)(end_id - start_id + 1);
    int *covered = calloc(span_len, sizeof(int));
    if (!covered) {
        *out_len = 0;
        return;
    }
    
    // Build list of pattern applications with their spans
    typedef struct {
        uint64_t pattern_id;
        uint64_t anchor_id;
        size_t span_len;
        float quality;  // For tie-breaking
    } PatternMatch;
    
    PatternMatch matches[256];  // Max matches
    size_t num_matches = 0;
    
    for (size_t i = 0; i < exp->count && num_matches < 256; i++) {
        uint64_t pattern_id = exp->apps[i].pattern_id;
        uint64_t anchor_id = exp->apps[i].anchor_id;
        
        Node *pattern = graph_find_node_by_id(g, pattern_id);
        if (!pattern || pattern->kind != NODE_PATTERN) continue;
        
        size_t span = get_pattern_span_length(g, pattern);
        if (span == 0) continue;
        
        // Check if anchor is within our span
        if (anchor_id < start_id || anchor_id > end_id) continue;
        
        matches[num_matches].pattern_id = pattern_id;
        matches[num_matches].anchor_id = anchor_id;
        matches[num_matches].span_len = span;
        matches[num_matches].quality = pattern->q;
        num_matches++;
    }
    
    // Sort matches by: 1) anchor position, 2) span length (descending), 3) quality (descending)
    for (size_t i = 0; i < num_matches; i++) {
        for (size_t j = i + 1; j < num_matches; j++) {
            if (matches[j].anchor_id < matches[i].anchor_id ||
                (matches[j].anchor_id == matches[i].anchor_id &&
                 (matches[j].span_len > matches[i].span_len ||
                  (matches[j].span_len == matches[i].span_len && matches[j].quality > matches[i].quality)))) {
                PatternMatch tmp = matches[i];
                matches[i] = matches[j];
                matches[j] = tmp;
            }
        }
    }
    
    // Greedy cover: pick longest pattern at each position
    uint16_t seq_len = 0;
    uint64_t pos = start_id;
    
    while (pos <= end_id && seq_len < max_len) {
        // Find longest pattern starting at pos
        size_t best_match_idx = SIZE_MAX;
        size_t best_span = 0;
        
        for (size_t i = 0; i < num_matches; i++) {
            if (matches[i].anchor_id == pos) {
                // Check if this match is still uncovered
                size_t anchor_offset = (size_t)(matches[i].anchor_id - start_id);
                int all_uncovered = 1;
                for (size_t j = 0; j < matches[i].span_len && anchor_offset + j < span_len; j++) {
                    if (covered[anchor_offset + j]) {
                        all_uncovered = 0;
                        break;
                    }
                }
                
                if (all_uncovered && matches[i].span_len > best_span) {
                    best_match_idx = i;
                    best_span = matches[i].span_len;
                }
            }
        }
        
        if (best_match_idx != SIZE_MAX) {
            // Use pattern
            out_seq[seq_len++] = matches[best_match_idx].pattern_id;
            
            // Mark positions as covered
            size_t anchor_offset = (size_t)(matches[best_match_idx].anchor_id - start_id);
            for (size_t j = 0; j < matches[best_match_idx].span_len && anchor_offset + j < span_len; j++) {
                covered[anchor_offset + j] = 1;
            }
            
            pos += matches[best_match_idx].span_len;
        } else {
            // Fall back to DATA node
            out_seq[seq_len++] = pos;
            size_t pos_offset = (size_t)(pos - start_id);
            if (pos_offset < span_len) {
                covered[pos_offset] = 1;
            }
            pos++;
        }
    }
    
    free(covered);
    *out_len = seq_len;
}

// Create pattern from two symbol sequences, using blanks where they differ
Node *graph_create_pattern_from_sequences(Graph *g,
                                          const uint64_t *seq1, uint16_t len1,
                                          const uint64_t *seq2, uint16_t len2) {
    if (!g || !seq1 || !seq2 || len1 == 0 || len2 == 0) return NULL;
    
    // Find minimum length
    uint16_t min_len = (len1 < len2) ? len1 : len2;
    if (min_len == 0) return NULL;
    
    // Build pattern atoms
    PatternAtom atoms[64];  // Max pattern length
    size_t num_atoms = 0;
    int16_t delta = 0;
    
    for (uint16_t i = 0; i < min_len && num_atoms < 64; i++) {
        if (seq1[i] == seq2[i]) {
            // Same node ID - use concrete reference
            // For DATA nodes, we need to extract the byte value
            // For PATTERN nodes, we reference the pattern node ID
            Node *node = graph_find_node_by_id(g, seq1[i]);
            if (!node) continue;
            
            if (node->kind == NODE_DATA) {
                // DATA node: extract byte value
                if (node->payload_len > 0) {
                    atoms[num_atoms].delta = delta;
                    atoms[num_atoms].mode = 0;  // CONST_BYTE
                    atoms[num_atoms].value = g->blob[node->payload_offset];
                    num_atoms++;
                    delta++;
                }
            } else if (node->kind == NODE_PATTERN) {
                // PATTERN node: we can't directly reference it in a PatternAtom
                // Instead, we use a blank (patterns reference patterns via edges, not atoms)
                atoms[num_atoms].delta = delta;
                atoms[num_atoms].mode = 1;  // BLANK (accepts any value)
                atoms[num_atoms].value = 0;
                num_atoms++;
                delta++;
            }
        } else {
            // Different node IDs - use blank
            atoms[num_atoms].delta = delta;
            atoms[num_atoms].mode = 1;  // BLANK
            atoms[num_atoms].value = 0;
            num_atoms++;
            delta++;
        }
    }
    
    if (num_atoms == 0) return NULL;
    
    // Create pattern
    return graph_add_pattern(g, atoms, num_atoms, 0.5f);
}

// Build symbol sequence from EPISODE node (graph-native version)
void build_symbol_sequence_from_episode_node(Graph *g,
                                            uint64_t episode_node_id,
                                            uint64_t *out_seq,
                                            uint16_t *out_len,
                                            uint16_t max_len) {
    if (!g || !out_seq || !out_len || max_len == 0) {
        if (out_len) *out_len = 0;
        return;
    }
    
    Node *episode_node = graph_find_node_by_id(g, episode_node_id);
    if (!episode_node || episode_node->kind != NODE_EPISODE) {
        if (out_len) *out_len = 0;
        return;
    }
    
    // Extract start_id and end_id from episode payload
    if (episode_node->payload_len < 2 * sizeof(uint64_t)) {
        if (out_len) *out_len = 0;
        return;
    }
    
    uint64_t *ids = (uint64_t *)(g->blob + episode_node->payload_offset);
    uint64_t start_id = ids[0];
    uint64_t end_id = ids[1];
    
    if (start_id > end_id) {
        if (out_len) *out_len = 0;
        return;
    }
    
    // Collect APPLICATION nodes connected to this EPISODE
    // Walk incoming edges to EPISODE node
    typedef struct {
        uint64_t pattern_id;
        uint64_t anchor_id;
        size_t span_len;
        float quality;
        Node *app_node;
    } PatternMatch;
    
    PatternMatch matches[256];
    size_t num_matches = 0;
    
    uint32_t eid = episode_node->first_in_edge;
    uint32_t visited = 0;
    while (eid != UINT32_MAX && eid < g->num_edges && visited < 1000 && num_matches < 256) {
        visited++;
        Edge *e = &g->edges[eid];
        Node *app_node = graph_find_node_by_id(g, e->src);
        
        if (app_node && app_node->kind == NODE_APPLICATION) {
            // Extract pattern_id, anchor_id, score from APPLICATION payload
            if (app_node->payload_len >= sizeof(uint64_t) * 2 + sizeof(float)) {
                uint64_t *pattern_id_ptr = (uint64_t *)(g->blob + app_node->payload_offset);
                uint64_t *anchor_id_ptr = pattern_id_ptr + 1;
                
                uint64_t pattern_id = *pattern_id_ptr;
                uint64_t anchor_id = *anchor_id_ptr;
                
                Node *pattern = graph_find_node_by_id(g, pattern_id);
                if (pattern && pattern->kind == NODE_PATTERN) {
                    size_t span = get_pattern_span_length(g, pattern);
                    if (span > 0 && anchor_id >= start_id && anchor_id <= end_id) {
                        matches[num_matches].pattern_id = pattern_id;
                        matches[num_matches].anchor_id = anchor_id;
                        matches[num_matches].span_len = span;
                        matches[num_matches].quality = pattern->q;
                        matches[num_matches].app_node = app_node;
                        num_matches++;
                    }
                }
            }
        }
        
        eid = e->next_in_edge;
    }
    
    // Sort matches (same as before)
    for (size_t i = 0; i < num_matches; i++) {
        for (size_t j = i + 1; j < num_matches; j++) {
            if (matches[j].anchor_id < matches[i].anchor_id ||
                (matches[j].anchor_id == matches[i].anchor_id &&
                 (matches[j].span_len > matches[i].span_len ||
                  (matches[j].span_len == matches[i].span_len && matches[j].quality > matches[i].quality)))) {
                PatternMatch tmp = matches[i];
                matches[i] = matches[j];
                matches[j] = tmp;
            }
        }
    }
    
    // Greedy cover (same algorithm as before)
    size_t span_len = (size_t)(end_id - start_id + 1);
    int *covered = calloc(span_len, sizeof(int));
    if (!covered) {
        if (out_len) *out_len = 0;
        return;
    }
    
    uint16_t seq_len = 0;
    uint64_t pos = start_id;
    
    while (pos <= end_id && seq_len < max_len) {
        size_t best_match_idx = SIZE_MAX;
        size_t best_span = 0;
        
        for (size_t i = 0; i < num_matches; i++) {
            if (matches[i].anchor_id == pos) {
                size_t anchor_offset = (size_t)(matches[i].anchor_id - start_id);
                int all_uncovered = 1;
                for (size_t j = 0; j < matches[i].span_len && anchor_offset + j < span_len; j++) {
                    if (covered[anchor_offset + j]) {
                        all_uncovered = 0;
                        break;
                    }
                }
                
                if (all_uncovered && matches[i].span_len > best_span) {
                    best_match_idx = i;
                    best_span = matches[i].span_len;
                }
            }
        }
        
        if (best_match_idx != SIZE_MAX) {
            out_seq[seq_len++] = matches[best_match_idx].pattern_id;
            size_t anchor_offset = (size_t)(matches[best_match_idx].anchor_id - start_id);
            for (size_t j = 0; j < matches[best_match_idx].span_len && anchor_offset + j < span_len; j++) {
                covered[anchor_offset + j] = 1;
            }
            pos += matches[best_match_idx].span_len;
        } else {
            out_seq[seq_len++] = pos;
            size_t pos_offset = (size_t)(pos - start_id);
            if (pos_offset < span_len) {
                covered[pos_offset] = 1;
            }
            pos++;
        }
    }
    
    free(covered);
    *out_len = seq_len;
}

// Graph-native learning: run local rules for active LEARNER nodes
void graph_run_local_rules(Graph *g) {
    if (!g) return;
    
    // Find all active LEARNER nodes
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        Node *learner = &g->nodes[i];
        if (learner->kind != NODE_LEARNER) continue;
        if (learner->a < 0.1f) continue;  // Must be active
        
        // Get learning parameters from VALUE nodes
        float min_pattern_len = graph_get_value(g, "min_pattern_len", 2.0f);
        float max_pattern_len = graph_get_value(g, "max_pattern_len", 16.0f);
        
        // Find EPISODE nodes connected to this LEARNER
        // Walk outgoing edges from LEARNER
        uint32_t eid = learner->first_out_edge;
        uint32_t visited = 0;
        Node *episodes[16];
        size_t num_episodes = 0;
        
        while (eid != UINT32_MAX && eid < g->num_edges && visited < 100 && num_episodes < 16) {
            visited++;
            Edge *e = &g->edges[eid];
            Node *episode = graph_find_node_by_id(g, e->dst);
            
            if (episode && episode->kind == NODE_EPISODE) {
                episodes[num_episodes++] = episode;
            }
            
            eid = e->next_out_edge;
        }
        
        // For each pair of episodes, try to create a pattern
        for (size_t i = 0; i < num_episodes; i++) {
            for (size_t j = i + 1; j < num_episodes; j++) {
                // Build symbol sequences from both episodes
                uint64_t seq1[64], seq2[64];
                uint16_t len1 = 0, len2 = 0;
                
                build_symbol_sequence_from_episode_node(g, episodes[i]->id, seq1, &len1, 64);
                build_symbol_sequence_from_episode_node(g, episodes[j]->id, seq2, &len2, 64);
                
                if (len1 >= (uint16_t)min_pattern_len && len2 >= (uint16_t)min_pattern_len &&
                    len1 <= (uint16_t)max_pattern_len && len2 <= (uint16_t)max_pattern_len) {
                    // Create pattern from sequences
                    Node *new_pattern = graph_create_pattern_from_sequences(g, seq1, len1, seq2, len2);
                    if (new_pattern) {
                        fprintf(stderr, "[graph_learn] created pattern %llu from EPISODE %llu and EPISODE %llu via LEARNER %llu\n",
                                (unsigned long long)new_pattern->id,
                                (unsigned long long)episodes[i]->id,
                                (unsigned long long)episodes[j]->id,
                                (unsigned long long)learner->id);
                    }
                }
            }
        }
    }
    
    // Find all active MAINTENANCE nodes and apply maintenance rules
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        Node *maintenance = &g->nodes[i];
        if (maintenance->kind != NODE_MAINTENANCE) continue;
        if (maintenance->a < 0.1f) continue;  // Must be active
        
        // Get maintenance parameters from VALUE nodes
        float min_pattern_usage = graph_get_value(g, "min_pattern_usage", 3.0f);
        float min_edge_usage = graph_get_value(g, "min_edge_usage", 2.0f);
        float max_age_without_activation = graph_get_value(g, "max_age_without_activation", 1000.0f);
        float edge_decay_rate = graph_get_value(g, "edge_decay_rate", 0.99f);
        float min_edge_weight = graph_get_value(g, "min_edge_weight", 0.01f);
        float maintenance_work_budget = graph_get_value(g, "maintenance_work_budget", 10.0f);
        
        // Track work done (bounded per tick)
        size_t work_done = 0;
        size_t max_work = (size_t)maintenance_work_budget;
        
        // Find patterns and edges connected to this MAINTENANCE node
        // Walk outgoing edges from MAINTENANCE
        uint32_t eid = maintenance->first_out_edge;
        uint32_t visited = 0;
        // uint32_t patterns_checked = 0;  // Unused - kept for future use
        
        while (eid != UINT32_MAX && eid < g->num_edges && visited < 1000 && work_done < max_work) {
            visited++;
            Edge *e = &g->edges[eid];
            Node *target = graph_find_node_by_id(g, e->dst);
            
            if (target) {
                if (target->kind == NODE_PATTERN) {
                    patterns_checked++;
                    // Check pattern usage (quality can proxy for usage)
                    // Low quality = candidate for removal
                    if (target->q < min_pattern_usage) {
                        // Mark pattern for removal by setting a flag or zeroing quality
                        // Decay quality further (get decay rate from VALUE node)
                        float pattern_decay_rate = graph_get_value(g, "pattern_decay_rate", 0.9f);
                        target->q *= pattern_decay_rate;
                        if (target->q < 0.001f) {
                            // Effectively remove by making it unreachable
                            // We can't actually delete nodes yet, so just mark as dead
                            target->flags |= 1;  // Mark as deleted
                        }
                        work_done++;
                    }
                } else if (target->kind == NODE_DATA || target->kind == NODE_PATTERN) {
                    // Check edges from this target node
                    uint32_t out_eid = target->first_out_edge;
                    uint32_t out_visited = 0;
                    
                    while (out_eid != UINT32_MAX && out_eid < g->num_edges && 
                           out_visited < 50 && work_done < max_work) {
                        out_visited++;
                        Edge *out_e = &g->edges[out_eid];
                        
                        // Decay low-weight edges
                        if (out_e->w < min_edge_weight) {
                            // Remove edge if weight is too low
                            graph_remove_edge(g, out_eid);
                            work_done++;
                        } else if (out_e->w < 0.5f) {
                            // Decay weak edges
                            out_e->w *= edge_decay_rate;
                            if (out_e->w < min_edge_weight) {
                                graph_remove_edge(g, out_eid);
                            }
                            work_done++;
                        }
                        
                        out_eid = out_e->next_out_edge;
                    }
                }
            }
            
            eid = e->next_out_edge;
        }
        
        if (work_done > 0) {
            fprintf(stderr, "[maintenance] node %llu processed %zu items\n",
                    (unsigned long long)maintenance->id, work_done);
        }
    }
}

// Self-consistency API (Phase 2)

size_t graph_collect_data_span(const Graph *g,
                                uint64_t start_id,
                                uint8_t *out,
                                size_t max_len) {
    if (!g || !out || max_len == 0) return 0;
    
    size_t count = 0;
    for (size_t i = 0; i < max_len; i++) {
        uint64_t id = start_id + i;
        Node *n = graph_find_node_by_id((Graph *)g, id);
        if (!n || n->kind != NODE_DATA || n->payload_len < 1) {
            break;
        }
        out[count++] = g->blob[n->payload_offset];
    }
    return count;
}

size_t pattern_reconstruct_segment(const Graph *g,
                                   const Node *pattern,
                                   uint64_t anchor_id,
                                   uint8_t *out,
                                   size_t segment_len) {
    (void)anchor_id; // anchor_id is part of API but not used in reconstruction
    if (!g || !pattern || !out || segment_len == 0) return 0;
    if (pattern->kind != NODE_PATTERN) return 0;
    
    size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
    if (num_atoms == 0) return 0;
    
    const PatternAtom *atoms =
        (const PatternAtom *)(g->blob + pattern->payload_offset);
    
    size_t written = 0;
    for (size_t i = 0; i < num_atoms; i++) {
        PatternAtom atom = atoms[i];
        if (atom.mode == 0) { // CONST_BYTE
            int64_t idx = (int64_t)atom.delta;
            if (idx < 0) continue; // ignore negative deltas for now in reconstruction
            if ((size_t)idx >= segment_len) continue;
            
            out[idx] = atom.value;
            written++;
        }
        // mode == BLANK: do nothing, leave out[idx] as-is
    }
    return written;
}

float graph_self_consistency_pass(Graph *g,
                                   Node *pattern,
                                   uint64_t start_id,
                                   uint64_t end_id,
                                   float match_threshold,
                                   float lr_q) {
    if (!g || !pattern || start_id >= end_id) return 0.0f;
    if (pattern->kind != NODE_PATTERN) return 0.0f;
    
    size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
    if (num_atoms == 0) return 0.0f;
    
    size_t total_errors = 0;
    size_t total_positions = 0;
    
    // For simplicity, define a local segment length equal to
    // the max positive delta + 1
    int16_t max_delta = 0;
    const PatternAtom *atoms =
        (const PatternAtom *)(g->blob + pattern->payload_offset);
    for (size_t i = 0; i < num_atoms; i++) {
        if (atoms[i].delta > max_delta) max_delta = atoms[i].delta;
    }
    size_t segment_len = (size_t)(max_delta + 1);
    if (segment_len == 0) return 0.0f;
    
    uint8_t *actual  = malloc(segment_len);
    uint8_t *pred    = malloc(segment_len);
    if (!actual || !pred) {
        free(actual);
        free(pred);
        return 0.0f;
    }
    
    for (uint64_t anchor_id = start_id; anchor_id <= end_id; anchor_id++) {
        float score = pattern_match_score(g, pattern, anchor_id);
        if (score < match_threshold) continue;
        
        // 1) Collect actual bytes under this anchor
        for (size_t i = 0; i < segment_len; i++) {
            actual[i] = 0x00;
            pred[i]   = 0x00;
        }
        
        size_t got = graph_collect_data_span(g, anchor_id, actual, segment_len);
        if (got == 0) continue;
        
        // 2) Let the pattern write its predicted bytes into pred[]
        size_t written = pattern_reconstruct_segment(g, pattern, anchor_id, pred, segment_len);
        if (written == 0) continue;
        
        // 3) Compare actual vs pred at positions where pred wrote something
        size_t positions = 0;
        size_t errors    = 0;
        for (size_t i = 0; i < segment_len; i++) {
            if (pred[i] == 0x00) continue; // only look at positions we predicted
            positions++;
            if (i >= got) {
                errors++;
                continue;
            }
            if (pred[i] != actual[i]) {
                errors++;
            }
        }
        
        if (positions == 0) continue;
        
        total_errors    += errors;
        total_positions += positions;
    }
    
    if (total_positions == 0) {
        free(actual);
        free(pred);
        return 0.0f;
    }
    
    float avg_error = (float)total_errors / (float)total_positions;
    
    // Use 1 - error as "goodness" and nudge pattern quality q
    float goodness = 1.0f - avg_error;
    pattern->q += lr_q * (goodness - pattern->q);
    if (pattern->q < 0.0f) pattern->q = 0.0f;
    if (pattern->q > 1.0f) pattern->q = 1.0f;
    
    free(actual);
    free(pred);
    return avg_error;
}

// Explanation API (Phase 3)

void explanation_init(Explanation *exp) {
    if (!exp) return;
    exp->apps = NULL;
    exp->count = 0;
    exp->capacity = 0;
}

void explanation_free(Explanation *exp) {
    if (!exp) return;
    free(exp->apps);
    exp->apps = NULL;
    exp->count = 0;
    exp->capacity = 0;
}

void explanation_add(Explanation *exp, uint64_t pattern_id, uint64_t anchor_id) {
    if (!exp) return;
    if (exp->count == exp->capacity) {
        size_t new_cap = exp->capacity ? exp->capacity * 2 : 8;
        PatternApplication *new_apps =
            (PatternApplication *)realloc(exp->apps, new_cap * sizeof(PatternApplication));
        if (!new_apps) return;
        exp->apps = new_apps;
        exp->capacity = new_cap;
    }
    exp->apps[exp->count].pattern_id = pattern_id;
    exp->apps[exp->count].anchor_id  = anchor_id;
    exp->count++;
}

void graph_build_explanation_single_pattern(const Graph *g,
                                            const Node *pattern,
                                            uint64_t start_id,
                                            uint64_t end_id,
                                            float match_threshold,
                                            Explanation *out) {
    if (!g || !pattern || !out) return;
    if (pattern->kind != NODE_PATTERN) return;
    if (start_id > end_id) return;
    
    for (uint64_t anchor_id = start_id; anchor_id <= end_id; anchor_id++) {
        float score = pattern_match_score(g, pattern, anchor_id);
        if (score >= match_threshold) {
            explanation_add(out, pattern->id, anchor_id);
        }
    }
}

size_t graph_reconstruct_from_explanation(const Graph *g,
                                          const Explanation *exp,
                                          uint64_t start_id,
                                          uint64_t end_id,
                                          uint8_t *out,
                                          size_t out_cap) {
    if (!g || !exp || !out) return 0;
    if (start_id > end_id) return 0;
    
    size_t len = (size_t)(end_id - start_id + 1);
    if (out_cap < len) return 0;
    
    // initialize with 0x00 meaning "no prediction"
    for (size_t i = 0; i < len; i++) {
        out[i] = 0x00;
    }
    
    size_t written = 0;
    
    for (size_t i = 0; i < exp->count; i++) {
        uint64_t pattern_id = exp->apps[i].pattern_id;
        uint64_t anchor_id  = exp->apps[i].anchor_id;
        
        Node *pattern = graph_find_node_by_id((Graph *)g, pattern_id);
        if (!pattern || pattern->kind != NODE_PATTERN) continue;
        
        // Determine max positive delta to figure local segment size
        size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
        if (num_atoms == 0) continue;
        
        const PatternAtom *atoms =
            (const PatternAtom *)(g->blob + pattern->payload_offset);
        
        int16_t max_delta = 0;
        for (size_t j = 0; j < num_atoms; j++) {
            if (atoms[j].delta > max_delta) max_delta = atoms[j].delta;
        }
        
        size_t local_len = (size_t)(max_delta + 1);
        if (local_len == 0) continue;
        
        // local buffer for pattern-predicted bytes
        uint8_t *local = (uint8_t *)malloc(local_len);
        if (!local) continue;
        for (size_t j = 0; j < local_len; j++) {
            local[j] = 0x00;
        }
        
        pattern_reconstruct_segment(g, pattern, anchor_id, local, local_len);
        
        // Merge into global segment (last-writer-wins)
        for (size_t j = 0; j < local_len; j++) {
            if (local[j] == 0x00) continue; // pattern didn't predict here
            
            uint64_t global_pos = anchor_id + (uint64_t)j;
            if (global_pos < start_id || global_pos > end_id) continue;
            
            size_t idx = (size_t)(global_pos - start_id);
            if (out[idx] == 0x00) written++;
            out[idx] = local[j];
        }
        
        free(local);
    }
    
    return written;
}

float graph_self_consistency_episode_single_pattern(Graph *g,
                                                     Node *pattern,
                                                     uint64_t start_id,
                                                     uint64_t end_id,
                                                     float match_threshold,
                                                     float lr_q) {
    if (!g || !pattern) return 0.0f;
    if (pattern->kind != NODE_PATTERN) return 0.0f;
    if (start_id > end_id) return 0.0f;
    
    size_t len = (size_t)(end_id - start_id + 1);
    if (len == 0) return 0.0f;
    
    // 1) Collect actual bytes for the segment
    uint8_t *actual = (uint8_t *)malloc(len);
    uint8_t *pred   = (uint8_t *)malloc(len);
    if (!actual || !pred) {
        free(actual);
        free(pred);
        return 0.0f;
    }
    
    size_t got = graph_collect_data_span(g, start_id, actual, len);
    if (got < len) {
        // If we didn't get the full span, adjust len to what we got
        len = got;
    }
    if (len == 0) {
        free(actual);
        free(pred);
        return 0.0f;
    }
    
    // 2) Build explanation using the single pattern
    Explanation exp;
    explanation_init(&exp);
    
    graph_build_explanation_single_pattern(g,
                                          pattern,
                                          start_id,
                                          start_id + len - 1,
                                          match_threshold,
                                          &exp);
    
    // 3) Reconstruct from explanation
    (void)graph_reconstruct_from_explanation(g,
                                             &exp,
                                             start_id,
                                             start_id + len - 1,
                                             pred,
                                             len);
    
    // 4) Compare actual vs predicted where we have predictions
    size_t positions = 0;
    size_t errors    = 0;
    
    for (size_t i = 0; i < len; i++) {
        if (pred[i] == 0x00) continue; // no prediction here
        positions++;
        if (pred[i] != actual[i]) {
            errors++;
        }
    }
    
    float avg_error = 0.0f;
    if (positions > 0) {
        avg_error = (float)errors / (float)positions;
        
        // 5) Update pattern quality toward (1 - avg_error)
        // Only update if we actually made predictions (positions > 0)
        float target_q = 1.0f - avg_error;   // good if error small
        pattern->q += lr_q * (target_q - pattern->q);
        if (pattern->q < 0.0f) pattern->q = 0.0f;
        if (pattern->q > 1.0f) pattern->q = 1.0f;
    }
    // If positions == 0, no predictions were made, so don't update quality
    
    explanation_free(&exp);
    free(actual);
    free(pred);
    
    return avg_error; // in [0,1], 0 = perfect consistency
}

// ============================================================================
// LEGACY GLOBAL LEARNING (TO BE REPLACED)
// ============================================================================
// This section contains non-local, O(patterns × anchors) learning code.
// It MUST NOT run in runtime mode.
// Training-enabled runs may call this, but it is a known scalability risk.
//
// These functions perform global scans over patterns and anchors,
// violating the "C is frozen hardware, graph is the brain" principle.
// They are kept here for backward compatibility during training,
// but should eventually be replaced with graph-native, local learning rules.
// ============================================================================

// Helper: compute application score for sorting
static float legacy_compute_app_score(const Graph *g,
                                      uint64_t pattern_id,
                                      uint64_t anchor_id,
                                      float epsilon) {
    Node *pattern = graph_find_node_by_id((Graph *)g, pattern_id);
    if (!pattern || pattern->kind != NODE_PATTERN) return 0.0f;
    
    float match_score = pattern_match_score(g, pattern, anchor_id);
    float q = pattern->q;
    
    return match_score * (epsilon + q);
}

// Legacy multi-pattern candidate collection
// WARNING: Performs O(patterns × anchors) scan - only use in training mode
void legacy_collect_candidates_multi_pattern(const Graph *g,
                                            Node *const *patterns,
                                            size_t num_patterns,
                                            uint64_t start_id,
                                            uint64_t end_id,
                                            float match_threshold,
                                            Explanation *out_candidates) {
    if (!g || !patterns || !out_candidates) return;
    if (start_id > end_id) return;
    
    // Runtime mode: skip heavy learning work
    if (!g_sys.training_enabled) {
        return;  // Fast inference mode - no pattern scanning
    }
    
    for (size_t p = 0; p < num_patterns; p++) {
        Node *pattern = patterns[p];
        if (!pattern || pattern->kind != NODE_PATTERN) continue;
        
        for (uint64_t anchor_id = start_id; anchor_id <= end_id; anchor_id++) {
            float score = pattern_match_score(g, pattern, anchor_id);
            if (score >= match_threshold) {
                explanation_add(out_candidates, pattern->id, anchor_id);
            }
        }
    }
}

// Legacy multi-pattern self-consistency episode
// WARNING: Performs O(patterns × anchors) scan - only use in training mode
float legacy_self_consistency_episode_multi_pattern(Graph *g,
                                                   Node *const *patterns,
                                                   size_t num_patterns,
                                                   uint64_t start_id,
                                                   uint64_t end_id,
                                                   float match_threshold,
                                                   float lr_q) {
    if (!g || !patterns || num_patterns == 0) return 0.0f;
    if (start_id > end_id) return 0.0f;
    
    // Runtime mode: skip heavy learning work
    if (!g_sys.training_enabled) {
        return 0.0f;  // Fast inference mode - no learning episodes
    }
    
    size_t len = (size_t)(end_id - start_id + 1);
    if (len == 0) return 0.0f;
    
    // 1) Collect actual bytes
    uint8_t *actual = malloc(len);
    if (!actual) return 0.0f;
    
    size_t got = graph_collect_data_span(g, start_id, actual, len);
    if (got < len) len = got;
    if (len == 0) {
        free(actual);
        return 0.0f;
    }
    
    // 2) Collect candidates from all patterns
    Explanation candidates;
    explanation_init(&candidates);
    legacy_collect_candidates_multi_pattern(g, patterns, num_patterns,
                                          start_id, end_id,
                                          match_threshold, &candidates);
    
    // 3) Select consistent subset
    Explanation selected;
    explanation_init(&selected);
    explanation_select_greedy_consistent(g, &candidates,
                                        start_id, end_id, &selected);
    
    // 4) Reconstruct from selected explanation
    uint8_t *pred = malloc(len);
    if (!pred) {
        explanation_free(&candidates);
        explanation_free(&selected);
        free(actual);
        return 0.0f;
    }
    
    graph_reconstruct_from_explanation(g, &selected,
                                      start_id, end_id,
                                      pred, len);
    
    // 5) Compute global error
    size_t positions = 0;
    size_t errors = 0;
    for (size_t i = 0; i < len; i++) {
        if (pred[i] != 0x00) {
            positions++;
            if (pred[i] != actual[i]) {
                errors++;
            }
        }
    }
    
    float avg_error = (positions > 0) ? (float)errors / (float)positions : 1.0f;
    
    // 6) Update each pattern's quality based on its contribution
    // For simplicity, update all patterns equally based on global error
    // (A more sophisticated version would track per-pattern contributions)
    for (size_t p = 0; p < num_patterns; p++) {
        Node *pattern = patterns[p];
        if (!pattern || pattern->kind != NODE_PATTERN) continue;
        
        // Update quality toward (1 - error)
        float target_q = 1.0f - avg_error;
        pattern->q += lr_q * (target_q - pattern->q);
        if (pattern->q < 0.0f) pattern->q = 0.0f;
        if (pattern->q > 1.0f) pattern->q = 1.0f;
    }
    
    explanation_free(&candidates);
    explanation_free(&selected);
    free(actual);
    free(pred);
    
    return avg_error; // in [0,1], 0 = perfect consistency
}

// Multi-pattern explanation API (Phase 4)
// Wrapper: delegates to legacy learning when training is enabled
void graph_collect_candidates_multi_pattern(const Graph *g,
                                            Node *const *patterns,
                                            size_t num_patterns,
                                            uint64_t start_id,
                                            uint64_t end_id,
                                            float match_threshold,
                                            Explanation *out_candidates) {
    // Only call legacy learning if training is enabled
    if (g_sys.training_enabled) {
        legacy_collect_candidates_multi_pattern(g, patterns, num_patterns,
                                               start_id, end_id,
                                               match_threshold, out_candidates);
    }
    // Runtime mode: do nothing (no global scans)
}

// Helper: compute application score for sorting
static float compute_app_score(const Graph *g,
                                uint64_t pattern_id,
                                uint64_t anchor_id,
                                float epsilon) {
    Node *pattern = graph_find_node_by_id((Graph *)g, pattern_id);
    if (!pattern || pattern->kind != NODE_PATTERN) return 0.0f;
    
    float match_score = pattern_match_score(g, pattern, anchor_id);
    float q = pattern->q;
    
    return match_score * (epsilon + q);
}

void explanation_select_greedy_consistent(const Graph *g,
                                           const Explanation *candidates,
                                           uint64_t start_id,
                                           uint64_t end_id,
                                           Explanation *out_selected) {
    if (!g || !candidates || !out_selected) return;
    if (start_id > end_id) return;
    
    size_t len = (size_t)(end_id - start_id + 1);
    if (len == 0) return;
    
    // Initialize prediction buffer
    uint8_t *pred = calloc(len, sizeof(uint8_t));
    if (!pred) return;
    
    // Create index array for sorting
    size_t *indices = malloc(candidates->count * sizeof(size_t));
    if (!indices) {
        free(pred);
        return;
    }
    
    for (size_t i = 0; i < candidates->count; i++) {
        indices[i] = i;
    }
    
    // Sort indices by app_score descending using quicksort-like approach
    // Pre-compute scores to avoid repeated expensive pattern_match_score calls
    const float epsilon = 0.1f;
    float *scores = malloc(candidates->count * sizeof(float));
    if (scores) {
        // Pre-compute all scores once (O(n) instead of O(n²))
        for (size_t i = 0; i < candidates->count; i++) {
            scores[i] = compute_app_score(g,
                                          candidates->apps[i].pattern_id,
                                          candidates->apps[i].anchor_id,
                                          epsilon);
        }
        
        // Sort indices by pre-computed scores (simple insertion sort for small n, or use qsort)
        for (size_t i = 1; i < candidates->count; i++) {
            size_t j = i;
            while (j > 0 && scores[indices[j-1]] < scores[indices[j]]) {
                size_t tmp = indices[j];
                indices[j] = indices[j-1];
                indices[j-1] = tmp;
                j--;
            }
        }
        
        free(scores);
    } else {
        // Fallback: original O(n²) method if allocation fails
        for (size_t i = 0; i < candidates->count; i++) {
            for (size_t j = i + 1; j < candidates->count; j++) {
                float score_i = compute_app_score(g,
                                                   candidates->apps[indices[i]].pattern_id,
                                                   candidates->apps[indices[i]].anchor_id,
                                                   epsilon);
                float score_j = compute_app_score(g,
                                                   candidates->apps[indices[j]].pattern_id,
                                                   candidates->apps[indices[j]].anchor_id,
                                                   epsilon);
                if (score_j > score_i) {
                    size_t tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }
    }
    
    // Greedy selection: accept apps that don't conflict
    for (size_t idx = 0; idx < candidates->count; idx++) {
        size_t i = indices[idx];
        uint64_t pattern_id = candidates->apps[i].pattern_id;
        uint64_t anchor_id = candidates->apps[i].anchor_id;
        
        Node *pattern = graph_find_node_by_id((Graph *)g, pattern_id);
        if (!pattern || pattern->kind != NODE_PATTERN) continue;
        
        // Determine local segment size
        size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
        if (num_atoms == 0) continue;
        
        const PatternAtom *atoms =
            (const PatternAtom *)(g->blob + pattern->payload_offset);
        
        int16_t max_delta = 0;
        for (size_t j = 0; j < num_atoms; j++) {
            if (atoms[j].delta > max_delta) max_delta = atoms[j].delta;
        }
        
        size_t local_len = (size_t)(max_delta + 1);
        if (local_len == 0) continue;
        
        // Build local prediction for this app
        uint8_t *local = malloc(local_len);
        if (!local) continue;
        for (size_t j = 0; j < local_len; j++) {
            local[j] = 0x00;
        }
        
        pattern_reconstruct_segment(g, pattern, anchor_id, local, local_len);
        
        // Check for conflicts
        int has_conflict = 0;
        for (size_t j = 0; j < local_len; j++) {
            if (local[j] == 0x00) continue; // no prediction here
            
            uint64_t global_pos = anchor_id + (uint64_t)j;
            if (global_pos < start_id || global_pos > end_id) continue;
            
            size_t pos_idx = (size_t)(global_pos - start_id);
            if (pred[pos_idx] != 0x00 && pred[pos_idx] != local[j]) {
                has_conflict = 1;
                break;
            }
        }
        
        // If no conflict, accept this app
        if (!has_conflict) {
            explanation_add(out_selected, pattern_id, anchor_id);
            
            // Write predictions into global buffer
            for (size_t j = 0; j < local_len; j++) {
                if (local[j] == 0x00) continue;
                
                uint64_t global_pos = anchor_id + (uint64_t)j;
                if (global_pos < start_id || global_pos > end_id) continue;
                
                size_t pos_idx = (size_t)(global_pos - start_id);
                pred[pos_idx] = local[j];
            }
        }
        
        free(local);
    }
    
    free(indices);
    free(pred);
}

// Wrapper: delegates to legacy learning when training is enabled
float graph_self_consistency_episode_multi_pattern(Graph *g,
                                                   Node *const *patterns,
                                                   size_t num_patterns,
                                                   uint64_t start_id,
                                                   uint64_t end_id,
                                                   float match_threshold,
                                                   float lr_q) {
    // Only call legacy learning if training is enabled
    if (g_sys.training_enabled) {
        return legacy_self_consistency_episode_multi_pattern(g, patterns, num_patterns,
                                                             start_id, end_id,
                                                             match_threshold, lr_q);
    }
    // Runtime mode: return 0 (no learning)
    return 0.0f;
}

// Pattern binding API (Phase 4b)

// Remove edge from graph, unlinking from both adjacency lists
// Note: This doesn't free the edge slot; it just unlinks it.
// For now, we don't have a free list for edges, so edges are never truly removed.
// This function is prepared for future use with edge removal/pruning.
// Hardware operation: remove edge by index (no cognitive decisions)
void graph_remove_edge(Graph *g, uint32_t eid) {
    if (!g || eid >= g->num_edges) return;
    
    Edge *e = &g->edges[eid];
    uint64_t src = e->src;
    uint64_t dst = e->dst;
    
    // Unlink from src's outgoing list
    Node *src_node = graph_find_node_by_id(g, src);
    if (src_node) {
        uint32_t *cur = &src_node->first_out_edge;
        while (*cur != UINT32_MAX) {
            if (*cur == eid) {
                *cur = e->next_out_edge;
                break;
            }
            if (*cur >= g->num_edges) break;  // Safety check
            cur = &g->edges[*cur].next_out_edge;
        }
    }
    
    // Unlink from dst's incoming list
    Node *dst_node = graph_find_node_by_id(g, dst);
    if (dst_node) {
        uint32_t *cur = &dst_node->first_in_edge;
        while (*cur != UINT32_MAX) {
            if (*cur == eid) {
                *cur = e->next_in_edge;
                break;
            }
            if (*cur >= g->num_edges) break;  // Safety check
            cur = &g->edges[*cur].next_in_edge;
        }
    }
    
    // Clear edge fields (but keep it in the array for now)
    e->next_out_edge = UINT32_MAX;
    e->next_in_edge  = UINT32_MAX;
}

// Rebuild adjacency lists after loading from file
// Also rebuilds the ID→index mapping from scratch
static void graph_rebuild_adjacency(Graph *g) {
    if (!g) return;
    
    // First, rebuild the ID→index mapping from all nodes (O(N))
    // Clear existing map
    if (g->id_to_index) {
        free(g->id_to_index);
        g->id_to_index = NULL;
    }
    g->id_map_size = 0;
    g->id_map_count = 0;
    graph_ensure_id_map_capacity(g);
    
    // Register all nodes in the map
    // Show progress for large graphs (always, not just DEBUG)
    if (g->num_nodes > 5000) {
        fprintf(stderr, "Loading graph: %llu nodes, %llu edges...\n",
                (unsigned long long)g->num_nodes, (unsigned long long)g->num_edges);
    }
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        graph_id_map_insert(g, g->nodes[i].id, (uint32_t)i);
    }
    
    // Reset all node adjacency lists
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        g->nodes[i].first_out_edge = UINT32_MAX;
        g->nodes[i].first_in_edge  = UINT32_MAX;
    }
    
    // Reset all edge next pointers
    for (uint32_t eid = 0; eid < g->num_edges; eid++) {
        g->edges[eid].next_out_edge = UINT32_MAX;
        g->edges[eid].next_in_edge  = UINT32_MAX;
    }
    
    // Rebuild adjacency by scanning all edges (now O(E) with O(1) lookups)
    // Show progress for large graphs (always, not just DEBUG)
    if (g->num_edges > 100000) {
        fprintf(stderr, "Rebuilding adjacency: %llu edges...\n", 
                (unsigned long long)g->num_edges);
    }
    for (uint32_t eid = 0; eid < g->num_edges; eid++) {
        // Show progress every 100K edges for large graphs
        if (g->num_edges > 100000 && eid % 100000 == 0 && eid > 0) {
            fprintf(stderr, "  %u/%llu edges\n", eid, (unsigned long long)g->num_edges);
        }
        
        Edge *e = &g->edges[eid];
        
        // O(1) lookup for source node
        uint32_t src_idx = graph_id_map_lookup(g, e->src);
        if (src_idx != UINT32_MAX && src_idx < g->num_nodes) {
            Node *src_node = &g->nodes[src_idx];
            e->next_out_edge = src_node->first_out_edge;
            src_node->first_out_edge = eid;
        } else {
            e->next_out_edge = UINT32_MAX;
        }
        
        // O(1) lookup for destination node
        uint32_t dst_idx = graph_id_map_lookup(g, e->dst);
        if (dst_idx != UINT32_MAX && dst_idx < g->num_nodes) {
            Node *dst_node = &g->nodes[dst_idx];
            e->next_in_edge = dst_node->first_in_edge;
            dst_node->first_in_edge = eid;
        } else {
            e->next_in_edge = UINT32_MAX;
        }
    }
    #ifdef DEBUG
    fprintf(stderr, "Adjacency lists rebuilt successfully.\n");
    #endif
}

// Find edge using adjacency list (local, O(out_degree(src)) instead of O(E))
static Edge *graph_find_edge(Graph *g, uint64_t src, uint64_t dst) {
    if (!g) return NULL;
    
    // O(1) lookup for source node
    uint32_t src_idx = graph_id_map_lookup(g, src);
    if (src_idx == UINT32_MAX || src_idx >= g->num_nodes) return NULL;
    Node *src_node = &g->nodes[src_idx];
    
    // Traverse source node's outgoing adjacency list
    uint32_t eid = src_node->first_out_edge;
    uint32_t visited_count = 0;
    uint32_t max_visit = 1024;  // Debug safety limit
    
    while (eid != UINT32_MAX) {
        // Safety check: invalid edge index
        if (eid >= g->num_edges) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Invalid edge index %u in adjacency list\n", eid);
            #endif
            break;
        }
        
        // Debug-only cycle detection
        if (visited_count++ > max_visit) {
            #ifdef DEBUG
            fprintf(stderr, "ERROR: Possible cycle in adjacency list for node %llu\n", 
                    (unsigned long long)src);
            #endif
            break;
        }
        
        Edge *e = &g->edges[eid];
        if (e->dst == dst) {
            return e;
        }
        eid = e->next_out_edge;
    }
    
    return NULL;
}

void graph_reinforce_pattern_data_edge(Graph *g,
                                       uint64_t pattern_id,
                                       uint64_t data_id,
                                       float delta_w) {
    if (!g) return;
    
    Node *p = graph_find_node_by_id(g, pattern_id);
    Node *d = graph_find_node_by_id(g, data_id);
    if (!p || !d) return;
    if (p->kind != NODE_PATTERN || d->kind != NODE_DATA) return;
    
    Edge *e = graph_find_edge(g, pattern_id, data_id);
    if (!e) {
        e = graph_add_edge(g, pattern_id, data_id, 0.0f);
        if (!e) return;
    }
    e->w += delta_w;
}

void graph_bind_explanation_to_graph(Graph *g,
                                     const Explanation *exp,
                                     uint64_t start_id,
                                     uint64_t end_id,
                                     float delta_w) {
    if (!g || !exp) return;
    if (start_id > end_id) return;
    
    for (size_t i = 0; i < exp->count; i++) {
        uint64_t pattern_id = exp->apps[i].pattern_id;
        uint64_t anchor_id  = exp->apps[i].anchor_id;
        
        Node *pattern = graph_find_node_by_id(g, pattern_id);
        if (!pattern || pattern->kind != NODE_PATTERN) continue;
        
        size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
        if (num_atoms == 0) continue;
        
        const PatternAtom *atoms =
            (const PatternAtom *)(g->blob + pattern->payload_offset);
        
        for (size_t j = 0; j < num_atoms; j++) {
            PatternAtom atom = atoms[j];
            if (atom.mode != 0) continue; // only CONST_BYTE define bindings
            
            uint64_t data_id = anchor_id + (int64_t)atom.delta;
            if (data_id < start_id || data_id > end_id) continue;
            
            graph_reinforce_pattern_data_edge(g, pattern_id, data_id, delta_w);
        }
    }
}

void graph_debug_print_pattern_bindings(const Graph *g, size_t max_bindings_per_pattern) {
    if (!g) return;
    
    printf("=== Pattern bindings ===\n");
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        const Node *p = &g->nodes[i];
        if (p->kind != NODE_PATTERN) continue;
        
        printf("PATTERN id=%llu q=%.3f\n", (unsigned long long)p->id, p->q);
        
        size_t printed = 0;
        // Use adjacency list instead of scanning all edges
        uint32_t eid = p->first_out_edge;
        uint32_t visited_count = 0;
        uint32_t max_visit = 1024;  // Debug safety limit
        
        while (eid != UINT32_MAX && printed < max_bindings_per_pattern) {
            if (eid >= g->num_edges) break;  // Safety check
            if (visited_count++ > max_visit) {
                #ifdef DEBUG
                fprintf(stderr, "ERROR: Possible cycle in pattern %llu adjacency list\n",
                        (unsigned long long)p->id);
                #endif
                break;
            }
            const Edge *edge = &g->edges[eid];
            
            const Node *d = graph_find_node_by_id((Graph *)g, edge->dst);
            if (!d || d->kind != NODE_DATA) {
                eid = edge->next_out_edge;
                continue;
            }
            
            if (d->payload_len < 1) {
                eid = edge->next_out_edge;
                continue;
            }
            uint8_t b = g->blob[d->payload_offset];
            
            printf("  -> DATA id=%llu byte=%c (0x%02X) w=%.3f\n",
                   (unsigned long long)d->id,
                   (b >= 32 && b <= 126) ? (char)b : '.',
                   b,
                   edge->w);
            
            printed++;
            eid = edge->next_out_edge;
        }
        
        if (printed >= max_bindings_per_pattern && eid != UINT32_MAX) {
            printf("  ... (truncated)\n");
        }
        
        if (printed == 0) {
            printf("  (no bindings yet)\n");
        }
    }
    printf("========================\n");
}

// Graph persistence

int graph_save_to_file(const Graph *g, const char *filename) {
    if (!g || !filename) return 0;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return 0;
    
    // Write header: magic + version
    uint32_t magic = 0x4D454C56; // "MELV"
    uint32_t version = 1;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    
    // Write graph metadata
    fwrite(&g->num_nodes, sizeof(g->num_nodes), 1, f);
    fwrite(&g->num_edges, sizeof(g->num_edges), 1, f);
    fwrite(&g->blob_used, sizeof(g->blob_used), 1, f);
    fwrite(&g->next_data_pos, sizeof(g->next_data_pos), 1, f);
    fwrite(&g->next_pattern_id, sizeof(g->next_pattern_id), 1, f);
    fwrite(&g->blank_id, sizeof(g->blank_id), 1, f);
    
    // Write nodes
    if (g->num_nodes > 0) {
        fwrite(g->nodes, sizeof(Node), g->num_nodes, f);
    }
    
    // Write edges
    if (g->num_edges > 0) {
        fwrite(g->edges, sizeof(Edge), g->num_edges, f);
    }
    
    // Write blob
    if (g->blob_used > 0) {
        fwrite(g->blob, 1, g->blob_used, f);
    }
    
    fclose(f);
    return 1;
}

Graph *graph_load_from_file(const char *filename) {
    if (!filename) return NULL;
    
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    // Read header
    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x4D454C56) {
        fclose(f);
        return NULL;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != 1) {
        fclose(f);
        return NULL;
    }
    
    // Allocate graph
    Graph *g = calloc(1, sizeof(Graph));
    if (!g) {
        fclose(f);
        return NULL;
    }
    
    // Read metadata
    if (fread(&g->num_nodes, sizeof(g->num_nodes), 1, f) != 1 ||
        fread(&g->num_edges, sizeof(g->num_edges), 1, f) != 1 ||
        fread(&g->blob_used, sizeof(g->blob_used), 1, f) != 1 ||
        fread(&g->next_data_pos, sizeof(g->next_data_pos), 1, f) != 1 ||
        fread(&g->next_pattern_id, sizeof(g->next_pattern_id), 1, f) != 1 ||
        fread(&g->blank_id, sizeof(g->blank_id), 1, f) != 1) {
        free(g);
        fclose(f);
        return NULL;
    }
    
    // Allocate arrays
    g->nodes_cap = g->num_nodes > 0 ? g->num_nodes : 16;
    g->edges_cap = g->num_edges > 0 ? g->num_edges : 16;
    g->blob_cap = g->blob_used > 0 ? g->blob_used : 1024;
    
    g->nodes = malloc(g->nodes_cap * sizeof(Node));
    g->edges = malloc(g->edges_cap * sizeof(Edge));
    g->blob = malloc(g->blob_cap);
    
    if (!g->nodes || !g->edges || !g->blob) {
        graph_destroy(g);
        fclose(f);
        return NULL;
    }
    
    // Read nodes
    if (g->num_nodes > 0) {
        if (fread(g->nodes, sizeof(Node), g->num_nodes, f) != g->num_nodes) {
            graph_destroy(g);
            fclose(f);
            return NULL;
        }
    }
    
    // Read edges
    if (g->num_edges > 0) {
        if (fread(g->edges, sizeof(Edge), g->num_edges, f) != g->num_edges) {
            graph_destroy(g);
            fclose(f);
            return NULL;
        }
    }
    
    // Read blob
    if (g->blob_used > 0) {
        if (fread(g->blob, 1, g->blob_used, f) != g->blob_used) {
            graph_destroy(g);
            fclose(f);
            return NULL;
        }
    }
    
    fclose(f);
    
    // Rebuild adjacency lists (they're not stored in file format)
    // Add progress feedback for large graphs
    #ifdef DEBUG
    fprintf(stderr, "Loading graph: %llu nodes, %llu edges, rebuilding adjacency...\n",
            (unsigned long long)g->num_nodes, (unsigned long long)g->num_edges);
    #endif
    graph_rebuild_adjacency(g);
    #ifdef DEBUG
    fprintf(stderr, "Graph loaded successfully.\n");
    #endif
    
    return g;
}

// Fast snapshot format implementation (memory-mapped style)
// Saves graph as a contiguous memory image - adjacency lists and ID map are already correct

int graph_save_snapshot(const Graph *g, const char *filename) {
    if (!g || !filename) return 0;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return 0;
    
    // Compute offsets
    GraphSnapshotHeader hdr = {0};
    hdr.magic = MELVIN_SNAPSHOT_MAGIC;
    hdr.version = MELVIN_SNAPSHOT_VERSION;
    
    hdr.num_nodes = g->num_nodes;
    hdr.num_edges = g->num_edges;
    hdr.blob_used = g->blob_used;
    hdr.next_data_pos = g->next_data_pos;
    hdr.next_pattern_id = g->next_pattern_id;
    hdr.blank_id = g->blank_id;
    hdr.id_map_size = g->id_map_size;
    hdr.id_map_count = g->id_map_count;
    
    // Get training_enabled from global config
    extern SystemConfig g_sys;
    hdr.training_enabled = g_sys.training_enabled;
    
    // Compute offsets (all relative to start of file)
    hdr.nodes_offset = sizeof(GraphSnapshotHeader);
    hdr.edges_offset = hdr.nodes_offset + (g->num_nodes * sizeof(Node));
    hdr.blob_offset = hdr.edges_offset + (g->num_edges * sizeof(Edge));
    hdr.id_map_offset = hdr.blob_offset + g->blob_used;
    
    // Write header
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return 0;
    }
    
    // Write nodes array (single write, no loops)
    if (g->num_nodes > 0) {
        if (fwrite(g->nodes, sizeof(Node), g->num_nodes, f) != g->num_nodes) {
            fclose(f);
            return 0;
        }
    }
    
    // Write edges array (single write, no loops)
    if (g->num_edges > 0) {
        if (fwrite(g->edges, sizeof(Edge), g->num_edges, f) != g->num_edges) {
            fclose(f);
            return 0;
        }
    }
    
    // Write blob (single write)
    if (g->blob_used > 0) {
        if (fwrite(g->blob, 1, g->blob_used, f) != g->blob_used) {
            fclose(f);
            return 0;
        }
    }
    
    // Write ID map hash table (single write)
    if (g->id_map_size > 0 && g->id_to_index) {
        size_t id_map_bytes = g->id_map_size * sizeof(IdMapEntry);
        if (fwrite(g->id_to_index, 1, id_map_bytes, f) != id_map_bytes) {
            fclose(f);
            return 0;
        }
    }
    
    // Flush and sync
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    
    return 1;
}

Graph *graph_load_snapshot(const char *filename) {
    if (!filename) return NULL;
    
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    // Read header
    GraphSnapshotHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    
    // Verify magic and version
    if (hdr.magic != MELVIN_SNAPSHOT_MAGIC) {
        fclose(f);
        return NULL;
    }
    if (hdr.version != MELVIN_SNAPSHOT_VERSION) {
        fprintf(stderr, "[melvin] Snapshot version mismatch: file=%u, expected=%u\n",
                hdr.version, MELVIN_SNAPSHOT_VERSION);
        fclose(f);
        return NULL;
    }
    
    // Validate sizes (sanity checks)
    if (hdr.num_nodes > 100000000 || hdr.num_edges > 100000000 || hdr.blob_used > 1000000000) {
        fprintf(stderr, "[melvin] Snapshot has unreasonable sizes\n");
        fclose(f);
        return NULL;
    }
    
    // Allocate graph structure
    Graph *g = calloc(1, sizeof(Graph));
    if (!g) {
        fclose(f);
        return NULL;
    }
    
    // Set counts and state
    g->num_nodes = hdr.num_nodes;
    g->num_edges = hdr.num_edges;
    g->blob_used = hdr.blob_used;
    g->next_data_pos = hdr.next_data_pos;
    g->next_pattern_id = hdr.next_pattern_id;
    g->blank_id = hdr.blank_id;
    g->id_map_size = hdr.id_map_size;
    g->id_map_count = hdr.id_map_count;
    
    // Set capacities (use actual sizes, no extra allocation)
    g->nodes_cap = hdr.num_nodes > 0 ? hdr.num_nodes : 16;
    g->edges_cap = hdr.num_edges > 0 ? hdr.num_edges : 16;
    g->blob_cap = hdr.blob_used > 0 ? hdr.blob_used : 1024;
    
    // Allocate all arrays in one contiguous block (for better cache locality)
    // But we'll allocate separately for simplicity (can optimize later with mmap)
    g->nodes = malloc(g->nodes_cap * sizeof(Node));
    g->edges = malloc(g->edges_cap * sizeof(Edge));
    g->blob = malloc(g->blob_cap);
    
    if (!g->nodes || !g->edges || !g->blob) {
        graph_destroy(g);
        fclose(f);
        return NULL;
    }
    
    // Read nodes array (single read)
    if (g->num_nodes > 0) {
        if (fread(g->nodes, sizeof(Node), g->num_nodes, f) != g->num_nodes) {
            graph_destroy(g);
            fclose(f);
            return NULL;
        }
    }
    
    // Read edges array (single read)
    if (g->num_edges > 0) {
        if (fread(g->edges, sizeof(Edge), g->num_edges, f) != g->num_edges) {
            graph_destroy(g);
            fclose(f);
            return NULL;
        }
    }
    
    // Read blob (single read)
    if (g->blob_used > 0) {
        if (fread(g->blob, 1, g->blob_used, f) != g->blob_used) {
            graph_destroy(g);
            fclose(f);
            return NULL;
        }
    }
    
    // Read ID map hash table (single read)
    if (g->id_map_size > 0) {
        g->id_to_index = malloc(g->id_map_size * sizeof(IdMapEntry));
        if (!g->id_to_index) {
            graph_destroy(g);
            fclose(f);
            return NULL;
        }
        size_t id_map_bytes = g->id_map_size * sizeof(IdMapEntry);
        if (fread(g->id_to_index, 1, id_map_bytes, f) != id_map_bytes) {
            graph_destroy(g);
            fclose(f);
            return NULL;
        }
    } else {
        g->id_to_index = NULL;
    }
    
    fclose(f);
    
    // Restore system config
    extern SystemConfig g_sys;
    g_sys.training_enabled = hdr.training_enabled;
    
    // NO REBUILDING - adjacency lists and ID map are already correct!
    // The snapshot contains the exact in-memory state
    
    return g;
}

// Auto-detect format: try snapshot first, fall back to legacy
Graph *graph_load_auto(const char *filename) {
    if (!filename) return NULL;
    
    // Try snapshot format first (fast)
    Graph *g = graph_load_snapshot(filename);
    if (g) {
        return g;
    }
    
    // Fall back to legacy format (slower, but compatible)
    return graph_load_from_file(filename);
}


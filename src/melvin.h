#ifndef MELVIN_H
#define MELVIN_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    NODE_DATA       = 0,
    NODE_PATTERN    = 1,
    NODE_BLANK      = 2,
    NODE_OUTPUT     = 3,  // Output node - represents output bytes/actions
    NODE_ERROR      = 4,  // Error node - activated when failures occur (graph-native feedback)
    NODE_EPISODE    = 5,  // Episode node - represents a learning episode/span
    NODE_APPLICATION= 6,  // Application node - represents a pattern match/application
    NODE_VALUE      = 7,  // Value node - stores cognitive parameters (thresholds, rates, etc.)
    NODE_LEARNER    = 8,  // Learner node - triggers learning when active
    NODE_MAINTENANCE= 9   // Maintenance node - triggers pruning/decay/compression when active
} NodeKind;

typedef struct {
    uint64_t id;             // identity; for DATA: also global position
    uint32_t payload_offset; // offset into graph blob
    uint16_t payload_len;    // length in bytes of payload
    uint8_t  kind;           // NodeKind
    uint8_t  flags;          // reserved for later, keep small
    float    a;              // activation
    float    q;              // quality / consistency score (for PATTERN); 0 for others
    uint32_t first_out_edge; // index into edges[] for outgoing adjacency list, UINT32_MAX if none
    uint32_t first_in_edge;  // index into edges[] for incoming adjacency list, UINT32_MAX if none
} Node;

// Universal edge type
typedef struct {
    uint64_t src;            // src node id
    uint64_t dst;            // dst node id
    float    w;              // weight / influence
    uint32_t next_out_edge;  // next edge in src's outgoing adjacency list, UINT32_MAX if none
    uint32_t next_in_edge;   // next edge in dst's incoming adjacency list, UINT32_MAX if none
} Edge;

// Pattern atom encoding a single constraint relative to an anchor position
typedef struct {
    int16_t  delta;          // offset from anchor DATA id (can be negative)
    uint8_t  mode;           // 0 = CONST_BYTE, 1 = BLANK
    uint8_t  value;          // byte value if mode == CONST_BYTE
} PatternAtom;

// Hash table entry for ID→index mapping
typedef struct {
    uint64_t id;      // node ID (UINT64_MAX if empty slot)
    uint32_t index;   // index into nodes[] array
} IdMapEntry;

// System configuration - hardware/runtime only (no cognitive parameters)
typedef struct {
    int training_enabled;  // 0 = runtime (fast inference), 1 = training (slow learning)
    size_t max_output_bytes_per_tick;  // Maximum bytes to emit per tick (default: 4)
} SystemConfig;

// Global system config (default: runtime mode, no learning)
extern SystemConfig g_sys;

typedef struct {
    Node    *nodes;
    Edge    *edges;
    uint8_t *blob;
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t nodes_cap;
    uint64_t edges_cap;
    uint64_t blob_cap;
    uint64_t blob_used;
    uint64_t next_data_pos;      // next DATA id / position
    uint64_t next_pattern_id;    // next PATTERN id (start in high range)
    uint64_t blank_id;           // id of the single BLANK node
    // ID→index mapping (hash table for O(1) lookups)
    IdMapEntry *id_to_index;     // hash table for node ID → index mapping
    uint32_t id_map_size;        // current hash table size (power of 2)
    uint32_t id_map_count;       // number of entries in hash table
} Graph;

// Snapshot format for fast loading (memory-mapped style)
#define MELVIN_SNAPSHOT_MAGIC 0x4D454C56u  // 'MELV'
#define MELVIN_SNAPSHOT_VERSION 1

typedef struct {
    uint32_t magic;      // MELVIN_SNAPSHOT_MAGIC
    uint32_t version;     // MELVIN_SNAPSHOT_VERSION
    
    // Basic counts
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t blob_used;
    
    // Graph state
    uint64_t next_data_pos;
    uint64_t next_pattern_id;
    uint64_t blank_id;
    
    // ID map hash table
    uint32_t id_map_size;
    uint32_t id_map_count;
    
    // System config
    int training_enabled;
    
    // Offsets from start of file to each array (in bytes)
    uint64_t nodes_offset;
    uint64_t edges_offset;
    uint64_t blob_offset;
    uint64_t id_map_offset;
    
    // Reserved for future use
    uint64_t reserved[8];
} GraphSnapshotHeader;

// Public API (implement in melvin.c)
Graph  *graph_create(size_t node_cap, size_t edge_cap, size_t blob_cap);
void    graph_destroy(Graph *g);
// Hardware: Only these 3 node types can be created by C
Node   *graph_add_blank_node(Graph *g);                // Hardware: creates BLANK node
Node   *graph_add_data_byte(Graph *g, uint8_t b);      // Hardware: creates DATA node
Node   *graph_add_pattern(Graph *g,
                          const PatternAtom *atoms,
                          size_t num_atoms,
                          float initial_q);  // Hardware: creates PATTERN node

// Generic hardware function: creates a node (no cognitive decisions - pure hardware)
// This is called by graph-native rules, not directly by application code
Node   *graph_create_node(Graph *g, NodeKind kind, uint64_t id, const void *payload, size_t payload_len);
float   graph_get_value(const Graph *g, const char *value_name, float default_val);  // Get cognitive parameter from VALUE node (graph-native lookup)
Edge   *graph_add_edge(Graph *g, uint64_t src, uint64_t dst, float w);
void    graph_remove_edge(Graph *g, uint32_t eid);  // Hardware operation: remove edge by index
Node   *graph_find_node_by_id(const Graph *g, uint64_t id);  // O(1) via hash table
void    graph_propagate(Graph *g, size_t steps);
float   pattern_match_score(const Graph *g, const Node *pattern, uint64_t anchor_id);
void    graph_update_pattern_quality(Graph *g,
                                     Node *pattern,
                                     float delta_quality);
void    graph_update_edge_from_error(Graph *g,
                                     uint64_t src_id,
                                     uint64_t dst_id,
                                     float error,
                                     float lr);
void    graph_print_stats(const Graph *g);
void    graph_log_core_stats(Graph *g);  // Lightweight stats for debugging
void    graph_emit_output(Graph *g, size_t max_bytes, int fd);  // Emit active OUTPUT nodes
void    local_update_pattern_to_output(Graph *g, Node *pattern_node, Node *output_node);  // Local learning rule

// Self-consistency API (Phase 2)
size_t  graph_collect_data_span(const Graph *g,
                                 uint64_t start_id,
                                 uint8_t *out,
                                 size_t max_len);
size_t  pattern_reconstruct_segment(const Graph *g,
                                    const Node *pattern,
                                    uint64_t anchor_id,
                                    uint8_t *out,
                                    size_t segment_len);
float   graph_self_consistency_pass(Graph *g,
                                     Node *pattern,
                                     uint64_t start_id,
                                     uint64_t end_id,
                                     float match_threshold,
                                     float lr_q);

// Explanation API (Phase 3)
// A single application of a pattern at some anchor DATA id
typedef struct {
    uint64_t pattern_id;
    uint64_t anchor_id;
} PatternApplication;

// A simple "explanation code": a list of pattern applications over a segment
typedef struct {
    PatternApplication *apps;
    size_t count;
    size_t capacity;
} Explanation;

// Legacy global learning (training-only, guarded by training_enabled)
// WARNING: Performs O(patterns × anchors) scans - only use in training mode
void    legacy_collect_candidates_multi_pattern(const Graph *g,
                                                Node *const *patterns,
                                                size_t num_patterns,
                                                uint64_t start_id,
                                                uint64_t end_id,
                                                float match_threshold,
                                                Explanation *out_candidates);
float   legacy_self_consistency_episode_multi_pattern(Graph *g,
                                                      Node *const *patterns,
                                                      size_t num_patterns,
                                                      uint64_t start_id,
                                                      uint64_t end_id,
                                                      float match_threshold,
                                                      float lr_q);

// Initialize / free explanation
void explanation_init(Explanation *exp);
void explanation_free(Explanation *exp);

// Add an application (pattern_id, anchor_id) to explanation
void explanation_add(Explanation *exp, uint64_t pattern_id, uint64_t anchor_id);

// Convert Explanation to graph nodes (EPISODE and APPLICATION)
void explanation_to_graph(Graph *g, const Explanation *exp, uint64_t episode_node_id);

// Make an explanation for a segment [start_id, end_id] using a single pattern:
// scan anchors, and whenever pattern_match_score >= match_threshold, record
// (pattern_id, anchor_id) in the Explanation.
void graph_build_explanation_single_pattern(const Graph *g,
                                            const Node *pattern,
                                            uint64_t start_id,
                                            uint64_t end_id,
                                            float match_threshold,
                                            Explanation *out);

// Given an explanation and a segment [start_id, end_id], reconstruct predicted
// bytes by applying each pattern at each anchor and merging their predictions.
// For now, last-writer-wins if multiple patterns write same position.
// Writes into 'out' which must have capacity >= (end_id - start_id + 1) bytes.
// Returns number of positions actually written (where a predicted byte exists).
size_t graph_reconstruct_from_explanation(const Graph *g,
                                          const Explanation *exp,
                                          uint64_t start_id,
                                          uint64_t end_id,
                                          uint8_t *out,
                                          size_t out_cap);

// Run a full self-consistency episode over [start_id, end_id]:
//  1) Build explanation using single pattern
//  2) Reconstruct from that explanation
//  3) Compare to actual DATA bytes
//  4) Compute average error in [0,1]
//  5) Update pattern->q toward (1 - error) with lr_q
float graph_self_consistency_episode_single_pattern(Graph *g,
                                                     Node *pattern,
                                                     uint64_t start_id,
                                                     uint64_t end_id,
                                                     float match_threshold,
                                                     float lr_q);

// Multi-pattern explanation API (Phase 4)
// Collect candidate pattern applications from multiple patterns
void graph_collect_candidates_multi_pattern(const Graph *g,
                                            Node *const *patterns,
                                            size_t num_patterns,
                                            uint64_t start_id,
                                            uint64_t end_id,
                                            float match_threshold,
                                            Explanation *out_candidates);

// Select a consistent subset of candidates using greedy selection
// Avoids conflicts: no two apps predict different bytes at same position
void explanation_select_greedy_consistent(const Graph *g,
                                           const Explanation *candidates,
                                           uint64_t start_id,
                                           uint64_t end_id,
                                           Explanation *out_selected);

// Run self-consistency episode with multiple patterns
// Updates each pattern's q based on its individual contribution
float graph_self_consistency_episode_multi_pattern(Graph *g,
                                                   Node *const *patterns,
                                                   size_t num_patterns,
                                                   uint64_t start_id,
                                                   uint64_t end_id,
                                                   float match_threshold,
                                                   float lr_q);

// Pattern binding API (Phase 4b)
// Find or create an edge (pattern_id -> data_id) and adjust its weight by delta_w.
// Used to bind patterns to the DATA nodes they help explain.
void graph_reinforce_pattern_data_edge(Graph *g,
                                       uint64_t pattern_id,
                                       uint64_t data_id,
                                       float delta_w);

// For each selected pattern application in an explanation, bind the PATTERN
// to the DATA nodes corresponding to its CONST_BYTE atoms in [start_id, end_id].
// delta_w > 0 strengthens bindings when explanations are accepted.
void graph_bind_explanation_to_graph(Graph *g,
                                     const Explanation *exp,
                                     uint64_t start_id,
                                     uint64_t end_id,
                                     float delta_w);

// Debug helper: print each PATTERN, its quality q, and a sample of which DATA
// nodes it's bound to (id + byte). This is how we "see" patterns in the graph.
void graph_debug_print_pattern_bindings(const Graph *g, size_t max_bindings_per_pattern);

// Pattern-on-pattern learning: build symbol sequences from episodes
// Version 1: Uses Explanation struct (temporary, for migration)
void build_symbol_sequence_from_episode(Graph *g,
                                        const Explanation *exp,
                                        uint64_t start_id,
                                        uint64_t end_id,
                                        uint64_t *out_seq,
                                        uint16_t *out_len,
                                        uint16_t max_len);
// Version 2: Uses EPISODE/APPLICATION nodes (graph-native)
void build_symbol_sequence_from_episode_node(Graph *g,
                                             uint64_t episode_node_id,
                                             uint64_t *out_seq,
                                             uint16_t *out_len,
                                             uint16_t max_len);
// Create a pattern from two symbol sequences, using blanks where they differ
Node *graph_create_pattern_from_sequences(Graph *g,
                                          const uint64_t *seq1, uint16_t len1,
                                          const uint64_t *seq2, uint16_t len2);
// Graph-native learning: run local rules for active LEARNER nodes
void graph_run_local_rules(Graph *g);  // Triggers learning from graph activations

// Graph persistence (for growing global graph)
// Save graph to file (binary format)
int graph_save_to_file(const Graph *g, const char *filename);
// Load graph from file
Graph *graph_load_from_file(const char *filename);

// Fast snapshot format (memory-mapped style, no rebuilding)
int graph_save_snapshot(const Graph *g, const char *filename);
Graph *graph_load_snapshot(const char *filename);
// Auto-detect format and load (tries snapshot first, falls back to legacy)
Graph *graph_load_auto(const char *filename);

#endif // MELVIN_H


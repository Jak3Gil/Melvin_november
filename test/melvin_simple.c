/*
 * MELVIN SIMPLE INTERFACE - Implementation
 * 
 * Just wraps melvin.c - that's all we need.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "melvin_simple.h"

// Include melvin.c (the physics engine)
// Use guard to prevent double inclusion
#ifndef MELVIN_C_INCLUDED_BY_SIMPLE
#define MELVIN_C_INCLUDED_BY_SIMPLE
#include "melvin.c"
#endif

// Stub EXEC functions (not needed for basic learning tests)
// These are only called if EXEC nodes fire, which won't happen in simple tests
void melvin_exec_add32(MelvinFile *g, uint64_t self_id) { (void)g; (void)self_id; }
void melvin_exec_mul32(MelvinFile *g, uint64_t self_id) { (void)g; (void)self_id; }
void melvin_exec_select_add_or_mul(MelvinFile *g, uint64_t self_id) { (void)g; (void)self_id; }

// Helper to create brain
int melvin_simple_create_brain(const char *path) {
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    // TUNED: Using increased learning rate (0.025) for stronger 1-step associations
    params.learning_rate = 0.025f;  // Increased from 0.01 to match tuning
    return melvin_m_init_new_file(path, &params);
}

// Context structure
struct MelvinSimple {
    MelvinFile file;
    MelvinRuntime runtime;
};

MelvinSimple* melvin_open(const char *brain_path) {
    if (!brain_path) return NULL;
    
    MelvinSimple *m = calloc(1, sizeof(MelvinSimple));
    if (!m) return NULL;
    
    // Map the brain file
    if (melvin_m_map(brain_path, &m->file) < 0) {
        free(m);
        return NULL;
    }
    
    // Initialize runtime
    if (runtime_init(&m->runtime, &m->file) < 0) {
        close_file(&m->file);
        free(m);
        return NULL;
    }
    
    return m;
}

void melvin_close(MelvinSimple *m) {
    if (!m) return;
    
    melvin_m_sync(&m->file);
    runtime_cleanup(&m->runtime);
    close_file(&m->file);
    free(m);
}

void melvin_feed(MelvinSimple *m, uint64_t channel, uint8_t byte) {
    if (!m) return;
    
    // Feed byte to graph - that's it
    ingest_byte(&m->runtime, channel, byte, 1.0f);
}

void melvin_tick(MelvinSimple *m, uint64_t num_events) {
    if (!m) return;
    
    // Tick physics - graph learns, updates, evolves
    melvin_process_n_events(&m->runtime, num_events);
}

float melvin_read_byte(MelvinSimple *m, uint8_t byte) {
    if (!m) return 0.0f;
    
    // DATA node ID = byte + 1000000
    uint64_t node_id = (uint64_t)byte + 1000000ULL;
    uint64_t node_idx = find_node_index_by_id(&m->file, node_id);
    
    if (node_idx == UINT64_MAX) return 0.0f;
    
    NodeDisk *node = &m->file.nodes[node_idx];
    return node->state;  // Activation
}

void melvin_stats(MelvinSimple *m, MelvinStats *out) {
    if (!m || !out) return;
    
    GraphHeaderDisk *gh = m->file.graph_header;
    out->num_nodes = gh->num_nodes;
    out->num_edges = gh->num_edges;
    out->avg_activation = gh->avg_activation;
}

float melvin_get_edge_weight(MelvinSimple *m, uint8_t src_byte, uint8_t dst_byte) {
    if (!m) return 0.0f;
    
    // Convert bytes to node IDs (same mapping as ingest_byte)
    uint64_t src_id = (uint64_t)src_byte + 1000000ULL;
    uint64_t dst_id = (uint64_t)dst_byte + 1000000ULL;
    
    // Find source node
    uint64_t src_idx = find_node_index_by_id(&m->file, src_id);
    if (src_idx == UINT64_MAX) return 0.0f;
    
    NodeDisk *src_node = &m->file.nodes[src_idx];
    if (src_node->first_out_edge == UINT64_MAX) return 0.0f;
    
    // Traverse adjacency list to find edge
    GraphHeaderDisk *gh = m->file.graph_header;
    EdgeDisk *edges = m->file.edges;
    uint64_t edge_idx = src_node->first_out_edge;
    
    while (edge_idx != UINT64_MAX && edge_idx < gh->edge_capacity) {
        EdgeDisk *e = &edges[edge_idx];
        if (e->src == UINT64_MAX) break;  // Invalid edge
        if (e->src == src_id && e->dst == dst_id) {
            return e->weight;  // Found it!
        }
        edge_idx = e->next_out_edge;
    }
    
    return 0.0f;  // Edge not found
}


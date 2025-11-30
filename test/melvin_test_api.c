/*
 * MELVIN TEST API Implementation
 * 
 * Contract-compliant wrapper around melvin.c
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "melvin_test_api.h"

// Include melvin.c for type definitions
// Tests should NOT include melvin.c directly - use this API instead
// Use a guard to prevent double inclusion
#ifndef MELVIN_C_INCLUDED_BY_API
#define MELVIN_C_INCLUDED_BY_API
#include "melvin.c"
#include "instincts.c"
#include "melvin_exec_helpers.c"  // Include after melvin.c (as per its design)
#endif

// Fixed channel IDs for testing (these are just channel labels)
#define CH_INPUT  1  // Input channel for all test data
#define CH_OUTPUT 2  // Output channel (if graph learns to emit)

// Input port node ID (created by instincts or first byte ingestion)
static uint64_t get_input_port_node_id(void) {
    return 3000000ULL;  // Fixed ID for input port (would be set up by instincts)
}

// Output port node ID
static uint64_t get_output_port_node_id(void) {
    return 3000001ULL;  // Fixed ID for output port
}

// Context structure
struct MelvinCtx {
    MelvinFile file;
    MelvinRuntime runtime;
    char brain_path[256];
};

bool melvin_open(const char *path, MelvinCtx **ctx_out) {
    if (!path || !ctx_out) return false;
    
    MelvinCtx *ctx = calloc(1, sizeof(MelvinCtx));
    if (!ctx) return false;
    
    strncpy(ctx->brain_path, path, sizeof(ctx->brain_path) - 1);
    
    // Map the file
    if (melvin_m_map(path, &ctx->file) < 0) {
        free(ctx);
        return false;
    }
    
    // Initialize runtime
    if (runtime_init(&ctx->runtime, &ctx->file) < 0) {
        close_file(&ctx->file);
        free(ctx);
        return false;
    }
    
    *ctx_out = ctx;
    return true;
}

void melvin_close(MelvinCtx *ctx) {
    if (!ctx) return;
    
    melvin_m_sync(&ctx->file);
    runtime_cleanup(&ctx->runtime);
    close_file(&ctx->file);
    free(ctx);
}

void melvin_ingest_byte(MelvinCtx *ctx, uint8_t byte, float energy) {
    if (!ctx) return;
    
    // Ingest byte on input channel
    // DATA node will be created automatically if it doesn't exist
    ingest_byte(&ctx->runtime, CH_INPUT, byte, energy);
}

void melvin_tick(MelvinCtx *ctx, uint64_t num_events) {
    if (!ctx) return;
    
    melvin_process_n_events(&ctx->runtime, num_events);
}

bool melvin_pop_output(MelvinCtx *ctx, uint8_t *byte_out) {
    if (!ctx || !byte_out) return false;
    
    // For now, output is read by checking if a DATA node has high activation
    // This is a simplified interface - in full implementation, graph would
    // emit bytes to output channel when patterns activate
    
    // TODO: Implement proper output mechanism if graph learns to emit
    // For now, return false (no output mechanism yet)
    return false;
}

void melvin_get_stats(MelvinCtx *ctx, MelvinStats *stats_out) {
    if (!ctx || !stats_out) return;
    
    GraphHeaderDisk *gh = ctx->file.graph_header;
    stats_out->num_nodes = gh->num_nodes;
    stats_out->num_edges = gh->num_edges;
    stats_out->avg_activation = gh->avg_activation;
    // Note: prediction_error and reward are per-node metrics
    // For global stats, would need to iterate and average
}

float melvin_get_data_node_activation(MelvinCtx *ctx, uint8_t byte) {
    if (!ctx) return 0.0f;
    
    // DATA node ID = byte + 1000000
    uint64_t node_id = (uint64_t)byte + 1000000ULL;
    uint64_t node_idx = find_node_index_by_id(&ctx->file, node_id);
    
    if (node_idx == UINT64_MAX) return 0.0f;
    
    NodeDisk *node = &ctx->file.nodes[node_idx];
    return node->state;  // Activation/state
}

float melvin_get_edge_weight(MelvinCtx *ctx, uint8_t from_byte, uint8_t to_byte) {
    if (!ctx) return 0.0f;
    
    uint64_t from_id = (uint64_t)from_byte + 1000000ULL;
    uint64_t to_id = (uint64_t)to_byte + 1000000ULL;
    
    // Find edge
    GraphHeaderDisk *gh = ctx->file.graph_header;
    EdgeDisk *edges = ctx->file.edges;
    
    for (uint64_t i = 0; i < gh->num_edges; i++) {
        EdgeDisk *e = &edges[i];
        if (e->src == UINT64_MAX) continue;
        
        // Check if this edge connects from->to
        uint64_t from_idx = find_node_index_by_id(&ctx->file, from_id);
        uint64_t to_idx = find_node_index_by_id(&ctx->file, to_id);
        
        if (from_idx != UINT64_MAX && to_idx != UINT64_MAX) {
            // Need to check if edge connects these nodes
            // This is simplified - full implementation would check node indices
            // For now, return 0.0
            // TODO: Implement proper edge lookup
        }
    }
    
    return 0.0f;
}

// Helper functions for test initialization
int melvin_test_create_brain(const char *path) {
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.01f;
    
    if (melvin_m_init_new_file(path, &params) < 0) {
        return -1;
    }
    return 0;
}

int melvin_test_inject_instincts(const char *path) {
    MelvinFile file;
    if (melvin_m_map(path, &file) < 0) {
        return -1;
    }
    melvin_inject_instincts(&file);
    melvin_m_sync(&file);
    close_file(&file);
    return 0;
}


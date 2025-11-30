/*
 * MELVIN TEST API
 * 
 * Minimal, contract-compliant interface for testing Melvin's learning capabilities.
 * 
 * This API enforces TESTING_CONTRACT.md:
 * - Only bytes in/out
 * - No direct graph manipulation
 * - All learning is internal
 */

#ifndef MELVIN_TEST_API_H
#define MELVIN_TEST_API_H

#include <stdint.h>
#include <stdbool.h>

// ========================================================================
// CONTEXT: Opaque handle to Melvin runtime
// ========================================================================
typedef struct MelvinCtx MelvinCtx;

// ========================================================================
// STATS: Read-only statistics from the graph
// ========================================================================
typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    float avg_activation;
    // Note: prediction_error and reward are per-node, not global
    // To get global stats, would need to iterate nodes
} MelvinStats;

// ========================================================================
// API FUNCTIONS (Contract-Compliant)
// ========================================================================

// Open melvin.m file
// Returns: true on success, false on failure
bool melvin_open(const char *path, MelvinCtx **ctx_out);

// Close and sync melvin.m
void melvin_close(MelvinCtx *ctx);

// Ingest a byte (sends to fixed input port node)
// This is the ONLY way to send data into the graph
void melvin_ingest_byte(MelvinCtx *ctx, uint8_t byte, float energy);

// Tick the graph (advance physics and learning)
void melvin_tick(MelvinCtx *ctx, uint64_t num_events);

// Try to pop a byte from output port
// Returns: true if byte was available, false if no output yet
bool melvin_pop_output(MelvinCtx *ctx, uint8_t *byte_out);

// Get read-only statistics
void melvin_get_stats(MelvinCtx *ctx, MelvinStats *stats_out);

// Get activation of a specific DATA node (by byte value)
// This is for probing - not for modifying
float melvin_get_data_node_activation(MelvinCtx *ctx, uint8_t byte);

// Get edge weight between two DATA nodes (by byte values)
// Returns 0.0 if edge doesn't exist
float melvin_get_edge_weight(MelvinCtx *ctx, uint8_t from_byte, uint8_t to_byte);

// Helper: Create new brain file (exposed for test initialization)
// This is needed because GraphParams is defined in melvin.c
int melvin_test_create_brain(const char *path);
// Helper: Inject instincts (exposed for test initialization)
int melvin_test_inject_instincts(const char *path);

#endif // MELVIN_TEST_API_H


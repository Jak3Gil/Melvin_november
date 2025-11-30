/*
 * MELVIN DIAGNOSTICS - Instrumentation for Physics Verification
 * 
 * This header provides diagnostic logging capabilities to verify:
 * - Prediction error tracking
 * - Learning rule implementation
 * - Information capture (structured vs random)
 * - Control loop wiring
 */

#ifndef MELVIN_DIAGNOSTICS_H
#define MELVIN_DIAGNOSTICS_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Diagnostic mode flag
extern bool g_diagnostics_enabled;

// Diagnostic log files
extern FILE *g_node_diag_log;
extern FILE *g_edge_diag_log;
extern FILE *g_global_diag_log;

// Initialize diagnostics
void diagnostics_init(const char *output_dir);

// Cleanup diagnostics
void diagnostics_cleanup(void);

// Node-level diagnostics
void diagnostics_log_node_update(
    uint64_t event_index,
    uint64_t node_id,
    float state_before,
    float state_after,
    float prediction_before,
    float prediction_after,
    float prediction_error,
    float fe_inst,
    float fe_ema,
    float traffic_ema
);

// Edge-level diagnostics
void diagnostics_log_edge_update(
    uint64_t event_index,
    uint64_t src_id,
    uint64_t dst_id,
    float weight_before,
    float weight_after,
    float delta_w,
    float eligibility,
    float usage,
    float last_energy
);

// Global summary snapshot
void diagnostics_log_global_snapshot(
    uint64_t event_index,
    float mean_state,
    float var_state,
    float mean_prediction_error,
    float var_prediction_error,
    float mean_fe_ema,
    float var_fe_ema,
    float mean_weight,
    float var_weight,
    float frac_strong_edges,
    uint64_t num_pattern_nodes,
    uint64_t num_seq_edges,
    uint64_t num_chan_edges,
    uint64_t num_bonds
);

#endif // MELVIN_DIAGNOSTICS_H


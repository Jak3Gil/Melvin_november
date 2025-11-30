/*
 * DIAGNOSTIC EXPERIMENT A: Single-Node Sanity Check
 * 
 * Purpose: Verify prediction, error, and FE behave correctly on smallest system
 * 
 * Setup:
 * - Graph with 1 node, no edges
 * - Each event: inject constant external delta
 * - Run normal update (message passing is trivial)
 * 
 * Log:
 * - state, prediction, prediction_error, fe_inst, fe_ema
 * 
 * Check:
 * 1. Boundedness: state settles into bounded range
 * 2. Prediction tracking: prediction moves toward state, error shrinks
 * 3. FE behavior: fe_inst and fe_ema stabilize
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "melvin.c"
#include "melvin_diagnostics.c"

#define TEST_FILE "diag_a_single_node.m"
#define CHANNEL_ID 1
#define STEADY_DELTA 0.5f
#define MAX_EVENTS 2000
#define SNAPSHOT_INTERVAL 100

int main() {
    printf("========================================\n");
    printf("DIAGNOSTIC EXPERIMENT A: Single-Node Sanity\n");
    printf("========================================\n\n");
    
    // Initialize diagnostics
    diagnostics_init("diag_a_results");
    printf("Diagnostics enabled - logging to diag_a_results/\n\n");
    
    // Create test file
    GraphParams params;
    params.decay_rate = 0.95f;
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.01f;
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create test file\n");
        diagnostics_cleanup();
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        diagnostics_cleanup();
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        diagnostics_cleanup();
        return 1;
    }
    
    // Create single DATA node
    uint8_t test_byte = 65;  // 'A'
    ingest_byte(&rt, CHANNEL_ID, test_byte, STEADY_DELTA);
    melvin_process_n_events(&rt, 10);
    
    uint64_t node_id = (uint64_t)test_byte + 1000000ULL;
    uint64_t node_idx = find_node_index_by_id(&file, node_id);
    
    if (node_idx == UINT64_MAX) {
        fprintf(stderr, "ERROR: Node not found\n");
        runtime_cleanup(&rt);
        close_file(&file);
        diagnostics_cleanup();
        return 1;
    }
    
    printf("Created single node: ID=%llu\n", (unsigned long long)node_id);
    printf("Running %d events with steady delta=%.3f\n\n", MAX_EVENTS, STEADY_DELTA);
    
    // Track for analysis
    float state_history[MAX_EVENTS];
    float prediction_history[MAX_EVENTS];
    float error_history[MAX_EVENTS];
    float fe_history[MAX_EVENTS];
    
    // Run event stream
    for (int event = 0; event < MAX_EVENTS; event++) {
        diagnostics_increment_event_counter();
        uint64_t event_idx = diagnostics_get_event_counter();
        
        // Get state before
        NodeDisk *node = &file.nodes[node_idx];
        float state_before = node->state;
        float prediction_before = node->prediction;
        float fe_ema_before = node->fe_ema;
        
        // Inject steady delta
        MelvinEvent ev = {
            .type = EV_NODE_DELTA,
            .node_id = node_id,
            .value = STEADY_DELTA
        };
        melvin_event_enqueue(&rt.evq, &ev);
        
        // Process events
        melvin_process_n_events(&rt, 10);
        
        // Get state after
        node = &file.nodes[node_idx];
        float state_after = node->state;
        float prediction_after = node->prediction;
        float prediction_error = fabsf(node->prediction_error);
        float fe_inst = node->fe_ema;  // Approximate instantaneous FE
        float fe_ema = node->fe_ema;
        float traffic_ema = node->traffic_ema;
        
        // Log node update
        diagnostics_log_node_update(
            event_idx,
            node_id,
            state_before,
            state_after,
            prediction_before,
            prediction_after,
            prediction_error,
            fe_inst,
            fe_ema,
            traffic_ema
        );
        
        // Store for analysis
        state_history[event] = state_after;
        prediction_history[event] = prediction_after;
        error_history[event] = prediction_error;
        fe_history[event] = fe_ema;
        
        // Periodic snapshots
        if (event % SNAPSHOT_INTERVAL == 0 || event == MAX_EVENTS - 1) {
            // Compute global stats (just for this one node)
            float mean_state = state_after;
            float var_state = 0.0f;
            float mean_error = prediction_error;
            float var_error = 0.0f;
            float mean_fe = fe_ema;
            float var_fe = 0.0f;
            
            diagnostics_log_global_snapshot(
                event_idx,
                mean_state,
                var_state,
                mean_error,
                var_error,
                mean_fe,
                var_fe,
                0.0f,  // mean_weight (no edges)
                0.0f,  // var_weight
                0.0f,  // frac_strong_edges
                0,     // num_pattern_nodes
                0,     // num_seq_edges
                0,     // num_chan_edges
                0      // num_bonds
            );
            
            printf("Event %d: state=%.6f, prediction=%.6f, error=%.6f, FE=%.6f\n",
                   event, state_after, prediction_after, prediction_error, fe_ema);
        }
        
        // Check for NaN/Inf
        if (isnan(state_after) || isinf(state_after) || 
            isnan(prediction_after) || isinf(prediction_after)) {
            fprintf(stderr, "ERROR: NaN/Inf detected at event %d\n", event);
            runtime_cleanup(&rt);
            close_file(&file);
            diagnostics_cleanup();
            return 1;
        }
    }
    
    // Analysis
    printf("\n========================================\n");
    printf("ANALYSIS:\n");
    printf("========================================\n");
    
    // Check boundedness
    float max_state = -1e6f;
    float min_state = 1e6f;
    for (int i = 0; i < MAX_EVENTS; i++) {
        if (state_history[i] > max_state) max_state = state_history[i];
        if (state_history[i] < min_state) min_state = state_history[i];
    }
    printf("State range: [%.6f, %.6f]\n", min_state, max_state);
    if (fabsf(max_state) > 10.0f || fabsf(min_state) > 10.0f) {
        printf("❌ FAIL: State not bounded\n");
    } else {
        printf("✓ PASS: State is bounded\n");
    }
    
    // Check prediction tracking (last 500 events)
    int tail_start = MAX_EVENTS - 500;
    if (tail_start < 0) tail_start = 0;
    float avg_error_tail = 0.0f;
    for (int i = tail_start; i < MAX_EVENTS; i++) {
        avg_error_tail += error_history[i];
    }
    avg_error_tail /= (MAX_EVENTS - tail_start);
    printf("Average prediction error (last 500 events): %.6f\n", avg_error_tail);
    
    // Check FE stability
    float fe_first_half = 0.0f;
    float fe_second_half = 0.0f;
    int half = MAX_EVENTS / 2;
    for (int i = 0; i < half; i++) {
        fe_first_half += fe_history[i];
    }
    fe_first_half /= half;
    for (int i = half; i < MAX_EVENTS; i++) {
        fe_second_half += fe_history[i];
    }
    fe_second_half /= (MAX_EVENTS - half);
    printf("FE first half: %.6f, second half: %.6f\n", fe_first_half, fe_second_half);
    if (fabsf(fe_first_half - fe_second_half) < 0.1f) {
        printf("✓ PASS: FE stabilized\n");
    } else {
        printf("⚠ WARNING: FE may not have stabilized\n");
    }
    
    printf("\nResults logged to diag_a_results/\n");
    printf("Check node_diagnostics.csv for detailed time-series\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    diagnostics_cleanup();
    
    return 0;
}


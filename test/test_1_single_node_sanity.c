/*
 * TEST 1: Single-Node Sanity (Continuous Equilibrium)
 * 
 * Goal: Show a single node driven by a steady input relaxes to a stable state.
 * 
 * Setup:
 * - Graph: 1 node, no edges
 * - External process delivers steady event stream (constant delta injection)
 * 
 * Check over event trajectory:
 * - state converges to bounded value
 * - prediction converges to state
 * - prediction_error shrinks toward 0
 * - FE converges to stable range
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "melvin.c"

#define TEST_FILE "test_1_single_node.m"
#define CHANNEL_ID 1
#define STEADY_DELTA 0.5f
#define MAX_EVENTS 1000
#define CONVERGENCE_THRESHOLD 0.01f

int main() {
    printf("========================================\n");
    printf("TEST 1: Single-Node Sanity\n");
    printf("========================================\n\n");
    
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
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    
    // Create a single DATA node (byte value 65 = 'A')
    uint8_t test_byte = 65;
    ingest_byte(&rt, CHANNEL_ID, test_byte, STEADY_DELTA);
    melvin_process_n_events(&rt, 10);
    
    // Find the node ID (DATA node ID = byte_value + 1000000)
    uint64_t node_id = (uint64_t)test_byte + 1000000ULL;
    
    uint64_t node_idx = find_node_index_by_id(&file, node_id);
    if (node_idx == UINT64_MAX) {
        fprintf(stderr, "ERROR: Node not found after creation\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    printf("Created single node: ID=%llu\n", (unsigned long long)node_id);
    printf("Running %d events with steady delta=%.3f\n\n", MAX_EVENTS, STEADY_DELTA);
    
    // Track convergence metrics
    float prev_state = 0.0f;
    float prev_prediction = 0.0f;
    float prev_error = 0.0f;
    float prev_fe = 0.0f;
    
    int state_converged = 0;
    int prediction_converged = 0;
    int error_converged = 0;
    int fe_converged = 0;
    
    // Run event stream
    for (int event = 0; event < MAX_EVENTS; event++) {
        // Inject steady delta (constant external field)
        MelvinEvent ev = {
            .type = EV_NODE_DELTA,
            .node_id = node_id,
            .value = STEADY_DELTA
        };
        melvin_event_enqueue(&rt.evq, &ev);
        
        // Process events (message passing, activation update, learning)
        melvin_process_n_events(&rt, 10);
        
        // Read node state
        NodeDisk *node = &file.nodes[node_idx];
        float state = node->state;
        float prediction = node->prediction;
        float error = fabsf(node->prediction_error);
        float fe = node->fe_ema;
        
        // Check convergence (every 100 events)
        if (event % 100 == 0 || event == MAX_EVENTS - 1) {
            float state_change = fabsf(state - prev_state);
            float pred_change = fabsf(prediction - prev_prediction);
            float error_change = fabsf(error - prev_error);
            float fe_change = fabsf(fe - prev_fe);
            
            printf("Event %d:\n", event);
            printf("  state=%.6f (change=%.6f)\n", state, state_change);
            printf("  prediction=%.6f (change=%.6f)\n", prediction, pred_change);
            printf("  error=%.6f (change=%.6f)\n", error, error_change);
            printf("  FE_ema=%.6f (change=%.6f)\n", fe, fe_change);
            
            if (state_change < CONVERGENCE_THRESHOLD && !state_converged) {
                printf("  ✓ STATE CONVERGED at event %d\n", event);
                state_converged = 1;
            }
            if (pred_change < CONVERGENCE_THRESHOLD && !prediction_converged) {
                printf("  ✓ PREDICTION CONVERGED at event %d\n", event);
                prediction_converged = 1;
            }
            if (error < CONVERGENCE_THRESHOLD && !error_converged) {
                printf("  ✓ ERROR CONVERGED at event %d\n", event);
                error_converged = 1;
            }
            if (fe_change < CONVERGENCE_THRESHOLD && !fe_converged) {
                printf("  ✓ FE CONVERGED at event %d\n", event);
                fe_converged = 1;
            }
            printf("\n");
            
            prev_state = state;
            prev_prediction = prediction;
            prev_error = error;
            prev_fe = fe;
        }
        
        // Check for NaN/Inf
        if (isnan(state) || isinf(state) || isnan(prediction) || isinf(prediction)) {
            fprintf(stderr, "ERROR: NaN/Inf detected at event %d\n", event);
            runtime_cleanup(&rt);
            close_file(&file);
            return 1;
        }
    }
    
    // Final check
    NodeDisk *node = &file.nodes[node_idx];
    float final_state = node->state;
    float final_prediction = node->prediction;
    float final_error = fabsf(node->prediction_error);
    float final_fe = node->fe_ema;
    
    printf("========================================\n");
    printf("FINAL RESULTS:\n");
    printf("========================================\n");
    printf("Final state: %.6f\n", final_state);
    printf("Final prediction: %.6f\n", final_prediction);
    printf("Final error: %.6f\n", final_error);
    printf("Final FE_ema: %.6f\n", final_fe);
    printf("\n");
    
    // Verify bounded state
    if (fabsf(final_state) > 10.0f) {
        printf("❌ FAIL: State not bounded (|state| > 10)\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    // Verify prediction matches state
    if (fabsf(final_state - final_prediction) > 0.1f) {
        printf("❌ FAIL: Prediction doesn't match state (diff=%.6f)\n", 
               fabsf(final_state - final_prediction));
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    // Verify error is small
    if (final_error > 0.1f) {
        printf("❌ FAIL: Prediction error too large (error=%.6f)\n", final_error);
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    printf("✓ PASS: Single-node system converges to stable equilibrium\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    return 0;
}


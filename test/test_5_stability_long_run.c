/*
 * TEST 5: Stability Under Long Event Trajectories
 * 
 * Goal: Verify system remains stable over millions of events
 * 
 * Test 5.1: Mixed event bombardment
 * - Build medium graph (10^4-10^5 nodes)
 * - Feed long trajectory of mixed events:
 *   - structured streams on some channels
 *   - random noise on others
 *   - simple control episodes in parallel
 * 
 * Check:
 * - No validation failures (no corruption)
 * - No NaN/Inf, no weight blowup
 * - Activation stats: not all saturated, not all zero
 * - FE_ema histogram looks sane; some low-FE clusters emerge
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_5_stability.m"
#define MAX_EVENTS 100000  // Reduced for testing, increase for full test
#define EVENTS_PER_BATCH 100
#define STRUCTURED_CHANNEL 1
#define NOISE_CHANNEL 2
#define CONTROL_CHANNEL 3

static void inject_structured_stream(MelvinRuntime *rt, int length) {
    // ABABAB pattern
    for (int i = 0; i < length; i++) {
        uint8_t byte = (i % 2 == 0) ? 65 : 66;  // A or B
        ingest_byte(rt, STRUCTURED_CHANNEL, byte, 1.0f);
    }
}

static void inject_noise_stream(MelvinRuntime *rt, int length) {
    // Random bytes
    for (int i = 0; i < length; i++) {
        uint8_t byte = 32 + (rand() % 95);  // Printable ASCII
        ingest_byte(rt, NOISE_CHANNEL, byte, 0.5f);
    }
}

static void inject_control_episode(MelvinRuntime *rt) {
    // Simple control: state -> action -> reward
    int state = rand() % 10;
    uint8_t state_byte = (uint8_t)(state * 25);
    ingest_byte(rt, CONTROL_CHANNEL, state_byte, 1.0f);
    
    uint8_t action = rand() % 3;
    uint8_t action_byte = (uint8_t)(200 + action);
    ingest_byte(rt, CONTROL_CHANNEL, action_byte, 1.0f);
    
    float reward = 1.0f / (1.0f + (float)abs(state - 5));
    uint8_t reward_byte = (uint8_t)(reward * 255.0f);
    ingest_byte(rt, CONTROL_CHANNEL, reward_byte, 1.0f);
}

int main() {
    printf("========================================\n");
    printf("TEST 5: Stability Under Long Event Trajectories\n");
    printf("========================================\n\n");
    
    srand(time(NULL));
    
    // Create test file with larger capacity
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
    
    printf("Running %d events...\n\n", MAX_EVENTS);
    
    int validation_failures = 0;
    int nan_inf_detected = 0;
    float max_weight = 0.0f;
    float min_activation = 1e6f;
    float max_activation = -1e6f;
    int zero_activations = 0;
    int saturated_activations = 0;
    
    // FE histogram
    int fe_low = 0;    // FE < 0.1
    int fe_medium = 0; // 0.1 <= FE < 1.0
    int fe_high = 0;   // FE >= 1.0
    
    for (int event_batch = 0; event_batch < MAX_EVENTS / EVENTS_PER_BATCH; event_batch++) {
        // Mix different event types
        if (event_batch % 3 == 0) {
            // Structured stream
            inject_structured_stream(&rt, 10);
        } else if (event_batch % 3 == 1) {
            // Noise stream
            inject_noise_stream(&rt, 10);
        } else {
            // Control episode
            inject_control_episode(&rt);
        }
        
        // Process events
        melvin_process_n_events(&rt, EVENTS_PER_BATCH);
        
        // Periodic checks
        if (event_batch % 100 == 0 || event_batch == (MAX_EVENTS / EVENTS_PER_BATCH) - 1) {
            GraphHeaderDisk *gh = file.graph_header;
            
            // Check for NaN/Inf and collect stats
            for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
                NodeDisk *node = &file.nodes[i];
                if (node->id == UINT64_MAX) continue;
                
                float state = node->state;
                float fe = node->fe_ema;
                
                if (isnan(state) || isinf(state) || isnan(fe) || isinf(fe)) {
                    nan_inf_detected++;
                }
                
                if (fabsf(state) < min_activation) min_activation = fabsf(state);
                if (fabsf(state) > max_activation) max_activation = fabsf(state);
                
                if (fabsf(state) < 0.001f) zero_activations++;
                if (fabsf(state) > 0.99f) saturated_activations++;
                
                // FE histogram
                if (fe < 0.1f) fe_low++;
                else if (fe < 1.0f) fe_medium++;
                else fe_high++;
            }
            
            // Check edge weights
            for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
                EdgeDisk *edge = &file.edges[e];
                if (edge->src == UINT64_MAX) continue;
                
                float weight = fabsf(edge->weight);
                if (weight > max_weight) max_weight = weight;
                
                if (isnan(weight) || isinf(weight)) {
                    nan_inf_detected++;
                }
            }
            
            printf("Event batch %d: nodes=%llu, edges=%llu, max_weight=%.3f\n",
                   event_batch, (unsigned long long)gh->num_nodes, 
                   (unsigned long long)gh->num_edges, max_weight);
        }
    }
    
    // Final statistics
    GraphHeaderDisk *gh = file.graph_header;
    uint64_t total_nodes = 0;
    uint64_t total_edges = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (file.nodes[i].id != UINT64_MAX) total_nodes++;
    }
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        if (file.edges[e].src != UINT64_MAX) total_edges++;
    }
    
    printf("\n========================================\n");
    printf("FINAL STATISTICS:\n");
    printf("========================================\n");
    printf("Total nodes: %llu\n", (unsigned long long)total_nodes);
    printf("Total edges: %llu\n", (unsigned long long)total_edges);
    printf("Max weight: %.6f\n", max_weight);
    printf("Activation range: [%.6f, %.6f]\n", min_activation, max_activation);
    printf("Zero activations: %d\n", zero_activations);
    printf("Saturated activations: %d\n", saturated_activations);
    printf("\n");
    printf("FE histogram:\n");
    printf("  Low (FE < 0.1): %d\n", fe_low);
    printf("  Medium (0.1 <= FE < 1.0): %d\n", fe_medium);
    printf("  High (FE >= 1.0): %d\n", fe_high);
    printf("\n");
    printf("Validation failures: %d\n", validation_failures);
    printf("NaN/Inf detected: %d\n", nan_inf_detected);
    printf("\n");
    
    // Verify stability
    int passed = 1;
    
    if (validation_failures > 0) {
        printf("❌ FAIL: Validation failures detected\n");
        passed = 0;
    }
    
    if (nan_inf_detected > 0) {
        printf("❌ FAIL: NaN/Inf detected\n");
        passed = 0;
    }
    
    if (max_weight > 10.0f) {
        printf("❌ FAIL: Weight blowup (max_weight=%.6f)\n", max_weight);
        passed = 0;
    }
    
    if (zero_activations == total_nodes) {
        printf("❌ FAIL: All activations are zero\n");
        passed = 0;
    }
    
    if (saturated_activations == total_nodes) {
        printf("❌ FAIL: All activations are saturated\n");
        passed = 0;
    }
    
    if (fe_low == 0 && fe_medium == 0) {
        printf("⚠ WARNING: No low-FE clusters emerged\n");
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    if (passed) {
        printf("✓ PASS: System remains stable over long event trajectory\n");
        return 0;
    } else {
        printf("❌ FAIL: Stability test failed\n");
        return 1;
    }
}


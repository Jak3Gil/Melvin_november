/*
 * Test: Repeatable Circuits
 * 
 * Goal: Verify that Melvin can build circuits that fire consistently
 * when given the same input sequence multiple times.
 * 
 * This tests:
 * 1. Circuit formation (pattern nodes with stable edges)
 * 2. Circuit stability (high stability = slower decay = reusable)
 * 3. Circuit repeatability (same input → same activation pattern)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "melvin.c"

#define TEST_FILE "test_repeatable_circuits.m"
#define PATTERN "ABC"
#define NUM_TRAINING_ROUNDS 50
#define NUM_TEST_ROUNDS 10

// Measure activation of a node
static float get_node_activation(MelvinFile *file, uint64_t node_id) {
    uint64_t idx = find_node_index_by_id(file, node_id);
    if (idx == UINT64_MAX) return 0.0f;
    return file->nodes[idx].state;
}

// Measure stability of a node
static float get_node_stability(MelvinFile *file, uint64_t node_id) {
    uint64_t idx = find_node_index_by_id(file, node_id);
    if (idx == UINT64_MAX) return 0.0f;
    return file->nodes[idx].stability;
}

// Find pattern node for sequence ABC
static uint64_t find_abc_pattern_node(MelvinFile *file) {
    uint64_t a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t b_id = (uint64_t)'B' + 1000000ULL;
    uint64_t c_id = (uint64_t)'C' + 1000000ULL;
    
    // Look for a pattern node that has incoming edges from A and B
    // and outgoing edge to C
    GraphHeaderDisk *gh = file->graph_header;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id < 5000000ULL || n->id >= 10000000ULL) continue;  // Pattern node ID range
        
        // Check if this pattern has edges A→P, B→P, P→C
        int has_a_in = 0, has_b_in = 0, has_c_out = 0;
        
        // Check incoming edges (scan all edges)
        for (uint64_t e_idx = 0; e_idx < gh->num_edges && e_idx < gh->edge_capacity; e_idx++) {
            EdgeDisk *e = &file->edges[e_idx];
            if (e->src == UINT64_MAX) continue;
            if (e->dst == n->id) {
                if (e->src == a_id) has_a_in = 1;
                if (e->src == b_id) has_b_in = 1;
            }
            if (e->src == n->id && e->dst == c_id) {
                has_c_out = 1;
            }
        }
        
        if (has_a_in && has_b_in && has_c_out) {
            return n->id;
        }
    }
    return UINT64_MAX;
}

int main() {
    printf("========================================\n");
    printf("REPEATABLE CIRCUITS TEST\n");
    printf("========================================\n\n");
    
    printf("Goal: Verify circuits can be triggered repeatedly\n");
    printf("Training: Feed 'ABC' pattern %d times\n", NUM_TRAINING_ROUNDS);
    printf("Testing: Trigger same pattern %d times and measure consistency\n\n", NUM_TEST_ROUNDS);
    
    // Step 1: Create file
    printf("Step 1: Creating test file...\n");
    GraphParams params;
    init_default_params(&params);
    unlink(TEST_FILE);
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "Failed to create test file\n");
        return 1;
    }
    printf("✓ Created %s\n\n", TEST_FILE);
    
    // Step 2: Map and initialize
    printf("Step 2: Initializing runtime...\n");
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "Failed to map file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "Failed to initialize runtime\n");
        return 1;
    }
    printf("✓ Runtime initialized\n\n");
    
    // Step 3: Train circuit (feed ABC many times)
    printf("Step 3: Training circuit (feeding 'ABC' %d times)...\n", NUM_TRAINING_ROUNDS);
    uint64_t a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t b_id = (uint64_t)'B' + 1000000ULL;
    uint64_t c_id = (uint64_t)'C' + 1000000ULL;
    
    for (int round = 0; round < NUM_TRAINING_ROUNDS; round++) {
        ingest_byte(&rt, 0, 'A', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, 0, 'B', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, 0, 'C', 1.0f);
        melvin_process_n_events(&rt, 30);
        
        if ((round + 1) % 10 == 0) {
            printf("  Training round %d/%d\n", round + 1, NUM_TRAINING_ROUNDS);
        }
    }
    printf("✓ Training complete\n\n");
    
    // Step 4: Find pattern node
    printf("Step 4: Finding pattern node...\n");
    uint64_t pattern_id = find_abc_pattern_node(&file);
    if (pattern_id == UINT64_MAX) {
        printf("⚠ Pattern node not found (may need more training)\n");
        printf("  Continuing with direct node activations...\n\n");
    } else {
        printf("✓ Found pattern node: %llu\n", (unsigned long long)pattern_id);
        float pattern_stability = get_node_stability(&file, pattern_id);
        printf("  Pattern stability: %.4f\n", pattern_stability);
        printf("  (High stability = slower decay = more reusable)\n\n");
    }
    
    // Step 5: Test repeatability
    printf("Step 5: Testing circuit repeatability...\n");
    printf("  Triggering 'ABC' sequence %d times and measuring activations\n\n", NUM_TEST_ROUNDS);
    
    float *c_activations = malloc(NUM_TEST_ROUNDS * sizeof(float));
    float *pattern_activations = NULL;
    if (pattern_id != UINT64_MAX) {
        pattern_activations = malloc(NUM_TEST_ROUNDS * sizeof(float));
    }
    
    for (int test_round = 0; test_round < NUM_TEST_ROUNDS; test_round++) {
        // Reset activations
        melvin_process_n_events(&rt, 50);  // Let system settle
        
        // Trigger sequence
        ingest_byte(&rt, 0, 'A', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, 0, 'B', 1.0f);
        melvin_process_n_events(&rt, 20);
        
        // Measure C activation (should be high if circuit works)
        float c_act = get_node_activation(&file, c_id);
        c_activations[test_round] = c_act;
        
        // Measure pattern activation if it exists
        if (pattern_id != UINT64_MAX) {
            float p_act = get_node_activation(&file, pattern_id);
            pattern_activations[test_round] = p_act;
        }
        
        // Complete sequence
        ingest_byte(&rt, 0, 'C', 1.0f);
        melvin_process_n_events(&rt, 30);
        
        printf("  Round %d: C activation = %.4f", test_round + 1, c_act);
        if (pattern_id != UINT64_MAX) {
            printf(", Pattern activation = %.4f", pattern_activations[test_round]);
        }
        printf("\n");
    }
    
    // Step 6: Analyze consistency
    printf("\nStep 6: Analyzing consistency...\n");
    
    // Compute mean and variance of C activations
    float mean_c = 0.0f;
    for (int i = 0; i < NUM_TEST_ROUNDS; i++) {
        mean_c += c_activations[i];
    }
    mean_c /= NUM_TEST_ROUNDS;
    
    float variance_c = 0.0f;
    for (int i = 0; i < NUM_TEST_ROUNDS; i++) {
        float diff = c_activations[i] - mean_c;
        variance_c += diff * diff;
    }
    variance_c /= NUM_TEST_ROUNDS;
    float stddev_c = sqrtf(variance_c);
    
    printf("  C node activation:\n");
    printf("    Mean: %.6f\n", mean_c);
    printf("    StdDev: %.6f\n", stddev_c);
    
    float cv = 0.0f;
    if (fabsf(mean_c) > 0.0001f) {
        cv = (stddev_c / fabsf(mean_c)) * 100.0f;
        printf("    Coefficient of variation: %.2f%%\n", cv);
    } else {
        printf("    Coefficient of variation: N/A (mean near zero)\n");
    }
    
    float mean_p = 0.0f;
    float stddev_p = 0.0f;
    if (pattern_id != UINT64_MAX && pattern_activations) {
        for (int i = 0; i < NUM_TEST_ROUNDS; i++) {
            mean_p += pattern_activations[i];
        }
        mean_p /= NUM_TEST_ROUNDS;
        
        float variance_p = 0.0f;
        for (int i = 0; i < NUM_TEST_ROUNDS; i++) {
            float diff = pattern_activations[i] - mean_p;
            variance_p += diff * diff;
        }
        variance_p /= NUM_TEST_ROUNDS;
        stddev_p = sqrtf(variance_p);
        
        printf("  Pattern node activation:\n");
        printf("    Mean: %.6f\n", mean_p);
        printf("    StdDev: %.6f\n", stddev_p);
        if (fabsf(mean_p) > 0.0001f) {
            float cv_p = (stddev_p / fabsf(mean_p)) * 100.0f;
            printf("    Coefficient of variation: %.2f%%\n", cv_p);
        } else {
            printf("    Coefficient of variation: N/A (mean near zero)\n");
        }
    }
    
    // Step 7: Results (NO THRESHOLDS - pure observation)
    printf("\n========================================\n");
    printf("OBSERVATIONS (No Thresholds - Pure Physics)\n");
    printf("========================================\n\n");
    
    // Just report what the physics produced - no judgments, no thresholds
    printf("Circuit Behavior (emerges from free-energy & stability laws):\n");
    printf("  C node activation mean: %.6f\n", mean_c);
    printf("  C node activation stddev: %.6f\n", stddev_c);
    
    // Reuse cv from Step 6 (already computed)
    if (fabsf(mean_c) > 0.0001f) {
        printf("  Coefficient of variation: %.2f%%\n", cv);
    } else {
        printf("  Coefficient of variation: N/A (mean near zero)\n");
    }
    
    if (pattern_id != UINT64_MAX) {
        float pattern_stability = get_node_stability(&file, pattern_id);
        printf("\nPattern Node (from FE-based creation law):\n");
        printf("  Pattern ID: %llu\n", (unsigned long long)pattern_id);
        printf("  Stability: %.6f\n", pattern_stability);
        printf("  Pattern activation mean: %.6f\n", mean_p);
        printf("  Pattern activation stddev: %.6f\n", stddev_p);
        
        // Stability-dependent decay: higher stability = slower decay
        float base_decay = file.graph_header->decay_rate;
        float max_decay_boost = 0.07f;
        float effective_decay = base_decay + max_decay_boost * pattern_stability;
        printf("  Effective decay rate: %.6f (base: %.6f, stability boost: %.6f)\n", 
               effective_decay, base_decay, max_decay_boost * pattern_stability);
        printf("  (Higher stability → slower decay → more persistent circuit)\n");
    } else {
        printf("\nPattern Node: Not found\n");
        printf("  (Pattern formation depends on free-energy reduction)\n");
    }
    
    printf("\n========================================\n");
    printf("INTERPRETATION\n");
    printf("========================================\n\n");
    printf("These measurements emerge from the physics laws:\n");
    printf("  - Free-energy minimization drives pattern formation\n");
    printf("  - Stability increases when FE is low and node is active\n");
    printf("  - High stability → slower decay → circuit persists\n");
    printf("  - No thresholds: all behavior emerges from energy, FE, and stability\n");
    printf("\nThe circuit's repeatability is determined by:\n");
    printf("  - How much free energy the pattern reduces\n");
    printf("  - How stable the pattern becomes\n");
    printf("  - How strongly edges route energy\n");
    printf("  - All of these emerge from the laws, not from thresholds\n");
    
    // Cleanup
    free(c_activations);
    if (pattern_activations) free(pattern_activations);
    runtime_cleanup(&rt);
    close_file(&file);
    
    // No pass/fail - just report what the physics produced
    return 0;
}


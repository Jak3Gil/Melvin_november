/*
 * TEST 2: Simple Pattern Learning from Event Stream
 * 
 * Goal: Learn "A predicts B" from a structured stream vs random noise
 * 
 * Test 2.1: Learn ABABAB... pattern
 * - Two DATA nodes (A, B)
 * - Event stream: A, B, A, B, A, B...
 * - Check: edge A→B appears and strengthens
 * - Check: When A activates, B's prediction/state increases
 * - Check: Prediction error at B goes down
 * - Check: FE_ema around B goes down
 * 
 * Test 2.2: Random bytes (control)
 * - Same setup but random byte stream
 * - Compare: patterns created, FE_ema, edge weights
 * - Expect: structured stream → structure + lower FE
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE_STRUCTURED "test_2_structured.m"
#define TEST_FILE_RANDOM "test_2_random.m"
#define CHANNEL_ID 1
#define BYTE_A 65  // 'A'
#define BYTE_B 66  // 'B'
#define PATTERN_LENGTH 100  // ABABAB... repeated 50 times
#define RANDOM_LENGTH 100
#define EVENTS_PER_BYTE 5

typedef struct {
    uint64_t node_a_id;
    uint64_t node_b_id;
    uint64_t node_a_idx;
    uint64_t node_b_idx;
    float edge_weight;
    float fe_before;
    float fe_after;
    float error_before;
    float error_after;
    int pattern_count;
} TestResults;

static TestResults run_pattern_test(const char *test_name, const char *file_path, 
                                    uint8_t *byte_stream, int stream_length, int is_structured) {
    printf("\n========================================\n");
    printf("TEST: %s\n", test_name);
    printf("========================================\n\n");
    
    TestResults results = {0};
    
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
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create test file\n");
        return results;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return results;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return results;
    }
    
    // Find or create DATA nodes for A and B
    uint64_t node_a_id = (uint64_t)BYTE_A + 1000000ULL;
    uint64_t node_b_id = (uint64_t)BYTE_B + 1000000ULL;
    
    // Ingest first A to create node
    ingest_byte(&rt, CHANNEL_ID, BYTE_A, 1.0f);
    melvin_process_n_events(&rt, EVENTS_PER_BYTE);
    
    // Ingest first B to create node
    ingest_byte(&rt, CHANNEL_ID, BYTE_B, 1.0f);
    melvin_process_n_events(&rt, EVENTS_PER_BYTE);
    
    // Find node indices
    results.node_a_idx = find_node_index_by_id(&file, node_a_id);
    results.node_b_idx = find_node_index_by_id(&file, node_b_id);
    results.node_a_id = node_a_id;
    results.node_b_id = node_b_id;
    
    if (results.node_a_idx == UINT64_MAX || results.node_b_idx == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to find nodes\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return results;
    }
    
    // Get initial state
    NodeDisk *node_b = &file.nodes[results.node_b_idx];
    results.error_before = fabsf(node_b->prediction_error);
    results.fe_before = node_b->fe_ema;
    
    // Check if edge exists
    results.edge_weight = 0.0f;
    if (edge_exists_between(&file, node_a_id, node_b_id)) {
        // Find edge weight
        EdgeDisk *edges = file.edges;
        GraphHeaderDisk *gh = file.graph_header;
        for (uint64_t e = 0; e < gh->num_edges; e++) {
            if (edges[e].src == node_a_id && edges[e].dst == node_b_id) {
                results.edge_weight = edges[e].weight;
                break;
            }
        }
    }
    
    printf("Initial state:\n");
    printf("  Edge A→B weight: %.6f\n", results.edge_weight);
    printf("  B prediction error: %.6f\n", results.error_before);
    printf("  B FE_ema: %.6f\n", results.fe_before);
    printf("\n");
    
    // Process byte stream
    printf("Processing %d bytes...\n", stream_length);
    for (int i = 0; i < stream_length; i++) {
        uint8_t byte = byte_stream[i];
        ingest_byte(&rt, CHANNEL_ID, byte, 1.0f);
        melvin_process_n_events(&rt, EVENTS_PER_BYTE);
        
        if (i % 20 == 0) {
            // Check edge weight
            if (edge_exists_between(&file, node_a_id, node_b_id)) {
                EdgeDisk *edges = file.edges;
                GraphHeaderDisk *gh = file.graph_header;
                for (uint64_t e = 0; e < gh->num_edges; e++) {
                    if (edges[e].src == node_a_id && edges[e].dst == node_b_id) {
                        results.edge_weight = edges[e].weight;
                        break;
                    }
                }
            }
            
            node_b = &file.nodes[results.node_b_idx];
            printf("  Event %d: edge_weight=%.6f, error=%.6f, FE=%.6f\n",
                   i, results.edge_weight, fabsf(node_b->prediction_error), node_b->fe_ema);
        }
    }
    
    // Final state
    node_b = &file.nodes[results.node_b_idx];
    results.error_after = fabsf(node_b->prediction_error);
    results.fe_after = node_b->fe_ema;
    
    // Count pattern nodes (ID range 5000000-9999999)
    GraphHeaderDisk *gh = file.graph_header;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (file.nodes[i].id >= 5000000ULL && file.nodes[i].id < 10000000ULL) {
            results.pattern_count++;
        }
    }
    
    printf("\nFinal state:\n");
    printf("  Edge A→B weight: %.6f\n", results.edge_weight);
    printf("  B prediction error: %.6f (change: %.6f)\n", 
           results.error_after, results.error_after - results.error_before);
    printf("  B FE_ema: %.6f (change: %.6f)\n", 
           results.fe_after, results.fe_after - results.fe_before);
    printf("  Pattern nodes created: %d\n", results.pattern_count);
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    return results;
}

int main() {
    printf("========================================\n");
    printf("TEST 2: Pattern Learning from Event Stream\n");
    printf("========================================\n");
    
    // Test 2.1: Structured stream (ABABAB...)
    uint8_t structured_stream[PATTERN_LENGTH];
    for (int i = 0; i < PATTERN_LENGTH; i++) {
        structured_stream[i] = (i % 2 == 0) ? BYTE_A : BYTE_B;
    }
    
    TestResults structured = run_pattern_test(
        "Structured Stream (ABABAB...)",
        TEST_FILE_STRUCTURED,
        structured_stream,
        PATTERN_LENGTH,
        1
    );
    
    // Test 2.2: Random stream
    srand(time(NULL));
    uint8_t random_stream[RANDOM_LENGTH];
    for (int i = 0; i < RANDOM_LENGTH; i++) {
        random_stream[i] = 65 + (rand() % 26);  // Random A-Z
    }
    
    TestResults random = run_pattern_test(
        "Random Stream (Control)",
        TEST_FILE_RANDOM,
        random_stream,
        RANDOM_LENGTH,
        0
    );
    
    // Compare results
    printf("\n========================================\n");
    printf("COMPARISON:\n");
    printf("========================================\n");
    printf("Structured stream:\n");
    printf("  Edge weight: %.6f\n", structured.edge_weight);
    printf("  FE reduction: %.6f\n", structured.fe_before - structured.fe_after);
    printf("  Error reduction: %.6f\n", structured.error_before - structured.error_after);
    printf("  Patterns: %d\n", structured.pattern_count);
    printf("\n");
    printf("Random stream:\n");
    printf("  Edge weight: %.6f\n", random.edge_weight);
    printf("  FE reduction: %.6f\n", random.fe_before - random.fe_after);
    printf("  Error reduction: %.6f\n", random.error_before - random.error_after);
    printf("  Patterns: %d\n", random.pattern_count);
    printf("\n");
    
    // Verify expectations
    int passed = 1;
    
    // Structured should have stronger edge
    if (structured.edge_weight <= random.edge_weight) {
        printf("❌ FAIL: Structured stream should have stronger edge\n");
        passed = 0;
    }
    
    // Structured should have better FE reduction
    float structured_fe_reduction = structured.fe_before - structured.fe_after;
    float random_fe_reduction = random.fe_before - random.fe_after;
    if (structured_fe_reduction <= random_fe_reduction) {
        printf("❌ FAIL: Structured stream should have better FE reduction\n");
        passed = 0;
    }
    
    // Structured should have more patterns
    if (structured.pattern_count <= random.pattern_count) {
        printf("⚠ WARNING: Structured stream should create more patterns\n");
    }
    
    if (passed) {
        printf("✓ PASS: Pattern learning works correctly\n");
        return 0;
    } else {
        printf("❌ FAIL: Pattern learning test failed\n");
        return 1;
    }
}


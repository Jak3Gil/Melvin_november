/*
 * DIAGNOSTIC EXPERIMENT B: Pattern Learning - Structured vs Random
 * 
 * Purpose: Verify information is actually being captured
 * 
 * Setup:
 * - Two runs: structured (ABABAB...) vs random bytes
 * - Track nodes A, B, edge A→B, any pattern nodes
 * 
 * Log:
 * - weight(A→B) trajectory
 * - prediction_error at B after A
 * - fe_ema for A, B, pattern nodes
 * - number of patterns
 * 
 * Check:
 * - Structured should show stronger edge, lower error, lower FE
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "melvin.c"
#include "melvin_diagnostics.c"

#define TEST_FILE_STRUCTURED "diag_b_structured.m"
#define TEST_FILE_RANDOM "diag_b_random.m"
#define CHANNEL_ID 1
#define BYTE_A 65
#define BYTE_B 66
#define STREAM_LENGTH 500
#define EVENTS_PER_BYTE 5
#define SNAPSHOT_INTERVAL 50

typedef struct {
    uint64_t node_a_id;
    uint64_t node_b_id;
    uint64_t node_a_idx;
    uint64_t node_b_idx;
    float edge_weight;
    float error_at_b;
    float fe_a;
    float fe_b;
    int pattern_count;
} PatternMetrics;

static PatternMetrics run_pattern_diagnostic(const char *test_name, const char *file_path, 
                                             uint8_t *byte_stream, int stream_length, 
                                             const char *diag_dir) {
    printf("\n========================================\n");
    printf("RUN: %s\n", test_name);
    printf("========================================\n\n");
    
    diagnostics_init(diag_dir);
    
    PatternMetrics metrics = {0};
    
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
        diagnostics_cleanup();
        return metrics;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        diagnostics_cleanup();
        return metrics;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        diagnostics_cleanup();
        return metrics;
    }
    
    metrics.node_a_id = (uint64_t)BYTE_A + 1000000ULL;
    metrics.node_b_id = (uint64_t)BYTE_B + 1000000ULL;
    
    // Process byte stream
    printf("Processing %d bytes...\n", stream_length);
    
    for (int i = 0; i < stream_length; i++) {
        diagnostics_increment_event_counter();
        uint64_t event_idx = diagnostics_get_event_counter();
        
        uint8_t byte = byte_stream[i];
        
        // Get state before ingestion
        metrics.node_a_idx = find_node_index_by_id(&file, metrics.node_a_id);
        metrics.node_b_idx = find_node_index_by_id(&file, metrics.node_b_id);
        
        float state_a_before = 0.0f;
        float state_b_before = 0.0f;
        float prediction_a_before = 0.0f;
        float prediction_b_before = 0.0f;
        
        if (metrics.node_a_idx != UINT64_MAX) {
            NodeDisk *node_a = &file.nodes[metrics.node_a_idx];
            state_a_before = node_a->state;
            prediction_a_before = node_a->prediction;
        }
        if (metrics.node_b_idx != UINT64_MAX) {
            NodeDisk *node_b = &file.nodes[metrics.node_b_idx];
            state_b_before = node_b->state;
            prediction_b_before = node_b->prediction;
        }
        
        // Ingest byte
        ingest_byte(&rt, CHANNEL_ID, byte, 1.0f);
        melvin_process_n_events(&rt, EVENTS_PER_BYTE);
        
        // Get state after
        metrics.node_a_idx = find_node_index_by_id(&file, metrics.node_a_id);
        metrics.node_b_idx = find_node_index_by_id(&file, metrics.node_b_id);
        
        if (metrics.node_a_idx != UINT64_MAX) {
            NodeDisk *node_a = &file.nodes[metrics.node_a_idx];
            float state_a_after = node_a->state;
            float prediction_a_after = node_a->prediction;
            float error_a = fabsf(node_a->prediction_error);
            float fe_a_inst = node_a->fe_ema;
            float fe_a_ema = node_a->fe_ema;
            float traffic_a = node_a->traffic_ema;
            
            diagnostics_log_node_update(
                event_idx,
                metrics.node_a_id,
                state_a_before,
                state_a_after,
                prediction_a_before,
                prediction_a_after,
                error_a,
                fe_a_inst,
                fe_a_ema,
                traffic_a
            );
            
            metrics.fe_a = fe_a_ema;
        }
        
        if (metrics.node_b_idx != UINT64_MAX) {
            NodeDisk *node_b = &file.nodes[metrics.node_b_idx];
            float state_b_after = node_b->state;
            float prediction_b_after = node_b->prediction;
            float error_b = fabsf(node_b->prediction_error);
            float fe_b_inst = node_b->fe_ema;
            float fe_b_ema = node_b->fe_ema;
            float traffic_b = node_b->traffic_ema;
            
            diagnostics_log_node_update(
                event_idx,
                metrics.node_b_id,
                state_b_before,
                state_b_after,
                prediction_b_before,
                prediction_b_after,
                error_b,
                fe_b_inst,
                fe_b_ema,
                traffic_b
            );
            
            metrics.fe_b = fe_b_ema;
            metrics.error_at_b = error_b;
        }
        
        // Check edge A→B
        if (edge_exists_between(&file, metrics.node_a_id, metrics.node_b_id)) {
            EdgeDisk *edges = file.edges;
            GraphHeaderDisk *gh = file.graph_header;
            
            for (uint64_t e = 0; e < gh->num_edges; e++) {
                if (edges[e].src == metrics.node_a_id && 
                    edges[e].dst == metrics.node_b_id) {
                    float weight_before = metrics.edge_weight;
                    float weight_after = edges[e].weight;
                    float delta_w = weight_after - weight_before;
                    
                    diagnostics_log_edge_update(
                        event_idx,
                        metrics.node_a_id,
                        metrics.node_b_id,
                        weight_before,
                        weight_after,
                        delta_w,
                        edges[e].eligibility,
                        edges[e].usage,
                        edges[e].last_energy
                    );
                    
                    metrics.edge_weight = weight_after;
                    break;
                }
            }
        }
        
        // Periodic snapshots
        if (i % SNAPSHOT_INTERVAL == 0 || i == stream_length - 1) {
            GraphHeaderDisk *gh = file.graph_header;
            
            // Count patterns
            int pattern_count = 0;
            int seq_edges = 0;
            int chan_edges = 0;
            int bonds = 0;
            
            for (uint64_t n = 0; n < gh->num_nodes; n++) {
                if (file.nodes[n].id >= 5000000ULL && file.nodes[n].id < 10000000ULL) {
                    pattern_count++;
                }
            }
            
            for (uint64_t e = 0; e < gh->num_edges; e++) {
                if (file.edges[e].src == UINT64_MAX) continue;
                if (file.edges[e].flags & 0x01) seq_edges++;  // SEQ flag
                if (file.edges[e].flags & 0x02) chan_edges++;  // CHAN flag
                if (file.edges[e].is_bond) bonds++;
            }
            
            metrics.pattern_count = pattern_count;
            
            printf("  Event %d: edge_weight=%.6f, error_B=%.6f, FE_A=%.6f, FE_B=%.6f, patterns=%d\n",
                   i, metrics.edge_weight, metrics.error_at_b, metrics.fe_a, metrics.fe_b, pattern_count);
        }
    }
    
    printf("\nFinal metrics:\n");
    printf("  Edge A→B weight: %.6f\n", metrics.edge_weight);
    printf("  Error at B: %.6f\n", metrics.error_at_b);
    printf("  FE at A: %.6f\n", metrics.fe_a);
    printf("  FE at B: %.6f\n", metrics.fe_b);
    printf("  Patterns: %d\n", metrics.pattern_count);
    
    runtime_cleanup(&rt);
    close_file(&file);
    diagnostics_cleanup();
    
    return metrics;
}

int main() {
    printf("========================================\n");
    printf("DIAGNOSTIC EXPERIMENT B: Pattern Learning\n");
    printf("========================================\n");
    
    // Structured stream
    uint8_t structured_stream[STREAM_LENGTH];
    for (int i = 0; i < STREAM_LENGTH; i++) {
        structured_stream[i] = (i % 2 == 0) ? BYTE_A : BYTE_B;
    }
    
    PatternMetrics structured = run_pattern_diagnostic(
        "Structured Stream (ABABAB...)",
        TEST_FILE_STRUCTURED,
        structured_stream,
        STREAM_LENGTH,
        "diag_b_structured_results"
    );
    
    // Random stream
    srand(time(NULL));
    uint8_t random_stream[STREAM_LENGTH];
    for (int i = 0; i < STREAM_LENGTH; i++) {
        random_stream[i] = 65 + (rand() % 26);  // Random A-Z
    }
    
    PatternMetrics random = run_pattern_diagnostic(
        "Random Stream (Control)",
        TEST_FILE_RANDOM,
        random_stream,
        STREAM_LENGTH,
        "diag_b_random_results"
    );
    
    // Comparison
    printf("\n========================================\n");
    printf("COMPARISON:\n");
    printf("========================================\n");
    printf("Structured:\n");
    printf("  Edge weight: %.6f\n", structured.edge_weight);
    printf("  Error at B: %.6f\n", structured.error_at_b);
    printf("  FE at A: %.6f\n", structured.fe_a);
    printf("  FE at B: %.6f\n", structured.fe_b);
    printf("  Patterns: %d\n", structured.pattern_count);
    printf("\n");
    printf("Random:\n");
    printf("  Edge weight: %.6f\n", random.edge_weight);
    printf("  Error at B: %.6f\n", random.error_at_b);
    printf("  FE at A: %.6f\n", random.fe_a);
    printf("  FE at B: %.6f\n", random.fe_b);
    printf("  Patterns: %d\n", random.pattern_count);
    printf("\n");
    
    // Analysis
    int differences = 0;
    
    if (structured.edge_weight > random.edge_weight) {
        printf("✓ Structured has stronger edge\n");
        differences++;
    } else {
        printf("❌ Structured edge not stronger\n");
    }
    
    if (structured.error_at_b < random.error_at_b) {
        printf("✓ Structured has lower error at B\n");
        differences++;
    } else {
        printf("❌ Structured error not lower\n");
    }
    
    if (structured.fe_b < random.fe_b) {
        printf("✓ Structured has lower FE at B\n");
        differences++;
    } else {
        printf("❌ Structured FE not lower\n");
    }
    
    printf("\nDifferences detected: %d/3\n", differences);
    
    if (differences >= 2) {
        printf("✓ PASS: Information is being captured\n");
        return 0;
    } else {
        printf("❌ FAIL: No clear information capture\n");
        printf("Check edge_diagnostics.csv for learning rule behavior\n");
        return 1;
    }
}


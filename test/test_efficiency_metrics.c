/*
 * TEST: Efficiency Metrics - Verify efficiency-aware FE is working
 * 
 * This test measures:
 * 1. Pattern node complexity vs EXEC node complexity
 * 2. Efficiency scores (FE per traffic)
 * 3. Stability scores (should favor efficient nodes)
 * 4. Graph size (should penalize large inefficient structures)
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include "melvin.c"

#define TEST_FILE "test_efficiency_metrics.m"

// Measure efficiency metrics for a node
static void measure_node_efficiency(MelvinFile *file, uint64_t node_id, const char *label) {
    uint64_t node_idx = find_node_index_by_id(file, node_id);
    if (node_idx == UINT64_MAX) {
        printf("  %s: Node not found\n", label);
        return;
    }
    
    NodeDisk *n = &file->nodes[node_idx];
    
    // Count edges
    GraphHeaderDisk *gh = file->graph_header;
    uint64_t deg_in = 0;
    EdgeDisk *edges = file->edges;
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        if (edges[e].src == UINT64_MAX) continue;
        if (edges[e].dst == node_id) deg_in++;
    }
    uint64_t deg_out = n->out_degree;
    uint64_t total_degree = deg_in + deg_out;
    
    // Calculate complexity (approximate - using defaults)
    float k_deg = 0.1f;
    float k_payload = 0.01f;
    float complexity = k_deg * (float)total_degree + k_payload * (float)n->payload_len;
    
    // Efficiency: FE_ema / (traffic_ema + eps)
    float eps_eff = 0.001f;
    float efficiency = 0.0f;
    if (n->traffic_ema > 0.0f || n->fe_ema > 0.0f) {
        efficiency = n->fe_ema / (n->traffic_ema + eps_eff);
    }
    
    printf("  %s:\n", label);
    printf("    Node ID: %llu\n", (unsigned long long)node_id);
    printf("    Degree: %llu (in: %llu, out: %llu)\n", 
           (unsigned long long)total_degree, (unsigned long long)deg_in, (unsigned long long)deg_out);
    printf("    Payload: %u bytes\n", n->payload_len);
    printf("    Complexity: %.6f\n", complexity);
    printf("    Activation: %.6f\n", n->state);
    printf("    Traffic EMA: %.6f\n", n->traffic_ema);
    printf("    FE EMA: %.6f\n", n->fe_ema);
    printf("    Efficiency: %.6f (lower = better)\n", efficiency);
    printf("    Stability: %.6f (higher = better)\n", n->stability);
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("EFFICIENCY METRICS TEST\n");
    printf("========================================\n\n");
    
    printf("Goal: Verify efficiency-aware FE penalizes large structures\n");
    printf("      and rewards efficient EXEC nodes\n\n");
    
    srand(time(NULL));
    unlink(TEST_FILE);
    
    // Initialize
    printf("Initializing...\n");
    GraphParams params;
    init_default_params(&params);
    params.decay_rate = 0.90f;
    params.learning_rate = 0.02f;
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
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
        return 1;
    }
    printf("  ✓ Initialized\n\n");
    
    // Step 1: Create EXEC node (should be efficient)
    printf("Step 1: Creating EXEC node (efficient computation)...\n");
    const uint8_t ARM64_ADD[] = {
        0x00, 0x00, 0x01, 0x8b,  // add x0, x0, x1
        0xc0, 0x03, 0x5f, 0xd6   // ret
    };
    
    uint64_t add_offset = melvin_write_machine_code(&file, ARM64_ADD, sizeof(ARM64_ADD));
    uint64_t exec_id = melvin_create_executable_node(&file, add_offset, sizeof(ARM64_ADD));
    printf("  ✓ EXEC node created: %llu\n", (unsigned long long)exec_id);
    printf("\n");
    
    // Step 2: Feed lots of examples (creates large pattern graph)
    printf("Step 2: Feeding 200 addition examples (creates large pattern graph)...\n");
    for (int i = 0; i < 200; i++) {
        int a = rand() % 100;
        int b = rand() % 100;
        if (a == 50 && b == 50) a = 49;
        int sum = a + b;
        
        char problem[64];
        snprintf(problem, sizeof(problem), "%d+%d=%d", a, b, sum);
        
        for (int j = 0; problem[j] != '\0'; j++) {
            ingest_byte(&rt, 0, problem[j], 1.0f);
            melvin_process_n_events(&rt, 3);
        }
        
        if ((i + 1) % 50 == 0) {
            GraphHeaderDisk *gh = file.graph_header;
            printf("  [%3d examples] Nodes: %4llu, Edges: %4llu\n",
                   i + 1,
                   (unsigned long long)gh->num_nodes,
                   (unsigned long long)gh->num_edges);
        }
    }
    printf("  ✓ Training complete\n\n");
    
    // Step 3: Process more events to let efficiency metrics stabilize
    printf("Step 3: Processing events to stabilize efficiency metrics...\n");
    melvin_process_n_events(&rt, 500);
    printf("  ✓ Processed\n\n");
    
    // Step 4: Measure efficiency of pattern nodes vs EXEC node
    printf("Step 4: Measuring efficiency metrics...\n\n");
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Find a pattern node (one with ID in pattern range)
    uint64_t pattern_node_id = UINT64_MAX;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file.nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            pattern_node_id = n->id;
            break;
        }
    }
    
    if (pattern_node_id != UINT64_MAX) {
        measure_node_efficiency(&file, pattern_node_id, "Pattern Node (example)");
    }
    
    measure_node_efficiency(&file, exec_id, "EXEC Node");
    
    // Step 5: Summary statistics
    printf("Step 5: Summary statistics...\n\n");
    
    uint64_t pattern_count = 0;
    float total_pattern_complexity = 0.0f;
    float total_pattern_efficiency = 0.0f;
    float total_pattern_stability = 0.0f;
    
    EdgeDisk *edges = file.edges;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file.nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            pattern_count++;
            
            // Count edges
            uint64_t deg_in = 0;
            for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
                if (edges[e].src == UINT64_MAX) continue;
                if (edges[e].dst == n->id) deg_in++;
            }
            float complexity = 0.1f * (float)(deg_in + n->out_degree) + 0.01f * (float)n->payload_len;
            total_pattern_complexity += complexity;
            
            float eps_eff = 0.001f;
            float efficiency = n->fe_ema / (n->traffic_ema + eps_eff);
            total_pattern_efficiency += efficiency;
            
            total_pattern_stability += n->stability;
        }
    }
    
    printf("  Pattern nodes: %llu\n", (unsigned long long)pattern_count);
    if (pattern_count > 0) {
        printf("  Avg pattern complexity: %.6f\n", total_pattern_complexity / pattern_count);
        printf("  Avg pattern efficiency: %.6f\n", total_pattern_efficiency / pattern_count);
        printf("  Avg pattern stability: %.6f\n", total_pattern_stability / pattern_count);
    }
    
    // EXEC node stats
    uint64_t exec_idx = find_node_index_by_id(&file, exec_id);
    if (exec_idx != UINT64_MAX) {
        NodeDisk *exec = &file.nodes[exec_idx];
        uint64_t deg_in = 0;
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            if (edges[e].src == UINT64_MAX) continue;
            if (edges[e].dst == exec_id) deg_in++;
        }
        float exec_complexity = 0.1f * (float)(deg_in + exec->out_degree) + 0.01f * (float)exec->payload_len;
        float eps_eff = 0.001f;
        float exec_efficiency = exec->fe_ema / (exec->traffic_ema + eps_eff);
        
        printf("\n  EXEC node:\n");
        printf("    Complexity: %.6f\n", exec_complexity);
        printf("    Efficiency: %.6f\n", exec_efficiency);
        printf("    Stability: %.6f\n", exec->stability);
    }
    
    printf("\n");
    printf("========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("Expected behavior with efficiency-aware FE:\n");
    printf("  1. EXEC node should have LOWER complexity (fewer edges)\n");
    printf("  2. EXEC node should have LOWER efficiency score (if it's efficient)\n");
    printf("  3. EXEC node should have HIGHER stability (if efficient)\n");
    printf("  4. Pattern nodes should have HIGHER complexity (many edges)\n");
    printf("  5. Pattern nodes with high complexity should have LOWER stability\n");
    printf("\n");
    
    printf("If efficiency-aware FE is working:\n");
    printf("  - Large pattern graphs will be penalized by complexity\n");
    printf("  - Small EXEC nodes will be rewarded for efficiency\n");
    printf("  - System will prefer compact, efficient structures\n");
    printf("\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    return 0;
}


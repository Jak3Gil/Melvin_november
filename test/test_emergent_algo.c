#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/mman.h>

// Include the implementation
#include "melvin.c"

// ========================================================================
// TEST: Emergent Algorithm Formation
// 
// Tests if repeated patterns form reusable computational structures.
// This tests if patterns can become functional abstractions.
// ========================================================================

int main(int argc, char **argv) {
    const char *file_path = "test_emergent_algo.m";
    
    printf("========================================\n");
    printf("EMERGENT ALGORITHM FORMATION TEST\n");
    printf("========================================\n\n");
    printf("Goal: Test if repeated patterns form reusable structures\n");
    printf("This tests algorithm discovery through pattern formation.\n\n");
    
    // Create new file
    GraphParams params;
    params.decay_rate = 0.05f;  // Lower decay to maintain patterns
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.exec_threshold = 1.0f;
    params.learning_rate = 0.002f;  // Higher learning rate
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    unlink(file_path);
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return 1;
    }
    printf("✓ Created %s\n", file_path);
    
    // Map file
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    printf("✓ Mapped file\n");
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n");
    
    // Create repeated patterns to form "algorithms"
    printf("\nStep 1: Creating repeated patterns...\n");
    const uint64_t CH_TEXT = 1;
    
    // Pattern 1: "ABC" repeated many times (should form strong edges)
    printf("  Pattern 1: 'ABC' (repeated 100 times)...\n");
    for (int i = 0; i < 100; i++) {
        ingest_byte(&rt, CH_TEXT, 'A', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, CH_TEXT, 'B', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, CH_TEXT, 'C', 1.0f);
        melvin_process_n_events(&rt, 50);
    }
    
    // Pattern 2: "XYZ" repeated many times
    printf("  Pattern 2: 'XYZ' (repeated 100 times)...\n");
    for (int i = 0; i < 100; i++) {
        ingest_byte(&rt, CH_TEXT, 'X', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, CH_TEXT, 'Y', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, CH_TEXT, 'Z', 1.0f);
        melvin_process_n_events(&rt, 50);
    }
    
    printf("✓ Patterns ingested\n");
    
    // Analyze graph structure
    printf("\nStep 2: Analyzing graph structure for emergent algorithms...\n");
    
    GraphHeaderDisk *gh = file.graph_header;
    NodeDisk *nodes = file.nodes;
    EdgeDisk *edges = file.edges;
    
    // Find strong edge sequences (potential algorithms)
    printf("\n  Strong edge sequences (weight > 0.5):\n");
    
    // Find A, B, C nodes
    uint64_t a_node_id = (uint64_t)'A' + 1000000ULL;
    uint64_t b_node_id = (uint64_t)'B' + 1000000ULL;
    uint64_t c_node_id = (uint64_t)'C' + 1000000ULL;
    
    uint64_t a_idx = find_node_index_by_id(&file, a_node_id);
    uint64_t b_idx = find_node_index_by_id(&file, b_node_id);
    uint64_t c_idx = find_node_index_by_id(&file, c_node_id);
    
    float w_ab = 0.0f, w_bc = 0.0f;
    
    if (a_idx != UINT64_MAX && a_idx < gh->node_capacity) {
        NodeDisk *a_node = &nodes[a_idx];
        uint64_t edge_idx = a_node->first_out_edge;
        
        for (uint32_t i = 0; i < a_node->out_degree && edge_idx != UINT64_MAX; i++) {
            if (edge_idx >= gh->edge_capacity) break;
            EdgeDisk *e = &edges[edge_idx];
            if (e->dst == b_node_id && (e->flags & EDGE_FLAG_SEQ)) {
                w_ab = e->weight;
                break;
            }
            edge_idx = e->next_out_edge;
        }
    }
    
    if (b_idx != UINT64_MAX && b_idx < gh->node_capacity) {
        NodeDisk *b_node = &nodes[b_idx];
        uint64_t edge_idx = b_node->first_out_edge;
        
        for (uint32_t i = 0; i < b_node->out_degree && edge_idx != UINT64_MAX; i++) {
            if (edge_idx >= gh->edge_capacity) break;
            EdgeDisk *e = &edges[edge_idx];
            if (e->dst == c_node_id && (e->flags & EDGE_FLAG_SEQ)) {
                w_bc = e->weight;
                break;
            }
            edge_idx = e->next_out_edge;
        }
    }
    
    printf("    A -> B: weight = %.4f\n", w_ab);
    printf("    B -> C: weight = %.4f\n", w_bc);
    
    // Check if pattern forms a "routine" (strong chain)
    int pattern_formed = (w_ab > 0.5f && w_bc > 0.5f);
    
    // Find X, Y, Z nodes
    uint64_t x_node_id = (uint64_t)'X' + 1000000ULL;
    uint64_t y_node_id = (uint64_t)'Y' + 1000000ULL;
    uint64_t z_node_id = (uint64_t)'Z' + 1000000ULL;
    
    uint64_t x_idx = find_node_index_by_id(&file, x_node_id);
    uint64_t y_idx = find_node_index_by_id(&file, y_node_id);
    uint64_t z_idx = find_node_index_by_id(&file, z_node_id);
    
    float w_xy = 0.0f, w_yz = 0.0f;
    
    if (x_idx != UINT64_MAX && x_idx < gh->node_capacity) {
        NodeDisk *x_node = &nodes[x_idx];
        uint64_t edge_idx = x_node->first_out_edge;
        
        for (uint32_t i = 0; i < x_node->out_degree && edge_idx != UINT64_MAX; i++) {
            if (edge_idx >= gh->edge_capacity) break;
            EdgeDisk *e = &edges[edge_idx];
            if (e->dst == y_node_id && (e->flags & EDGE_FLAG_SEQ)) {
                w_xy = e->weight;
                break;
            }
            edge_idx = e->next_out_edge;
        }
    }
    
    if (y_idx != UINT64_MAX && y_idx < gh->node_capacity) {
        NodeDisk *y_node = &nodes[y_idx];
        uint64_t edge_idx = y_node->first_out_edge;
        
        for (uint32_t i = 0; i < y_node->out_degree && edge_idx != UINT64_MAX; i++) {
            if (edge_idx >= gh->edge_capacity) break;
            EdgeDisk *e = &edges[edge_idx];
            if (e->dst == z_node_id && (e->flags & EDGE_FLAG_SEQ)) {
                w_yz = e->weight;
                break;
            }
            edge_idx = e->next_out_edge;
        }
    }
    
    printf("    X -> Y: weight = %.4f\n", w_xy);
    printf("    Y -> Z: weight = %.4f\n", w_yz);
    
    int pattern2_formed = (w_xy > 0.5f && w_yz > 0.5f);
    
    // Test if patterns are "reusable" (activating A predicts C)
    printf("\nStep 3: Testing pattern reusability...\n");
    
    // Activate A and see if energy flows to C
    float initial_c_activation = 0.0f;
    if (c_idx != UINT64_MAX && c_idx < gh->node_capacity) {
        initial_c_activation = nodes[c_idx].state;
    }
    
    // Inject energy into A
    MelvinEvent ev = {
        .type = EV_NODE_DELTA,
        .node_id = a_node_id,
        .value = 1.0f
    };
    melvin_event_enqueue(&rt.evq, &ev);
    melvin_process_n_events(&rt, 100);
    
    // Check if C activated
    float final_c_activation = 0.0f;
    if (c_idx != UINT64_MAX && c_idx < gh->node_capacity) {
        final_c_activation = nodes[c_idx].state;
    }
    
    float c_activation_delta = final_c_activation - initial_c_activation;
    
    printf("  Activating A node...\n");
    printf("    C activation before: %.6f\n", initial_c_activation);
    printf("    C activation after: %.6f\n", final_c_activation);
    printf("    C activation delta: %.6f\n", c_activation_delta);
    
    int pattern_reusable = (c_activation_delta > 0.1f);
    
    // Evaluate results
    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    
    int success = 1;
    
    if (pattern_formed) {
        printf("✓ PASS: Pattern 'ABC' formed strong edges (algorithm-like structure)\n");
        printf("  A->B: %.4f, B->C: %.4f\n", w_ab, w_bc);
    } else {
        printf("✗ FAIL: Pattern 'ABC' did not form strong edges\n");
        printf("  A->B: %.4f, B->C: %.4f (need > 0.5)\n", w_ab, w_bc);
        success = 0;
    }
    
    if (pattern2_formed) {
        printf("✓ PASS: Pattern 'XYZ' formed strong edges\n");
        printf("  X->Y: %.4f, Y->Z: %.4f\n", w_xy, w_yz);
    } else {
        printf("⚠ WARNING: Pattern 'XYZ' did not form strong edges\n");
        printf("  X->Y: %.4f, Y->Z: %.4f\n", w_xy, w_yz);
    }
    
    if (pattern_reusable) {
        printf("✓ PASS: Pattern is reusable (activating A predicts C)\n");
        printf("  Energy flows through pattern: A -> B -> C\n");
    } else {
        printf("✗ FAIL: Pattern is not reusable\n");
        printf("  Activating A did not significantly activate C (delta: %.6f)\n", c_activation_delta);
        success = 0;
    }
    
    // Final evaluation
    printf("\n========================================\n");
    if (success && pattern_formed && pattern_reusable) {
        printf("✅ TEST PASSED: Emergent algorithm formation works!\n");
        printf("Repeated patterns form reusable computational structures.\n");
        printf("This enables algorithm discovery through pattern formation.\n");
    } else {
        printf("❌ TEST FAILED: Emergent algorithm formation did not work\n");
        printf("Patterns may not form reusable structures.\n");
    }
    printf("========================================\n");
    
    // Final stats
    printf("\nFinal graph state:\n");
    printf("  Nodes: %llu\n", (unsigned long long)gh->num_nodes);
    printf("  Edges: %llu\n", (unsigned long long)gh->num_edges);
    printf("  Avg activation: %.6f\n", gh->avg_activation);
    
    // Sync and close
    melvin_m_sync(&file);
    runtime_cleanup(&rt);
    close_file(&file);
    
    return success ? 0 : 1;
}


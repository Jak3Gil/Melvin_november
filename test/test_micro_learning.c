// BRUTAL MICRO-TEST: Does weight actually move off 0.200 when pre=post=1 and error≠0?
// This test isolates the learning law to prove weights CAN move, or find hidden clamps.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

// Include melvin.c directly to get all functions
#include "melvin.c"

int main() {
    printf("========================================\n");
    printf("MICRO-LEARNING TEST: Weight Movement\n");
    printf("========================================\n");
    printf("Goal: Prove weights CAN move off 0.200 when:\n");
    printf("  - pre=1.0, post=1.0 (co-active)\n");
    printf("  - prediction_error=1.0\n");
    printf("  - Run 1000 learning steps\n");
    printf("========================================\n\n");
    
    const char *file_path = "test_micro_learning.m";
    unlink(file_path);
    
    // Create minimal graph
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create test file\n");
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to init runtime\n");
        close_file(&file);
        return 1;
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Ensure capacity
    if (gh->num_nodes + 2 > gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 2);
        gh = file.graph_header;
    }
    if (gh->num_edges + 1 > gh->edge_capacity) {
        melvin_m_ensure_edge_capacity(&file, gh->num_edges + 1);
        gh = file.graph_header;
    }
    
    // Create two nodes: A (pre) and B (post)
    uint64_t node_a_id = 10000ULL;
    uint64_t node_b_id = 10001ULL;
    
    uint64_t node_a_idx = gh->num_nodes++;
    NodeDisk *node_a = &file.nodes[node_a_idx];
    memset(node_a, 0, sizeof(NodeDisk));
    node_a->id = node_a_id;
    node_a->state = 1.0f;  // Pre always active
    node_a->prediction = 0.0f;
    node_a->prediction_error = 0.0f;
    
    uint64_t node_b_idx = gh->num_nodes++;
    NodeDisk *node_b = &file.nodes[node_b_idx];
    memset(node_b, 0, sizeof(NodeDisk));
    node_b->id = node_b_id;
    node_b->state = 1.0f;  // Post always active
    node_b->prediction = 0.0f;
    node_b->prediction_error = 1.0f;  // Fixed prediction error
    
    // Create edge A->B with initial weight = 0.2 using create_edge_between
    // This properly links the edge to nodes
    if (create_edge_between(&file, node_a_id, node_b_id, 0.2f) < 0) {
        fprintf(stderr, "ERROR: Failed to create edge\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    // Find the edge we just created
    EdgeDisk *edge = NULL;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_a_id && file.edges[i].dst == node_b_id) {
            edge = &file.edges[i];
            break;
        }
    }
    
    if (!edge) {
        fprintf(stderr, "ERROR: Edge not found after creation\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    printf("Initial state:\n");
    printf("  Node A: id=%llu, state=%.6f\n", (unsigned long long)node_a_id, node_a->state);
    printf("  Node B: id=%llu, state=%.6f, prediction_error=%.6f\n", 
           (unsigned long long)node_b_id, node_b->state, node_b->prediction_error);
    printf("  Edge A->B: weight=%.6f, eligibility=%.6f\n", edge->weight, edge->eligibility);
    printf("\n");
    
    // Run 1000 learning steps
    printf("Running 1000 learning steps...\n");
    float weight_before = edge->weight;
    
    for (int step = 0; step < 1000; step++) {
        // Force co-activity: both nodes active
        node_a->state = 1.0f;
        node_b->state = 1.0f;
        node_b->prediction_error = 1.0f;  // Positive error should strengthen
        
        // Update eligibility: eligibility = decay * eligibility + pre * post
        // From melvin.c line 3586
        // Read eligibility_decay from g_params
        edge->eligibility = g_params.eligibility_decay * edge->eligibility + 
                           node_a->state * node_b->state;
        
        // Call the actual learning function
        strengthen_edges_with_prediction_and_reward(&rt);
        
        // Also call message_passing which updates weights
        message_passing(&rt);
        
        // Print every 100 steps
        if ((step + 1) % 100 == 0) {
            printf("  Step %d: weight=%.9f, eligibility=%.9f, delta=%.9f\n",
                   step + 1, edge->weight, edge->eligibility, edge->weight - weight_before);
        }
        
        // Safety: if weight becomes NaN or Inf, abort
        if (isnan(edge->weight) || isinf(edge->weight)) {
            fprintf(stderr, "ERROR: Weight became NaN/Inf at step %d\n", step);
            break;
        }
    }
    
    float weight_after = edge->weight;
    float weight_change = weight_after - weight_before;
    
    printf("\n");
    printf("Final state:\n");
    printf("  Weight before: %.9f\n", weight_before);
    printf("  Weight after:  %.9f\n", weight_after);
    printf("  Weight change: %.9f\n", weight_change);
    printf("  Eligibility:   %.9f\n", edge->eligibility);
    printf("\n");
    
    // BRUTAL ASSERTION: Weight MUST have moved
    bool weight_moved = fabsf(weight_change) > 0.000001f;
    bool weight_increased = weight_after > weight_before;
    
    printf("========================================\n");
    if (weight_moved) {
        printf("✅ PASS: Weight moved off 0.200\n");
        printf("   Change: %.9f\n", weight_change);
        if (weight_increased) {
            printf("   Direction: INCREASED (strengthening)\n");
        } else {
            printf("   Direction: DECREASED (weakening)\n");
        }
        printf("\n");
        printf("CONCLUSION: Learning law works. Tests need co-activity + error.\n");
    } else {
        printf("❌ FAIL: Weight STUCK at 0.200\n");
        printf("   Change: %.9f (effectively zero)\n", weight_change);
        printf("\n");
        printf("CONCLUSION: Hidden clamp/reset detected. Search for:\n");
        printf("  - Hard clamp at 0.2f\n");
        printf("  - Normalization resetting weights\n");
        printf("  - Template copy overwriting updates\n");
    }
    printf("========================================\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
    
    return weight_moved ? 0 : 1;
}


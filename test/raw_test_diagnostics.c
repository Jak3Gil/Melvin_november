#define _POSIX_C_SOURCE 200809L

/*
 * raw_test_diagnostics.c
 * 
 * Captures RAW state data during failing tests to understand WHY they fail.
 * No assumptions, just data.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "melvin.c"

// ========================================================================
// Test 0.2.2: Exec Subtracts Cost - RAW DATA
// ========================================================================

static void diagnose_test_0_2_2_raw() {
    printf("\n");
    printf("========================================\n");
    printf("TEST 0.2.2: Exec Subtracts Cost - RAW DATA\n");
    printf("========================================\n\n");
    
    const char *file_path = "diag_exec_cost.m";
    unlink(file_path);
    
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cp melvin.m %s", file_path);
    system(cmd);
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        printf("[ERROR] Failed to map file\n");
        return;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        printf("[ERROR] Failed to init runtime\n");
        close_file(&file);
        return;
    }
    
    printf("[RAW] Runtime state:\n");
    printf("  exec_cost: %.6f\n", rt.exec_cost);
    printf("  gpu_cost_multiplier: %.6f\n", rt.gpu_cost_multiplier);
    printf("  exec_calls: %llu\n", (unsigned long long)rt.exec_calls);
    printf("  g_params.exec_factor_min: %.6f\n", g_params.exec_factor_min);
    printf("  g_params.exec_k: %.6f\n", g_params.exec_k);
    printf("  g_params.exec_center: %.6f\n", g_params.exec_center);
    
    GraphHeaderDisk *gh = file.graph_header;
    printf("  gh->exec_threshold: %.6f\n", gh->exec_threshold);
    
    // Create EXECUTABLE node
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 2000ULL;
    node->state = 2.0f;
    node->flags = NODE_FLAG_EXECUTABLE;
    
    #if defined(__aarch64__) || defined(_M_ARM64)
    static const uint8_t stub[] = {0x40, 0x08, 0x80, 0xD2, 0xC0, 0x03, 0x5F, 0xD6};
    #else
    static const uint8_t stub[] = {0x48, 0xC7, 0xC0, 0x42, 0x00, 0x00, 0x00, 0xC3};
    #endif
    
    uint64_t code_offset = melvin_write_machine_code(&file, stub, sizeof(stub));
    if (code_offset == UINT64_MAX) {
        printf("[ERROR] Failed to write code\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return;
    }
    
    node->payload_offset = code_offset;
    node->payload_len = sizeof(stub);
    
    printf("\n[RAW] Node state BEFORE execution:\n");
    printf("  node->id: %llu\n", (unsigned long long)node->id);
    printf("  node->state: %.6f\n", node->state);
    printf("  node->flags: 0x%x (EXECUTABLE=%d)\n", node->flags, 
           (node->flags & NODE_FLAG_EXECUTABLE) ? 1 : 0);
    printf("  node->payload_offset: %llu\n", (unsigned long long)node->payload_offset);
    printf("  node->payload_len: %llu\n", (unsigned long long)node->payload_len);
    
    // Compute exec_factor (melvin.c line 4617-4623)
    float exec_center = gh->exec_threshold;
    float exec_k = g_params.exec_k;
    float x = exec_k * (node->state - exec_center);
    float exec_factor = 1.0f / (1.0f + expf(-x));
    printf("\n[RAW] Execution probability calculation:\n");
    printf("  exec_center (gh->exec_threshold): %.6f\n", exec_center);
    printf("  exec_k: %.6f\n", exec_k);
    printf("  node->state: %.6f\n", node->state);
    printf("  x = exec_k * (state - center): %.6f\n", x);
    printf("  exec_factor: %.6f\n", exec_factor);
    printf("  exec_factor_min: %.6f\n", g_params.exec_factor_min);
    printf("  Will execute? %s\n", (exec_factor >= g_params.exec_factor_min) ? "YES" : "NO");
    
    float state_before = node->state;
    uint64_t exec_calls_before = rt.exec_calls;
    
    printf("\n[RAW] Triggering pulse and processing events...\n");
    inject_pulse(&rt, 2000ULL, 2.0f);
    
    printf("  State after pulse: %.6f\n", node->state);
    
    // Process events and log what happens
    printf("\n[RAW] Processing 200 events, logging every 50:\n");
    for (int i = 0; i < 200; i++) {
        if (i % 50 == 0) {
            printf("  Event %d: state=%.6f, exec_calls=%llu\n", 
                   i, node->state, (unsigned long long)rt.exec_calls);
        }
        melvin_process_n_events(&rt, 1);
    }
    
    float state_after = node->state;
    uint64_t exec_calls_after = rt.exec_calls;
    float cost_applied = state_before - state_after;
    float expected_cost = rt.exec_cost * rt.gpu_cost_multiplier;
    
    printf("\n[RAW] Final state:\n");
    printf("  state_before: %.6f\n", state_before);
    printf("  state_after: %.6f\n", state_after);
    printf("  cost_applied: %.6f\n", cost_applied);
    printf("  expected_cost: %.6f\n", expected_cost);
    printf("  exec_calls: %llu -> %llu (delta: %llu)\n",
           (unsigned long long)exec_calls_before,
           (unsigned long long)exec_calls_after,
           (unsigned long long)(exec_calls_after - exec_calls_before));
    
    // Check if node is still in graph
    uint64_t found_idx = find_node_index_by_id(&file, 2000ULL);
    printf("  Node still in graph? %s (idx=%llu)\n",
           (found_idx != UINT64_MAX) ? "YES" : "NO",
           (unsigned long long)found_idx);
    
    if (found_idx != UINT64_MAX) {
        NodeDisk *found_node = &file.nodes[found_idx];
        printf("  Found node state: %.6f\n", found_node->state);
        printf("  Found node flags: 0x%x\n", found_node->flags);
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// Test 0.5.1: Learning Only Prediction Error - RAW DATA
// ========================================================================

static void diagnose_test_0_5_1_raw() {
    printf("\n");
    printf("========================================\n");
    printf("TEST 0.5.1: Learning Only Prediction Error - RAW DATA\n");
    printf("========================================\n\n");
    
    const char *file_path = "diag_learning.m";
    unlink(file_path);
    
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cp melvin.m %s", file_path);
    system(cmd);
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    printf("[RAW] Graph state:\n");
    printf("  learning_rate: %.6f\n", gh->learning_rate);
    printf("  reward_lambda: %.6f\n", gh->reward_lambda);
    
    // Create nodes
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 2);
        gh = file.graph_header;
    }
    
    uint64_t node1_idx = gh->num_nodes++;
    NodeDisk *node1 = &file.nodes[node1_idx];
    node1->id = 7000ULL;
    node1->state = 1.0f;
    node1->prediction = 0.5f;
    node1->prediction_error = 0.0f;
    
    uint64_t node2_idx = gh->num_nodes++;
    NodeDisk *node2 = &file.nodes[node2_idx];
    node2->id = 7001ULL;
    node2->state = 0.0f;
    node2->prediction = 0.0f;
    node2->prediction_error = 0.0f;
    
    // Create edge
    if (gh->num_edges >= gh->edge_capacity) {
        melvin_m_ensure_edge_capacity(&file, gh->num_edges + 1);
        gh = file.graph_header;
    }
    
    uint64_t edge_idx = gh->num_edges++;
    EdgeDisk *edge = &file.edges[edge_idx];
    edge->src = 7000ULL;
    edge->dst = 7001ULL;
    edge->weight = 0.5f;
    edge->eligibility = 0.0f;
    edge->trace = 0.0f;
    
    printf("\n[RAW] Initial state:\n");
    printf("  Node1: id=%llu, state=%.6f, prediction=%.6f, prediction_error=%.6f\n",
           (unsigned long long)node1->id, node1->state, node1->prediction, node1->prediction_error);
    printf("  Node2: id=%llu, state=%.6f, prediction=%.6f, prediction_error=%.6f\n",
           (unsigned long long)node2->id, node2->state, node2->prediction, node2->prediction_error);
    printf("  Edge: src=%llu, dst=%llu, weight=%.6f, eligibility=%.6f, trace=%.6f\n",
           (unsigned long long)edge->src, (unsigned long long)edge->dst,
           edge->weight, edge->eligibility, edge->trace);
    
    // Set prediction_error
    float target = 1.0f;
    melvin_set_epsilon_for_node(&rt, 7001ULL, target);
    
    printf("\n[RAW] After setting prediction_error:\n");
    printf("  target: %.6f\n", target);
    printf("  Node2 prediction: %.6f\n", node2->prediction);
    printf("  Node2 prediction_error: %.6f\n", node2->prediction_error);
    
    float weight_before = edge->weight;
    
    printf("\n[RAW] Processing 100 events, logging every 20:\n");
    for (int i = 0; i < 100; i++) {
        if (i % 20 == 0) {
            printf("  Event %d: weight=%.6f, eligibility=%.6f, trace=%.6f, node2_error=%.6f\n",
                   i, edge->weight, edge->eligibility, edge->trace, node2->prediction_error);
        }
        melvin_process_n_events(&rt, 1);
    }
    
    float weight_after = edge->weight;
    float weight_change = weight_after - weight_before;
    
    printf("\n[RAW] Final state:\n");
    printf("  weight_before: %.6f\n", weight_before);
    printf("  weight_after: %.6f\n", weight_after);
    printf("  weight_change: %.6f\n", weight_change);
    printf("  eligibility: %.6f\n", edge->eligibility);
    printf("  trace: %.6f\n", edge->trace);
    printf("  Node1 state: %.6f\n", node1->state);
    printf("  Node2 state: %.6f\n", node2->state);
    printf("  Node2 prediction_error: %.6f\n", node2->prediction_error);
    
    // Check learning parameters
    printf("\n[RAW] Learning parameters:\n");
    printf("  g_params.eligibility_decay: %.6f\n", g_params.eligibility_decay);
    printf("  g_params.elig_scale: %.6f\n", g_params.elig_scale);
    printf("  g_params.trace_strength: %.6f\n", g_params.trace_strength);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// HARD-5: Multi-Step Reasoning - RAW DATA
// ========================================================================

static void diagnose_hard_5_raw() {
    printf("\n");
    printf("========================================\n");
    printf("HARD-5: Multi-Step Reasoning - RAW DATA\n");
    printf("========================================\n\n");
    
    const char *file_path = "diag_hard5.m";
    unlink(file_path);
    
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cp melvin.m %s", file_path);
    system(cmd);
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    printf("[RAW] Initial graph: %llu nodes, %llu edges\n",
           (unsigned long long)gh->num_nodes, (unsigned long long)gh->num_edges);
    
    uint64_t node_a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t node_b_id = (uint64_t)'B' + 1000000ULL;
    uint64_t node_c_id = (uint64_t)'C' + 1000000ULL;
    uint64_t node_d_id = (uint64_t)'D' + 1000000ULL;
    
    printf("\n[RAW] Training on A->B->C->D (10 iterations, logging each):\n");
    for (int i = 0; i < 10; i++) {
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'D', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        // Check nodes and edges after each iteration
        uint64_t a_idx = find_node_index_by_id(&file, node_a_id);
        uint64_t b_idx = find_node_index_by_id(&file, node_b_id);
        uint64_t c_idx = find_node_index_by_id(&file, node_c_id);
        uint64_t d_idx = find_node_index_by_id(&file, node_d_id);
        
        uint64_t ab_edges = 0, bc_edges = 0, cd_edges = 0;
        float ab_weight = 0.0f, bc_weight = 0.0f, cd_weight = 0.0f;
        
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            EdgeDisk *edge = &file.edges[e];
            if (edge->src == node_a_id && edge->dst == node_b_id) {
                ab_edges++;
                ab_weight = edge->weight;
            }
            if (edge->src == node_b_id && edge->dst == node_c_id) {
                bc_edges++;
                bc_weight = edge->weight;
            }
            if (edge->src == node_c_id && edge->dst == node_d_id) {
                cd_edges++;
                cd_weight = edge->weight;
            }
        }
        
        printf("  Iter %d: nodes[A=%llu B=%llu C=%llu D=%llu] edges[AB=%llu(%.3f) BC=%llu(%.3f) CD=%llu(%.3f)]\n",
               i,
               (unsigned long long)a_idx, (unsigned long long)b_idx,
               (unsigned long long)c_idx, (unsigned long long)d_idx,
               (unsigned long long)ab_edges, ab_weight,
               (unsigned long long)bc_edges, bc_weight,
               (unsigned long long)cd_edges, cd_weight);
    }
    
    printf("\n[RAW] Final graph state:\n");
    printf("  Total nodes: %llu\n", (unsigned long long)gh->num_nodes);
    printf("  Total edges: %llu\n", (unsigned long long)gh->num_edges);
    
    // Final edge check
    uint64_t ab = 0, bc = 0, cd = 0;
    float ab_w = 0.0f, bc_w = 0.0f, cd_w = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &file.edges[i];
        if (e->src == node_a_id && e->dst == node_b_id) {
            ab++;
            ab_w = e->weight;
        }
        if (e->src == node_b_id && e->dst == node_c_id) {
            bc++;
            bc_w = e->weight;
        }
        if (e->src == node_c_id && e->dst == node_d_id) {
            cd++;
            cd_w = e->weight;
        }
    }
    
    printf("  A->B: %llu edge(s), weight=%.6f\n", (unsigned long long)ab, ab_w);
    printf("  B->C: %llu edge(s), weight=%.6f\n", (unsigned long long)bc, bc_w);
    printf("  C->D: %llu edge(s), weight=%.6f\n", (unsigned long long)cd, cd_w);
    printf("  Test requires: all edges exist AND weight > 0.3\n");
    printf("  Result: AB=%s BC=%s CD=%s\n",
           (ab > 0 && ab_w > 0.3f) ? "PASS" : "FAIL",
           (bc > 0 && bc_w > 0.3f) ? "PASS" : "FAIL",
           (cd > 0 && cd_w > 0.3f) ? "PASS" : "FAIL");
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// Main
// ========================================================================

int main() {
    printf("========================================\n");
    printf("RAW TEST DIAGNOSTICS\n");
    printf("Capturing actual state data from failing tests\n");
    printf("========================================\n");
    
    diagnose_test_0_2_2_raw();
    diagnose_test_0_5_1_raw();
    diagnose_hard_5_raw();
    
    printf("\n========================================\n");
    printf("END OF RAW DIAGNOSTICS\n");
    printf("========================================\n");
    
    return 0;
}


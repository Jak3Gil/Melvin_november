#define _POSIX_C_SOURCE 200809L

/*
 * analyze_failing_tests.c
 * 
 * Deep analysis of why the 10 failing tests aren't working:
 * - 0.2.2: Exec Subtracts Cost
 * - 0.5.1: Learning Only Prediction Error
 * - HARD-1, HARD-2, HARD-5, HARD-7, HARD-9, HARD-11, HARD-12
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "melvin.c"

// ========================================================================
// Test 0.2.2: Exec Subtracts Cost
// ========================================================================

static void analyze_test_0_2_2() {
    printf("\n=== ANALYZING: Test 0.2.2 (Exec Subtracts Cost) ===\n\n");
    
    const char *file_path = "analyze_exec_cost.m";
    unlink(file_path);
    
    // Copy melvin.m to test file
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cp melvin.m %s", file_path);
    if (system(cmd) != 0) {
        printf("[ERROR] Failed to copy melvin.m\n");
        return;
    }
    
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
    
    printf("[1] Runtime exec_cost: %.6f\n", rt.exec_cost);
    printf("[2] GPU cost multiplier: %.6f\n", rt.gpu_cost_multiplier);
    
    // Create EXECUTABLE node
    GraphHeaderDisk *gh = file.graph_header;
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 2000ULL;
    node->state = 2.0f;
    node->flags = NODE_FLAG_EXECUTABLE;
    
    // Write stub code
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
    
    printf("[3] Created executable node:\n");
    printf("    ID: %llu\n", (unsigned long long)node->id);
    printf("    State before: %.6f\n", node->state);
    printf("    Flags: 0x%x (EXECUTABLE=%d)\n", node->flags, 
           (node->flags & NODE_FLAG_EXECUTABLE) ? 1 : 0);
    
    // Check if node will execute (activation above threshold)
    float exec_center = g_params.exec_center;
    float exec_k = g_params.exec_k;
    float exec_factor = 1.0f / (1.0f + expf(-exec_k * (node->state - exec_center)));
    printf("[4] Execution probability:\n");
    printf("    exec_center: %.6f\n", exec_center);
    printf("    exec_k: %.6f\n", exec_k);
    printf("    exec_factor: %.6f\n", exec_factor);
    
    // Trigger execution
    printf("\n[5] Triggering execution...\n");
    float state_before = node->state;
    inject_pulse(&rt, 2000ULL, 2.0f);
    
    printf("    State after pulse: %.6f\n", node->state);
    
    // Process events
    printf("\n[6] Processing events...\n");
    uint64_t exec_calls_before = rt.exec_calls;
    melvin_process_n_events(&rt, 100);
    uint64_t exec_calls_after = rt.exec_calls;
    
    float state_after = node->state;
    float cost_applied = state_before - state_after;
    
    printf("\n[7] Results:\n");
    printf("    State before: %.6f\n", state_before);
    printf("    State after: %.6f\n", state_after);
    printf("    Cost applied: %.6f\n", cost_applied);
    printf("    Expected cost: %.6f\n", rt.exec_cost * rt.gpu_cost_multiplier);
    printf("    Exec calls: %llu -> %llu (delta: %llu)\n",
           (unsigned long long)exec_calls_before,
           (unsigned long long)exec_calls_after,
           (unsigned long long)(exec_calls_after - exec_calls_before));
    
    if (exec_calls_after == exec_calls_before) {
        printf("\n[PROBLEM] Node did NOT execute!\n");
        printf("    - Check if exec_factor is too low\n");
        printf("    - Check if node activation is above threshold\n");
        printf("    - Check if EXECUTABLE flag is set correctly\n");
    } else if (cost_applied <= 0.0f) {
        printf("\n[PROBLEM] Cost was NOT applied!\n");
        printf("    - Execution happened but cost wasn't subtracted\n");
    } else {
        printf("\n[SUCCESS] Cost was applied correctly!\n");
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// Test 0.5.1: Learning Only Prediction Error
// ========================================================================

static void analyze_test_0_5_1() {
    printf("\n=== ANALYZING: Test 0.5.1 (Learning Only Prediction Error) ===\n\n");
    
    const char *file_path = "analyze_learning.m";
    unlink(file_path);
    
    // Copy melvin.m to test file
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cp melvin.m %s", file_path);
    if (system(cmd) != 0) {
        printf("[ERROR] Failed to copy melvin.m\n");
        return;
    }
    
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
    
    GraphHeaderDisk *gh = file.graph_header;
    printf("[1] Learning rate: %.6f\n", gh->learning_rate);
    printf("[2] Reward lambda: %.6f\n", gh->reward_lambda);
    
    // Create two nodes with an edge
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 2);
        gh = file.graph_header;
    }
    
    uint64_t node1_idx = gh->num_nodes++;
    NodeDisk *node1 = &file.nodes[node1_idx];
    node1->id = 7000ULL;
    node1->state = 1.0f;
    node1->prediction = 0.5f;
    node1->prediction_error = 0.0f;  // NOT SET!
    
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
    
    printf("\n[3] Initial state:\n");
    printf("    Node1: state=%.3f, prediction=%.3f, prediction_error=%.3f\n",
           node1->state, node1->prediction, node1->prediction_error);
    printf("    Node2: state=%.3f, prediction=%.3f, prediction_error=%.3f\n",
           node2->state, node2->prediction, node2->prediction_error);
    printf("    Edge: weight=%.3f, eligibility=%.3f, trace=%.3f\n",
           edge->weight, edge->eligibility, edge->trace);
    
    printf("\n[4] PROBLEM: prediction_error is NOT set!\n");
    printf("    Test sets prediction=0.5 but never computes prediction_error\n");
    printf("    prediction_error should be: target - prediction\n");
    printf("    If target=1.0, then prediction_error = 1.0 - 0.5 = 0.5\n");
    
    // Set prediction_error manually
    printf("\n[5] Setting prediction_error manually...\n");
    float target = 1.0f;
    node2->prediction_error = target - node2->prediction;  // 1.0 - 0.0 = 1.0
    printf("    Target: %.3f\n", target);
    printf("    Prediction: %.3f\n", node2->prediction);
    printf("    prediction_error: %.3f\n", node2->prediction_error);
    
    float weight_before = edge->weight;
    
    printf("\n[6] Processing events (should trigger learning)...\n");
    melvin_process_n_events(&rt, 100);
    
    float weight_after = edge->weight;
    float weight_change = weight_after - weight_before;
    
    printf("\n[7] Results:\n");
    printf("    Weight before: %.6f\n", weight_before);
    printf("    Weight after: %.6f\n", weight_after);
    printf("    Weight change: %.6f\n", weight_change);
    printf("    Eligibility: %.6f\n", edge->eligibility);
    printf("    Trace: %.6f\n", edge->trace);
    
    if (fabsf(weight_change) < 0.0001f) {
        printf("\n[PROBLEM] No learning occurred!\n");
        printf("    Possible causes:\n");
        printf("    - Eligibility is 0 (need pre/post activation)\n");
        printf("    - Learning rate too low\n");
        printf("    - prediction_error not being used\n");
    } else {
        printf("\n[SUCCESS] Learning occurred!\n");
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// HARD Tests: Pattern Learning
// ========================================================================

static void analyze_hard_tests() {
    printf("\n=== ANALYZING: HARD Tests (Pattern Learning) ===\n\n");
    
    const char *file_path = "analyze_hard.m";
    unlink(file_path);
    
    // Copy melvin.m to test file
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cp melvin.m %s", file_path);
    if (system(cmd) != 0) {
        printf("[ERROR] Failed to copy melvin.m\n");
        return;
    }
    
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
    
    printf("[1] Testing byte ingestion...\n");
    
    // Ingest bytes A, B, C
    ingest_byte(&rt, 1ULL, 'A', 1.0f);
    melvin_process_n_events(&rt, 10);
    
    ingest_byte(&rt, 1ULL, 'B', 1.0f);
    melvin_process_n_events(&rt, 10);
    
    ingest_byte(&rt, 1ULL, 'C', 1.0f);
    melvin_process_n_events(&rt, 10);
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Find nodes for A, B, C
    uint64_t node_a = UINT64_MAX, node_b = UINT64_MAX, node_c = UINT64_MAX;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (file.nodes[i].id == (uint64_t)'A') node_a = i;
        if (file.nodes[i].id == (uint64_t)'B') node_b = i;
        if (file.nodes[i].id == (uint64_t)'C') node_c = i;
    }
    
    printf("\n[2] Node creation:\n");
    printf("    Node A: %s (idx=%llu)\n", 
           node_a != UINT64_MAX ? "found" : "missing",
           (unsigned long long)node_a);
    printf("    Node B: %s (idx=%llu)\n", 
           node_b != UINT64_MAX ? "found" : "missing",
           (unsigned long long)node_b);
    printf("    Node C: %s (idx=%llu)\n", 
           node_c != UINT64_MAX ? "found" : "missing",
           (unsigned long long)node_c);
    
    if (node_a == UINT64_MAX || node_b == UINT64_MAX || node_c == UINT64_MAX) {
        printf("\n[PROBLEM] Nodes not created by byte ingestion!\n");
        printf("    Check ingest_byte() function\n");
        return;
    }
    
    // Check edges
    uint64_t ab_edges = 0, bc_edges = 0;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &file.edges[i];
        if (e->src == (uint64_t)'A' && e->dst == (uint64_t)'B') ab_edges++;
        if (e->src == (uint64_t)'B' && e->dst == (uint64_t)'C') bc_edges++;
    }
    
    printf("\n[3] Edge creation:\n");
    printf("    A->B edges: %llu\n", (unsigned long long)ab_edges);
    printf("    B->C edges: %llu\n", (unsigned long long)bc_edges);
    
    if (ab_edges == 0 && bc_edges == 0) {
        printf("\n[PROBLEM] Edges not created!\n");
        printf("    - Co-activation edges should be created\n");
        printf("    - SEQ edges should be created by ingest_byte\n");
        printf("    - Check edge formation laws\n");
    }
    
    // Check prediction_error
    printf("\n[4] Prediction error:\n");
    printf("    Node A prediction_error: %.6f\n", file.nodes[node_a].prediction_error);
    printf("    Node B prediction_error: %.6f\n", file.nodes[node_b].prediction_error);
    printf("    Node C prediction_error: %.6f\n", file.nodes[node_c].prediction_error);
    
    if (file.nodes[node_a].prediction_error == 0.0f &&
        file.nodes[node_b].prediction_error == 0.0f &&
        file.nodes[node_c].prediction_error == 0.0f) {
        printf("\n[PROBLEM] prediction_error is 0 for all nodes!\n");
        printf("    - Tests don't set targets\n");
        printf("    - prediction_error needs to be computed from targets\n");
        printf("    - Or use trace-based learning (doesn't need epsilon)\n");
    }
    
    // Check if activation propagates
    printf("\n[5] Testing activation propagation...\n");
    file.nodes[node_a].state = 1.0f;
    melvin_process_n_events(&rt, 50);
    
    printf("    Node A state: %.6f\n", file.nodes[node_a].state);
    printf("    Node B state: %.6f\n", file.nodes[node_b].state);
    printf("    Node C state: %.6f\n", file.nodes[node_c].state);
    
    if (file.nodes[node_c].state < 0.1f) {
        printf("\n[PROBLEM] Activation did NOT propagate to C!\n");
        printf("    - Chain may be too weak\n");
        printf("    - Edges may not exist\n");
        printf("    - Weights may be too low\n");
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// Main
// ========================================================================

int main() {
    printf("=== DEEP ANALYSIS OF FAILING TESTS ===\n");
    
    analyze_test_0_2_2();
    analyze_test_0_5_1();
    analyze_hard_tests();
    
    printf("\n=== SUMMARY ===\n\n");
    printf("Key findings:\n");
    printf("1. Test 0.2.2: May not execute if exec_factor too low\n");
    printf("2. Test 0.5.1: prediction_error is NOT set by test\n");
    printf("3. HARD tests: prediction_error is 0, no targets set\n");
    printf("4. HARD tests: May rely on trace-based learning instead\n");
    printf("\nRecommendations:\n");
    printf("- Tests should set prediction_error explicitly\n");
    printf("- Or use trace-based learning (doesn't need epsilon)\n");
    printf("- Or compute epsilon from energy efficiency (lower energy = reward)\n");
    
    return 0;
}


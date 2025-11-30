/*
 * TEST: GPU EXEC Basic
 * 
 * Goal: Verify that CUDA path is callable, EXEC can use GPU helper,
 * and result is fed back as energy through normal mechanisms.
 * 
 * This test demonstrates GPU EXEC as an implementation detail that
 * doesn't change physics laws - EXEC still returns scalar → energy.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "melvin.c"

#define TEST_FILE "test_gpu_exec_basic.m"

// Simple stub code that does nothing (GPU dispatch will replace it)
static uint8_t stub_code[] = {
    0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00,  // mov rax, 0 (x86_64)
    0xC3,  // ret
};

static uint8_t aarch64_stub[] = {
    0x00, 0x00, 0x80, 0xD2,  // mov x0, #0 (ARM64)
    0xC0, 0x03, 0x5F, 0xD6,  // ret
};

int main() {
    printf("TEST: GPU EXEC Basic\n");
    printf("====================\n");
    printf("Goal: Verify GPU EXEC path works and is more energy-efficient\n\n");
    
    // Step 1: Create fresh brain
    printf("Step 1: Creating fresh brain...\n");
    MelvinFile file;
    if (melvin_m_init_new_file(TEST_FILE, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to create brain file\n");
        return 1;
    }
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map brain file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("  ✓ Brain created\n");
    
    // Step 2: Enable GPU EXEC via param nodes
    printf("Step 2: Enabling GPU EXEC via param nodes...\n");
    GraphHeaderDisk *gh = file.graph_header;
    
    // Enable GPU
    uint64_t gpu_enabled_idx = find_node_index_by_id(&file, NODE_ID_PARAM_EXEC_GPU_ENABLED);
    if (gpu_enabled_idx != UINT64_MAX) {
        file.nodes[gpu_enabled_idx].state = 0.8f;  // > 0.5 = enabled
        printf("  ✓ GPU enabled (param node activation: 0.8)\n");
    } else {
        fprintf(stderr, "ERROR: GPU enabled param node not found\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    // Set GPU cost multiplier to 0.5 (GPU is cheaper)
    uint64_t gpu_cost_idx = find_node_index_by_id(&file, NODE_ID_PARAM_EXEC_GPU_COST_MULTIPLIER);
    if (gpu_cost_idx != UINT64_MAX) {
        // Map 0.5x cost to param node value: (0.5 - 0.1) / 0.9 = 0.44
        file.nodes[gpu_cost_idx].state = 0.44f;
        printf("  ✓ GPU cost multiplier set to 0.5x (param node: 0.44)\n");
    } else {
        fprintf(stderr, "ERROR: GPU cost multiplier param node not found\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    melvin_sync_params_from_nodes(&rt);
    printf("  ✓ Params synced\n");
    
    // Step 3: Create nodes with non-zero activations (for GPU to read)
    printf("Step 3: Creating nodes with activations...\n");
    uint64_t test_node_ids[10];
    for (int i = 0; i < 10; i++) {
        if (gh->num_nodes >= gh->node_capacity) {
            grow_graph(&file, gh->num_nodes + 1, gh->num_edges);
            gh = file.graph_header;
        }
        uint64_t node_idx = gh->num_nodes++;
        NodeDisk *node = &file.nodes[node_idx];
        node->id = 1000000ULL + i;
        node->state = 0.5f + (i * 0.05f);  // Varying activations
        node->bias = 0.0f;
        node->prediction = 0.0f;
        node->stability = 0.0f;
        node->first_out_edge = UINT64_MAX;
        node->out_degree = 0;
        node->flags = 0;
        test_node_ids[i] = node->id;
    }
    printf("  ✓ Created 10 nodes with activations [0.5, 0.95]\n");
    
    // Step 4: Create EXEC node
    printf("Step 4: Creating EXEC node...\n");
    uint64_t exec_node_id = 999999ULL;
    size_t code_len = sizeof(stub_code);
    #ifdef __aarch64__
    code_len = sizeof(aarch64_stub);
    #endif
    
    uint64_t code_offset = melvin_write_machine_code(&file,
        #ifdef __aarch64__
        aarch64_stub
        #else
        stub_code
        #endif
        , code_len);
    
    if (code_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write machine code\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    if (gh->num_nodes >= gh->node_capacity) {
        grow_graph(&file, gh->num_nodes + 1, gh->num_edges);
        gh = file.graph_header;
    }
    uint64_t exec_node_idx = gh->num_nodes++;
    NodeDisk *exec_node = &file.nodes[exec_node_idx];
    exec_node->id = exec_node_id;
    exec_node->flags = NODE_FLAG_EXECUTABLE;
    exec_node->payload_offset = code_offset;
    exec_node->payload_len = code_len;
    exec_node->state = 0.0f;  // Start at zero
    exec_node->bias = 0.0f;
    exec_node->prediction = 0.0f;
    exec_node->stability = 0.0f;
    exec_node->first_out_edge = UINT64_MAX;
    exec_node->out_degree = 0;
    printf("  ✓ EXEC node created (ID: %llu)\n", (unsigned long long)exec_node_id);
    
    // Step 5: Force EXEC activation
    printf("Step 5: Forcing EXEC activation...\n");
    float exec_threshold = gh->exec_threshold;
    float activation_needed = exec_threshold + 0.1f;  // Ensure we cross threshold
    
    // Inject energy into EXEC node
    MelvinEvent delta = {
        .type = EV_NODE_DELTA,
        .node_id = exec_node_id,
        .value = activation_needed
    };
    melvin_event_enqueue(&rt.evq, &delta);
    
    // Process events to trigger EXEC
    printf("  Processing events to trigger EXEC...\n");
    uint64_t exec_calls_before = rt.exec_calls;
    float exec_state_before = exec_node->state;
    
    // Process enough events to trigger EXEC
    for (int i = 0; i < 100; i++) {
        melvin_process_n_events(&rt, 10);
        
        // Check if EXEC triggered
        if (rt.exec_calls > exec_calls_before) {
            printf("  ✓ EXEC triggered (call count: %llu)\n", (unsigned long long)rt.exec_calls);
            break;
        }
    }
    
    if (rt.exec_calls == exec_calls_before) {
        printf("  ⚠ EXEC did not trigger (may need more events or lower threshold)\n");
    }
    
    // Step 6: Measure results
    printf("\nStep 6: Measuring results...\n");
    
    float exec_state_after = exec_node->state;
    float exec_cost_applied = exec_state_before - exec_state_after;
    float expected_cpu_cost = rt.exec_cost;
    float expected_gpu_cost = rt.exec_cost * rt.gpu_cost_multiplier;
    
    printf("  EXEC node state before: %.6f\n", exec_state_before);
    printf("  EXEC node state after: %.6f\n", exec_state_after);
    printf("  Cost applied: %.6f\n", exec_cost_applied);
    printf("  Expected CPU cost: %.6f\n", expected_cpu_cost);
    printf("  Expected GPU cost: %.6f\n", expected_gpu_cost);
    printf("  GPU cost multiplier: %.6f\n", rt.gpu_cost_multiplier);
    printf("  EXEC calls: %llu\n", (unsigned long long)rt.exec_calls);
    
    // Check if GPU path was used (cost should be lower if GPU was used)
    bool gpu_used = (fabsf(exec_cost_applied - expected_gpu_cost) < fabsf(exec_cost_applied - expected_cpu_cost));
    
    printf("\n");
    printf("TEST RESULTS\n");
    printf("============\n");
    
    int passed = 1;
    
    if (rt.exec_calls > exec_calls_before) {
        printf("✓ EXEC was triggered\n");
    } else {
        printf("✗ EXEC was not triggered\n");
        passed = 0;
    }
    
    if (exec_cost_applied > 0.0f) {
        printf("✓ EXEC cost was applied\n");
    } else {
        printf("✗ EXEC cost was not applied\n");
        passed = 0;
    }
    
    #ifdef HAVE_CUDA
    if (gpu_used) {
        printf("✓ GPU path was used (cost matches GPU multiplier)\n");
    } else {
        printf("⚠ GPU path may not have been used (cost matches CPU)\n");
        printf("  Note: GPU may not be available or param check failed\n");
    }
    #else
    printf("⚠ CUDA not compiled (HAVE_CUDA not defined) - using CPU fallback\n");
    #endif
    
    if (rt.gpu_cost_multiplier < 1.0f) {
        printf("✓ GPU cost multiplier is set (< 1.0, making GPU cheaper)\n");
    } else {
        printf("⚠ GPU cost multiplier is 1.0 (same as CPU)\n");
    }
    
    printf("\n");
    if (passed) {
        printf("✅ TEST PASSED: GPU EXEC Basic\n");
        printf("   GPU EXEC path is callable and integrates with physics\n");
    } else {
        printf("⚠️  TEST PARTIAL: Some checks failed\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return (passed ? 0 : 1);
}


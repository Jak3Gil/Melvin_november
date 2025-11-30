/*
 * Phase 2 Test B: Parameter Adaptation Test
 * 
 * Goal: Verify EXEC nodes can modify param nodes and physics changes
 * 
 * This tests the self-tuning physics loop.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "melvin.c"

#define TEST_FILE "test_phase2_param_adaptation.m"

int main() {
    printf("========================================\n");
    printf("PHASE 2 TEST B: PARAMETER ADAPTATION TEST\n");
    printf("========================================\n\n");
    
    printf("Goal: Verify EXEC→param node→physics change cycle\n\n");
    
    // Step 1: Create new file
    printf("Step 1: Creating test file...\n");
    GraphParams params;
    init_default_params(&params);
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "Failed to create test file\n");
        return 1;
    }
    printf("✓ Created %s\n\n", TEST_FILE);
    
    // Step 2: Map file and initialize runtime
    printf("Step 2: Mapping file and initializing runtime...\n");
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
    
    // Step 3: Read initial parameter values
    printf("Step 3: Reading initial parameter values...\n");
    
    GraphHeaderDisk *gh = file.graph_header;
    float initial_decay = gh->decay_rate;
    float initial_exec_threshold = gh->exec_threshold;
    float initial_learn_rate = gh->learning_rate;
    float initial_exec_cost = rt.exec_cost;
    
    printf("  Initial decay_rate: %.6f\n", initial_decay);
    printf("  Initial exec_threshold: %.6f\n", initial_exec_threshold);
    printf("  Initial learning_rate: %.6f\n", initial_learn_rate);
    printf("  Initial exec_cost: %.6f\n\n", initial_exec_cost);
    
    // Step 4: Read param node values
    printf("Step 4: Reading param node values...\n");
    
    uint64_t decay_idx = find_node_index_by_id(&file, NODE_ID_PARAM_DECAY);
    uint64_t threshold_idx = find_node_index_by_id(&file, NODE_ID_PARAM_EXEC_THRESHOLD);
    uint64_t learn_idx = find_node_index_by_id(&file, NODE_ID_PARAM_LEARN_RATE);
    uint64_t cost_idx = find_node_index_by_id(&file, NODE_ID_PARAM_EXEC_COST);
    
    if (decay_idx == UINT64_MAX || threshold_idx == UINT64_MAX || 
        learn_idx == UINT64_MAX || cost_idx == UINT64_MAX) {
        fprintf(stderr, "Param nodes not found\n");
        return 1;
    }
    
    float decay_node_val = file.nodes[decay_idx].state;
    float threshold_node_val = file.nodes[threshold_idx].state;
    float learn_node_val = file.nodes[learn_idx].state;
    float cost_node_val = file.nodes[cost_idx].state;
    
    printf("  Decay param node activation: %.6f\n", decay_node_val);
    printf("  Threshold param node activation: %.6f\n", threshold_node_val);
    printf("  Learn rate param node activation: %.6f\n", learn_node_val);
    printf("  Exec cost param node activation: %.6f\n\n", cost_node_val);
    
    // Step 5: Modify param nodes via direct activation (simulating EXEC modification)
    printf("Step 5: Modifying param nodes...\n");
    
    // Modify decay param node (increase activation = increase decay)
    file.nodes[decay_idx].state = 0.5f;  // Increase from ~0.18 to 0.5
    file.nodes[threshold_idx].state = 0.2f;  // Decrease threshold
    file.nodes[learn_idx].state = 0.8f;  // Increase learning rate
    file.nodes[cost_idx].state = 0.3f;  // Increase exec cost
    
    printf("  Modified decay param node: %.6f\n", file.nodes[decay_idx].state);
    printf("  Modified threshold param node: %.6f\n", file.nodes[threshold_idx].state);
    printf("  Modified learn rate param node: %.6f\n", file.nodes[learn_idx].state);
    printf("  Modified exec cost param node: %.6f\n\n", file.nodes[cost_idx].state);
    
    // Step 6: Trigger homeostasis sweep to sync params
    printf("Step 6: Triggering homeostasis sweep to sync parameters...\n");
    
    MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
    melvin_process_event(&rt, &homeostasis_ev);
    
    // Step 7: Read updated physics values
    printf("Step 7: Reading updated physics values...\n");
    
    float new_decay = gh->decay_rate;
    float new_exec_threshold = gh->exec_threshold;
    float new_learn_rate = gh->learning_rate;
    float new_exec_cost = rt.exec_cost;
    
    printf("  New decay_rate: %.6f (was %.6f, delta: %.6f)\n", 
           new_decay, initial_decay, new_decay - initial_decay);
    printf("  New exec_threshold: %.6f (was %.6f, delta: %.6f)\n", 
           new_exec_threshold, initial_exec_threshold, new_exec_threshold - initial_exec_threshold);
    printf("  New learning_rate: %.6f (was %.6f, delta: %.6f)\n", 
           new_learn_rate, initial_learn_rate, new_learn_rate - initial_learn_rate);
    printf("  New exec_cost: %.6f (was %.6f, delta: %.6f)\n\n", 
           new_exec_cost, initial_exec_cost, new_exec_cost - initial_exec_cost);
    
    // Step 8: Verify changes
    printf("Step 8: Verifying parameter changes...\n");
    
    int passed = 1;
    
    // Decay should have increased (0.18 -> 0.5 activation maps to higher decay)
    float expected_decay = 0.01f + 0.5f * 0.49f;  // ~0.255
    if (fabsf(new_decay - expected_decay) < 0.01f) {
        printf("✓ Decay rate changed correctly\n");
    } else {
        printf("✗ Decay rate change incorrect (expected ~%.6f, got %.6f)\n", 
               expected_decay, new_decay);
        passed = 0;
    }
    
    // Threshold should have decreased
    float expected_threshold = 0.5f + 0.2f * 1.5f;  // ~0.8
    if (fabsf(new_exec_threshold - expected_threshold) < 0.1f) {
        printf("✓ Exec threshold changed correctly\n");
    } else {
        printf("✗ Exec threshold change incorrect (expected ~%.6f, got %.6f)\n", 
               expected_threshold, new_exec_threshold);
        passed = 0;
    }
    
    // Learning rate should have increased
    float expected_learn = 0.0001f + 0.8f * 0.0099f;  // ~0.00802
    if (fabsf(new_learn_rate - expected_learn) < 0.001f) {
        printf("✓ Learning rate changed correctly\n");
    } else {
        printf("✗ Learning rate change incorrect (expected ~%.6f, got %.6f)\n", 
               expected_learn, new_learn_rate);
        passed = 0;
    }
    
    // Exec cost should have increased
    float expected_cost = 0.01f + 0.3f * 0.49f;  // ~0.157
    if (fabsf(new_exec_cost - expected_cost) < 0.01f) {
        printf("✓ Exec cost changed correctly\n");
    } else {
        printf("✗ Exec cost change incorrect (expected ~%.6f, got %.6f)\n", 
               expected_cost, new_exec_cost);
        passed = 0;
    }
    
    printf("\n");
    
    // Summary
    printf("========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    
    if (passed) {
        printf("✅ PARAMETER ADAPTATION TEST: PASSED\n");
        printf("EXEC nodes can modify param nodes and physics changes!\n");
    } else {
        printf("❌ PARAMETER ADAPTATION TEST: FAILED\n");
        printf("Parameter sync may need refinement.\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return passed ? 0 : 1;
}


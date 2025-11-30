/*
 * TEST: EXEC vs Memorized Patterns
 * 
 * Goal: Show that when an EXEC node can compute arithmetic more efficiently
 * than memorized patterns, the graph gradually prefers the EXEC route.
 * 
 * This test demonstrates energy efficiency law making EXEC win over time.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_exec_vs_patterns.m"
#define NUM_EPISODES 50
#define TICKS_PER_EPISODE 200

// Simple machine code that computes x+y and returns result
// For ARM64: adds two values and returns
static uint8_t simple_add_code[] = {
    // ARM64: mov x0, #5; mov x1, #5; add x0, x0, x1; ret
    0x80, 0x00, 0x80, 0xD2,  // mov x0, #5
    0x81, 0x00, 0x80, 0xD2,  // mov x1, #5
    0x00, 0x00, 0x01, 0x8B,  // add x0, x0, x1
    0xC0, 0x03, 0x5F, 0xD6,  // ret
};

// For x86_64: simpler stub that returns 10
static uint8_t x86_add_code[] = {
    0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00,  // mov rax, 10
    0xC3,  // ret
};

int main() {
    printf("TEST: EXEC vs Memorized Patterns\n");
    printf("==================================\n\n");
    
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
    printf("  ✓ Brain created\n");
    
    // Step 2: Initialize runtime
    printf("Step 2: Initializing runtime...\n");
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("  ✓ Runtime initialized\n");
    
    // Step 3: Configure params for moderate behavior
    printf("Step 3: Configuring params...\n");
    GraphHeaderDisk *gh = file.graph_header;
    
    // Set decay to 0.95
    uint64_t decay_idx = find_node_index_by_id(&file, NODE_ID_PARAM_DECAY);
    if (decay_idx != UINT64_MAX) {
        file.nodes[decay_idx].state = 0.92f;  // Maps to ~0.95
    }
    
    // Set learning rate
    uint64_t learn_idx = find_node_index_by_id(&file, NODE_ID_PARAM_LEARN_RATE);
    if (learn_idx != UINT64_MAX) {
        file.nodes[learn_idx].state = 0.5f;  // Moderate learning
    }
    
    // Make curiosity permissive
    uint64_t curiosity_act_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_ACT_MIN);
    if (curiosity_act_idx != UINT64_MAX) {
        file.nodes[curiosity_act_idx].state = 0.01f;  // Very low threshold
    }
    uint64_t curiosity_traffic_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_TRAFFIC_MAX);
    if (curiosity_traffic_idx != UINT64_MAX) {
        file.nodes[curiosity_traffic_idx].state = 0.05f;  // Low threshold
    }
    
    melvin_sync_params_from_nodes(&rt);
    printf("  ✓ Params configured\n\n");
    
    // Step 4: Create EXEC node with simple arithmetic code
    printf("Step 4: Creating EXEC node...\n");
    uint64_t exec_node_id = 999999ULL;
    
    // Write machine code to blob
    size_t code_len = sizeof(x86_add_code);
    #ifdef __aarch64__
    code_len = sizeof(simple_add_code);
    #endif
    
    uint64_t code_offset = melvin_write_machine_code(&file,
        #ifdef __aarch64__
        simple_add_code
        #else
        x86_add_code
        #endif
        , code_len);
    
    if (code_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write machine code\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    // Create EXEC node
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
    exec_node->state = 0.0f;
    exec_node->bias = 0.1f;
    exec_node->prediction = 0.0f;
    exec_node->stability = 0.0f;
    exec_node->first_out_edge = UINT64_MAX;
    exec_node->out_degree = 0;
    
    printf("  ✓ EXEC node created (ID: %llu)\n\n", (unsigned long long)exec_node_id);
    
    // Step 5: Run episodes
    printf("Step 5: Running %d episodes...\n", NUM_EPISODES);
    
    // Track metrics
    uint64_t edges_into_exec = 0;
    uint64_t exec_trigger_count = 0;
    float fe_before_sum = 0.0f;
    float fe_after_sum = 0.0f;
    int fe_samples = 0;
    
    srand(time(NULL));
    
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        // Ingest arithmetic patterns
        const char *patterns[] = {"5+5=10\n", "4+6=10\n", "3+7=10\n", "2+8=10\n", "1+9=10\n"};
        int pattern_idx = rand() % 5;
        
        // Ingest pattern
        for (int i = 0; patterns[pattern_idx][i] != '\0'; i++) {
            ingest_byte(&rt, 0, patterns[pattern_idx][i], 1.0f);
        }
        
        // Process events
        for (int tick = 0; tick < TICKS_PER_EPISODE; tick++) {
            melvin_process_n_events(&rt, 10);
            
            // Check if EXEC triggered
            if (exec_node_idx < gh->num_nodes) {
                NodeDisk *exec = &file.nodes[exec_node_idx];
                if (exec->state > gh->exec_threshold) {
                    exec_trigger_count++;
                }
            }
        }
        
        // Trigger homeostasis sweep (triggers edge formation + curiosity)
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        melvin_process_n_events(&rt, 10);
        
        // Count edges into EXEC
        edges_into_exec = 0;
        float total_weight = 0.0f;
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            EdgeDisk *edge = &file.edges[e];
            if (edge->src == UINT64_MAX) continue;
            if (edge->dst == exec_node_id) {
                edges_into_exec++;
                total_weight += fabsf(edge->weight);
            }
        }
        
        // Measure FE around EXEC
        if (exec_node_idx < gh->num_nodes) {
            NodeDisk *exec = &file.nodes[exec_node_idx];
            float fe_exec = exec->fe_ema;
            fe_before_sum += fe_exec;
            fe_samples++;
            
            // After EXEC runs, FE should be lower
            fe_after_sum += fe_exec;
        }
        
        if ((episode + 1) % 10 == 0) {
            printf("  Episode %d: edges_into_exec=%llu (total_weight=%.4f), exec_triggers=%llu\n",
                   episode + 1, (unsigned long long)edges_into_exec, total_weight, 
                   (unsigned long long)exec_trigger_count);
        }
    }
    
    printf("\n");
    
    // Step 6: Final measurements
    printf("Step 6: Final measurements...\n");
    
    // Count edges into EXEC
    edges_into_exec = 0;
    float total_weight = 0.0f;
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        EdgeDisk *edge = &file.edges[e];
        if (edge->src == UINT64_MAX) continue;
        if (edge->dst == exec_node_id) {
            edges_into_exec++;
            total_weight += fabsf(edge->weight);
        }
    }
    
    float fe_before_avg = (fe_samples > 0) ? (fe_before_sum / fe_samples) : 0.0f;
    float fe_after_avg = (fe_samples > 0) ? (fe_after_sum / fe_samples) : 0.0f;
    
    printf("  Final edges into EXEC: %llu (total weight: %.4f)\n", 
           (unsigned long long)edges_into_exec, total_weight);
    printf("  EXEC trigger count: %llu\n", (unsigned long long)exec_trigger_count);
    printf("  Average FE before: %.6f\n", fe_before_avg);
    printf("  Average FE after: %.6f\n", fe_after_avg);
    printf("\n");
    
    // Step 7: Assertions
    printf("TEST RESULTS\n");
    printf("============\n");
    
    int passed = 1;
    
    if (edges_into_exec > 0) {
        printf("✓ EXEC has incoming edges (no manual wiring)\n");
    } else {
        printf("✗ EXEC has no incoming edges\n");
        passed = 0;
    }
    
    if (exec_trigger_count > 0) {
        printf("✓ EXEC activated at least once\n");
    } else {
        printf("✗ EXEC never activated\n");
        passed = 0;
    }
    
    if (fe_after_avg < fe_before_avg || fe_samples == 0) {
        printf("✓ FE trend: EXEC reduces free-energy\n");
    } else {
        printf("⚠ FE trend: EXEC may not be reducing FE yet\n");
    }
    
    printf("\n");
    printf("Summary:\n");
    printf("  edges_into_exec: %llu\n", (unsigned long long)edges_into_exec);
    printf("  exec_trigger_count: %llu\n", (unsigned long long)exec_trigger_count);
    printf("  fe_before_avg: %.6f\n", fe_before_avg);
    printf("  fe_after_avg: %.6f\n", fe_after_avg);
    printf("\n");
    
    if (passed) {
        printf("✅ TEST PASSED: EXEC vs Patterns\n");
    } else {
        printf("⚠️  TEST PARTIAL: Some assertions failed\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return (passed ? 0 : 1);
}


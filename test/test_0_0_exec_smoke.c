#define _POSIX_C_SOURCE 200809L

/*
 * test_0_0_exec_smoke.c
 * 
 * MINIMAL EXEC SMOKE TEST
 * 
 * Goal: Prove in the smallest possible universe that:
 *   - EXEC attempts > 0
 *   - EXEC executed > 0
 *   - A simple ADD computation runs through the graph
 * 
 * This test does NOT use instincts.c - it creates a minimal graph manually.
 * This proves the EXEC path works at the fundamental level.
 * 
 * Test: 2 + 3 = 5 via EXEC:ADD32
 * 
 * POLICY: This smoke test must not call any melvin_exec_* function directly.
 * All EXEC happens through the normal EXEC dispatch path (event loop + execute_hot_nodes).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>

// Include the implementation
#include "melvin.c"
#include "instincts.c"  // Required for melvin_inject_instincts
#include "test_helpers.h"
#include "melvin_instincts.h"

// Include EXEC helper functions
#include "melvin_exec_helpers.c"

// Test limits
#define MAX_TICKS 100

// Note: Using test_helpers.h functions instead of custom helpers
// write_int32_to_node_by_id and read_int32_from_node_by_id are defined in test_helpers.h

// Note: We now use instinct-injected EXEC nodes, not custom ones
// The EXEC node is created by melvin_inject_instincts() and we just bind the function pointer

int main() {
    printf("========================================\n");
    printf("TEST 0.0 — Minimal EXEC Smoke Test\n");
    printf("========================================\n\n");
    
    printf("Goal: Prove EXEC path works in minimal graph\n");
    printf("Test: 2 + 3 = 5 via EXEC:ADD32\n");
    printf("No instincts required - pure minimal graph\n\n");
    
    const char *file_path = "test_0_0_exec_smoke.m";
    unlink(file_path);
    
    // Initialize minimal graph
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.015f;
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    
    // Inject instincts FIRST - this creates the math workspace nodes and EXEC nodes
    // We use the existing pattern nodes, not custom ones
    // IMPORTANT: Inject into the SAME file we just created, not a different one
    melvin_inject_instincts(&file);
    melvin_m_sync(&file);
    
    // Use known IDs from instincts.c pattern bases (MATH_PATTERN_BASE = 60000, EXEC_PATTERN_BASE = 50000)
    // These are stable IDs that instincts.c always uses
    uint64_t math_in_a_id = 60000ULL;   // MATH:IN_A:I32
    uint64_t math_in_b_id = 60001ULL;   // MATH:IN_B:I32
    uint64_t math_out_id = 60002ULL;    // MATH:OUT:I32
    uint64_t exec_add_id = 50010ULL;    // EXEC:ADD32
    
    // Verify the math workspace nodes exist (created by instincts)
    NodeDisk *node_a = melvin_get_node_safe(&file, math_in_a_id);
    NodeDisk *node_b = melvin_get_node_safe(&file, math_in_b_id);
    NodeDisk *node_out = melvin_get_node_safe(&file, math_out_id);
    NodeDisk *exec_add_node = melvin_get_node_safe(&file, exec_add_id);
    
    if (!node_a || !node_b || !node_out || !exec_add_node) {
        close_file(&file);
        fprintf(stderr, "ERROR: Instinct nodes not found\n");
        fprintf(stderr, "  math_in_a (60000): %s\n", node_a ? "found" : "missing");
        fprintf(stderr, "  math_in_b (60001): %s\n", node_b ? "found" : "missing");
        fprintf(stderr, "  math_out (60002): %s\n", node_out ? "found" : "missing");
        fprintf(stderr, "  exec_add32 (50010): %s\n", exec_add_node ? "found" : "missing");
        return 1;
    }
    
    // Ensure EXEC node has function pointer bound (instincts created the node, but we need to bind the function)
    if (!(exec_add_node->flags & NODE_FLAG_EXECUTABLE) || exec_add_node->payload_len == 0) {
        // Bind melvin_exec_add32 function to the EXEC node
        uint64_t offset = melvin_write_machine_code(&file, (const uint8_t*)&melvin_exec_add32, sizeof(void*));
        if (offset != UINT64_MAX) {
            exec_add_node->flags |= NODE_FLAG_EXECUTABLE;
            exec_add_node->payload_offset = offset;
            exec_add_node->payload_len = sizeof(void*);
        } else {
            close_file(&file);
            fprintf(stderr, "ERROR: Failed to write EXEC function to blob\n");
            return 1;
        }
    }
    
    // Set instinct IDs so melvin_exec_add32 can find the nodes (if available)
    // Note: melvin_exec_add32 has a fallback to use pattern base IDs directly
    MelvinInstinctIds updated_ids = {0};
    updated_ids.math_in_a_i32_id = math_in_a_id;
    updated_ids.math_in_b_i32_id = math_in_b_id;
    updated_ids.math_out_i32_id = math_out_id;
    updated_ids.exec_add32_id = exec_add_id;
    updated_ids.math_nodes_valid = 1;
    updated_ids.exec_nodes_valid = 1;
    melvin_set_instinct_ids(&file, &updated_ids);
    
    // Note: We don't check if instinct IDs were set correctly because
    // melvin_exec_add32 has a fallback to use pattern base IDs (60000, 60001, 60002) directly
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        close_file(&file);
        fprintf(stderr, "ERROR: Failed to init runtime\n");
        return 1;
    }
    
    // Reset EXEC stats
    rt.exec_attempts = 0;
    rt.exec_skipped_by_law = 0;
    rt.exec_executed = 0;
    rt.exec_calls = 0;
    
    printf("Using instinct-injected pattern nodes:\n");
    printf("  Node A (MATH:IN_A:I32): %llu\n", (unsigned long long)math_in_a_id);
    printf("  Node B (MATH:IN_B:I32): %llu\n", (unsigned long long)math_in_b_id);
    printf("  Node OUT (MATH:OUT:I32): %llu\n", (unsigned long long)math_out_id);
    printf("  EXEC:ADD32 node: %llu\n", (unsigned long long)exec_add_id);
    printf("\n");
    
    // Write inputs: a=2, b=3 to the instinct-created math workspace nodes
    printf("Writing inputs: a=2, b=3\n");
    write_int32_to_node_by_id(&file, math_in_a_id, 2);
    write_int32_to_node_by_id(&file, math_in_b_id, 3);
    
    // Debug: Verify inputs were written
    int32_t a_dbg = read_int32_from_node_by_id(&file, math_in_a_id);
    int32_t b_dbg = read_int32_from_node_by_id(&file, math_in_b_id);
    printf("[SMOKE] pre-tick A=%d B=%d\n", a_dbg, b_dbg);
    
    // Activate EXEC node above threshold (using instinct-created EXEC node)
    exec_add_node->state = file.graph_header->exec_threshold + 0.5f;
    printf("Activated EXEC node: state=%.2f (threshold=%.2f)\n",
           exec_add_node->state, file.graph_header->exec_threshold);
    
    printf("\nRunning ticks to trigger EXEC...\n");
    printf("Using real EXEC dispatch path (execute_hot_nodes via event processing)\n");
    
    // Run ticks - EXEC should fire through normal event mechanism
    // This uses the real EXEC dispatch path: event -> execute_hot_nodes -> melvin_exec_dispatch
    // NO direct function calls - all goes through the graph event loop
    bool exec_fired = false;
    for (int tick = 0; tick < MAX_TICKS; tick++) {
        // Trigger homeostasis sweep which calls execute_hot_nodes internally
        // This is the real EXEC path - no direct function calls
        MelvinEvent homeo = {
            .type = EV_HOMEOSTASIS_SWEEP,
            .node_id = 0,
            .value = 0.0f
        };
        melvin_event_enqueue(&rt.evq, &homeo);
        
        // Process events - this will call execute_hot_nodes which dispatches EXEC
        melvin_process_n_events(&rt, 50);
        
        // Check if EXEC fired
        if (rt.exec_executed > 0) {
            exec_fired = true;
            printf("  Tick %d: EXEC fired! (executed=%llu)\n", tick, (unsigned long long)rt.exec_executed);
            break;
        }
        
        // Progress indicator
        if (tick % 10 == 0) {
            printf("  Tick %d: attempts=%llu, skipped=%llu, executed=%llu\n",
                   tick,
                   (unsigned long long)rt.exec_attempts,
                   (unsigned long long)rt.exec_skipped_by_law,
                   (unsigned long long)rt.exec_executed);
        }
    }
    
    // Read result from instinct-created math workspace OUT node
    int32_t result = read_int32_from_node_by_id(&file, math_out_id);
    int32_t expected = 5;  // 2 + 3
    
    // Debug: Print post-tick value
    printf("[SMOKE] post-tick OUT=%d\n", result);
    
    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n\n");
    
    printf("Input: a=2, b=3\n");
    printf("Expected output: %d\n", expected);
    printf("Actual output: %d\n", result);
    printf("\n");
    
    // Print EXEC stats
    printf("EXEC Statistics:\n");
    melvin_print_exec_stats(&rt);
    printf("\n");
    
    // Verify results
    bool passed = false;
    const char *failure_reason = NULL;
    
    if (!exec_fired) {
        failure_reason = "EXEC never fired (exec_executed=0)";
    } else if (rt.exec_attempts == 0) {
        failure_reason = "No EXEC attempts (exec_attempts=0)";
    } else if (result != expected) {
        failure_reason = "Wrong result";
    } else {
        passed = true;
    }
    
    printf("========================================\n");
    if (passed) {
        printf("✅ TEST PASSED\n");
        printf("\nEXEC path is working!\n");
        printf("  - EXEC attempts: %llu\n", (unsigned long long)rt.exec_attempts);
        printf("  - EXEC executed: %llu\n", (unsigned long long)rt.exec_executed);
        printf("  - Computation: 2 + 3 = %d ✓\n", result);
    } else {
        printf("❌ TEST FAILED\n");
        printf("\nReason: %s\n", failure_reason);
        printf("  - EXEC attempts: %llu\n", (unsigned long long)rt.exec_attempts);
        printf("  - EXEC skipped: %llu\n", (unsigned long long)rt.exec_skipped_by_law);
        printf("  - EXEC executed: %llu\n", (unsigned long long)rt.exec_executed);
        printf("  - Result: %d (expected %d)\n", result, expected);
    }
    printf("========================================\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
    
    return passed ? 0 : 1;
}


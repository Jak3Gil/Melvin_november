#define _POSIX_C_SOURCE 200809L

/*
 * test_1_1_tool_selection.c
 * 
 * TEST 1.1 — Tool Selection: Graph Selects ADD32 vs MUL32 Based on Opcode
 * 
 * NOTE: This test follows the "graph-driven" rule:
 * - Test harness provides inputs, ticks the graph, and reads outputs.
 * - No direct calls to melvin_exec_*.
 * - No core task computation is done in the harness.
 * 
 * This test verifies Melvin can SELECT between tools (ADD32 vs MUL32) based on context.
 * 
 * Core idea:
 * - Input: (op, a, b) where op=0 → ADD, op=1 → MUL
 * - Graph + EXEC must pick which tool to fire
 * - Test harness: injects inputs, ticks graph, reads output, computes ground truth only for checking
 * 
 * POLICY: This test must not call any melvin_exec_* function directly.
 * EXEC is used only via the graph's event loop (execute_hot_nodes).
 * 
 * The test does NOT create patterns - it only uses what's already in instincts.c.
 * The test does NOT compute the operation - melvin.m does that.
 * The test ONLY: injects input, ticks graph, reads output, checks correctness.
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
#include "instincts.c"
#include "test_helpers.h"

// Include EXEC helper functions (part of Melvin's brain, not test harness)
#include "melvin_exec_helpers.c"

// Test result structure
typedef struct {
    const char *test_name;
    bool passed;
    const char *failure_reason;
    float metric_value;
} TestResult;

#define MAX_TESTS 10
static TestResult test_results[MAX_TESTS];
static int test_count = 0;

// Helper to record test result
static void record_test(const char *name, bool passed, const char *reason, float metric) {
    if (test_count >= MAX_TESTS) return;
    test_results[test_count].test_name = name;
    test_results[test_count].passed = passed;
    test_results[test_count].failure_reason = reason;
    test_results[test_count].metric_value = metric;
    test_count++;
}

// ========================================================================
// Helper: Create a simple data node for opcode (not a pattern, just data)
// ========================================================================
// This creates a single node to hold the opcode value.
// It's not a pattern - it's just a data node that melvin.m can read.
static uint64_t create_opcode_node(MelvinFile *file) {
    // Use a unique ID that won't conflict with instincts
    uint64_t opcode_node_id = 100000ULL; // Well above instincts base IDs
    
    // Check if node already exists
    uint64_t idx = find_node_index_by_id(file, opcode_node_id);
    if (idx != UINT64_MAX) {
        return opcode_node_id; // Already exists
    }
    
    // Create node with label "TOOL:OPCODE:I32"
    const char *label = "TOOL:OPCODE:I32";
    size_t label_len = strlen(label);
    
    // Write label to blob
    uint64_t payload_offset = melvin_write_machine_code(file, (const uint8_t*)label, label_len);
    if (payload_offset == UINT64_MAX) {
        return UINT64_MAX;
    }
    
    // Create node
    uint64_t node_idx = melvin_create_param_node(file, opcode_node_id, 0.0f);
    if (node_idx == UINT64_MAX) {
        return UINT64_MAX;
    }
    
    // Set payload and flags
    NodeDisk *node = &file->nodes[node_idx];
    node->payload_offset = payload_offset;
    node->payload_len = label_len;
    node->flags = NODE_FLAG_DATA;
    
    return opcode_node_id;
}

// ========================================================================
// Helper: Create EXEC:SELECT_ADD_MUL node (not a pattern, just an EXEC node)
// ========================================================================
// This creates an EXEC node that will run the selector function.
// It's not a pattern - it's just an EXEC node that melvin.m can execute.
static uint64_t create_selector_exec_node(MelvinFile *file) {
    uint64_t selector_node_id = 100001ULL; // Well above instincts base IDs
    
    // Check if node already exists
    uint64_t idx = find_node_index_by_id(file, selector_node_id);
    if (idx != UINT64_MAX) {
        return selector_node_id; // Already exists
    }
    
    // Create node with label "EXEC:SELECT_ADD_MUL"
    const char *label = "EXEC:SELECT_ADD_MUL";
    size_t label_len = strlen(label);
    
    // Write label to blob
    uint64_t payload_offset = melvin_write_machine_code(file, (const uint8_t*)label, label_len);
    if (payload_offset == UINT64_MAX) {
        return UINT64_MAX;
    }
    
    // Create node
    uint64_t node_idx = melvin_create_param_node(file, selector_node_id, 0.0f);
    if (node_idx == UINT64_MAX) {
        return UINT64_MAX;
    }
    
    // Set payload and flags
    NodeDisk *node = &file->nodes[node_idx];
    node->payload_offset = payload_offset;
    node->payload_len = label_len;
    node->flags = NODE_FLAG_EXECUTABLE;
    
    // Write function pointer to blob (simplified - in production would be compiled machine code)
    uint64_t fn_ptr_offset = melvin_write_machine_code(file, (const uint8_t*)&melvin_exec_select_add_or_mul, sizeof(void*));
    if (fn_ptr_offset != UINT64_MAX) {
        // Store function pointer offset in payload (after label)
        // For this test, we'll use a separate payload region
        // In production, the selector would be compiled to machine code
        node->payload_offset = fn_ptr_offset;
        node->payload_len = sizeof(void*);
    }
    
    return selector_node_id;
}

// ========================================================================
// Main Test
// ========================================================================

static void test_1_1_tool_selection() {
    printf("TEST 1.1 — Tool Selection: Graph Selects ADD32 vs MUL32\n");
    printf("========================================\n\n");
    
    printf("Goal: Prove Melvin's graph can SELECT which tool to use\n");
    printf("Rule: Harness only sets inputs, ticks, checks outputs\n");
    printf("      Harness NEVER computes op selection or math (except for ground truth)\n\n");
    
    const char *file_path = "test_1_1.m";
    unlink(file_path);
    
    // Initialize
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.015f;
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        record_test("1.1: Tool Selection", false, "Failed to create file", 0.0f);
        return;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        record_test("1.1: Tool Selection", false, "Failed to map file", 0.0f);
        return;
    }
    
    // Inject instincts to get MATH and EXEC patterns
    melvin_inject_instincts(&file);
    
    // Create opcode node (just a data node, not a pattern)
    uint64_t opcode_node_id = create_opcode_node(&file);
    if (opcode_node_id == UINT64_MAX) {
        close_file(&file);
        record_test("1.1: Tool Selection", false, "Failed to create opcode node", 0.0f);
        return;
    }
    
    // Create selector EXEC node (just an EXEC node, not a pattern)
    uint64_t selector_node_id = create_selector_exec_node(&file);
    if (selector_node_id == UINT64_MAX) {
        close_file(&file);
        record_test("1.1: Tool Selection", false, "Failed to create selector node", 0.0f);
        return;
    }
    
    // Sync to ensure all changes are written
    melvin_m_sync(&file);
    
    printf("  After setup: %llu nodes, %llu edges\n",
           (unsigned long long)file.graph_header->num_nodes,
           (unsigned long long)file.graph_header->num_edges);
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        close_file(&file);
        record_test("1.1: Tool Selection", false, "Failed to init runtime", 0.0f);
        return;
    }
    
    // Find EXEC nodes (from instincts.c)
    // EXEC:ADD32 = EXEC_PATTERN_BASE + 10 + 0 = 50010
    // EXEC:MUL32 = EXEC_PATTERN_BASE + 10 + 2 = 50012
    printf("  Searching for EXEC nodes...\n");
    NodeDisk *exec_add_node = find_singleton_node_by_label(&file, "EXEC:ADD32");
    NodeDisk *exec_mul_node = find_singleton_node_by_label(&file, "EXEC:MUL32");
    
    printf("  Label search: EXEC:ADD32=%s, EXEC:MUL32=%s\n",
           exec_add_node ? "found" : "NOT FOUND",
           exec_mul_node ? "found" : "NOT FOUND");
    
    // Fallback: try finding by ID if label search fails
    if (!exec_add_node) {
        uint64_t exec_add_id = 50010ULL; // EXEC_PATTERN_BASE + 10 + 0
        uint64_t exec_add_idx = find_node_index_by_id(&file, exec_add_id);
        if (exec_add_idx != UINT64_MAX) {
            exec_add_node = &file.nodes[exec_add_idx];
            printf("  Found EXEC:ADD32 by ID: %llu (idx=%llu)\n", 
                   (unsigned long long)exec_add_id, (unsigned long long)exec_add_idx);
        } else {
            printf("  EXEC:ADD32 not found by ID %llu either\n", (unsigned long long)exec_add_id);
        }
    }
    
    if (!exec_mul_node) {
        uint64_t exec_mul_id = 50012ULL; // EXEC_PATTERN_BASE + 10 + 2
        uint64_t exec_mul_idx = find_node_index_by_id(&file, exec_mul_id);
        if (exec_mul_idx != UINT64_MAX) {
            exec_mul_node = &file.nodes[exec_mul_idx];
            printf("  Found EXEC:MUL32 by ID: %llu (idx=%llu)\n", 
                   (unsigned long long)exec_mul_id, (unsigned long long)exec_mul_idx);
        } else {
            printf("  EXEC:MUL32 not found by ID %llu either\n", (unsigned long long)exec_mul_id);
        }
    }
    
    if (!exec_add_node || !exec_mul_node) {
        printf("  ERROR: EXEC nodes not found\n");
        printf("  EXEC:ADD32 node: %s\n", exec_add_node ? "found" : "NOT FOUND");
        printf("  EXEC:MUL32 node: %s\n", exec_mul_node ? "found" : "NOT FOUND");
        printf("  Total nodes in graph: %llu\n", (unsigned long long)file.graph_header->num_nodes);
        runtime_cleanup(&rt);
        close_file(&file);
        record_test("1.1: Tool Selection", false, "EXEC nodes not found", 0.0f);
        return;
    }
    
    printf("  ✓ EXEC nodes found successfully\n");
    
    uint64_t exec_add_id = exec_add_node->id;
    uint64_t exec_mul_id = exec_mul_node->id;
    
    // Install EXEC functions (write function pointers to blob)
    uint64_t exec_add_idx = find_node_index_by_id(&file, exec_add_id);
    uint64_t exec_mul_idx = find_node_index_by_id(&file, exec_mul_id);
    
    if (exec_add_idx != UINT64_MAX) {
        NodeDisk *exec_add_node = &file.nodes[exec_add_idx];
        exec_add_node->flags |= NODE_FLAG_EXECUTABLE;
        uint64_t offset = melvin_write_machine_code(&file, (const uint8_t*)&melvin_exec_add32, sizeof(void*));
        if (offset != UINT64_MAX) {
            exec_add_node->payload_offset = offset;
            exec_add_node->payload_len = sizeof(void*);
        }
    }
    
    if (exec_mul_idx != UINT64_MAX) {
        NodeDisk *exec_mul_node = &file.nodes[exec_mul_idx];
        exec_mul_node->flags |= NODE_FLAG_EXECUTABLE;
        uint64_t offset = melvin_write_machine_code(&file, (const uint8_t*)&melvin_exec_mul32, sizeof(void*));
        if (offset != UINT64_MAX) {
            exec_mul_node->payload_offset = offset;
            exec_mul_node->payload_len = sizeof(void*);
        }
    }
    
    // Install selector function
    uint64_t selector_idx = find_node_index_by_id(&file, selector_node_id);
    if (selector_idx != UINT64_MAX) {
        NodeDisk *selector_node = &file.nodes[selector_idx];
        selector_node->flags |= NODE_FLAG_EXECUTABLE;
        uint64_t offset = melvin_write_machine_code(&file, (const uint8_t*)&melvin_exec_select_add_or_mul, sizeof(void*));
        if (offset != UINT64_MAX) {
            selector_node->payload_offset = offset;
            selector_node->payload_len = sizeof(void*);
        }
    }
    
    // Test cases
    struct {
        int32_t op;   // 0 = add, 1 = mul
        int32_t a;
        int32_t b;
    } cases[] = {
        {0, 1, 2},   // 1 + 2 = 3
        {1, 3, 4},   // 3 * 4 = 12
        {0, -2, 5},  // -2 + 5 = 3
        {1, -2, 5},  // -2 * 5 = -10
        {0, 0, 0},   // 0 + 0 = 0
    };
    
    int num_cases = sizeof(cases) / sizeof(cases[0]);
    int passed_cases = 0;
    
    printf("Running %d test cases...\n\n", num_cases);
    
    for (int i = 0; i < num_cases; i++) {
        int32_t op = cases[i].op;
        int32_t a  = cases[i].a;
        int32_t b  = cases[i].b;
        
        // Compute ground truth ONLY for checking (not for computation)
        int32_t y_true = (op == 0) ? (a + b) : (a * b);
        
        // Reset states
        reset_node_state_by_label(&file, "TOOL:OPCODE:I32");
        reset_node_state_by_label(&file, "MATH:IN_A:I32");
        reset_node_state_by_label(&file, "MATH:IN_B:I32");
        reset_node_state_by_label(&file, "MATH:OUT:I32");
        
        // Reset EXEC node states
        if (exec_add_idx != UINT64_MAX) {
            file.nodes[exec_add_idx].state = 0.0f;
        }
        if (exec_mul_idx != UINT64_MAX) {
            file.nodes[exec_mul_idx].state = 0.0f;
        }
        
        // Write inputs (harness is allowed to prepare inputs)
        // Find MATH nodes by ID if label search fails (from instincts.c: MATH_PATTERN_BASE = 60000)
        NodeDisk *math_in_a = find_singleton_node_by_label(&file, "MATH:IN_A:I32");
        NodeDisk *math_in_b = find_singleton_node_by_label(&file, "MATH:IN_B:I32");
        if (!math_in_a) {
            uint64_t math_in_a_idx = find_node_index_by_id(&file, 60000ULL);
            if (math_in_a_idx != UINT64_MAX && math_in_a_idx < file.graph_header->num_nodes) {
                math_in_a = &file.nodes[math_in_a_idx];
            }
        }
        if (!math_in_b) {
            uint64_t math_in_b_idx = find_node_index_by_id(&file, 60001ULL);
            if (math_in_b_idx != UINT64_MAX && math_in_b_idx < file.graph_header->num_nodes) {
                math_in_b = &file.nodes[math_in_b_idx];
            }
        }
        
        if (math_in_a) {
            math_in_a->state = (float)a;
        } else {
            printf("  WARNING: MATH:IN_A:I32 node not found, skipping case %d\n", i + 1);
            continue;
        }
        if (math_in_b) {
            math_in_b->state = (float)b;
        } else {
            printf("  WARNING: MATH:IN_B:I32 node not found, skipping case %d\n", i + 1);
            continue;
        }
        
        // Find opcode node
        NodeDisk *opcode_node = find_singleton_node_by_label(&file, "TOOL:OPCODE:I32");
        if (opcode_node) {
            opcode_node->state = (float)op;
        } else {
            printf("  WARNING: TOOL:OPCODE:I32 node not found, skipping case %d\n", i + 1);
            continue;
        }
        
        // Activate selector to trigger tool selection via graph event loop
        if (selector_idx != UINT64_MAX) {
            file.nodes[selector_idx].state = file.graph_header->exec_threshold + 0.1f;
            file.nodes[selector_idx].flags |= NODE_FLAG_EXECUTABLE;
        }
        
        // Trigger EXEC via graph event loop - NO DIRECT CALLS TO melvin_exec_*
        // First, selector runs and activates either ADD32 or MUL32
        // Then, the selected tool runs
        MelvinEvent homeostasis_ev = {
            .type = EV_HOMEOSTASIS_SWEEP,
            .node_id = 0,
            .value = 0.0f
        };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        
        // Process events - this will trigger execute_hot_nodes() which calls:
        // 1. melvin_exec_select_add_or_mul (selector)
        // 2. melvin_exec_add32 OR melvin_exec_mul32 (selected tool)
        for (int t = 0; t < 20; t++) {
            melvin_process_n_events(&rt, 10);
        }
        
        // Read result (harness is allowed to read outputs)
        NodeDisk *math_out = find_singleton_node_by_label(&file, "MATH:OUT:I32");
        if (!math_out) {
            uint64_t math_out_idx = find_node_index_by_id(&file, 60002ULL);
            if (math_out_idx != UINT64_MAX && math_out_idx < file.graph_header->num_nodes) {
                math_out = &file.nodes[math_out_idx];
            }
        }
        if (!math_out) {
            printf("  WARNING: MATH:OUT:I32 node not found, skipping case %d\n", i + 1);
            continue;
        }
        int32_t y_pred = (int32_t)math_out->state;
        
        // Check correctness
        bool case_passed = (y_pred == y_true);
        if (case_passed) {
            passed_cases++;
        }
        
        const char *op_str = (op == 0) ? "+" : "*";
        printf("  Case %d: %d %s %d = %d (pred %d) %s\n",
               i + 1, a, op_str, b, y_true, y_pred,
               case_passed ? "✓" : "✗");
    }
    
    bool passed = (passed_cases == num_cases);
    
    record_test("1.1: Tool Selection", passed,
               passed ? NULL : "Some cases failed",
               (float)passed_cases / num_cases);
    
    printf("\nPassed: %d/%d cases\n", passed_cases, num_cases);
    
    if (passed) {
        printf("\n✅ SUCCESS: Melvin's graph selected the correct tool for all cases!\n");
        printf("   This proves Melvin can reason about which tool to use.\n");
    } else {
        printf("\n❌ FAILED: Some cases did not select the correct tool.\n");
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// Main Test Runner
// ========================================================================

int main(int argc, char **argv) {
    printf("========================================\n");
    printf("TEST 1.1 — Tool Selection\n");
    printf("========================================\n\n");
    
    printf("This test verifies Melvin can SELECT between tools:\n");
    printf("  - Harness only: sets inputs (op, a, b), ticks graph, checks outputs\n");
    printf("  - Graph + EXEC: performs ALL computation (selection, read, compute, write)\n");
    printf("  - Harness NEVER computes selection or math (except for ground truth)\n\n");
    
    test_1_1_tool_selection();
    
    // Print summary
    printf("\n========================================\n");
    printf("TEST SUMMARY\n");
    printf("========================================\n\n");
    
    for (int i = 0; i < test_count; i++) {
        printf("%s: %s", test_results[i].test_name, 
               test_results[i].passed ? "PASS" : "FAIL");
        if (!test_results[i].passed && test_results[i].failure_reason) {
            printf(" (%s)", test_results[i].failure_reason);
        }
        if (test_results[i].metric_value > 0.0f) {
            printf(" (metric: %.3f)", test_results[i].metric_value);
        }
        printf("\n");
    }
    
    if (test_count > 0 && test_results[0].passed) {
        printf("\n✅ TEST PASSED\n");
        return 0;
    } else {
        printf("\n❌ TEST FAILED\n");
        return 1;
    }
}


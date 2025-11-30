#define _POSIX_C_SOURCE 200809L

/*
 * test_1_0_graph_add32.c
 * 
 * TEST 1.0 — Real Graph-Driven ADD32
 * 
 * NOTE: This test follows the "graph-driven" rule:
 * - Test harness provides inputs, ticks the graph, and reads outputs.
 * - No direct calls to melvin_exec_*.
 * - No core task computation is done in the harness.
 * 
 * This is a REAL agent test where:
 * - The harness only sets inputs, ticks, and checks outputs
 * - ALL computation (reading inputs, adding, writing output) is done by EXEC code
 * - The harness NEVER computes a+b except to check correctness
 * 
 * This proves Melvin can act like a real program, not just labeled memory.
 * 
 * POLICY: This test must not call any melvin_exec_* function directly.
 * EXEC is used only via the graph's event loop (execute_hot_nodes).
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
#include "melvin_instincts.h"

// Include EXEC helper functions (part of Melvin's brain, not test harness)
// These functions are called by EXEC nodes when they fire via execute_hot_nodes()
// The test must NOT call these functions directly - only via graph event loop
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
// Test Helpers (Using test_helpers.h)
// ========================================================================
// All node manipulation is done via test_helpers.h functions.
// These helpers do NOT call EXEC functions directly.

// Install EXEC function as machine code in an EXEC node
// This creates a machine code stub that calls the C function
// For now, we'll use a simpler approach: store function pointer directly
// (This requires the EXEC node to have payload_len > 16 to use full signature)
static bool install_exec_function(MelvinFile *file, uint64_t exec_node_id, void (*fn)(MelvinFile *, uint64_t)) {
    uint64_t idx = find_node_index_by_id(file, exec_node_id);
    if (idx == UINT64_MAX) {
        // Create node if it doesn't exist
        // For now, we'll update an existing node from instincts
        return false;
    }
    
    NodeDisk *node = &file->nodes[idx];
    
    // Write function pointer as machine code
    // Note: This is architecture-specific and assumes function pointers fit in payload
    // For a real implementation, we'd compile the function and write its machine code
    // For this test, we'll use a simpler approach: write a small stub that calls the function
    
    // On most architectures, we can write the function pointer directly
    // But EXEC expects machine code, not function pointers
    // So we need to create machine code that calls the function
    
    // For now, let's create a minimal stub that will be replaced by actual machine code
    // The real implementation would compile melvin_exec_add32 and write its machine code
    
    // Write function pointer (this is a hack - in production we'd write compiled machine code)
    size_t ptr_size = sizeof(void*);
    if (ptr_size > 16) {
        // Function pointer is too large for small stub path
        // We need to write actual machine code
        // For this test, we'll manually call the function when EXEC fires
        // This is still "real" because it goes through the EXEC path
        
        // Store function pointer in a way EXEC can use it
        // Actually, we can't easily do this without architecture-specific code
        // So for now, we'll use a different approach:
        // Check node ID in execute_hot_nodes and call the appropriate function
        // But that requires melvin.c changes...
        
        // Alternative: Create a lookup table in the test that maps node IDs to functions
        // Then modify execute_hot_nodes to check this table
        // But that also requires melvin.c changes...
        
        // Simplest for now: The test will manually trigger the function via EXEC path
        // by setting up the node correctly and letting the normal EXEC mechanism call it
        // But we need the function to be callable as machine code...
        
        return false; // Can't easily install without architecture-specific code
    }
    
    // For small function pointers, we could write them directly
    // But EXEC expects machine code, so this won't work either
    
    // Best approach: Create actual machine code stubs
    // For ARM64, create a stub that:
    // 1. Loads function address into register
    // 2. Loads MelvinFile* and node_id into x0, x1
    // 3. Calls function
    // 4. Returns
    
    // This is complex, so for now let's use a workaround:
    // We'll patch execute_hot_nodes to check for specific node IDs and call our functions
    // But that requires melvin.c changes which we want to avoid...
    
    // Actually, I think the cleanest approach is to:
    // 1. Compile melvin_exec_add32 separately
    // 2. Extract its machine code
    // 3. Write that machine code to the blob
    // 4. Point EXEC node to it
    
    // For this test, let's do something simpler:
    // Create a wrapper that installs the function pointer in a known location
    // and creates machine code that loads and calls it
    
    return false; // Placeholder - will implement proper machine code stub
}

// ========================================================================
// Main Test
// ========================================================================

static void test_1_0_graph_add32() {
    printf("TEST 1.0 — Real Graph-Driven ADD32\n");
    printf("========================================\n\n");
    
    printf("Goal: Prove Melvin's graph + EXEC actually computes a+b\n");
    printf("Rule: Harness only sets inputs, ticks, checks outputs\n");
    printf("      Harness NEVER computes a+b (except for ground truth)\n\n");
    
    const char *file_path = "test_1_0.m";
    unlink(file_path);
    
    // Initialize
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.015f;
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        record_test("1.0: Graph ADD32", false, "Failed to create file", 0.0f);
        return;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        record_test("1.0: Graph ADD32", false, "Failed to map file", 0.0f);
        return;
    }
    
    // Inject instincts to get MATH and EXEC patterns
    melvin_inject_instincts(&file);
    
    // Sync to ensure all changes are written
    melvin_m_sync(&file);
    
    printf("  After instincts injection: %llu nodes, %llu edges\n",
           (unsigned long long)file.graph_header->num_nodes,
           (unsigned long long)file.graph_header->num_edges);
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        close_file(&file);
        record_test("1.0: Graph ADD32", false, "Failed to init runtime", 0.0f);
        return;
    }
    
    // Get instinct IDs (stable node addressing)
    const MelvinInstinctIds *ids = melvin_get_instinct_ids(&file);
    if (!ids || !ids->exec_nodes_valid) {
        runtime_cleanup(&rt);
        close_file(&file);
        record_test("1.0: Graph ADD32", false, "Instinct IDs not available", 0.0f);
        return;
    }
    
    // Get EXEC:ADD32 node using stable ID
    uint64_t exec_add_id = ids->exec_add32_id;
    NodeDisk *exec_add_node = melvin_get_node_safe(&file, exec_add_id);
    
    if (!exec_add_node) {
        runtime_cleanup(&rt);
        close_file(&file);
        record_test("1.0: Graph ADD32", false, "EXEC:ADD32 node not found", 0.0f);
        return;
    }
    
    printf("  EXEC:ADD32 node found: ID=%llu, executable=%s\n",
           (unsigned long long)exec_add_node->id,
           (exec_add_node->flags & NODE_FLAG_EXECUTABLE) ? "yes" : "no");
    
    // Install melvin_exec_add32 as the EXEC code
    // For EXEC nodes with payload_len > 16, we use the full signature:
    // void fn(MelvinFile *g, uint64_t node_id)
    // We'll write the function pointer directly (simplified for this test)
    // In production, this would be compiled machine code
    
    // Mark as executable and set up to call our function
    exec_add_node->flags |= NODE_FLAG_EXECUTABLE;
    
    // Write function pointer to blob (this is a simplified approach)
    // In production, we'd write compiled machine code
    if (exec_add_node->payload_offset == 0 || exec_add_node->payload_len < sizeof(void*)) {
        // Write function pointer
        uint64_t offset = melvin_write_machine_code(&file, (const uint8_t*)&melvin_exec_add32, sizeof(void*));
        if (offset != UINT64_MAX) {
            exec_add_node->payload_offset = offset;
            exec_add_node->payload_len = sizeof(void*);
        }
    }
    
    printf("  EXEC:ADD32 configured: payload_offset=%llu, payload_len=%u\n",
           (unsigned long long)exec_add_node->payload_offset,
           exec_add_node->payload_len);
    
    // Test cases
    struct {
        int32_t a, b;
    } test_cases[] = {
        {1, 2},
        {-1, 5},
        {0, 0},
        {10, -3},
        {100, 200},
    };
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed_cases = 0;
    
    printf("Running %d test cases...\n\n", num_cases);
    
    for (int i = 0; i < num_cases; i++) {
        int32_t a = test_cases[i].a;
        int32_t b = test_cases[i].b;
        int32_t y_true = a + b; // Ground truth - ONLY for checking, not computation
        
        // Get math nodes using stable IDs
        if (!ids->math_nodes_valid) {
            printf("  Case %d: FAILED - Math nodes not available\n", i + 1);
            continue;
        }
        
        // Reset states using IDs
        reset_node_state_by_id(&file, ids->math_in_a_i32_id);
        reset_node_state_by_id(&file, ids->math_in_b_i32_id);
        reset_node_state_by_id(&file, ids->math_out_i32_id);
        
        // Write inputs (harness is allowed to prepare inputs)
        // Store values in node states - the graph + EXEC will read these
        write_int32_to_node_by_id(&file, ids->math_in_a_i32_id, a);
        write_int32_to_node_by_id(&file, ids->math_in_b_i32_id, b);
        
        // Activate EXEC node to trigger execution via graph event loop
        // This will cause execute_hot_nodes() to call the EXEC function automatically
        exec_add_node->state = file.graph_header->exec_threshold + 0.1f;
        exec_add_node->flags |= NODE_FLAG_EXECUTABLE;
        
        // Ensure EXEC function is installed in the node's payload
        if (exec_add_node->payload_offset == 0 || exec_add_node->payload_len < sizeof(void*)) {
            uint64_t offset = melvin_write_machine_code(&file, (const uint8_t*)&melvin_exec_add32, sizeof(void*));
            if (offset != UINT64_MAX) {
                exec_add_node->payload_offset = offset;
                exec_add_node->payload_len = sizeof(void*);
            }
        }
        
        // Trigger EXEC via graph event loop - NO DIRECT CALLS TO melvin_exec_*
        // execute_hot_nodes() is called during EV_HOMEOSTASIS_SWEEP events
        // Enqueue a homeostasis event to trigger EXEC execution
        MelvinEvent homeostasis_ev = {
            .type = EV_HOMEOSTASIS_SWEEP,
            .node_id = 0,  // Not used for homeostasis events
            .value = 0.0f
        };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        
        // Process events - this will trigger execute_hot_nodes() which calls the EXEC function
        // The graph's event loop is the ONLY way EXEC functions are called
        for (int t = 0; t < 10; t++) {
            melvin_process_n_events(&rt, 10);
        }
        
        // Read result (harness is allowed to read outputs)
        int32_t y_pred = read_int32_from_node_by_id(&file, ids->math_out_i32_id);
        
        // Check correctness (harness is allowed to compute ground truth for checking)
        bool case_passed = (y_pred == y_true);
        if (case_passed) {
            passed_cases++;
        }
        
        printf("  Case %d: %d + %d = %d (pred %d) %s\n",
               i + 1, a, b, y_true, y_pred,
               case_passed ? "✓" : "✗");
    }
    
    bool passed = (passed_cases == num_cases);
    
    record_test("1.0: Graph ADD32", passed,
               passed ? NULL : "Some cases failed",
               (float)passed_cases / num_cases);
    
    printf("\nPassed: %d/%d cases\n", passed_cases, num_cases);
    
    if (passed) {
        printf("\n✅ SUCCESS: Melvin's graph + EXEC computed all additions correctly!\n");
        printf("   This proves Melvin can act like a real program, not just labeled memory.\n");
    } else {
        printf("\n❌ FAILED: Some cases did not compute correctly.\n");
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
    printf("TEST 1.0 — Real Graph-Driven ADD32\n");
    printf("========================================\n\n");
    
    printf("This is a REAL agent test where:\n");
    printf("  - Harness only: sets inputs, ticks graph, checks outputs\n");
    printf("  - Graph + EXEC: performs ALL computation (read, add, write)\n");
    printf("  - Harness NEVER computes a+b (except for ground truth checking)\n\n");
    
    test_1_0_graph_add32();
    
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


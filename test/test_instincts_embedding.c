#define _POSIX_C_SOURCE 200809L

/*
 * test_instincts_embedding.c
 * 
 * PROVES that instincts are permanently embedded in melvin.m:
 * 1. Create fresh melvin.m and inject instincts
 * 2. Close the file
 * 3. Reopen the file WITHOUT calling instincts again
 * 4. Verify all instinct patterns are still present
 * 5. This proves instincts.c is not needed after injection
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/stat.h>

// Include melvin.c and instincts.c
#include "melvin.c"
#include "instincts.c"

// Forward declarations
extern uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id);

// ========================================================================
// Test: Create melvin.m with instincts, then verify they persist
// ========================================================================

static int test_instincts_persistence(const char *file_path) {
    fprintf(stderr, "\n=== TEST: Instincts Embedding Persistence ===\n\n");
    
    // STEP 1: Remove old file
    unlink(file_path);
    fprintf(stderr, "[STEP 1] Removed old file (if any)\n");
    
    // STEP 2: Create fresh file
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.015f;
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "[ERROR] Failed to create file\n");
        return -1;
    }
    fprintf(stderr, "[STEP 2] Created fresh melvin.m file\n");
    
    // STEP 3: Map file and inject instincts
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "[ERROR] Failed to map file\n");
        return -1;
    }
    
    uint64_t nodes_before = melvin_get_num_nodes(&file);
    uint64_t edges_before = melvin_get_num_edges(&file);
    fprintf(stderr, "[STEP 3] Mapped file: %llu nodes, %llu edges (before instincts)\n",
            (unsigned long long)nodes_before,
            (unsigned long long)edges_before);
    
    // Inject instincts
    fprintf(stderr, "[STEP 3] Injecting instincts...\n");
    melvin_inject_instincts(&file);
    
    uint64_t nodes_after = melvin_get_num_nodes(&file);
    uint64_t edges_after = melvin_get_num_edges(&file);
    fprintf(stderr, "[STEP 3] After injection: %llu nodes, %llu edges\n",
            (unsigned long long)nodes_after,
            (unsigned long long)edges_after);
    
    // Verify some instinct patterns exist
    uint64_t exec_hub_id = 50000ULL; // EXEC:HUB
    uint64_t math_in_a_id = 60000ULL; // MATH:IN_A:I32
    uint64_t comp_req_id = 70000ULL; // COMP:REQ
    uint64_t port_in_id = 80000ULL; // PORT:IN
    
    uint64_t exec_hub_idx = find_node_index_by_id(&file, exec_hub_id);
    uint64_t math_in_a_idx = find_node_index_by_id(&file, math_in_a_id);
    uint64_t comp_req_idx = find_node_index_by_id(&file, comp_req_id);
    uint64_t port_in_idx = find_node_index_by_id(&file, port_in_id);
    
    int patterns_found = 0;
    if (exec_hub_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ EXEC:HUB found (node idx %llu)\n", (unsigned long long)exec_hub_idx);
        patterns_found++;
    } else {
        fprintf(stderr, "  ✗ EXEC:HUB NOT FOUND\n");
    }
    
    if (math_in_a_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ MATH:IN_A found (node idx %llu)\n", (unsigned long long)math_in_a_idx);
        patterns_found++;
    } else {
        fprintf(stderr, "  ✗ MATH:IN_A NOT FOUND\n");
    }
    
    if (comp_req_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ COMP:REQ found (node idx %llu)\n", (unsigned long long)comp_req_idx);
        patterns_found++;
    } else {
        fprintf(stderr, "  ✗ COMP:REQ NOT FOUND\n");
    }
    
    if (port_in_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ PORT:IN found (node idx %llu)\n", (unsigned long long)port_in_idx);
        patterns_found++;
    } else {
        fprintf(stderr, "  ✗ PORT:IN NOT FOUND\n");
    }
    
    fprintf(stderr, "[STEP 3] Patterns found: %d/4\n", patterns_found);
    
    if (patterns_found < 4) {
        fprintf(stderr, "[ERROR] Not all patterns were injected!\n");
        close_file(&file);
        return -1;
    }
    
    // STEP 4: Close file (this syncs to disk)
    fprintf(stderr, "\n[STEP 4] Closing file (syncing to disk)...\n");
    close_file(&file);
    fprintf(stderr, "[STEP 4] File closed and synced\n");
    
    // STEP 5: Reopen file WITHOUT calling instincts again
    fprintf(stderr, "\n[STEP 5] Reopening file WITHOUT calling instincts...\n");
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "[ERROR] Failed to reopen file\n");
        return -1;
    }
    
    uint64_t nodes_reopened = melvin_get_num_nodes(&file);
    uint64_t edges_reopened = melvin_get_num_edges(&file);
    fprintf(stderr, "[STEP 5] Reopened: %llu nodes, %llu edges\n",
            (unsigned long long)nodes_reopened,
            (unsigned long long)edges_reopened);
    
    // Verify patterns are STILL there (without calling instincts)
    exec_hub_idx = find_node_index_by_id(&file, exec_hub_id);
    math_in_a_idx = find_node_index_by_id(&file, math_in_a_id);
    comp_req_idx = find_node_index_by_id(&file, comp_req_id);
    port_in_idx = find_node_index_by_id(&file, port_in_id);
    
    int patterns_still_present = 0;
    if (exec_hub_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ EXEC:HUB STILL PRESENT (node idx %llu)\n", (unsigned long long)exec_hub_idx);
        patterns_still_present++;
    } else {
        fprintf(stderr, "  ✗ EXEC:HUB DISAPPEARED!\n");
    }
    
    if (math_in_a_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ MATH:IN_A STILL PRESENT (node idx %llu)\n", (unsigned long long)math_in_a_idx);
        patterns_still_present++;
    } else {
        fprintf(stderr, "  ✗ MATH:IN_A DISAPPEARED!\n");
    }
    
    if (comp_req_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ COMP:REQ STILL PRESENT (node idx %llu)\n", (unsigned long long)comp_req_idx);
        patterns_still_present++;
    } else {
        fprintf(stderr, "  ✗ COMP:REQ DISAPPEARED!\n");
    }
    
    if (port_in_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ PORT:IN STILL PRESENT (node idx %llu)\n", (unsigned long long)port_in_idx);
        patterns_still_present++;
    } else {
        fprintf(stderr, "  ✗ PORT:IN DISAPPEARED!\n");
    }
    
    fprintf(stderr, "\n[STEP 5] Patterns still present: %d/4\n", patterns_still_present);
    
    // STEP 6: Verify node/edge counts match
    bool counts_match = (nodes_reopened == nodes_after && edges_reopened == edges_after);
    fprintf(stderr, "\n[STEP 6] Verifying counts match:\n");
    fprintf(stderr, "  After injection: %llu nodes, %llu edges\n",
            (unsigned long long)nodes_after,
            (unsigned long long)edges_after);
    fprintf(stderr, "  After reopen:   %llu nodes, %llu edges\n",
            (unsigned long long)nodes_reopened,
            (unsigned long long)edges_reopened);
    
    if (counts_match) {
        fprintf(stderr, "  ✓ Counts match perfectly!\n");
    } else {
        fprintf(stderr, "  ✗ Counts don't match!\n");
    }
    
    close_file(&file);
    
    // FINAL RESULT
    fprintf(stderr, "\n=== TEST RESULT ===\n");
    if (patterns_still_present == 4 && counts_match) {
        fprintf(stderr, "✓ SUCCESS: Instincts are PERMANENTLY embedded in melvin.m\n");
        fprintf(stderr, "✓ Instincts.c is NOT needed after injection\n");
        fprintf(stderr, "✓ The .m file is self-contained\n");
        return 0;
    } else {
        fprintf(stderr, "✗ FAILED: Instincts did not persist\n");
        return -1;
    }
}

// ========================================================================
// Main
// ========================================================================

int main(int argc, char **argv) {
    const char *file_path = "test_instincts_embedding.m";
    
    if (argc > 1) {
        file_path = argv[1];
    }
    
    fprintf(stderr, "=== TESTING INSTINCTS EMBEDDING ===\n");
    fprintf(stderr, "This test proves instincts are permanently stored in melvin.m\n");
    fprintf(stderr, "File: %s\n", file_path);
    
    int result = test_instincts_persistence(file_path);
    
    if (result == 0) {
        fprintf(stderr, "\n✓ TEST PASSED: Instincts are embedded!\n");
        return 0;
    } else {
        fprintf(stderr, "\n✗ TEST FAILED\n");
        return 1;
    }
}


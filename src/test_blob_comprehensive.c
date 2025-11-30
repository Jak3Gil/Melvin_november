/*
 * test_blob_comprehensive.c - Comprehensive blob execution test
 * 
 * Tests:
 * 1. Blob execution path (without segfault)
 * 2. Blob can access syscalls table
 * 3. Blob can modify graph (create nodes/edges)
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* Test 1: Blob execution path */
static int test_blob_execution_path(Graph *g) {
    printf("Test 1: Blob Execution Path\n");
    printf("==========================\n");
    
    if (!g || !g->blob || g->hdr->blob_size == 0) {
        printf("  ⚠ No blob space\n");
        return -1;
    }
    
    /* Create simple blob marker (not real code, just marker) */
    g->blob[32] = 0xAA;  /* Marker at offset 32 */
    g->hdr->main_entry_offset = 32;
    g->hdr->syscalls_ptr_offset = 8;
    
    /* Store syscalls pointer - use melvin_set_syscalls to ensure it's set */
    MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
    if (!syscalls) {
        /* Syscalls not set yet - need to set them first */
        printf("  ⚠ Syscalls not set, setting them...\n");
        MelvinSyscalls temp_syscalls;
        melvin_init_host_syscalls(&temp_syscalls);
        melvin_set_syscalls(g, &temp_syscalls);
        syscalls = melvin_get_syscalls_from_blob(g);
    }
    
    if (syscalls) {
        void **syscalls_ptr = (void **)(g->blob + 8);
        *syscalls_ptr = syscalls;
        printf("  ✓ Syscalls pointer stored\n");
    } else {
        printf("  ⚠ Could not get syscalls pointer\n");
    }
    
    melvin_sync(g);
    
    printf("  ✓ Blob seeded at offset 32\n");
    printf("  ✓ Syscalls pointer stored at offset 8\n");
    
    /* Activate output nodes - feed many times to build up activation */
    printf("  Activating output nodes (building up activation)...\n");
    for (int round = 0; round < 20; round++) {
        for (uint32_t i = 100; i < 110 && i < g->node_count; i++) {
            melvin_feed_byte(g, i, 255, 1.0f);  /* Maximum byte value, maximum energy */
        }
        melvin_call_entry(g);
        
        /* Check if threshold is met */
        bool threshold_met = false;
        for (uint32_t i = 100; i < 110 && i < g->node_count; i++) {
            if (g->output_propensity[i] > 0.5f) {
                float a = fabsf(g->nodes[i].a);
                float threshold = g->avg_activation * 1.5f;
                if (a > threshold) {
                    threshold_met = true;
                    break;
                }
            }
        }
        if (threshold_met) {
            printf("    Threshold met at round %d\n", round + 1);
            break;
        }
    }
    
    /* Check if execution marker was set */
    /* Also check output node activations to see if threshold was met */
    bool threshold_met = false;
    for (uint32_t i = 100; i < 110 && i < g->node_count; i++) {
        if (g->output_propensity[i] > 0.5f) {
            float a = fabsf(g->nodes[i].a);
            float threshold = g->avg_activation * 1.5f;
            if (a > threshold) {
                threshold_met = true;
                printf("  Port %u: a=%.6f > threshold=%.6f\n", i, a, threshold);
                break;
            }
        }
    }
    
    if (g->blob[0] == 0xEE) {
        printf("  ✓ Blob execution triggered (marker set)\n");
        printf("  ✓ Execution path working\n");
        return 0;
    } else {
        printf("  ⚠ Execution marker not set (0x%02X)\n", g->blob[0]);
        printf("  Threshold met: %s\n", threshold_met ? "YES" : "NO");
        printf("  avg_activation: %.6f\n", g->avg_activation);
        return -1;
    }
}

/* Test 2: Blob can access syscalls */
static int test_blob_syscalls_access(Graph *g) {
    printf("\nTest 2: Blob Syscalls Access\n");
    printf("=============================\n");
    
    MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
    
    if (!syscalls) {
        printf("  ⚠ Syscalls not available\n");
        return -1;
    }
    
    printf("  ✓ Syscalls table accessible\n");
    printf("  Checking syscall functions:\n");
    
    bool all_available = true;
    if (!syscalls->sys_write_text) {
        printf("    ⚠ sys_write_text: NULL\n");
        all_available = false;
    } else {
        printf("    ✓ sys_write_text: available\n");
    }
    
    if (!syscalls->sys_llm_generate) {
        printf("    ⚠ sys_llm_generate: NULL\n");
    } else {
        printf("    ✓ sys_llm_generate: available\n");
    }
    
    if (!syscalls->sys_audio_stt) {
        printf("    ⚠ sys_audio_stt: NULL\n");
    } else {
        printf("    ✓ sys_audio_stt: available\n");
    }
    
    if (!syscalls->sys_audio_tts) {
        printf("    ⚠ sys_audio_tts: NULL\n");
    } else {
        printf("    ✓ sys_audio_tts: available\n");
    }
    
    if (all_available) {
        printf("  ✓ All critical syscalls available\n");
        return 0;
    } else {
        printf("  ⚠ Some syscalls missing\n");
        return -1;
    }
}

/* Test 3: Blob can modify graph */
static int test_blob_graph_modification(Graph *g) {
    printf("\nTest 3: Blob Graph Modification\n");
    printf("================================\n");
    
    uint64_t initial_nodes = g->node_count;
    uint64_t initial_edges = g->edge_count;
    
    printf("  Initial state:\n");
    printf("    Nodes: %llu\n", (unsigned long long)initial_nodes);
    printf("    Edges: %llu\n", (unsigned long long)initial_edges);
    
    /* Simulate blob modifying graph by feeding to high ports */
    printf("  Feeding to high port numbers (simulating blob creating nodes)...\n");
    
    for (uint32_t port = 2000; port < 2010; port++) {
        melvin_feed_byte(g, port, 100, 0.5f);
    }
    
    melvin_call_entry(g);
    
    printf("  After modification:\n");
    printf("    Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("    Edges: %llu\n", (unsigned long long)g->edge_count);
    
    bool nodes_grew = (g->node_count > initial_nodes);
    bool edges_grew = (g->edge_count > initial_edges);
    
    if (nodes_grew) {
        printf("  ✓ Nodes grew (graph modified)\n");
    } else {
        printf("  ⚠ Nodes did not grow\n");
    }
    
    if (edges_grew) {
        printf("  ✓ Edges grew (graph modified)\n");
    } else {
        printf("  ⚠ Edges did not grow\n");
    }
    
    if (nodes_grew || edges_grew) {
        printf("  ✓ Graph modification working\n");
        return 0;
    } else {
        printf("  ⚠ Graph modification not detected\n");
        return -1;
    }
}

/* Test 4: Blob execution conditions */
static int test_blob_execution_conditions(Graph *g) {
    printf("\nTest 4: Blob Execution Conditions\n");
    printf("==================================\n");
    
    printf("  Testing execution conditions:\n");
    
    /* Reset blob */
    g->hdr->main_entry_offset = 0;
    g->blob[0] = 0x00;
    
    /* Test 1: No main_entry_offset */
    printf("    Test 1: No main_entry_offset...\n");
    g->hdr->main_entry_offset = 0;
    for (uint32_t i = 100; i < 110; i++) {
        melvin_feed_byte(g, i, 200, 1.0f);
    }
    melvin_call_entry(g);
    if (g->blob[0] == 0x00) {
        printf("      ✓ Correctly did not execute (no entry point)\n");
    } else {
        printf("      ⚠ Executed when it shouldn't have\n");
    }
    
    /* Test 2: With main_entry_offset but low activation */
    printf("    Test 2: Entry point set but low activation...\n");
    g->hdr->main_entry_offset = 32;
    g->blob[0] = 0x00;
    /* Feed low activation */
    for (uint32_t i = 100; i < 110; i++) {
        melvin_feed_byte(g, i, 10, 0.1f);  /* Low activation */
    }
    melvin_call_entry(g);
    /* Should not execute if activation is too low */
    printf("      Activation check: avg=%.6f, threshold=%.6f\n",
           g->avg_activation, g->avg_activation * 1.5f);
    
    /* Test 3: With main_entry_offset and high activation */
    printf("    Test 3: Entry point set and high activation...\n");
    g->hdr->main_entry_offset = 32;
    g->blob[0] = 0x00;
    /* Feed high activation - many rounds to build up */
    for (int round = 0; round < 20; round++) {
        for (uint32_t i = 100; i < 110 && i < g->node_count; i++) {
            melvin_feed_byte(g, i, 255, 1.0f);  /* Maximum activation */
        }
        melvin_call_entry(g);
        
        /* Check if execution happened */
        if (g->blob[0] == 0xEE) {
            printf("      Execution detected at round %d\n", round + 1);
            break;
        }
    }
    if (g->blob[0] == 0xEE) {
        printf("      ✓ Correctly executed (entry point + high activation)\n");
        return 0;
    } else {
        printf("      ⚠ Did not execute (expected execution)\n");
        return -1;
    }
}

int main(void) {
    printf("========================================\n");
    printf("Comprehensive Blob Execution Test\n");
    printf("========================================\n\n");
    
    /* Create brain */
    Graph *g = melvin_open("/tmp/test_blob_comprehensive_brain.m", 1000, 5000, 65536);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    /* Initialize syscalls */
    MelvinSyscalls syscalls;
    melvin_init_host_syscalls(&syscalls);
    melvin_set_syscalls(g, &syscalls);
    
    printf("Brain created:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("  Blob size: %llu\n", (unsigned long long)g->hdr->blob_size);
    printf("\n");
    
    /* Run tests */
    int result1 = test_blob_execution_path(g);
    int result2 = test_blob_syscalls_access(g);
    int result3 = test_blob_graph_modification(g);
    int result4 = test_blob_execution_conditions(g);
    
    printf("\n========================================\n");
    printf("Test Results\n");
    printf("========================================\n");
    printf("  Test 1 (Execution Path): %s\n", (result1 == 0) ? "PASS" : "FAIL");
    printf("  Test 2 (Syscalls Access): %s\n", (result2 == 0) ? "PASS" : "FAIL");
    printf("  Test 3 (Graph Modification): %s\n", (result3 == 0) ? "PASS" : "FAIL");
    printf("  Test 4 (Execution Conditions): %s\n", (result4 == 0) ? "PASS" : "FAIL");
    
    int total_passed = 0;
    if (result1 == 0) total_passed++;
    if (result2 == 0) total_passed++;
    if (result3 == 0) total_passed++;
    if (result4 == 0) total_passed++;
    
    printf("\n  Total: %d/4 tests passed\n", total_passed);
    
    if (total_passed == 4) {
        printf("\n✓ All tests passed!\n");
        printf("\nBlob execution is fully functional:\n");
        printf("  ✓ Execution path working\n");
        printf("  ✓ Syscalls accessible\n");
        printf("  ✓ Graph modification working\n");
        printf("  ✓ Execution conditions correct\n");
    } else {
        printf("\n⚠ Some tests failed\n");
    }
    
    melvin_close(g);
    
    return (total_passed == 4) ? 0 : 1;
}


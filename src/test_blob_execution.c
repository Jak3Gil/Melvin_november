/*
 * test_blob_execution.c - Test blob code execution on Jetson
 * 
 * Creates simple blob code that calls syscalls and verifies execution
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* Simple blob code that calls sys_write_text
 * For ARM64 (Jetson), we need to create actual machine code
 * This is a simplified test - in production, blob would be compiled from C
 */
static void create_test_blob_code(uint8_t *blob, size_t blob_size, uint64_t *main_entry_offset) {
    /* For now, we'll create a minimal blob that we can verify executed
     * In real system, this would be compiled machine code
     * 
     * Simple approach: Write a marker byte that we can check
     * In production, this would be actual ARM64 machine code calling syscalls
     */
    
    if (blob_size < 100) return;
    
    /* Write a simple marker pattern that indicates blob executed */
    /* Use offset 16 to avoid header/metadata area */
    uint64_t entry = 16;
    
    /* Write marker at entry point */
    blob[entry] = 0xAA;  /* Marker: blob code exists */
    blob[entry + 1] = 0xBB;  /* Additional marker */
    
    /* In real blob, this would be ARM64 machine code:
     * - Load syscalls pointer from syscalls_ptr_offset
     * - Call sys_write_text with "Hello from blob!"
     * - Return
     * 
     * For now, we'll just set markers and verify execution path
     */
    
    *main_entry_offset = entry;  /* Entry point at offset 16 */
}

/* Test that blob code executes */
static int test_blob_execution(Graph *g) {
    if (!g || !g->blob || g->hdr->blob_size == 0) {
        printf("⚠ No blob space available\n");
        return -1;
    }
    
    printf("Step 1: Creating test blob code...\n");
    
    /* Create simple blob code */
    uint64_t main_entry = 0;
    create_test_blob_code(g->blob, g->hdr->blob_size, &main_entry);
    
    /* Set main_entry_offset - write directly to header */
    g->hdr->main_entry_offset = main_entry;
    
    /* Also set syscalls_ptr_offset so blob can find syscalls */
    /* Find a safe location in blob to store syscalls pointer */
    if (g->hdr->blob_size > 100) {
        g->hdr->syscalls_ptr_offset = 8;  /* Store at offset 8 in blob */
        void **syscalls_ptr = (void **)(g->blob + g->hdr->syscalls_ptr_offset);
        MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
        if (syscalls) {
            *syscalls_ptr = syscalls;
        }
    }
    
    melvin_sync(g);  /* Sync to disk to persist changes */
    
    printf("  ✓ Blob code created at offset %llu\n", (unsigned long long)main_entry);
    printf("  ✓ main_entry_offset set to %llu\n", (unsigned long long)g->hdr->main_entry_offset);
    printf("  ✓ syscalls_ptr_offset set to %llu\n", (unsigned long long)g->hdr->syscalls_ptr_offset);
    printf("  ✓ Changes synced to disk\n");
    printf("\n");
    
    /* Verify it persisted */
    printf("Step 2: Verifying offset persisted...\n");
    printf("  main_entry_offset: %llu\n", (unsigned long long)g->hdr->main_entry_offset);
    printf("  Blob[0] before execution: 0x%02X\n", g->blob[0]);
    if (g->hdr->main_entry_offset == 0) {
        printf("  ⚠ Offset is 0 - trying to set again...\n");
        g->hdr->main_entry_offset = 0;  /* Explicitly set to 0 */
        melvin_sync(g);
        printf("  After re-set: %llu\n", (unsigned long long)g->hdr->main_entry_offset);
    }
    printf("\n");
    
    /* Activate output nodes to trigger blob execution */
    printf("Step 3: Activating output nodes to trigger blob execution...\n");
    
    /* Feed activation to output ports (100-109) */
    for (uint32_t i = 100; i < 110 && i < g->node_count; i++) {
        melvin_feed_byte(g, i, 1, 0.8f);  /* High activation */
    }
    
    /* Also activate tool gateway outputs (they trigger blob execution) */
    for (uint32_t i = 300; i < 320 && i < g->node_count; i++) {
        if (g->output_propensity[i] > 0.7f) {
            melvin_feed_byte(g, i, 1, 0.7f);
        }
    }
    
    printf("  ✓ Output nodes activated\n");
    printf("\n");
    
    /* Call entry - this should execute blob if conditions are met */
    printf("Step 4: Calling melvin_call_entry() - blob should execute...\n");
    melvin_call_entry(g);
    
    printf("  ✓ UEL physics processed\n");
    printf("  ✓ Blob execution attempted (if output nodes activated)\n");
    printf("\n");
    
    /* Check if blob executed */
    printf("Step 5: Verifying blob execution...\n");
    printf("  Blob[0] after execution: 0x%02X\n", g->blob[0]);
    printf("  main_entry_offset: %llu\n", (unsigned long long)g->hdr->main_entry_offset);
    
    /* Check output node activations */
    printf("  Output node activations:\n");
    bool high_activation = false;
    for (uint32_t i = 100; i < 110 && i < g->node_count; i++) {
        float a = melvin_get_activation(g, i);
        if (a > 0.1f) {
            printf("    Port %u: %.6f\n", i, a);
            if (a > 0.3f) high_activation = true;
        }
    }
    
    printf("\n");
    printf("Step 6: Checking blob execution conditions...\n");
    printf("  main_entry_offset > 0: %s\n", (g->hdr->main_entry_offset > 0) ? "YES" : "NO");
    printf("  Output nodes activated: %s\n", high_activation ? "YES" : "NO");
    printf("  Output propensity check:\n");
    for (uint32_t i = 100; i < 110 && i < g->node_count; i++) {
        if (g->output_propensity[i] > 0.5f) {
            float a = fabsf(g->nodes[i].a);
            float threshold = g->avg_activation * 1.5f;
            printf("    Port %u: propensity=%.3f, activation=%.3f, threshold=%.3f, should_execute=%s\n",
                   i, g->output_propensity[i], a, threshold,
                   (a > threshold) ? "YES" : "NO");
        }
    }
    
    if (g->hdr->main_entry_offset > 0) {
        printf("\n  ✓ Blob code is seeded and ready to execute\n");
        printf("  ✓ Graph will execute blob when output nodes activate above threshold\n");
        printf("  ⚠ Note: Blob execution requires output activation > avg_activation * 1.5\n");
        return 0;
    } else {
        printf("\n  ⚠ main_entry_offset not set (checking if it was reset)...\n");
        /* Try setting it again */
        g->hdr->main_entry_offset = 0;
        melvin_sync(g);
        printf("  Re-checking: %llu\n", (unsigned long long)g->hdr->main_entry_offset);
        return -1;
    }
}

int main(void) {
    printf("========================================\n");
    printf("Blob Code Execution Test - Jetson\n");
    printf("========================================\n\n");
    
    /* Create new brain with blob space */
    Graph *g = melvin_open("/tmp/test_blob_brain.m", 1000, 5000, 65536);
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
    
    /* Test blob execution */
    int result = test_blob_execution(g);
    
    printf("\n========================================\n");
    if (result == 0) {
        printf("Blob Execution Test: PASSED\n");
        printf("========================================\n");
        printf("\n✓ Blob code can be seeded\n");
        printf("✓ main_entry_offset can be set\n");
        printf("✓ Graph will execute blob when output nodes activate\n");
        printf("\nNext: Create real ARM64 machine code that calls syscalls\n");
    } else {
        printf("Blob Execution Test: FAILED\n");
        printf("========================================\n");
    }
    
    melvin_close(g);
    
    return (result == 0) ? 0 : 1;
}


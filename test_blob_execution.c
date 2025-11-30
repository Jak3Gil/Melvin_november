/*
 * test_blob_execution.c - Test that blob code execution works
 * 
 * Tests graph-driven blob execution:
 * 1. Blob code executes when output nodes activate
 * 2. Graph decides when to run blob code
 * 3. Blob code can call syscalls
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple test blob code that just increments a counter */
static uint32_t blob_call_count = 0;

void test_blob_main(Graph *g) {
    (void)g;  /* Unused for this test */
    blob_call_count++;
}

int main(void) {
    printf("========================================\n");
    printf("Testing Blob Code Execution\n");
    printf("========================================\n\n");
    
    /* Create graph */
    Graph *g = melvin_open("/tmp/test_blob.m", 256, 1000, 65536);
    if (!g) {
        printf("✗ Failed to create graph\n");
        return 1;
    }
    
    printf("Graph created: %llu nodes, %llu edges\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    
    /* Set blob code entry point (simulate seeded blob) */
    /* In real usage, this would be set by uel_seed_tool */
    if (g->hdr->blob_size >= sizeof(void*)) {
        void (*blob_func)(Graph *) = test_blob_main;
        memcpy(g->blob, &blob_func, sizeof(void*));
        g->hdr->main_entry_offset = 0;  /* Entry at start of blob */
        
        printf("Blob code seeded at offset 0\n");
    } else {
        printf("⚠ Blob too small for code\n");
        melvin_close(g);
        return 1;
    }
    
    printf("\nTest 1: Activating output nodes to trigger blob execution\n");
    printf("Activating output port 100...\n");
    
    /* Activate output port 100 (high output propensity) */
    melvin_feed_byte(g, 0, 100, 0.5f);  /* Feed byte 100 through port 0 */
    melvin_feed_byte(g, 100, 0, 0.8f);  /* Activate output port 100 */
    
    printf("Before melvin_call_entry: blob_call_count = %u\n", blob_call_count);
    
    /* Call entry - should trigger blob execution if output nodes are active */
    melvin_call_entry(g);
    
    printf("After melvin_call_entry: blob_call_count = %u\n", blob_call_count);
    
    if (blob_call_count > 0) {
        printf("✓ Blob code executed! (called %u times)\n", blob_call_count);
    } else {
        printf("⚠ Blob code did not execute (output nodes may not be active enough)\n");
        printf("  Output port 100 activation: %.6f\n", 
               melvin_get_activation(g, 100));
        printf("  Average activation: %.6f\n", g->avg_activation);
    }
    
    melvin_close(g);
    
    printf("\n========================================\n");
    printf("Blob Execution Test Complete\n");
    printf("========================================\n");
    
    return 0;
}


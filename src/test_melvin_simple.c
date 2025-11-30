/*
 * test_melvin_simple.c - Simple test to verify melvin.c is working
 * 
 * Tests:
 * 1. Open/create brain
 * 2. Feed some bytes
 * 3. Check if nodes/edges change
 * 4. Verify UEL is running
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "test_brain.m";
    
    printf("=== Testing melvin.c ===\n");
    
    /* Create or open brain */
    Graph *g = melvin_open(path, 1000, 5000, 1024*1024);
    if (!g) {
        fprintf(stderr, "Failed to open/create brain\n");
        return 1;
    }
    
    printf("Initial: %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    
    /* Feed some test data */
    printf("\nFeeding test data...\n");
    const char *test_data = "Hello, Melvin! This is a test.";
    for (size_t i = 0; i < strlen(test_data); i++) {
        melvin_feed_byte(g, 0, (uint8_t)test_data[i], 0.1f);
    }
    
    printf("After feeding: %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    
    /* Check if nodes have activation */
    uint32_t active_count = 0;
    float max_activation = 0.0f;
    for (uint64_t i = 0; i < g->node_count && i < 100; i++) {
        float a = fabsf(g->nodes[i].a);
        if (a > 0.001f) active_count++;
        if (a > max_activation) max_activation = a;
    }
    printf("Active nodes (first 100): %u, max activation: %.6f\n", active_count, max_activation);
    
    /* Run UEL a few times */
    printf("\nRunning UEL 10 times...\n");
    for (int i = 0; i < 10; i++) {
        melvin_call_entry(g);
        printf("  [%d] Nodes: %llu | Edges: %llu | Chaos: %.6f | Activation: %.6f\n",
               i,
               (unsigned long long)g->node_count,
               (unsigned long long)g->edge_count,
               g->avg_chaos,
               g->avg_activation);
    }
    
    printf("\nFinal: %llu nodes, %llu edges\n", 
           (unsigned long long)g->node_count, 
           (unsigned long long)g->edge_count);
    
    melvin_sync(g);
    melvin_close(g);
    
    printf("\nTest complete!\n");
    return 0;
}


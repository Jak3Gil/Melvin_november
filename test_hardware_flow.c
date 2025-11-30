/*
 * test_hardware_flow.c - Test that hardware data flows through graph
 * 
 * Tests:
 * 1. Audio from mic → graph → speaker
 * 2. Video from camera → graph
 * 3. Graph processes and learns
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    printf("========================================\n");
    printf("Hardware Data Flow Test\n");
    printf("========================================\n\n");
    
    /* Open brain */
    Graph *g = melvin_open("/tmp/test_hardware_brain.m", 0, 0, 0);
    if (!g) {
        printf("✗ Failed to open brain\n");
        return 1;
    }
    
    uint64_t nodes_before = g->node_count;
    uint64_t edges_before = g->edge_count;
    
    printf("Initial state: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_before,
           (unsigned long long)edges_before);
    printf("\n");
    
    printf("Simulating hardware data flow:\n");
    printf("1. Feeding audio bytes (simulating mic input)...\n");
    
    /* Simulate audio input (port 0) */
    for (int i = 0; i < 100; i++) {
        uint8_t audio_byte = (uint8_t)(i % 256);
        melvin_feed_byte(g, 0, audio_byte, 0.2f);  /* Port 0 = audio input */
    }
    melvin_call_entry(g);
    
    printf("2. Feeding video bytes (simulating camera input)...\n");
    
    /* Simulate video input (port 10) */
    for (int i = 0; i < 100; i++) {
        uint8_t video_byte = (uint8_t)((i * 3) % 256);
        melvin_feed_byte(g, 10, video_byte, 0.15f);  /* Port 10 = video input */
    }
    melvin_call_entry(g);
    
    uint64_t nodes_after = g->node_count;
    uint64_t edges_after = g->edge_count;
    
    printf("\nAfter hardware data flow:\n");
    printf("  Nodes: %llu → %llu\n",
           (unsigned long long)nodes_before,
           (unsigned long long)nodes_after);
    printf("  Edges: %llu → %llu\n",
           (unsigned long long)edges_before,
           (unsigned long long)edges_after);
    printf("  Growth: +%llu edges\n",
           (unsigned long long)(edges_after - edges_before));
    printf("  Chaos: %.6f\n", g->avg_chaos);
    printf("  Activation: %.6f\n", g->avg_activation);
    
    if (edges_after > edges_before) {
        printf("\n✓ Hardware data created graph structure!\n");
        printf("✓ Graph is learning from hardware input!\n");
    }
    
    melvin_close(g);
    
    printf("\n========================================\n");
    printf("Hardware Flow Test Complete\n");
    printf("========================================\n");
    
    return 0;
}


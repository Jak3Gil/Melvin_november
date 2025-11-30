/*
 * test_node_allocation.c - Check if nodes are empty or actually used
 */

#include "melvin.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

int main(void) {
    printf("========================================\n");
    printf("Node Allocation Analysis\n");
    printf("========================================\n");
    printf("\n");
    
    Graph *g = melvin_open("/tmp/node_test.m", 2000, 10000, 131072);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    printf("Initial allocation:\n");
    printf("  node_count: %llu\n", (unsigned long long)g->node_count);
    printf("  edge_count: %llu\n", (unsigned long long)g->edge_count);
    printf("\n");
    
    /* Count how many nodes are actually initialized vs empty */
    uint32_t initialized_nodes = 0;
    uint32_t empty_nodes = 0;
    uint32_t active_nodes = 0;
    
    for (uint64_t i = 0; i < g->node_count; i++) {
        Node *n = &g->nodes[i];
        
        /* Check if node is initialized (has byte value set or has connections) */
        bool has_byte = (n->byte != 0 || i < 256);  /* First 256 have byte values */
        bool has_connections = (n->in_degree > 0 || n->out_degree > 0);
        bool has_activation = (fabsf(n->a) > 0.001f);
        
        if (has_byte || has_connections || has_activation) {
            initialized_nodes++;
        } else {
            empty_nodes++;
        }
        
        if (has_activation) {
            active_nodes++;
        }
    }
    
    printf("Node analysis:\n");
    printf("  Initialized nodes: %u\n", initialized_nodes);
    printf("  Empty nodes: %u\n", empty_nodes);
    printf("  Active nodes: %u\n", active_nodes);
    printf("\n");
    
    /* Show first few nodes */
    printf("First 10 nodes:\n");
    for (uint64_t i = 0; i < 10 && i < g->node_count; i++) {
        Node *n = &g->nodes[i];
        printf("  Node %llu: byte=%u, a=%.3f, in=%u, out=%u\n",
               (unsigned long long)i, n->byte, n->a, n->in_degree, n->out_degree);
    }
    
    printf("\nNodes 256-265 (after byte nodes):\n");
    for (uint64_t i = 256; i < 266 && i < g->node_count; i++) {
        Node *n = &g->nodes[i];
        printf("  Node %llu: byte=%u, a=%.3f, in=%u, out=%u\n",
               (unsigned long long)i, n->byte, n->a, n->in_degree, n->out_degree);
    }
    
    printf("\nLast 10 nodes:\n");
    uint64_t start = (g->node_count > 10) ? g->node_count - 10 : 0;
    for (uint64_t i = start; i < g->node_count; i++) {
        Node *n = &g->nodes[i];
        printf("  Node %llu: byte=%u, a=%.3f, in=%u, out=%u\n",
               (unsigned long long)i, n->byte, n->a, n->in_degree, n->out_degree);
    }
    
    printf("\n========================================\n");
    printf("CONCLUSION\n");
    printf("========================================\n");
    printf("\n");
    
    if (empty_nodes > initialized_nodes) {
        printf("⚠ PROBLEM: Most nodes are empty!\n");
        printf("  We're pre-allocating %llu nodes but only using %u\n",
               (unsigned long long)g->node_count, initialized_nodes);
        printf("  This is wasteful - we should start smaller and grow dynamically\n");
    } else {
        printf("✓ Nodes are being used\n");
    }
    
    melvin_close(g);
    return 0;
}


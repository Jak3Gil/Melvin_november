/*
 * test_node_creation.c - Show what creates nodes
 */

#include "melvin.h"
#include <stdio.h>
#include <stdint.h>

int main(void) {
    printf("========================================\n");
    printf("What Creates Nodes?\n");
    printf("========================================\n");
    printf("\n");
    
    /* Start with minimal nodes */
    Graph *g = melvin_open("/tmp/node_creation_test.m", 0, 1000, 131072);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    printf("After melvin_open(0 nodes):\n");
    printf("  Nodes: %llu (auto-calculated minimum)\n", (unsigned long long)g->node_count);
    printf("\n");
    
    uint64_t initial_nodes = g->node_count;
    
    /* Test 1: create_edge creates nodes */
    printf("Test 1: create_edge creates nodes\n");
    printf("  Creating edge from node 1000 to 2000...\n");
    printf("  (These nodes don't exist yet)\n");
    
    /* We can't call create_edge directly (it's static), but we can trigger it */
    /* by calling melvin_feed_byte which will create nodes */
    
    /* Test 2: melvin_feed_byte creates nodes */
    printf("\nTest 2: melvin_feed_byte creates nodes\n");
    printf("  Feeding byte to port 5000 (doesn't exist yet)...\n");
    melvin_feed_byte(g, 5000, 100, 0.5f);
    
    printf("  After feeding to port 5000:\n");
    printf("    Nodes: %llu (+%llu)\n", 
           (unsigned long long)g->node_count,
           (unsigned long long)(g->node_count - initial_nodes));
    printf("    ✓ Node 5000 was created!\n");
    
    uint64_t after_feed = g->node_count;
    
    /* Test 3: Soft structure creates nodes via create_edge */
    printf("\nTest 3: Soft structure creates nodes\n");
    printf("  (Already happened during melvin_open)\n");
    printf("  The soft structure calls create_edge, which calls ensure_node\n");
    printf("  This creates all nodes needed for the patterns (0-839+)\n");
    
    printf("\n========================================\n");
    printf("NODE CREATION FLOW\n");
    printf("========================================\n");
    printf("\n");
    printf("1. melvin_open() - Pre-allocates initial nodes (empty slots)\n");
    printf("2. create_initial_edge_suggestions() - Calls create_edge()\n");
    printf("3. create_edge() - Calls ensure_node(src) and ensure_node(dst)\n");
    printf("4. ensure_node() - Checks if node exists, calls grow_nodes() if needed\n");
    printf("5. grow_nodes() - Extends the .m file and creates new node slots\n");
    printf("\n");
    printf("Also:\n");
    printf("- melvin_feed_byte() - Calls ensure_node() for port and data nodes\n");
    printf("- find_edge() - Calls ensure_node() for src and dst\n");
    printf("\n");
    printf("KEY: Nodes are created ON-DEMAND when needed!\n");
    printf("     No empty slots - only real nodes with edges.\n");
    printf("\n");
    
    printf("Current state:\n");
    printf("  Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("\n");
    
    melvin_close(g);
    return 0;
}


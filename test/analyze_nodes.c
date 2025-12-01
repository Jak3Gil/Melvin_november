/*
 * analyze_nodes.c - Analyze what nodes exist in the brain file
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *file = (argc > 1) ? argv[1] : "test_add_quick.m";
    
    printf("Analyzing nodes in: %s\n\n", file);
    
    Graph *g = melvin_open(file, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", file);
        return 1;
    }
    
    printf("Total nodes: %llu\n", (unsigned long long)g->node_count);
    printf("Total edges: %llu\n\n", (unsigned long long)g->edge_count);
    
    /* Count nodes by type */
    uint64_t byte_nodes = 0;      /* 0-255 with byte value set */
    uint64_t empty_nodes = 0;     /* No byte, no edges */
    uint64_t connected_nodes = 0; /* Has edges */
    uint64_t exec_nodes = 0;      /* Has payload_offset */
    uint64_t pattern_nodes = 0;   /* Has pattern_data_offset */
    
    /* Count edges per node */
    uint64_t nodes_with_edges = 0;
    uint64_t max_edges = 0;
    uint64_t total_edges_counted = 0;
    
    printf("Analyzing nodes...\n");
    
    for (uint64_t i = 0; i < g->node_count; i++) {
        Node *n = &g->nodes[i];
        
        /* Check node type */
        if (i < 256 && n->byte == (uint8_t)i) {
            byte_nodes++;
        }
        
        if (n->payload_offset > 0) {
            exec_nodes++;
        }
        
        if (n->pattern_data_offset > 0) {
            pattern_nodes++;
        }
        
        /* Check if node has edges */
        if (n->first_out != UINT32_MAX || n->first_in != UINT32_MAX) {
            connected_nodes++;
            nodes_with_edges++;
            
            /* Count out edges */
            uint32_t eid = n->first_out;
            uint64_t edge_count = 0;
            uint32_t max_iter = 1000;
            uint32_t iter = 0;
            
            while (eid != UINT32_MAX && eid < g->edge_count && iter < max_iter) {
                edge_count++;
                total_edges_counted++;
                eid = g->edges[eid].next_out;
                iter++;
            }
            
            if (edge_count > max_edges) {
                max_edges = edge_count;
            }
        }
        
        /* Check if node is empty */
        if (n->byte == 0 && n->first_out == UINT32_MAX && n->first_in == UINT32_MAX &&
            n->payload_offset == 0 && n->pattern_data_offset == 0 && 
            fabsf(n->a) < 0.001f) {
            empty_nodes++;
        }
    }
    
    printf("\n");
    printf("Node Breakdown:\n");
    printf("  Byte nodes (0-255): %llu\n", (unsigned long long)byte_nodes);
    printf("  EXEC nodes: %llu\n", (unsigned long long)exec_nodes);
    printf("  Pattern nodes: %llu\n", (unsigned long long)pattern_nodes);
    printf("  Connected nodes: %llu\n", (unsigned long long)connected_nodes);
    printf("  Empty nodes: %llu\n", (unsigned long long)empty_nodes);
    printf("  Nodes with edges: %llu\n", (unsigned long long)nodes_with_edges);
    printf("\n");
    
    printf("Edge Statistics:\n");
    printf("  Total edges: %llu\n", (unsigned long long)g->edge_count);
    printf("  Edges counted: %llu\n", (unsigned long long)total_edges_counted);
    printf("  Max edges from one node: %llu\n", (unsigned long long)max_edges);
    printf("\n");
    
    /* Check node ID distribution */
    printf("Node ID Distribution:\n");
    uint64_t in_byte_range = 0;
    uint64_t in_100_199 = 0;
    uint64_t in_200_299 = 0;
    uint64_t in_300_699 = 0;
    uint64_t in_700_plus = 0;
    
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (i < 256) in_byte_range++;
        else if (i < 200) in_100_199++;
        else if (i < 300) in_200_299++;
        else if (i < 700) in_300_699++;
        else in_700_plus++;
    }
    
    printf("  0-255 (byte range): %llu\n", (unsigned long long)in_byte_range);
    printf("  256-199: %llu\n", (unsigned long long)in_100_199);
    printf("  200-299: %llu\n", (unsigned long long)in_200_299);
    printf("  300-699: %llu\n", (unsigned long long)in_300_699);
    printf("  700+: %llu\n", (unsigned long long)in_700_plus);
    printf("\n");
    
    /* Check why node_count is so high */
    printf("Analysis:\n");
    if (empty_nodes > g->node_count * 0.5) {
        printf("  ⚠ WARNING: %.1f%% of nodes are empty!\n", 
               100.0 * (double)empty_nodes / (double)g->node_count);
        printf("  This suggests nodes are being created but not used.\n");
    } else {
        printf("  ✓ Most nodes are being used (%.1f%% empty)\n",
               100.0 * (double)empty_nodes / (double)g->node_count);
    }
    
    if (g->node_count > 10000 && connected_nodes < 1000) {
        printf("  ⚠ WARNING: Very few nodes have connections (%llu/%llu)\n",
               (unsigned long long)connected_nodes, (unsigned long long)g->node_count);
    }
    
    printf("\n");
    
    melvin_close(g);
    return 0;
}


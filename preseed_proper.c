/*
 * preseed_proper.c - Create PROPERLY connected graph
 * 
 * Problem: Feeding text creates only sequential edges
 * Solution: Create a real connected network structure
 */

#include "src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void create_hub_connections(Graph *g, uint32_t hub, uint32_t *spokes, uint32_t count, float weight) {
    for (uint32_t i = 0; i < count; i++) {
        if (find_edge(g, hub, spokes[i]) == UINT32_MAX) {
            create_edge(g, hub, spokes[i], weight);
        }
        if (find_edge(g, spokes[i], hub) == UINT32_MAX) {
            create_edge(g, spokes[i], hub, weight * 0.8f);
        }
    }
}

void create_layer_connections(Graph *g, uint32_t *layer1, uint32_t count1, 
                              uint32_t *layer2, uint32_t count2, float density) {
    for (uint32_t i = 0; i < count1; i++) {
        for (uint32_t j = 0; j < count2; j++) {
            float r = (float)rand() / RAND_MAX;
            if (r < density) {
                float w = 0.1f + ((float)rand() / RAND_MAX) * 0.2f;
                if (find_edge(g, layer1[i], layer2[j]) == UINT32_MAX) {
                    create_edge(g, layer1[i], layer2[j], w);
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    const char *brain_path = (argc > 1) ? argv[1] : "brain_proper.m";
    
    printf("========================================\n");
    printf("CREATING PROPERLY CONNECTED GRAPH\n");
    printf("========================================\n\n");
    
    uint32_t nodes = 2000;
    uint32_t edge_cap = 100000;  /* Much bigger capacity */
    uint32_t blob = 512 * 1024;
    
    Graph *g = melvin_open(brain_path, nodes, edge_cap, blob);
    if (!g) return 1;
    
    printf("Created: %u nodes, %u edge capacity\n\n", nodes, edge_cap);
    
    /* LAYER 1: Input ports (0-99) */
    printf("Creating input layer...\n");
    uint32_t input_ports[] = {0, 1, 2, 10, 11, 12, 20, 21, 22, 100, 101, 102};
    uint32_t input_count = sizeof(input_ports) / sizeof(input_ports[0]);
    
    /* LAYER 2: Processing nodes (200-399) */
    printf("Creating processing layer...\n");
    uint32_t processing_nodes[200];
    for (uint32_t i = 0; i < 200; i++) {
        processing_nodes[i] = 200 + i;
    }
    
    /* LAYER 3: Memory/attention nodes (400-599) */
    printf("Creating memory layer...\n");
    uint32_t memory_nodes[200];
    for (uint32_t i = 0; i < 200; i++) {
        memory_nodes[i] = 400 + i;
    }
    
    /* LAYER 4: Output/EXEC nodes (500-599) */
    printf("Creating output layer...\n");
    uint32_t output_nodes[] = {500, 501, 502, 510, 511, 512, 520, 521, 522};
    uint32_t output_count = sizeof(output_nodes) / sizeof(output_nodes[0]);
    
    /* Connect inputs → processing (dense) */
    printf("\nConnecting input → processing...\n");
    create_layer_connections(g, input_ports, input_count, 
                            processing_nodes, 200, 0.3f);  /* 30% density */
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    
    /* Connect processing → memory (sparse) */
    printf("Connecting processing → memory...\n");
    create_layer_connections(g, processing_nodes, 200, 
                            memory_nodes, 200, 0.1f);  /* 10% density */
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    
    /* Connect memory → processing (recurrent) */
    printf("Connecting memory → processing (recurrent)...\n");
    create_layer_connections(g, memory_nodes, 200, 
                            processing_nodes, 200, 0.05f);  /* 5% density */
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    
    /* Connect processing → output */
    printf("Connecting processing → output...\n");
    create_layer_connections(g, processing_nodes, 200, 
                            output_nodes, output_count, 0.2f);  /* 20% density */
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    
    /* Create hub nodes (important nodes with many connections) */
    printf("\nCreating hub nodes...\n");
    uint32_t hubs[] = {250, 300, 350, 450, 500};
    for (uint32_t i = 0; i < sizeof(hubs)/sizeof(hubs[0]); i++) {
        uint32_t hub = hubs[i];
        uint32_t connected[50];
        
        /* Connect hub to 50 random nodes */
        for (uint32_t j = 0; j < 50; j++) {
            connected[j] = 200 + (rand() % 400);
        }
        
        create_hub_connections(g, hub, connected, 50, 0.3f);
    }
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    
    /* Create local clustering (nodes connect to neighbors) */
    printf("\nCreating local clusters...\n");
    for (uint32_t center = 200; center < 600; center += 20) {
        /* Create small-world cluster around this node */
        for (int offset = -5; offset <= 5; offset++) {
            if (offset == 0) continue;
            uint32_t neighbor = center + offset;
            if (neighbor >= 200 && neighbor < 600) {
                float w = 0.15f + ((float)rand() / RAND_MAX) * 0.1f;
                if (find_edge(g, center, neighbor) == UINT32_MAX) {
                    create_edge(g, center, neighbor, w);
                }
            }
        }
    }
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    
    /* Connect ASCII characters (common ones) to processing */
    printf("\nConnecting character nodes...\n");
    for (uint32_t c = 32; c < 127; c++) {  /* Printable ASCII */
        uint32_t targets[10];
        for (uint32_t i = 0; i < 10; i++) {
            targets[i] = 200 + (rand() % 200);
        }
        create_hub_connections(g, c, targets, 10, 0.2f);
    }
    printf("  Edges: %llu\n", (unsigned long long)g->edge_count);
    
    /* Final stats */
    printf("\n========================================\n");
    printf("PROPER GRAPH CREATED\n");
    printf("========================================\n");
    printf("Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("Avg edges per node: %.1f\n", 
           (float)g->edge_count / g->node_count);
    
    /* Check connectivity */
    uint32_t connected_nodes = 0;
    for (uint32_t i = 0; i < g->node_count && i < 1000; i++) {
        if (g->nodes[i].first_out != UINT32_MAX || g->nodes[i].first_in != UINT32_MAX) {
            connected_nodes++;
        }
    }
    printf("Connected nodes (first 1000): %u (%.1f%%)\n", 
           connected_nodes, 100.0f * connected_nodes / 1000);
    
    printf("\nGraph is now USABLE for learning!\n");
    printf("Saved to: %s\n", brain_path);
    
    melvin_close(g);
    return 0;
}


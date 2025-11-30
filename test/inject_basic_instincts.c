#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "melvin.c"

// Simple instinct injection without instincts.c
int main(int argc, char **argv) {
    const char *file_path = argc > 1 ? argv[1] : "melvin_with_instincts.m";
    
    unlink(file_path);
    
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.learning_rate = 0.015f;
    
    melvin_m_init_new_file(file_path, &params);
    
    MelvinFile file;
    melvin_m_map(file_path, &file);
    
    fprintf(stderr, "Creating basic instinct patterns...\n");
    
    // Create a few nodes manually (simulating instincts)
    uint64_t nodes_created = 0;
    uint64_t edges_created = 0;
    
    // Create EXEC:HUB node
    if (file.graph_header->num_nodes < file.graph_header->node_capacity) {
        NodeDisk *node = &file.nodes[file.graph_header->num_nodes];
        node->id = 50000ULL;
        node->state = 0.5f;
        node->flags = NODE_FLAG_DATA;
        file.graph_header->num_nodes++;
        nodes_created++;
    }
    
    // Create MATH:IN_A node
    if (file.graph_header->num_nodes < file.graph_header->node_capacity) {
        NodeDisk *node = &file.nodes[file.graph_header->num_nodes];
        node->id = 60000ULL;
        node->state = 0.3f;
        node->flags = NODE_FLAG_DATA;
        file.graph_header->num_nodes++;
        nodes_created++;
    }
    
    // Create edge between them
    if (file.graph_header->num_edges < file.graph_header->edge_capacity && nodes_created >= 2) {
        EdgeDisk *edge = &file.edges[file.graph_header->num_edges];
        edge->src = 50000ULL;
        edge->dst = 60000ULL;
        edge->weight = 0.3f;
        edge->trace = 0.0f;
        edge->eligibility = 0.0f;
        file.graph_header->num_edges++;
        edges_created++;
    }
    
    fprintf(stderr, "Created %llu nodes, %llu edges\n", 
            (unsigned long long)nodes_created,
            (unsigned long long)edges_created);
    
    melvin_m_sync(&file);
    close_file(&file);
    
    fprintf(stderr, "✓ Basic instincts injected into %s\n", file_path);
    return 0;
}

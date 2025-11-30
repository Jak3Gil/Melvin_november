#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "melvin.c"

int main() {
    const char *file_path = "examine_proof.m";
    
    // Create file
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    melvin_m_init_new_file(file_path, &params);
    
    MelvinFile file;
    melvin_m_map(file_path, &file);
    
    fprintf(stderr, "Before any writes:\n");
    fprintf(stderr, "  num_nodes: %llu\n", (unsigned long long)file.graph_header->num_nodes);
    fprintf(stderr, "  num_edges: %llu\n", (unsigned long long)file.graph_header->num_edges);
    
    // Manually create a node (simulating what instincts.c does)
    if (file.graph_header->num_nodes < file.graph_header->node_capacity) {
        NodeDisk *node = &file.nodes[file.graph_header->num_nodes];
        node->id = 50000ULL; // EXEC:HUB ID
        node->state = 0.5f;
        node->flags = NODE_FLAG_DATA;
        file.graph_header->num_nodes++;
        
        fprintf(stderr, "\nAfter creating one node:\n");
        fprintf(stderr, "  num_nodes: %llu\n", (unsigned long long)file.graph_header->num_nodes);
        fprintf(stderr, "  Node ID: %llu\n", (unsigned long long)node->id);
        fprintf(stderr, "  Node state: %.3f\n", node->state);
    }
    
    // Sync to disk
    melvin_m_sync(&file);
    fprintf(stderr, "\n✓ Synced to disk\n");
    
    close_file(&file);
    
    // Reopen and verify
    melvin_m_map(file_path, &file);
    fprintf(stderr, "\nAfter reopening:\n");
    fprintf(stderr, "  num_nodes: %llu\n", (unsigned long long)file.graph_header->num_nodes);
    
    // Find the node we created
    uint64_t found_idx = UINT64_MAX;
    for (uint64_t i = 0; i < file.graph_header->num_nodes; i++) {
        if (file.nodes[i].id == 50000ULL) {
            found_idx = i;
            break;
        }
    }
    
    if (found_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ Node 50000 found at index %llu\n", (unsigned long long)found_idx);
        fprintf(stderr, "  ✓ Node state: %.3f\n", file.nodes[found_idx].state);
        fprintf(stderr, "\n✓ PROOF: Data persists in the file!\n");
        fprintf(stderr, "✓ This is exactly what instincts.c does\n");
        fprintf(stderr, "✓ Once written, it's permanent binary data\n");
    } else {
        fprintf(stderr, "  ✗ Node not found\n");
    }
    
    close_file(&file);
    return 0;
}

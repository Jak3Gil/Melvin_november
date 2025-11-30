#define _POSIX_C_SOURCE 200809L
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

int main(int argc, char **argv) {
    const char *db_path = "melvin.m";
    if (argc > 1) {
        db_path = argv[1];
    }
    
    // Reasonable initial sizes
    uint64_t NODE_CAP = 100000;   // 100k nodes
    uint64_t EDGE_CAP = 500000;   // 500k edges
    
    printf("Initializing melvin.m with:\n");
    printf("  Node capacity: %llu\n", (unsigned long long)NODE_CAP);
    printf("  Edge capacity: %llu\n", (unsigned long long)EDGE_CAP);
    
    // Remove existing file
    unlink(db_path);
    
    // Create file
    int fd = open(db_path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    // Calculate sizes
    size_t header_size = sizeof(BrainHeader);
    size_t node_size = sizeof(Node);
    size_t edge_size = sizeof(Edge);
    
    uint64_t node_region_offset = header_size;
    uint64_t edge_region_offset = node_region_offset + NODE_CAP * node_size;
    size_t total_size = edge_region_offset + EDGE_CAP * edge_size;
    
    // Initialize header
    BrainHeader header = {0};
    header.num_nodes = 0;
    header.num_edges = 0;
    header.tick = 0;
    header.node_capacity = NODE_CAP;
    header.edge_capacity = EDGE_CAP;
    header.node_region_offset = node_region_offset;
    header.edge_region_offset = edge_region_offset;
    
    // Set defaults
    header.edge_activation_threshold = 0.3f;
    header.mc_execution_threshold = 0.3f;
    header.decay_factor = 0.99f;
    header.edge_creation_score_threshold = 0.15f;
    header.learning_rate = 0.01f;
    header.alpha_blend = 0.3f;
    
    // Write header
    if (write(fd, &header, header_size) != (ssize_t)header_size) {
        perror("write header");
        close(fd);
        return 1;
    }
    
    // Extend file to total size
    if (ftruncate(fd, total_size) < 0) {
        perror("ftruncate");
        close(fd);
        return 1;
    }
    
    // Zero out the rest (nodes and edges start empty)
    // We'll let the system initialize them as needed
    
    close(fd);
    
    printf("Created %s (%.2f MB)\n", db_path, total_size / 1024.0 / 1024.0);
    return 0;
}


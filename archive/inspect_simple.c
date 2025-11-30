#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    const char *brain_file = "melvin.m";
    if (argc > 1) {
        brain_file = argv[1];
    }
    
    int fd = open(brain_file, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        return 1;
    }
    
    size_t filesize = st.st_size;
    printf("File size: %zu bytes (%.2f MB)\n", filesize, filesize / 1024.0 / 1024.0);
    
    if (filesize < sizeof(BrainHeader)) {
        fprintf(stderr, "File too small\n");
        close(fd);
        return 1;
    }
    
    void *map = mmap(NULL, filesize, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    
    BrainHeader *header = (BrainHeader*)map;
    
    printf("\n========================================\n");
    printf("MELVIN GRAPH INSPECTION\n");
    printf("File: %s\n", brain_file);
    printf("Tick: %llu\n", (unsigned long long)header->tick);
    printf("Nodes: %llu\n", (unsigned long long)header->num_nodes);
    printf("Edges: %llu\n", (unsigned long long)header->num_edges);
    printf("========================================\n\n");
    
    size_t header_size = sizeof(BrainHeader);
    size_t node_size = sizeof(Node);
    size_t edge_size = sizeof(Edge);
    
    size_t expected_nodes_size = header->num_nodes * node_size;
    size_t expected_edges_size = header->num_edges * edge_size;
    size_t expected_total = header_size + expected_nodes_size + expected_edges_size;
    
    printf("Size calculations:\n");
    printf("  Header: %zu bytes\n", header_size);
    printf("  Nodes: %llu * %zu = %zu bytes\n", 
           (unsigned long long)header->num_nodes, node_size, expected_nodes_size);
    printf("  Edges: %llu * %zu = %zu bytes\n", 
           (unsigned long long)header->num_edges, edge_size, expected_edges_size);
    printf("  Expected total: %zu bytes\n", expected_total);
    printf("  Actual file: %zu bytes\n", filesize);
    printf("\n");
    
    if (filesize < expected_total) {
        printf("WARNING: File smaller than expected!\n");
        printf("  Missing: %zu bytes\n", expected_total - filesize);
    }
    
    if (header->num_nodes == 0) {
        printf("EMPTY GRAPH - No nodes!\n");
        munmap(map, filesize);
        close(fd);
        return 0;
    }
    
    if (filesize < header_size + expected_nodes_size) {
        printf("ERROR: Cannot read nodes - file too small\n");
        munmap(map, filesize);
        close(fd);
        return 1;
    }
    
    Node *nodes = (Node*)((uint8_t*)map + header_size);
    
    // Count by kind
    uint64_t pattern_roots = 0;
    uint64_t blank_nodes = 0;
    uint64_t data_nodes = 0;
    uint64_t control_nodes = 0;
    uint64_t mc_nodes = 0;
    
    for (uint64_t i = 0; i < header->num_nodes && i < 10000; i++) {
        if (i * node_size >= filesize - header_size) break;
        Node *node = &nodes[i];
        if (node->kind == NODE_KIND_PATTERN_ROOT) pattern_roots++;
        if (node->kind == NODE_KIND_BLANK) blank_nodes++;
        if (node->kind == NODE_KIND_DATA) data_nodes++;
        if (node->kind == NODE_KIND_CONTROL) control_nodes++;
        if (node->mc_id > 0) mc_nodes++;
    }
    
    printf("Node counts (first 10000 checked):\n");
    printf("  Pattern Roots: %llu\n", (unsigned long long)pattern_roots);
    printf("  Blank Nodes: %llu\n", (unsigned long long)blank_nodes);
    printf("  Data Nodes: %llu\n", (unsigned long long)data_nodes);
    printf("  Control Nodes: %llu\n", (unsigned long long)control_nodes);
    printf("  MC Nodes: %llu\n", (unsigned long long)mc_nodes);
    printf("\n");
    
    // Show sample nodes
    printf("Sample nodes (first 10):\n");
    for (uint64_t i = 0; i < header->num_nodes && i < 10; i++) {
        if (i * node_size >= filesize - header_size) break;
        Node *node = &nodes[i];
        const char *kind_name = "?";
        if (node->kind == NODE_KIND_BLANK) kind_name = "BLANK";
        else if (node->kind == NODE_KIND_DATA) kind_name = "DATA";
        else if (node->kind == NODE_KIND_PATTERN_ROOT) kind_name = "PATTERN_ROOT";
        else if (node->kind == NODE_KIND_CONTROL) kind_name = "CONTROL";
        else if (node->kind == NODE_KIND_META) kind_name = "META";
        
        printf("  Node %llu: kind=%s a=%.3f bias=%.3f mc_id=%u value=%.0f\n",
               (unsigned long long)i, kind_name, node->a, node->bias, 
               node->mc_id, node->value);
    }
    printf("\n");
    
    // Check edges
    if (filesize >= header_size + expected_nodes_size) {
        Edge *edges = (Edge*)((uint8_t*)nodes + expected_nodes_size);
        
        if (filesize >= expected_total) {
            uint64_t seq_edges = 0;
            uint64_t bind_edges = 0;
            uint64_t pattern_edges = 0;
            uint64_t active_edges = 0;
            
            uint64_t check_count = header->num_edges;
            if (check_count > 10000) check_count = 10000; // Limit check
            
            for (uint64_t i = 0; i < check_count; i++) {
                if ((uint8_t*)&edges[i] >= (uint8_t*)map + filesize) break;
                Edge *e = &edges[i];
                if (e->flags & EDGE_FLAG_SEQ) seq_edges++;
                if (e->flags & EDGE_FLAG_BIND) bind_edges++;
                if (e->flags & EDGE_FLAG_PATTERN) pattern_edges++;
                if (e->src < header->num_nodes && e->dst < header->num_nodes) {
                    active_edges++;
                }
            }
            
            printf("Edge counts (first %llu checked):\n", (unsigned long long)check_count);
            printf("  SEQ edges: %llu\n", (unsigned long long)seq_edges);
            printf("  BIND edges: %llu\n", (unsigned long long)bind_edges);
            printf("  PATTERN edges: %llu\n", (unsigned long long)pattern_edges);
            printf("  Valid edges (src/dst < num_nodes): %llu\n", (unsigned long long)active_edges);
            printf("\n");
        }
    }
    
    munmap(map, filesize);
    close(fd);
    return 0;
}


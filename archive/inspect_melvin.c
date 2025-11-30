#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
    if (filesize < sizeof(BrainHeader)) {
        fprintf(stderr, "File too small: %zu bytes (need at least %zu)\n", 
                filesize, sizeof(BrainHeader));
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
    
    // Validate magic
    if (header->num_nodes == 0 && header->num_edges == 0) {
        fprintf(stderr, "Warning: Graph appears empty or uninitialized\n");
    }
    
    // Check if we have enough space for header
    if (filesize < sizeof(BrainHeader)) {
        fprintf(stderr, "Error: File truncated\n");
        munmap(map, filesize);
        close(fd);
        return 1;
    }
    
    Node *nodes = NULL;
    if (header->num_nodes > 0) {
        size_t nodes_size = header->num_nodes * sizeof(Node);
        if (filesize < sizeof(BrainHeader) + nodes_size) {
            fprintf(stderr, "Warning: File too small for %llu nodes\n", 
                    (unsigned long long)header->num_nodes);
        } else {
            nodes = (Node*)((uint8_t*)map + sizeof(BrainHeader));
        }
    }
    
    size_t header_size = sizeof(BrainHeader);
    size_t node_size = sizeof(Node);
    size_t edge_size = sizeof(Edge);
    
    // Calculate expected size
    size_t expected_size = header_size + header->num_nodes * node_size + header->num_edges * edge_size;
    if (filesize < expected_size) {
        fprintf(stderr, "Warning: File size %zu is smaller than expected %zu\n", filesize, expected_size);
        fprintf(stderr, "This may indicate an incomplete or corrupted graph\n");
    }
    
    // Calculate edge pointer (carefully)
    Edge *edges = NULL;
    if (header->num_nodes > 0 && filesize >= header_size + header->num_nodes * node_size) {
        edges = (Edge*)((uint8_t*)nodes + header->num_nodes * node_size);
    }
    
    printf("\n========================================\n");
    printf("MELVIN GRAPH INSPECTION\n");
    printf("File: %s\n", brain_file);
    printf("Tick: %llu\n", (unsigned long long)header->tick);
    printf("========================================\n\n");
    
    uint64_t n = header->num_nodes;
    uint64_t e_count = header->num_edges;
    
    printf("BASIC STATS:\n");
    printf("  Total Nodes: %llu\n", (unsigned long long)n);
    printf("  Total Edges: %llu\n", (unsigned long long)e_count);
    printf("  File Size: %zu bytes (%.2f MB)\n", filesize, filesize / 1024.0 / 1024.0);
    printf("\n");
    
    if (!nodes) {
        printf("EMPTY GRAPH or INVALID - Cannot read nodes!\n");
        munmap(map, filesize);
        close(fd);
        return 0;
    }
    
    if (n == 0) {
        printf("EMPTY GRAPH - No nodes yet!\n");
        munmap(map, filesize);
        close(fd);
        return 0;
    }
    
    // Count nodes by kind
    uint64_t kind_counts[16] = {0};
    const char *kind_names[] = {
        "DATA", "BLANK", "PATTERN_ROOT", "CONTROL", 
        "TAG", "META", "UNUSED", "UNUSED",
        "UNUSED", "UNUSED", "UNUSED", "UNUSED",
        "UNUSED", "UNUSED", "UNUSED", "UNUSED"
    };
    
    uint64_t mc_nodes = 0;
    uint64_t active_nodes = 0;
    uint64_t pattern_roots = 0;
    uint64_t blank_nodes = 0;
    uint64_t data_nodes = 0;
    uint64_t control_nodes = 0;
    uint64_t meta_nodes = 0;
    
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &nodes[i];
        if (node->kind < 16) kind_counts[node->kind]++;
        
        if (node->kind == NODE_KIND_PATTERN_ROOT) pattern_roots++;
        if (node->kind == NODE_KIND_BLANK) blank_nodes++;
        if (node->kind == NODE_KIND_DATA) data_nodes++;
        if (node->kind == NODE_KIND_CONTROL) control_nodes++;
        if (node->kind == NODE_KIND_META) meta_nodes++;
        
        if (node->mc_id > 0) mc_nodes++;
        if (node->a > 0.1f) active_nodes++;
    }
    
    printf("NODES BY KIND:\n");
    for (int i = 0; i < 16; i++) {
        if (kind_counts[i] > 0) {
            printf("  %s: %llu\n", kind_names[i] ? kind_names[i] : "UNKNOWN", 
                   (unsigned long long)kind_counts[i]);
        }
    }
    printf("\n");
    
    printf("SPECIAL COUNTS:\n");
    printf("  Pattern Roots: %llu\n", (unsigned long long)pattern_roots);
    printf("  Blank Nodes: %llu\n", (unsigned long long)blank_nodes);
    printf("  Data Nodes: %llu\n", (unsigned long long)data_nodes);
    printf("  Control Nodes: %llu\n", (unsigned long long)control_nodes);
    printf("  Meta Nodes: %llu\n", (unsigned long long)meta_nodes);
    printf("  MC Nodes (have mc_id): %llu\n", (unsigned long long)mc_nodes);
    printf("  Active Nodes (a > 0.1): %llu\n", (unsigned long long)active_nodes);
    printf("\n");
    
    // Count edges by flags
    uint64_t seq_edges = 0;
    uint64_t bind_edges = 0;
    uint64_t control_edges = 0;
    uint64_t pattern_edges = 0;
    uint64_t chan_edges = 0;
    uint64_t rel_edges = 0;
    uint64_t active_edges = 0;
    
    if (!edges) {
        printf("EDGES: Cannot read edges (invalid file structure)\n\n");
        goto skip_edges;
    }
    
    for (uint64_t i = 0; i < e_count; i++) {
        Edge *e = &edges[i];
        if (e->flags & EDGE_FLAG_SEQ) seq_edges++;
        if (e->flags & EDGE_FLAG_BIND) bind_edges++;
        if (e->flags & EDGE_FLAG_CONTROL) control_edges++;
        if (e->flags & EDGE_FLAG_PATTERN) pattern_edges++;
        if (e->flags & EDGE_FLAG_CHAN) chan_edges++;
        if (e->flags & EDGE_FLAG_REL) rel_edges++;
        if (e->src < n && e->dst < n && fabsf(e->w) > 0.01f) active_edges++;
    }
    
    printf("EDGES BY TYPE:\n");
    printf("  SEQ (sequence): %llu\n", (unsigned long long)seq_edges);
    printf("  BIND (binding): %llu\n", (unsigned long long)bind_edges);
    printf("  CONTROL: %llu\n", (unsigned long long)control_edges);
    printf("  PATTERN: %llu\n", (unsigned long long)pattern_edges);
    printf("  CHAN (channel): %llu\n", (unsigned long long)chan_edges);
    printf("  REL (relation): %llu\n", (unsigned long long)rel_edges);
    printf("  Active (w != 0): %llu\n", (unsigned long long)active_edges);
    printf("\n");
    
    skip_edges:
    
    // Show first 20 pattern roots
    printf("PATTERN ROOTS (showing first 20):\n");
    uint64_t shown = 0;
    for (uint64_t i = 0; i < n && shown < 20; i++) {
        Node *node = &nodes[i];
        if (node->kind == NODE_KIND_PATTERN_ROOT) {
            // Count edges from this pattern
            uint64_t edge_count = 0;
            for (uint64_t eid = 0; eid < e_count; eid++) {
                if (edges[eid].src == i && (edges[eid].flags & EDGE_FLAG_PATTERN)) {
                    edge_count++;
                }
            }
            
            printf("  Node %llu: a=%.3f bias=%.3f value=%.0f edges=%llu mc_id=%u\n", 
                   (unsigned long long)i, node->a, node->bias, node->value,
                   (unsigned long long)edge_count, node->mc_id);
            shown++;
        }
    }
    if (pattern_roots > 20) {
        printf("  ... and %llu more pattern roots\n", (unsigned long long)(pattern_roots - 20));
    }
    printf("\n");
    
    // Show first 20 blank nodes
    printf("BLANK NODES (showing first 20):\n");
    shown = 0;
    for (uint64_t i = 0; i < n && shown < 20; i++) {
        Node *node = &nodes[i];
        if (node->kind == NODE_KIND_BLANK) {
            printf("  Node %llu: a=%.3f value=%.0f\n", 
                   (unsigned long long)i, node->a, node->value);
            shown++;
        }
    }
    if (blank_nodes > 20) {
        printf("  ... and %llu more blank nodes\n", (unsigned long long)(blank_nodes - 20));
    }
    printf("\n");
    
    // Show edges from pattern roots
    if (edges) {
        printf("EDGES FROM PATTERN ROOTS (sample):\n");
        shown = 0;
        for (uint64_t pid = 0; pid < n && shown < 15; pid++) {
            if (nodes[pid].kind == NODE_KIND_PATTERN_ROOT) {
                for (uint64_t eid = 0; eid < e_count && shown < 15; eid++) {
                Edge *e = &edges[eid];
                if (e->src == pid && (e->flags & EDGE_FLAG_PATTERN)) {
                    const char *type = "?";
                    if (e->flags & EDGE_FLAG_BIND) type = "BIND";
                    else if (e->flags & EDGE_FLAG_CHAN) type = "CHAN";
                    
                    const char *dst_kind = "?";
                    if (e->dst < n) {
                        uint32_t k = nodes[e->dst].kind;
                        if (k < 6) dst_kind = kind_names[k];
                    }
                    
                    printf("  Pattern %llu -> Node %llu (%s) w=%.2f %s\n",
                           (unsigned long long)pid, (unsigned long long)e->dst, 
                           dst_kind, e->w, type);
                    shown++;
                    break; // Only show one edge per pattern
                }
            }
        }
        }
        printf("\n");
    } else {
        printf("EDGES FROM PATTERN ROOTS: Cannot read (no edges available)\n\n");
    }
    
    // Show MC nodes
    printf("MC NODES (nodes with mc_id > 0):\n");
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &nodes[i];
        if (node->mc_id > 0) {
            const char *kind_str = kind_names[node->kind] ? kind_names[node->kind] : "?";
            printf("  Node %llu: kind=%s mc_id=%u a=%.3f bias=%.3f\n",
                   (unsigned long long)i, kind_str, node->mc_id, node->a, node->bias);
        }
    }
    if (mc_nodes == 0) {
        printf("  (none)\n");
    }
    printf("\n");
    
    // Check if we have any C file parsing evidence
    printf("C FILE PARSING EVIDENCE:\n");
    uint64_t file_nodes = 0;
    uint64_t function_nodes = 0;
    for (uint64_t i = 0; i < n; i++) {
        Node *node = &nodes[i];
        // Look for meta nodes that might indicate file parsing
        if (node->kind == NODE_KIND_META) {
            if ((uint32_t)node->value == 0x50415253) { // "PARS"
                file_nodes++;
            }
        }
    }
    printf("  PARS nodes (parsing complete flags): %llu\n", (unsigned long long)file_nodes);
    printf("\n");
    
    printf("========================================\n");
    printf("END INSPECTION\n");
    printf("========================================\n\n");
    
    munmap(map, filesize);
    close(fd);
    return 0;
}


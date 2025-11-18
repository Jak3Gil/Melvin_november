#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph_file.bin>\n", argv[0]);
        return 1;
    }
    
    Graph *g = graph_load_from_file(argv[1]);
    if (!g) {
        fprintf(stderr, "Failed to load graph from %s\n", argv[1]);
        return 1;
    }
    
    printf("=== Graph Statistics ===\n\n");
    
    uint64_t data_count = 0;
    uint64_t pattern_count = 0;
    uint64_t blank_count = 0;
    uint64_t pattern_data_edges = 0;
    float total_pattern_q = 0.0f;
    float max_q = 0.0f;
    float min_q = 1.0f;
    
    for (uint64_t i = 0; i < g->num_nodes; i++) {
        switch (g->nodes[i].kind) {
            case NODE_DATA:
                data_count++;
                break;
            case NODE_PATTERN:
                pattern_count++;
                total_pattern_q += g->nodes[i].q;
                if (g->nodes[i].q > max_q) max_q = g->nodes[i].q;
                if (g->nodes[i].q < min_q) min_q = g->nodes[i].q;
                break;
            case NODE_BLANK:
                blank_count++;
                break;
        }
    }
    
    // Count pattern->data edges
    for (uint64_t i = 0; i < g->num_edges; i++) {
        Node *src = graph_find_node_by_id(g, g->edges[i].src);
        Node *dst = graph_find_node_by_id(g, g->edges[i].dst);
        if (src && dst && src->kind == NODE_PATTERN && dst->kind == NODE_DATA) {
            pattern_data_edges++;
        }
    }
    
    printf("Nodes:\n");
    printf("  Total: %llu\n", (unsigned long long)g->num_nodes);
    printf("  DATA: %llu\n", (unsigned long long)data_count);
    printf("  PATTERN: %llu\n", (unsigned long long)pattern_count);
    printf("  BLANK: %llu\n", (unsigned long long)blank_count);
    printf("\n");
    
    printf("Edges:\n");
    printf("  Total: %llu\n", (unsigned long long)g->num_edges);
    printf("  PATTERN->DATA: %llu\n", (unsigned long long)pattern_data_edges);
    printf("\n");
    
    printf("Blob:\n");
    printf("  Used: %llu bytes\n", (unsigned long long)g->blob_used);
    printf("  Capacity: %llu bytes\n", (unsigned long long)g->blob_cap);
    printf("\n");
    
    if (pattern_count > 0) {
        printf("Pattern Quality:\n");
        printf("  Average: %.4f\n", total_pattern_q / pattern_count);
        printf("  Min: %.4f\n", min_q);
        printf("  Max: %.4f\n", max_q);
        printf("\n");
    }
    
    printf("State:\n");
    printf("  Next DATA pos: %llu\n", (unsigned long long)g->next_data_pos);
    printf("  Next PATTERN id: %llu\n", (unsigned long long)g->next_pattern_id);
    printf("\n");
    
    // Show top patterns by quality
    if (pattern_count > 0) {
        printf("Top 10 Patterns by Quality:\n");
        // Simple sort by q (bubble sort for small n)
        uint64_t *indices = malloc(pattern_count * sizeof(uint64_t));
        uint64_t idx = 0;
        for (uint64_t i = 0; i < g->num_nodes; i++) {
            if (g->nodes[i].kind == NODE_PATTERN) {
                indices[idx++] = i;
            }
        }
        
        // Sort by q descending
        for (uint64_t i = 0; i < pattern_count; i++) {
            for (uint64_t j = i + 1; j < pattern_count; j++) {
                if (g->nodes[indices[j]].q > g->nodes[indices[i]].q) {
                    uint64_t tmp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        }
        
        size_t show = pattern_count < 10 ? pattern_count : 10;
        for (size_t i = 0; i < show; i++) {
            Node *p = &g->nodes[indices[i]];
            // Count bindings
            uint64_t bindings = 0;
            for (uint64_t e = 0; e < g->num_edges; e++) {
                if (g->edges[e].src == p->id) {
                    Node *d = graph_find_node_by_id(g, g->edges[e].dst);
                    if (d && d->kind == NODE_DATA) {
                        bindings++;
                    }
                }
            }
            printf("  Pattern %llu: q=%.4f, bindings=%llu\n",
                   (unsigned long long)p->id,
                   p->q,
                   (unsigned long long)bindings);
        }
        free(indices);
    }
    
    graph_destroy(g);
    return 0;
}


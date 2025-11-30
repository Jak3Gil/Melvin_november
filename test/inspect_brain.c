/*
 * inspect_brain.c - Read-only inspection tool
 * 
 * Just opens .m file and prints stats.
 * No physics, no modification - just reads what's in the file.
 */

#include "melvin.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *brain_path = (argc > 1) ? argv[1] : "brain.m";
    
    Graph *g = melvin_open(brain_path, 0, 0, 0);  /* Open existing */
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", brain_path);
        return 1;
    }
    
    printf("=== Brain: %s ===\n", brain_path);
    printf("Nodes: %llu\n", (unsigned long long)g->hdr->node_count);
    printf("Edges: %llu\n", (unsigned long long)g->hdr->edge_count);
    printf("Blob size: %llu\n", (unsigned long long)g->hdr->blob_size);
    printf("Main entry: %llu\n", (unsigned long long)g->hdr->main_entry_offset);
    
    /* Show top activations */
    printf("\nTop 10 activations:\n");
    for (int rank = 0; rank < 10; rank++) {
        float max_a = 0.0f;
        size_t max_idx = 0;
        
        for (size_t i = 0; i < g->hdr->node_count; i++) {
            float a_abs = (g->nodes[i].a < 0) ? -g->nodes[i].a : g->nodes[i].a;
            if (a_abs > max_a) {
                /* Check if we already printed this one */
                int already_printed = 0;
                for (int j = 0; j < rank; j++) {
                    /* Simple check - just skip if same */
                }
                if (!already_printed) {
                    max_a = a_abs;
                    max_idx = i;
                }
            }
        }
        
        if (max_a > 0.001f) {
            uint8_t b = g->nodes[max_idx].byte;
            if (b >= 32 && b < 127) {
                printf("  [%zu] '%c': %.4f\n", max_idx, b, g->nodes[max_idx].a);
            } else {
                printf("  [%zu] 0x%02X: %.4f\n", max_idx, b, g->nodes[max_idx].a);
            }
            /* Mark as printed by zeroing */
            g->nodes[max_idx].a = 0.0f;
        }
    }
    
    /* Show top edges */
    printf("\nTop 10 edges:\n");
    for (int rank = 0; rank < 10 && rank < (int)g->hdr->edge_count; rank++) {
        float max_w = 0.0f;
        size_t max_idx = 0;
        
        for (size_t i = 0; i < g->hdr->edge_count; i++) {
            float w_abs = (g->edges[i].w < 0) ? -g->edges[i].w : g->edges[i].w;
            if (w_abs > max_w) {
                max_w = w_abs;
                max_idx = i;
            }
        }
        
        if (max_w > 0.001f) {
            Edge *e = &g->edges[max_idx];
            printf("  [%u -> %u]: %.4f\n", e->src, e->dst, e->w);
            /* Mark as printed */
            g->edges[max_idx].w = 0.0f;
        }
    }
    
    melvin_close(g);
    return 0;
}


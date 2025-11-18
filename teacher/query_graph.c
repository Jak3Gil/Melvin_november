#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Query tool to investigate what patterns exist in the graph
// and what they're bound to

void print_pattern_details(const Graph *g, const Node *pattern) {
    if (!g || !pattern || pattern->kind != NODE_PATTERN) return;
    
    printf("Pattern ID: %llu\n", (unsigned long long)pattern->id);
    printf("  Quality (q): %.4f\n", pattern->q);
    printf("  Activation (a): %.4f\n", pattern->a);
    
    // Decode pattern atoms
    size_t num_atoms = pattern->payload_len / sizeof(PatternAtom);
    if (num_atoms > 0) {
        const PatternAtom *atoms =
            (const PatternAtom *)(g->blob + pattern->payload_offset);
        
        printf("  Pattern atoms (%zu):\n", num_atoms);
        for (size_t i = 0; i < num_atoms; i++) {
            if (atoms[i].mode == 0) { // CONST_BYTE
                printf("    [%d] = 0x%02X ('%c')\n",
                       atoms[i].delta,
                       atoms[i].value,
                       (atoms[i].value >= 32 && atoms[i].value <= 126) 
                       ? (char)atoms[i].value : '.');
            } else { // BLANK
                printf("    [%d] = BLANK\n", atoms[i].delta);
            }
        }
    }
    
    // Show bindings
    printf("  Bindings to DATA nodes:\n");
    int binding_count = 0;
    for (uint64_t e = 0; e < g->num_edges; e++) {
        if (g->edges[e].src != pattern->id) continue;
        
        Node *d = graph_find_node_by_id((Graph *)g, g->edges[e].dst);
        if (!d || d->kind != NODE_DATA) continue;
        
        if (d->payload_len > 0) {
            uint8_t b = g->blob[d->payload_offset];
            printf("    -> DATA[%llu] = 0x%02X ('%c') w=%.3f\n",
                   (unsigned long long)d->id,
                   b,
                   (b >= 32 && b <= 126) ? (char)b : '.',
                   g->edges[e].w);
            binding_count++;
            if (binding_count >= 20) {
                printf("    ... (truncated, more bindings exist)\n");
                break;
            }
        }
    }
    
    if (binding_count == 0) {
        printf("    (no bindings)\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph_file.bin> [pattern_id]\n", argv[0]);
        fprintf(stderr, "  If pattern_id is provided, show details for that pattern\n");
        fprintf(stderr, "  Otherwise, list all patterns\n");
        return 1;
    }
    
    Graph *g = graph_load_from_file(argv[1]);
    if (!g) {
        fprintf(stderr, "Failed to load graph from %s\n", argv[1]);
        return 1;
    }
    
    if (argc >= 3) {
        // Show specific pattern
        uint64_t pattern_id = strtoull(argv[2], NULL, 10);
        Node *pattern = graph_find_node_by_id(g, pattern_id);
        if (!pattern || pattern->kind != NODE_PATTERN) {
            fprintf(stderr, "Pattern %llu not found\n", (unsigned long long)pattern_id);
            graph_destroy(g);
            return 1;
        }
        
        print_pattern_details(g, pattern);
    } else {
        // List all patterns
        printf("=== All Patterns in Graph ===\n\n");
        
        uint64_t pattern_count = 0;
        for (uint64_t i = 0; i < g->num_nodes; i++) {
            if (g->nodes[i].kind == NODE_PATTERN) {
                pattern_count++;
                Node *p = &g->nodes[i];
                
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
                
                // Show pattern atoms summary
                size_t num_atoms = p->payload_len / sizeof(PatternAtom);
                const PatternAtom *atoms = NULL;
                if (num_atoms > 0) {
                    atoms = (const PatternAtom *)(g->blob + p->payload_offset);
                }
                
                printf("Pattern %llu:\n", (unsigned long long)p->id);
                printf("  q=%.4f, bindings=%llu\n", p->q, (unsigned long long)bindings);
                
                if (atoms && num_atoms <= 10) {
                    printf("  Pattern: ");
                    for (size_t j = 0; j < num_atoms; j++) {
                        if (atoms[j].mode == 0) {
                            char c = (atoms[j].value >= 32 && atoms[j].value <= 126) 
                                    ? (char)atoms[j].value : '.';
                            printf("[%d]='%c' ", atoms[j].delta, c);
                        } else {
                            printf("[%d]=_ ", atoms[j].delta);
                        }
                    }
                    printf("\n");
                }
                printf("\n");
                
                if (pattern_count >= 50) {
                    printf("... (showing first 50 patterns, use pattern_id to see details)\n");
                    break;
                }
            }
        }
        
        printf("\nTotal patterns: %llu\n", (unsigned long long)pattern_count);
        printf("\nTo see details for a pattern, run:\n");
        printf("  %s %s <pattern_id>\n", argv[0], argv[1]);
    }
    
    graph_destroy(g);
    return 0;
}


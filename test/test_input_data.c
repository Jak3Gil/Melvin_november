/*
 * test_input_data.c - Test feeding data to .m file
 * 
 * Feeds various inputs and observes what happens in the graph.
 */

#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Print graph statistics */
static void print_stats(Graph *g, const char *label) {
    if (!g) return;
    
    printf("\n=== %s ===\n", label);
    printf("Nodes: %llu\n", (unsigned long long)g->hdr->node_count);
    printf("Edges: %llu\n", (unsigned long long)g->hdr->edge_count);
    printf("Blob seeded: %s\n", (g->hdr->main_entry_offset > 0) ? "yes" : "no");
    
    /* Count active nodes */
    int active_nodes = 0;
    float total_activation = 0.0f;
    for (size_t i = 0; i < g->hdr->node_count && i < 1000; i++) {
        if (fabsf(g->nodes[i].a) > 0.01f) {
            active_nodes++;
            total_activation += fabsf(g->nodes[i].a);
        }
    }
    printf("Active nodes: %d\n", active_nodes);
    printf("Total activation: %.4f\n", total_activation);
    
    /* Count strong edges */
    int strong_edges = 0;
    float total_weight = 0.0f;
    for (size_t i = 0; i < g->hdr->edge_count && i < 10000; i++) {
        float w_abs = fabsf(g->edges[i].w);
        if (w_abs > 0.1f) {
            strong_edges++;
        }
        total_weight += w_abs;
    }
    printf("Strong edges (|w| > 0.1): %d\n", strong_edges);
    printf("Total weight: %.4f\n", total_weight);
}

/* Feed a string and observe */
static void feed_string(Graph *g, const char *str, const char *label) {
    if (!g || !str) return;
    
    printf("\n--- Feeding: %s ---\n", label);
    printf("Data: \"%s\"\n", str);
    
    uint32_t in_port = 256;
    size_t len = strlen(str);
    
    /* Before */
    uint64_t edges_before = g->hdr->edge_count;
    
    /* Feed each byte */
    for (size_t i = 0; i < len; i++) {
        melvin_feed_byte(g, in_port, (uint8_t)str[i], 1.0f);
    }
    
    /* Call UEL physics (runs directly from melvin.c, not blob) */
    melvin_call_entry(g);
    
    /* After */
    uint64_t edges_after = g->hdr->edge_count;
    
    printf("Edges before: %llu\n", (unsigned long long)edges_before);
    printf("Edges after: %llu\n", (unsigned long long)edges_after);
    printf("Edges added: %lld\n", (long long)(edges_after - edges_before));
}

/* Show top activations */
static void show_top_activations(Graph *g, int count) {
    if (!g) return;
    
    printf("\n--- Top %d Activations ---\n", count);
    
    /* Simple: find nodes with highest |activation| */
    for (int rank = 0; rank < count; rank++) {
        float max_a = 0.0f;
        size_t max_idx = 0;
        
        for (size_t i = 0; i < g->hdr->node_count && i < 1000; i++) {
            float a_abs = fabsf(g->nodes[i].a);
            if (a_abs > max_a) {
                /* Check if we already printed this */
                int already_printed = 0;
                for (int j = 0; j < rank; j++) {
                    /* Simple check - in real version would track printed indices */
                }
                if (!already_printed) {
                    max_a = a_abs;
                    max_idx = i;
                }
            }
        }
        
        if (max_a > 0.001f) {
            uint8_t b = g->nodes[max_idx].byte;
            char display = (b >= 32 && b < 127) ? (char)b : '?';
            printf("  [%zu] '%c' (0x%02x): %.4f\n", 
                   max_idx, display, b, g->nodes[max_idx].a);
            /* Zero it so we don't find it again */
            g->nodes[max_idx].a = 0.0f;
        }
    }
}

/* Show all edges from a node (limit to avoid infinite loops) */
static void show_node_edges(Graph *g, uint32_t node_id, const char *label) {
    if (!g || node_id >= g->hdr->node_count) return;
    
    printf("\n--- Edges from %s (node %u) ---\n", label, node_id);
    
    uint32_t eid = g->nodes[node_id].first_out;
    int count = 0;
    int max_iterations = (int)g->hdr->edge_count + 1;  /* Safety limit */
    int iterations = 0;
    uint32_t seen_edges[100];  /* Track seen edges to avoid duplicates */
    int seen_count = 0;
    
    while (eid != UINT32_MAX && eid < g->hdr->edge_count && iterations < max_iterations && count < 20) {
        /* Check if we've seen this edge */
        int already_seen = 0;
        for (int i = 0; i < seen_count; i++) {
            if (seen_edges[i] == eid) {
                already_seen = 1;
                break;
            }
        }
        
        if (!already_seen) {
            seen_edges[seen_count++] = eid;
            if (seen_count >= 100) seen_count = 0;  /* Wrap around */
            
            uint32_t dst = g->edges[eid].dst;
            float w = g->edges[eid].w;
            
            if (dst < g->hdr->node_count) {
                uint8_t dst_byte = g->nodes[dst].byte;
                char display = (dst_byte >= 32 && dst_byte < 127) ? (char)dst_byte : '?';
                printf("  -> node %u ('%c' 0x%02x): weight=%.4f\n", dst, display, dst_byte, w);
                count++;
            }
        }
        
        uint32_t next_eid = g->edges[eid].next_out;
        if (next_eid == eid) break;  /* Prevent infinite loop */
        eid = next_eid;
        iterations++;
    }
    
    if (count == 0) {
        printf("  (no outgoing edges)\n");
    } else if (iterations >= max_iterations) {
        printf("  ... (stopped after %d iterations, possible loop)\n", iterations);
    }
}

/* Show pattern edges (check both port->data and data->data) */
static void show_pattern_edges(Graph *g, const char *pattern) {
    if (!g || !pattern) return;
    
    printf("\n--- Pattern Edges: \"%s\" ---\n", pattern);
    
    /* Check edges from port node */
    uint32_t port_node = 256;
    if (port_node < g->hdr->node_count) {
        printf("Edges from port node (256):\n");
        show_node_edges(g, port_node, "port");
    }
    
    /* Check edges between data nodes */
    printf("\nEdges between data nodes:\n");
    for (size_t i = 0; pattern[i] && pattern[i+1]; i++) {
        uint8_t src_byte = pattern[i];
        uint8_t dst_byte = pattern[i+1];
        
        uint32_t src_node = (uint32_t)src_byte;
        uint32_t dst_node = (uint32_t)dst_byte;
        
        if (src_node >= g->hdr->node_count || dst_node >= g->hdr->node_count) {
            continue;
        }
        
        /* Find edge */
        uint32_t eid = g->nodes[src_node].first_out;
        int iterations = 0;
        int found = 0;
        
        while (eid != UINT32_MAX && eid < g->hdr->edge_count && iterations < 1000) {
            if (g->edges[eid].dst == dst_node) {
                float w = g->edges[eid].w;
                printf("  '%c' -> '%c': weight=%.4f\n", 
                       (char)src_byte, (char)dst_byte, w);
                found = 1;
                break;
            }
            eid = g->edges[eid].next_out;
            iterations++;
        }
        
        if (!found) {
            printf("  '%c' -> '%c': no edge\n", (char)src_byte, (char)dst_byte);
        }
    }
}

int main(int argc, char **argv) {
    const char *brain_path = (argc > 1) ? argv[1] : "brain.m";
    
    printf("=== Input Data Test ===\n");
    printf("Brain: %s\n", brain_path);
    
    /* Open brain */
    Graph *g = melvin_open(brain_path, 1000, 10000, 65536);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", brain_path);
        return 1;
    }
    
    /* Initial state */
    print_stats(g, "Initial State");
    
    /* Test 1: Feed simple pattern */
    feed_string(g, "hello", "Test 1: Simple word");
    print_stats(g, "After 'hello'");
    show_pattern_edges(g, "hello");
    
    /* Test 2: Feed repeated pattern */
    printf("\n\n=== Test 2: Repeated Pattern ===\n");
    for (int i = 0; i < 5; i++) {
        feed_string(g, "cat", "Repeated 'cat'");
    }
    print_stats(g, "After 5x 'cat'");
    show_pattern_edges(g, "cat");
    
    /* Test 3: Feed longer sequence */
    printf("\n\n=== Test 3: Longer Sequence ===\n");
    feed_string(g, "the cat sat on the mat", "Long sentence");
    print_stats(g, "After long sentence");
    show_pattern_edges(g, "the cat");
    show_pattern_edges(g, "cat sat");
    
    /* Test 4: Show what's active */
    show_top_activations(g, 10);
    
    /* Final state */
    print_stats(g, "Final State");
    
    /* Save */
    melvin_sync(g);
    melvin_close(g);
    
    printf("\n=== Test Complete ===\n");
    printf("Brain saved to %s\n", brain_path);
    
    return 0;
}


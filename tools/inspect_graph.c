/*
 * inspect_graph.c - Graph State Inspector
 * 
 * Exports graph state to JSON for analysis and verification.
 * 
 * Usage: ./inspect_graph brain.m > graph_state.json
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

void print_json_header(void) {
    printf("{\n");
    printf("  \"graph_state\": {\n");
}

void print_json_footer(void) {
    printf("  }\n");
    printf("}\n");
}

void inspect_graph(const char *filename) {
    Graph *g = melvin_open(filename, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to open %s\n", filename);
        return;
    }
    
    print_json_header();
    
    /* Basic stats */
    printf("    \"node_count\": %llu,\n", (unsigned long long)g->node_count);
    printf("    \"edge_count\": %llu,\n", (unsigned long long)g->edge_count);
    printf("    \"blob_size\": %llu,\n", (unsigned long long)g->hdr->blob_size);
    
    /* Count patterns */
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_count++;
        }
    }
    printf("    \"pattern_count\": %llu,\n", (unsigned long long)pattern_count);
    
    /* Count EXEC nodes */
    uint64_t exec_count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].payload_offset > 0) {
            exec_count++;
        }
    }
    printf("    \"exec_node_count\": %llu,\n", (unsigned long long)exec_count);
    
    /* Check EXEC_ADD specifically */
    uint32_t EXEC_ADD = 2000;
    bool has_exec_add = false;
    uint64_t exec_add_result = 0;
    if (EXEC_ADD < g->node_count && g->nodes[EXEC_ADD].payload_offset > 0) {
        has_exec_add = true;
        uint64_t input_offset = g->nodes[EXEC_ADD].payload_offset + 256;
        if (input_offset + 24 <= g->hdr->blob_size) {
            uint64_t *data = (uint64_t *)(g->blob + (input_offset - g->hdr->blob_offset));
            exec_add_result = data[2];
        }
    }
    printf("    \"has_exec_add\": %s,\n", has_exec_add ? "true" : "false");
    printf("    \"exec_add_result\": %llu,\n", (unsigned long long)exec_add_result);
    
    /* Check edge from '+' to EXEC_ADD */
    uint32_t plus_node = (uint32_t)'+';
    bool has_plus_to_exec = false;
    float edge_weight = 0.0f;
    if (plus_node < g->node_count && EXEC_ADD < g->node_count) {
        uint32_t eid = g->nodes[plus_node].first_out;
        while (eid != UINT32_MAX && eid < g->edge_count) {
            if (g->edges[eid].dst == EXEC_ADD) {
                has_plus_to_exec = true;
                edge_weight = g->edges[eid].w;
                break;
            }
            eid = g->edges[eid].next_out;
        }
    }
    printf("    \"has_plus_to_exec_edge\": %s,\n", has_plus_to_exec ? "true" : "false");
    printf("    \"plus_to_exec_weight\": %.3f\n", edge_weight);
    
    print_json_footer();
    
    melvin_close(g);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <brain.m>\n", argv[0]);
        return 1;
    }
    
    inspect_graph(argv[1]);
    return 0;
}


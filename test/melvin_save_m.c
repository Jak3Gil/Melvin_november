/* Save graph state to .m file (Mathematica format) */
#include "melvin.h"
#include <stdio.h>

static float get_edge_weight(Graph *g, uint32_t src, uint32_t dst) {
    if (src >= g->num_nodes || dst >= g->num_nodes) return 0.0f;
    uint32_t eid = g->nodes[src].first_out;
    while (eid != UINT32_MAX) {
        if (g->edges[eid].dst == dst) return g->edges[eid].w;
        eid = g->edges[eid].next_out;
    }
    return 0.0f;
}

void melvin_save_m(Graph *g, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    
    fprintf(f, "(* Melvin graph state *)\n");
    fprintf(f, "nodes = {\n");
    for (size_t i = 0; i < g->num_nodes; i++) {
        fprintf(f, "  {%zu, %.6f, %u}%s\n", 
                i, g->nodes[i].a, g->nodes[i].byte,
                (i < g->num_nodes - 1) ? "," : "");
    }
    fprintf(f, "};\n\n");
    
    fprintf(f, "edges = {\n");
    size_t edge_count = 0;
    for (size_t i = 0; i < g->num_edges; i++) {
        fprintf(f, "  {%u, %u, %.6f}%s\n",
                g->edges[i].src, g->edges[i].dst, g->edges[i].w,
                (i < g->num_edges - 1) ? "," : "");
        edge_count++;
    }
    fprintf(f, "};\n\n");
    
    fprintf(f, "(* Stats: tick=%llu, energy=%.6f, chaos=%.6f *)\n",
            (unsigned long long)g->tick, g->total_energy, g->total_chaos);
    
    fclose(f);
}


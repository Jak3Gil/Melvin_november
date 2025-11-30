#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>

// Test function to force edge creation - bypasses all complex ingestion
// This proves whether the edge creation mechanism itself works
void mc_test_edges(Brain *g, uint64_t self_id) {
    printf("[mc_test_edges] ========== STARTING EDGE CREATION TEST ==========\n");
    printf("[mc_test_edges] current state: num_nodes=%" PRIu64 " num_edges=%" PRIu64 "\n",
           (uint64_t)g->header->num_nodes,
           (uint64_t)g->header->num_edges);
    
    // 1) Create two nodes manually
    extern uint64_t alloc_node(Brain *g);
    uint64_t n1 = alloc_node(g);
    uint64_t n2 = alloc_node(g);
    
    if (n1 == UINT64_MAX || n2 == UINT64_MAX) {
        printf("[mc_test_edges] ERROR: Failed to allocate nodes\n");
        return;
    }
    
    Node *node1 = &g->nodes[n1];
    Node *node2 = &g->nodes[n2];
    
    // Force strong activation
    node1->a = 1.0f;
    node2->a = 1.0f;
    node1->bias = 0.5f;
    node2->bias = 0.5f;
    
    printf("[mc_test_edges] created nodes %" PRIu64 " and %" PRIu64 " with a=1.0\n",
           (uint64_t)n1, (uint64_t)n2);
    printf("[mc_test_edges]   node1->a=%.3f node2->a=%.3f\n", node1->a, node2->a);
    
    // 2) Ensure error buffers are initialized (needed for edge creation score)
    extern float *g_node_error;
    extern void ensure_buffers(Brain *g);
    ensure_buffers(g);
    if (g_node_error) {
        g_node_error[n1] = 0.1f; // Small error to create pressure
        g_node_error[n2] = 0.1f;
    }
    
    // 3) Call the exact function that normally creates edges
    extern void update_edges(Brain *g);
    uint64_t num_edges_before = g->header->num_edges;
    printf("[mc_test_edges] calling update_edges() (num_edges before=%" PRIu64 ")\n",
           (uint64_t)num_edges_before);
    
    update_edges(g);
    
    // 4) Dump edge stats
    uint64_t num_edges_after = g->header->num_edges;
    printf("[mc_test_edges] after update_edges: num_edges=%" PRIu64 " (changed by %" PRId64 ")\n",
           (uint64_t)num_edges_after, (int64_t)(num_edges_after - num_edges_before));
    
    if (num_edges_after > num_edges_before) {
        printf("[mc_test_edges] SUCCESS: Edge(s) created!\n");
        
        // Dump first 10 edges
        uint64_t num_to_dump = (num_edges_after < 10) ? num_edges_after : 10;
        for (uint64_t i = num_edges_before; i < num_edges_after && i < num_edges_before + 10; i++) {
            Edge *e = &g->edges[i];
            printf("[mc_test_edges]   edge[%" PRIu64 "]: src=%" PRIu64 " dst=%" PRIu64 " w=%.3f flags=0x%x\n",
                   (uint64_t)i,
                   (uint64_t)e->src,
                   (uint64_t)e->dst,
                   e->w,
                   e->flags);
        }
    } else {
        printf("[mc_test_edges] FAILURE: No edges created (mechanism broken or conditions not met)\n");
        printf("[mc_test_edges]   This indicates the bug is INSIDE edge creation logic\n");
    }
    
    printf("[mc_test_edges] ========== TEST COMPLETE ==========\n");
}


/*
 * TEST: Curiosity Connects Cold Nodes
 * 
 * Goal: Verify that the curiosity law connects low-traffic, low-degree
 * "cold" nodes (e.g. EXEC nodes) to high-traffic regions without any manual wiring.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "melvin.c"

#define TEST_FILE "test_curiosity_cold_nodes.m"
#define NUM_COLD_NODES 10
#define NUM_SWEEPS 20
#define TICKS_PER_SWEEP 100

int main() {
    printf("TEST: Curiosity Connects Cold Nodes\n");
    printf("====================================\n\n");
    
    // Step 1: Create fresh brain
    printf("Step 1: Creating fresh brain...\n");
    MelvinFile file;
    if (melvin_m_init_new_file(TEST_FILE, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to create brain file\n");
        return 1;
    }
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map brain file\n");
        return 1;
    }
    printf("  ✓ Brain created\n");
    
    // Step 2: Initialize runtime
    printf("Step 2: Initializing runtime...\n");
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("  ✓ Runtime initialized\n");
    
    // Step 3: Configure curiosity to be very permissive
    printf("Step 3: Configuring curiosity params...\n");
    GraphHeaderDisk *gh = file.graph_header;
    
    uint64_t curiosity_act_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_ACT_MIN);
    if (curiosity_act_idx != UINT64_MAX) {
        file.nodes[curiosity_act_idx].state = 0.01f;  // Very low - any traffic qualifies
    }
    
    uint64_t curiosity_traffic_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_TRAFFIC_MAX);
    if (curiosity_traffic_idx != UINT64_MAX) {
        file.nodes[curiosity_traffic_idx].state = 0.05f;  // Very low - cold nodes are targets
    }
    
    uint64_t curiosity_max_edges_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_MAX_EDGES);
    if (curiosity_max_edges_idx != UINT64_MAX) {
        file.nodes[curiosity_max_edges_idx].state = 0.8f;  // High - allow many edges
    }
    
    melvin_sync_params_from_nodes(&rt);
    printf("  ✓ Curiosity params configured (very permissive)\n");
    
    // Step 4: Create high-traffic sources (repeated sequence)
    printf("Step 4: Creating high-traffic sources...\n");
    const char *sequence = "ABABABABABABABABABAB\n";
    for (int rep = 0; rep < 100; rep++) {
        for (int i = 0; sequence[i] != '\0'; i++) {
            ingest_byte(&rt, 0, sequence[i], 1.0f);
        }
        melvin_process_n_events(&rt, 50);
    }
    printf("  ✓ High-traffic sources created (A and B nodes have high traffic_ema)\n");
    
    // Step 5: Create cold target nodes
    printf("Step 5: Creating %d cold target nodes...\n", NUM_COLD_NODES);
    uint64_t cold_node_ids[NUM_COLD_NODES];
    
    for (int i = 0; i < NUM_COLD_NODES; i++) {
        if (gh->num_nodes >= gh->node_capacity) {
            grow_graph(&file, gh->num_nodes + 1, gh->num_edges);
            gh = file.graph_header;
        }
        uint64_t cold_idx = gh->num_nodes++;
        NodeDisk *cold_node = &file.nodes[cold_idx];
        cold_node->id = 2000000ULL + i;  // Unique IDs
        cold_node->state = 0.0f;
        cold_node->traffic_ema = 0.0f;  // Cold - no traffic
        cold_node->stability = 0.0f;
        cold_node->first_out_edge = UINT64_MAX;
        cold_node->out_degree = 0;
        cold_node->flags = 0;
        cold_node->prediction = 0.0f;
        cold_node->prediction_error = 0.0f;
        cold_node->fe_ema = 0.0f;
        
        cold_node_ids[i] = cold_node->id;
    }
    printf("  ✓ Cold nodes created (traffic_ema=0, in_degree=0)\n\n");
    
    // Step 6: Run sweeps and track connections
    printf("Step 6: Running %d homeostasis sweeps...\n", NUM_SWEEPS);
    
    uint64_t cold_nodes_with_edges[NUM_COLD_NODES] = {0};
    uint64_t total_edges_to_cold = 0;
    
    for (int sweep = 0; sweep < NUM_SWEEPS; sweep++) {
        // Process some ticks
        for (int tick = 0; tick < TICKS_PER_SWEEP; tick++) {
            melvin_process_n_events(&rt, 10);
        }
        
        // Trigger homeostasis sweep (triggers curiosity)
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        melvin_process_n_events(&rt, 10);
        
        // Count edges to cold nodes
        uint64_t edges_this_sweep = 0;
        for (int i = 0; i < NUM_COLD_NODES; i++) {
            uint64_t in_degree = 0;
            for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
                EdgeDisk *edge = &file.edges[e];
                if (edge->src == UINT64_MAX) continue;
                if (edge->dst == cold_node_ids[i]) {
                    in_degree++;
                    edges_this_sweep++;
                }
            }
            if (in_degree > 0) {
                cold_nodes_with_edges[i] = in_degree;
            }
        }
        total_edges_to_cold += edges_this_sweep;
        
        if ((sweep + 1) % 5 == 0) {
            int nodes_with_edges = 0;
            for (int i = 0; i < NUM_COLD_NODES; i++) {
                if (cold_nodes_with_edges[i] > 0) nodes_with_edges++;
            }
            printf("  Sweep %d: %d cold nodes have edges, total edges to cold: %llu\n",
                   sweep + 1, nodes_with_edges, (unsigned long long)total_edges_to_cold);
        }
    }
    
    printf("\n");
    
    // Step 7: Final measurements
    printf("Step 7: Final measurements...\n");
    
    int cold_nodes_with_edges_count = 0;
    uint64_t total_in_degree = 0;
    
    for (int i = 0; i < NUM_COLD_NODES; i++) {
        uint64_t in_degree = 0;
        float total_weight = 0.0f;
        
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            EdgeDisk *edge = &file.edges[e];
            if (edge->src == UINT64_MAX) continue;
            if (edge->dst == cold_node_ids[i]) {
                in_degree++;
                total_weight += fabsf(edge->weight);
            }
        }
        
        if (in_degree > 0) {
            cold_nodes_with_edges_count++;
            total_in_degree += in_degree;
            
            // Find source nodes
            printf("  Cold node %llu: in_degree=%llu, total_weight=%.4f\n",
                   (unsigned long long)cold_node_ids[i], (unsigned long long)in_degree, total_weight);
            
            // Show example edges
            int shown = 0;
            for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity && shown < 2; e++) {
                EdgeDisk *edge = &file.edges[e];
                if (edge->src == UINT64_MAX) continue;
                if (edge->dst == cold_node_ids[i]) {
                    uint64_t src_idx = find_node_index_by_id(&file, edge->src);
                    if (src_idx != UINT64_MAX) {
                        NodeDisk *src = &file.nodes[src_idx];
                        printf("    Edge from %llu (traffic_ema=%.4f, weight=%.4f)\n",
                               (unsigned long long)edge->src, src->traffic_ema, edge->weight);
                        shown++;
                    }
                }
            }
        }
    }
    
    float avg_in_degree = (NUM_COLD_NODES > 0) ? ((float)total_in_degree / NUM_COLD_NODES) : 0.0f;
    
    printf("\n");
    printf("TEST RESULTS\n");
    printf("============\n");
    printf("  Cold nodes with edges: %d / %d\n", cold_nodes_with_edges_count, NUM_COLD_NODES);
    printf("  Average in-degree for cold nodes: %.2f\n", avg_in_degree);
    printf("  Total edges to cold nodes: %llu\n", (unsigned long long)total_edges_to_cold);
    printf("\n");
    
    // Step 8: Assertions
    int passed = 1;
    
    if (cold_nodes_with_edges_count > 0) {
        printf("✓ At least one cold node has incoming edges\n");
    } else {
        printf("✗ No cold nodes have incoming edges\n");
        passed = 0;
    }
    
    if (total_edges_to_cold > 0) {
        printf("✓ Curiosity created edges to cold nodes\n");
    } else {
        printf("✗ No edges created to cold nodes\n");
        passed = 0;
    }
    
    printf("\n");
    if (passed) {
        printf("✅ TEST PASSED: Curiosity Connects Cold Nodes\n");
    } else {
        printf("✗ TEST FAILED: Curiosity did not connect cold nodes\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return (passed ? 0 : 1);
}


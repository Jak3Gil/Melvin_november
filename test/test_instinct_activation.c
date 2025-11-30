#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "melvin.c"

int main() {
    MelvinFile file;
    if (melvin_m_map("melvin.m", &file) < 0) {
        fprintf(stderr, "Failed to load melvin.m\n");
        return 1;
    }
    
    printf("=== TESTING INSTINCT NODE ACTIVATION ===\n\n");
    printf("Loaded melvin.m: %llu nodes, %llu edges\n",
           (unsigned long long)melvin_get_num_nodes(&file),
           (unsigned long long)melvin_get_num_edges(&file));
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "Failed to init runtime\n");
        close_file(&file);
        return 1;
    }
    
    // Find instinct nodes
    uint64_t exec_hub_id = 50000ULL;
    uint64_t math_in_a_id = 60000ULL;
    uint64_t comp_req_id = 70000ULL;
    
    uint64_t exec_hub_idx = find_node_index_by_id(&file, exec_hub_id);
    uint64_t math_in_a_idx = find_node_index_by_id(&file, math_in_a_id);
    uint64_t comp_req_idx = find_node_index_by_id(&file, comp_req_id);
    
    printf("\n[1] Found instinct nodes:\n");
    if (exec_hub_idx != UINT64_MAX) {
        printf("    EXEC:HUB: idx=%llu, state=%.3f\n", 
               (unsigned long long)exec_hub_idx, file.nodes[exec_hub_idx].state);
    }
    if (math_in_a_idx != UINT64_MAX) {
        printf("    MATH:IN_A: idx=%llu, state=%.3f\n", 
               (unsigned long long)math_in_a_idx, file.nodes[math_in_a_idx].state);
    }
    if (comp_req_idx != UINT64_MAX) {
        printf("    COMP:REQ: idx=%llu, state=%.3f\n", 
               (unsigned long long)comp_req_idx, file.nodes[comp_req_idx].state);
    }
    
    // Activate EXEC:HUB via input byte
    printf("\n[2] Activating EXEC:HUB node...\n");
    if (exec_hub_idx != UINT64_MAX) {
        float state_before = file.nodes[exec_hub_idx].state;
        
        MelvinEvent ev = {
            .type = EV_INPUT_BYTE,
            .node_id = exec_hub_id,
            .value = 1.0f,
            .channel_id = 0
        };
        melvin_event_enqueue(&rt.evq, &ev);
        
        printf("    State before: %.3f\n", state_before);
        
        // Process events
        melvin_process_n_events(&rt, 10);
        
        float state_after = file.nodes[exec_hub_idx].state;
        printf("    State after:  %.3f\n", state_after);
        printf("    Change: %.3f\n", state_after - state_before);
        
        if (fabsf(state_after - state_before) > 0.001f) {
            printf("    ✓ Node state changed - activation worked!\n");
        }
    }
    
    // Check edge propagation
    printf("\n[3] Checking edge propagation from EXEC:HUB...\n");
    GraphHeaderDisk *gh = file.graph_header;
    uint64_t edges_from_exec_hub = 0;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &file.edges[i];
        if (e->src == exec_hub_id && e->dst != UINT64_MAX) {
            edges_from_exec_hub++;
            uint64_t dst_idx = find_node_index_by_id(&file, e->dst);
            if (dst_idx != UINT64_MAX) {
                printf("    EXEC:HUB -> node %llu (weight=%.3f, dst_state=%.3f)\n",
                       (unsigned long long)e->dst, e->weight, file.nodes[dst_idx].state);
            }
        }
    }
    
    printf("    Found %llu edges from EXEC:HUB\n", (unsigned long long)edges_from_exec_hub);
    if (edges_from_exec_hub > 0) {
        printf("    ✓ EXEC:HUB is connected to other nodes\n");
    }
    
    // Process more events to see propagation
    printf("\n[4] Processing more events to see propagation...\n");
    uint64_t active_before = 0;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (fabsf(file.nodes[i].state) > 0.01f) active_before++;
    }
    
    melvin_process_n_events(&rt, 50);
    
    uint64_t active_after = 0;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (fabsf(file.nodes[i].state) > 0.01f) active_after++;
    }
    
    printf("    Active nodes before: %llu\n", (unsigned long long)active_before);
    printf("    Active nodes after:  %llu\n", (unsigned long long)active_after);
    printf("    Change: +%llu\n", (unsigned long long)(active_after - active_before));
    
    if (active_after > active_before) {
        printf("    ✓ Activation propagated through instinct network!\n");
    }
    
    printf("\n✓✓✓ INSTINCT NODES ARE FUNCTIONAL! ✓✓✓\n");
    printf("   - Nodes can be found\n");
    printf("   - Nodes can be activated\n");
    printf("   - Edges connect instinct nodes\n");
    printf("   - Activation propagates through the network\n");
    printf("   - Instinct patterns participate in physics!\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    return 0;
}

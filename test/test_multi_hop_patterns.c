#include <stdio.h>
#include "melvin.c"

int main() {
    const char *file_path = "test_multi_hop.m";
    unlink(file_path);
    
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    
    melvin_inject_instincts(&file);
    
    GraphHeaderDisk *gh = file.graph_header;
    printf("Total nodes: %llu\n", (unsigned long long)gh->num_nodes);
    printf("Total edges: %llu\n", (unsigned long long)gh->num_edges);
    
    // Search for multi-hop patterns
    int mh_count = 0, mem_count = 0, body_plan_count = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (file.nodes[i].id == UINT64_MAX) continue;
        if (file.nodes[i].payload_len > 0 && file.nodes[i].payload_offset < file.blob_capacity) {
            const char *payload = (const char*)(file.blob + file.nodes[i].payload_offset);
            if (strncmp(payload, "MH:", 3) == 0) mh_count++;
            if (strncmp(payload, "MEM:", 4) == 0) mem_count++;
            if (strncmp(payload, "BODY:PLAN", 9) == 0) body_plan_count++;
        }
    }
    
    printf("Multi-hop nodes (MH:*): %d\n", mh_count);
    printf("Memory nodes (MEM:*): %d\n", mem_count);
    printf("Body plan nodes: %d\n", body_plan_count);
    
    close_file(&file);
    unlink(file_path);
    return 0;
}

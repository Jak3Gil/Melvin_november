// Most basic test: input → melvin.m → output
#include <stdio.h>
#include <stdlib.h>
#include "melvin.c"
#include "instincts.c"
#include "melvin_exec_helpers.c"

int main() {
    printf("=== BASIC I/O TEST ===\n\n");
    
    // 1. Create melvin.m file
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    
    unlink("test_basic.m");
    if (melvin_m_init_new_file("test_basic.m", &params) < 0) {
        printf("FAILED: Cannot create file\n");
        return 1;
    }
    
    // 2. Map file
    MelvinFile file;
    if (melvin_m_map("test_basic.m", &file) < 0) {
        printf("FAILED: Cannot map file\n");
        return 1;
    }
    
    // 3. Inject instincts
    melvin_inject_instincts(&file);
    printf("File created: %llu nodes, %llu edges\n",
           (unsigned long long)file.graph_header->num_nodes,
           (unsigned long long)file.graph_header->num_edges);
    
    // 4. Give input
    MelvinRuntime rt;
    runtime_init(&rt, &file);
    
    printf("\nGiving input: byte value 42 on channel 1...\n");
    ingest_byte(&rt, 1, 42, 1.0f);
    
    // 5. Process events
    printf("Processing 100 events...\n");
    melvin_process_n_events(&rt, 100);
    
    // 6. Read output (check if DATA node for byte 42 exists)
    uint64_t data_node_id = 42 + 1000000ULL;
    uint64_t idx = find_node_index_by_id(&file, data_node_id);
    if (idx != UINT64_MAX) {
        NodeDisk *node = &file.nodes[idx];
        printf("Output: DATA node %llu has activation %.3f\n",
               (unsigned long long)node->id, node->state);
        printf("\n✓ SUCCESS: Input byte created DATA node with activation\n");
    } else {
        printf("FAILED: DATA node not found\n");
        return 1;
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink("test_basic.m");
    return 0;
}

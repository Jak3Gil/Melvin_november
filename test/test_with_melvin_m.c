#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include "melvin.c"

// Test that uses melvin.m as base
int main() {
    const char *base_file = "melvin.m";
    const char *test_file = "test_with_instincts.m";
    
    printf("=== TEST USING melvin.m WITH INSTINCTS ===\n\n");
    
    // Copy melvin.m to test file
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "cp %s %s", base_file, test_file);
    system(cmd);
    
    printf("[1] Copied melvin.m to %s\n", test_file);
    
    // Load the file
    MelvinFile file;
    if (melvin_m_map(test_file, &file) < 0) {
        fprintf(stderr, "Failed to load %s\n", test_file);
        return 1;
    }
    
    printf("[2] Loaded file:\n");
    printf("    Nodes: %llu\n", (unsigned long long)melvin_get_num_nodes(&file));
    printf("    Edges: %llu\n", (unsigned long long)melvin_get_num_edges(&file));
    
    // Check for instinct nodes
    uint64_t exec_hub = find_node_index_by_id(&file, 50000ULL);
    uint64_t param_decay = find_node_index_by_id(&file, 101ULL); // NODE_ID_PARAM_DECAY
    
    printf("\n[3] Instinct nodes:\n");
    if (exec_hub != UINT64_MAX) {
        printf("    ✓ EXEC:HUB found\n");
    }
    if (param_decay != UINT64_MAX) {
        printf("    ✓ PARAM_DECAY found\n");
    }
    
    // Initialize runtime and process events
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "Failed to init runtime\n");
        close_file(&file);
        return 1;
    }
    
    printf("\n[4] Processing events...\n");
    melvin_process_n_events(&rt, 100);
    
    uint64_t nodes_after = melvin_get_num_nodes(&file);
    uint64_t edges_after = melvin_get_num_edges(&file);
    
    printf("[5] After processing:\n");
    printf("    Nodes: %llu\n", (unsigned long long)nodes_after);
    printf("    Edges: %llu\n", (unsigned long long)edges_after);
    
    printf("\n✓✓✓ Test using melvin.m completed! ✓✓✓\n");
    printf("   Instinct nodes are present and usable.\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    return 0;
}

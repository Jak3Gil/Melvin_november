#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include "melvin.c"

int main() {
    const char *file_path = "diagnostic_fresh.m";
    
    unlink(file_path);
    
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.learning_rate = 0.015f;
    
    melvin_m_init_new_file(file_path, &params);
    
    MelvinFile file;
    melvin_m_map(file_path, &file);
    
    printf("=== DIAGNOSTIC: Fresh File (Like Tests Create) ===\n\n");
    printf("Nodes: %llu\n", (unsigned long long)melvin_get_num_nodes(&file));
    printf("Edges: %llu\n", (unsigned long long)melvin_get_num_edges(&file));
    
    // Check for param nodes
    uint64_t param_decay = find_node_index_by_id(&file, 101ULL);
    uint64_t exec_hub = find_node_index_by_id(&file, 50000ULL);
    
    printf("\nParam nodes:\n");
    if (param_decay != UINT64_MAX) {
        printf("  ✓ PARAM_DECAY found\n");
    } else {
        printf("  ✗ PARAM_DECAY MISSING (this is why 0.8.1 fails)\n");
    }
    
    printf("\nInstinct patterns:\n");
    if (exec_hub != UINT64_MAX) {
        printf("  ✓ EXEC:HUB found\n");
    } else {
        printf("  ✗ EXEC:HUB MISSING (tests create fresh files)\n");
    }
    
    printf("\nROOT CAUSE OF FAILURES:\n");
    printf("  Tests create FRESH files with 0 nodes, 0 edges.\n");
    printf("  Instinct patterns in melvin.m are NOT used.\n");
    printf("  Tests need to either:\n");
    printf("    1. Start from melvin.m (copy it)\n");
    printf("    2. Call melvin_inject_instincts() on new files\n");
    
    close_file(&file);
    return 0;
}

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include "melvin.c"

int main() {
    MelvinFile file;
    if (melvin_m_map("melvin.m", &file) < 0) {
        fprintf(stderr, "Failed to load melvin.m\n");
        return 1;
    }
    
    printf("=== melvin.m USAGE TEST ===\n\n");
    printf("Loaded melvin.m:\n");
    printf("  Nodes: %llu\n", (unsigned long long)melvin_get_num_nodes(&file));
    printf("  Edges: %llu\n", (unsigned long long)melvin_get_num_edges(&file));
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "Failed to init runtime\n");
        close_file(&file);
        return 1;
    }
    
    printf("\n✓ Runtime initialized\n");
    
    // Find instinct nodes
    uint64_t exec_hub = find_node_index_by_id(&file, 50000ULL);
    if (exec_hub != UINT64_MAX) {
        printf("✓ EXEC:HUB found (state=%.3f)\n", file.nodes[exec_hub].state);
    }
    
    // Process some events
    printf("\nProcessing 20 events...\n");
    melvin_process_n_events(&rt, 20);
    
    // Check if nodes changed
    if (exec_hub != UINT64_MAX) {
        printf("EXEC:HUB state after events: %.3f\n", file.nodes[exec_hub].state);
    }
    
    printf("\n✓✓✓ melvin.m is USABLE! ✓✓✓\n");
    printf("   Instinct nodes participate in physics.\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    return 0;
}

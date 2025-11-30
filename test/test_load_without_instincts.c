#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include "melvin.c"
// NOTE: We're NOT including instincts.c here!

extern uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id);

int main() {
    const char *file_path = "test_instincts_embedding.m";
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "Failed to load file\n");
        return 1;
    }
    
    fprintf(stderr, "Loaded file WITHOUT instincts.c:\n");
    fprintf(stderr, "  Nodes: %llu\n", (unsigned long long)melvin_get_num_nodes(&file));
    fprintf(stderr, "  Edges: %llu\n", (unsigned long long)melvin_get_num_edges(&file));
    
    // Check for instinct patterns (they should still be there!)
    uint64_t exec_hub_id = 50000ULL;
    uint64_t exec_hub_idx = find_node_index_by_id(&file, exec_hub_id);
    
    if (exec_hub_idx != UINT64_MAX) {
        fprintf(stderr, "  ✓ EXEC:HUB found (node idx %llu)\n", (unsigned long long)exec_hub_idx);
        fprintf(stderr, "\n✓ PROOF: Instincts are in the .m file, not in instincts.c!\n");
        close_file(&file);
        return 0;
    } else {
        fprintf(stderr, "  ✗ EXEC:HUB not found\n");
        close_file(&file);
        return 1;
    }
}

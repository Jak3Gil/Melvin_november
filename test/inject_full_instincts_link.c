#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Forward declare types and functions
typedef struct MelvinFile MelvinFile;
typedef struct GraphParams GraphParams;

extern int melvin_m_init_new_file(const char *path, const GraphParams *params);
extern int melvin_m_map(const char *path, MelvinFile *file);
extern void melvin_m_sync(MelvinFile *file);
extern void close_file(MelvinFile *file);
extern uint64_t melvin_get_num_nodes(MelvinFile *file);
extern uint64_t melvin_get_num_edges(MelvinFile *file);
extern uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id);
extern void melvin_inject_instincts(struct MelvinFile *file);

int main(int argc, char **argv) {
    const char *file_path = argc > 1 ? argv[1] : "melvin.m";
    
    printf("=== INJECTING FULL INSTINCTS INTO melvin.m ===\n\n");
    
    unlink(file_path);
    printf("[1] Removed old file\n");
    
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.015f;
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "[ERROR] Failed to create file\n");
        return 1;
    }
    printf("[2] Created fresh melvin.m file\n");
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "[ERROR] Failed to map file\n");
        return 1;
    }
    
    uint64_t nodes_before = melvin_get_num_nodes(&file);
    uint64_t edges_before = melvin_get_num_edges(&file);
    printf("[3] Mapped file: %llu nodes, %llu edges (before instincts)\n",
           (unsigned long long)nodes_before, (unsigned long long)edges_before);
    
    printf("[4] Injecting ALL instinct patterns...\n");
    melvin_inject_instincts((struct MelvinFile *)&file);
    
    uint64_t nodes_after = melvin_get_num_nodes(&file);
    uint64_t edges_after = melvin_get_num_edges(&file);
    printf("[5] After injection: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_after, (unsigned long long)edges_after);
    printf("     Added: %llu nodes, %llu edges\n",
           (unsigned long long)(nodes_after - nodes_before),
           (unsigned long long)(edges_after - edges_before));
    
    printf("\n[6] Verifying key patterns...\n");
    uint64_t exec_hub = find_node_index_by_id(&file, 50000ULL);
    uint64_t math_in_a = find_node_index_by_id(&file, 60000ULL);
    uint64_t comp_req = find_node_index_by_id(&file, 70000ULL);
    uint64_t port_in = find_node_index_by_id(&file, 80000ULL);
    
    int found = 0;
    if (exec_hub != UINT64_MAX) { printf("     ✓ EXEC:HUB\n"); found++; }
    if (math_in_a != UINT64_MAX) { printf("     ✓ MATH:IN_A\n"); found++; }
    if (comp_req != UINT64_MAX) { printf("     ✓ COMP:REQ\n"); found++; }
    if (port_in != UINT64_MAX) { printf("     ✓ PORT:IN\n"); found++; }
    printf("     Patterns verified: %d/4\n", found);
    
    printf("\n[7] Syncing to disk...\n");
    melvin_m_sync(&file);
    close_file(&file);
    
    printf("\n✓ SUCCESS: Full instincts injected into %s\n", file_path);
    printf("  File contains %llu nodes and %llu edges\n",
           (unsigned long long)nodes_after, (unsigned long long)edges_after);
    
    return 0;
}

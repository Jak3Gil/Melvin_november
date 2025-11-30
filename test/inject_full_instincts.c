#define _POSIX_C_SOURCE 200809L

/*
 * inject_full_instincts.c
 * 
 * Creates melvin.m with ALL instinct patterns injected:
 * - Param nodes
 * - Channel patterns (IN->PROC->OUT triplets)
 * - Code patterns (byte sequences)
 * - Reward patterns (reward hubs)
 * - Body patterns (sensor->internal->motor loops)
 * - EXEC patterns (EXEC:HUB, math ops, compile ops)
 * - Math patterns (MATH:IN_A, MATH:IN_B, etc.)
 * - Compile patterns (COMP:REQ, COMP:SRC, etc.)
 * - Port patterns (PORT:IN, PORT:OUT, etc.)
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "melvin.c"
#include "instincts.c"

int main(int argc, char **argv) {
    const char *file_path = argc > 1 ? argv[1] : "melvin.m";
    
    printf("=== INJECTING FULL INSTINCTS INTO melvin.m ===\n\n");
    
    // Remove old file
    unlink(file_path);
    printf("[1] Removed old file (if any)\n");
    
    // Create fresh file
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
    
    // Map file
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "[ERROR] Failed to map file\n");
        return 1;
    }
    
    uint64_t nodes_before = melvin_get_num_nodes(&file);
    uint64_t edges_before = melvin_get_num_edges(&file);
    printf("[3] Mapped file: %llu nodes, %llu edges (before instincts)\n",
           (unsigned long long)nodes_before,
           (unsigned long long)edges_before);
    
    // Inject full instincts
    printf("[4] Injecting ALL instinct patterns...\n");
    printf("     This includes:\n");
    printf("     - Param nodes (decay, learning rate, etc.)\n");
    printf("     - Channel patterns (7 channels: CODE_RAW, SENSOR, MOTOR, etc.)\n");
    printf("     - Code patterns (byte sequences: 'int ', 'return', etc.)\n");
    printf("     - Reward patterns (R+HUB, R-HUB, R-MIX)\n");
    printf("     - Body patterns (BODY:SENS, BODY:INT, BODY:MOTOR)\n");
    printf("     - EXEC patterns (EXEC:HUB, EXEC:ADD32, EXEC:COMPILE, etc.)\n");
    printf("     - Math patterns (MATH:IN_A, MATH:IN_B, MATH:OUT)\n");
    printf("     - Compile patterns (COMP:REQ, COMP:SRC, COMP:BIN)\n");
    printf("     - Port patterns (PORT:IN, PORT:OUT, PORT:BUF)\n");
    printf("\n");
    
    // Call instincts injection
    // instincts.c is included above, so melvin_inject_instincts is available
    melvin_inject_instincts(&file);
    
    uint64_t nodes_after = melvin_get_num_nodes(&file);
    uint64_t edges_after = melvin_get_num_edges(&file);
    
    printf("[5] After injection: %llu nodes, %llu edges\n",
           (unsigned long long)nodes_after,
           (unsigned long long)edges_after);
    printf("     Added: %llu nodes, %llu edges\n",
           (unsigned long long)(nodes_after - nodes_before),
           (unsigned long long)(edges_after - edges_before));
    
    // Verify some key patterns
    printf("\n[6] Verifying key patterns...\n");
    
    uint64_t patterns_found = 0;
    
    // Check for EXEC:HUB (50000)
    uint64_t exec_hub_idx = find_node_index_by_id(&file, 50000ULL);
    if (exec_hub_idx != UINT64_MAX) {
        printf("     ✓ EXEC:HUB found\n");
        patterns_found++;
    } else {
        printf("     ✗ EXEC:HUB missing\n");
    }
    
    // Check for MATH:IN_A (60000)
    uint64_t math_in_a_idx = find_node_index_by_id(&file, 60000ULL);
    if (math_in_a_idx != UINT64_MAX) {
        printf("     ✓ MATH:IN_A found\n");
        patterns_found++;
    } else {
        printf("     ✗ MATH:IN_A missing\n");
    }
    
    // Check for COMP:REQ (70000)
    uint64_t comp_req_idx = find_node_index_by_id(&file, 70000ULL);
    if (comp_req_idx != UINT64_MAX) {
        printf("     ✓ COMP:REQ found\n");
        patterns_found++;
    } else {
        printf("     ✗ COMP:REQ missing\n");
    }
    
    // Check for PORT:IN (80000)
    uint64_t port_in_idx = find_node_index_by_id(&file, 80000ULL);
    if (port_in_idx != UINT64_MAX) {
        printf("     ✓ PORT:IN found\n");
        patterns_found++;
    } else {
        printf("     ✗ PORT:IN missing\n");
    }
    
    printf("     Patterns verified: %llu/4 key patterns\n", (unsigned long long)patterns_found);
    
    // Sync to disk
    printf("\n[7] Syncing to disk...\n");
    melvin_m_sync(&file);
    close_file(&file);
    
    printf("\n✓ SUCCESS: Full instincts injected into %s\n", file_path);
    printf("  File contains %llu nodes and %llu edges\n",
           (unsigned long long)nodes_after,
           (unsigned long long)edges_after);
    printf("  All instinct patterns are permanently embedded!\n");
    
    return 0;
}


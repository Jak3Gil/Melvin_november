/*
 * TEST 2: Multi-Hop Reasoning - PERSISTENT BRAIN
 * 
 * Uses SAME melvin_brain.m from Test 1
 * Graph already knows A→B, now learns B→C, then tests A→C
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "melvin_simple.h"

#define BRAIN_FILE "melvin_brain.m"  // SAME FILE as Test 1
#define NUM_AB_EPISODES 50   // Less needed (graph may already know A→B)
#define NUM_BC_EPISODES 100

int main(int argc, char **argv) {
    bool fresh = false;
    if (argc > 1 && strcmp(argv[1], "--fresh") == 0) {
        fresh = true;
        unlink(BRAIN_FILE);
    }
    
    printf("========================================\n");
    printf("TEST 2: Multi-Hop Reasoning (A→B→C)\n");
    printf("Brain: %s (continuing from Test 1)\n", BRAIN_FILE);
    printf("========================================\n\n");
    
    if (fresh || access(BRAIN_FILE, F_OK) != 0) {
        printf("Creating new brain...\n");
        if (melvin_simple_create_brain(BRAIN_FILE) < 0) {
            fprintf(stderr, "FAILED: Cannot create brain\n");
            return 1;
        }
    } else {
        printf("Using existing brain (graph already knows A→B from Test 1)\n");
    }
    
    MelvinSimple *m = melvin_open(BRAIN_FILE);
    if (!m) {
        fprintf(stderr, "FAILED: Cannot open brain\n");
        return 1;
    }
    
    MelvinStats stats;
    melvin_stats(m, &stats);
    printf("Starting state: %llu nodes, %llu edges\n\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    
    // Phase 1: Reinforce A→B (graph may already know this)
    printf("[PHASE 1] Reinforcing A→B (%d episodes)...\n", NUM_AB_EPISODES);
    for (int i = 0; i < NUM_AB_EPISODES; i++) {
        melvin_feed(m, 1, 'A');
        melvin_tick(m, 50);
        melvin_feed(m, 1, 'B');
        melvin_tick(m, 50);
    }
    
    melvin_stats(m, &stats);
    printf("  After A→B: %llu nodes, %llu edges\n\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    
    // Phase 2: Learn B→C (NEW - graph doesn't know this yet)
    printf("[PHASE 2] Learning B→C (%d episodes)...\n", NUM_BC_EPISODES);
    for (int i = 0; i < NUM_BC_EPISODES; i++) {
        melvin_feed(m, 1, 'B');
        melvin_tick(m, 50);
        melvin_feed(m, 1, 'C');
        melvin_tick(m, 50);
    }
    
    melvin_stats(m, &stats);
    printf("  After B→C: %llu nodes, %llu edges\n\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    
    // Probe: Can graph chain A→B→C?
    printf("[PROBE] Testing multi-hop: A → ? → ?\n");
    melvin_feed(m, 1, 'A');
    melvin_tick(m, 300);  // Let activation propagate through chain
    
    float b_activation = melvin_read_byte(m, 'B');
    float c_activation = melvin_read_byte(m, 'C');
    
    printf("  After feeding 'A':\n");
    printf("    B activation (1-hop): %.4f\n", b_activation);
    printf("    C activation (2-hop): %.4f\n", c_activation);
    
    bool multi_hop_works = (c_activation > 0.2f);
    printf("\n%s\n", multi_hop_works ? "✓ SUCCESS: Multi-hop reasoning works!" : "✗ FAILED");
    printf("\n[Brain saved - next test will use this same graph]\n");
    
    melvin_close(m);
    return multi_hop_works ? 0 : 1;
}


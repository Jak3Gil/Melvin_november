/*
 * TEST 1: Simple Association (A→B) - PERSISTENT BRAIN
 * 
 * Uses persistent melvin_brain.m - all tests share the same graph
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "melvin_simple.h"

#define BRAIN_FILE "melvin_brain.m"  // PERSISTENT - shared by all tests
#define NUM_EPISODES 100

int main(int argc, char **argv) {
    bool fresh = false;
    if (argc > 1 && strcmp(argv[1], "--fresh") == 0) {
        fresh = true;
        unlink(BRAIN_FILE);
    }
    
    printf("========================================\n");
    printf("TEST 1: Can Melvin Learn A→B?\n");
    printf("Brain: %s (persistent)\n", BRAIN_FILE);
    printf("========================================\n\n");
    
    // Create brain only if needed
    if (fresh || access(BRAIN_FILE, F_OK) != 0) {
        printf("Creating new brain...\n");
        if (melvin_simple_create_brain(BRAIN_FILE) < 0) {
            fprintf(stderr, "FAILED: Cannot create brain\n");
            return 1;
        }
    } else {
        printf("Using existing brain (continuing from previous learning)\n");
    }
    
    // Open brain
    MelvinSimple *m = melvin_open(BRAIN_FILE);
    if (!m) {
        fprintf(stderr, "FAILED: Cannot open brain\n");
        return 1;
    }
    
    MelvinStats initial_stats;
    melvin_stats(m, &initial_stats);
    printf("Initial state: %llu nodes, %llu edges\n\n",
           (unsigned long long)initial_stats.num_nodes,
           (unsigned long long)initial_stats.num_edges);
    
    // Training: Feed A→B pattern with learning curve tracking
    printf("[TRAINING] Feeding A→B pattern %d times...\n", NUM_EPISODES);
    printf("\nLearning Curve (P(B|A) and edge weight A→B):\n");
    printf("Episode | P(B|A)    | Weight A→B\n");
    printf("--------|-----------|------------\n");
    
    for (int i = 0; i < NUM_EPISODES; i++) {
        melvin_feed(m, 1, 'A');
        melvin_tick(m, 50);
        melvin_feed(m, 1, 'B');
        melvin_tick(m, 50);
        
        // Log learning curve every 10 episodes
        if ((i + 1) % 10 == 0 || i == 0) {
            // Probe: feed A, check B activation
            melvin_feed(m, 1, 'A');
            melvin_tick(m, 100);
            float p_b_given_a = melvin_read_byte(m, 'B');
            float edge_weight = melvin_get_edge_weight(m, 'A', 'B');
            
            printf("  %3d   | %9.4f | %9.4f\n", i + 1, p_b_given_a, edge_weight);
        }
    }
    
    // Probe: Feed A, check if B activates
    printf("\n[PROBE] After feeding 'A', does 'B' activate?\n");
    melvin_feed(m, 1, 'A');
    melvin_tick(m, 200);
    
    float b_activation = melvin_read_byte(m, 'B');
    printf("  B activation: %.4f\n", b_activation);
    
    MelvinStats final_stats;
    melvin_stats(m, &final_stats);
    printf("\nFinal state: %llu nodes, %llu edges\n",
           (unsigned long long)final_stats.num_nodes,
           (unsigned long long)final_stats.num_edges);
    
    bool learned = (b_activation > 0.3f);
    printf("\n%s\n", learned ? "✓ SUCCESS: Graph learned A→B" : "✗ FAILED");
    printf("\n[Brain saved - next test will use this same graph]\n");
    
    melvin_close(m);
    return learned ? 0 : 1;
}


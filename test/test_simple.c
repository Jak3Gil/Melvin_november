/*
 * SIMPLEST POSSIBLE TEST
 * 
 * Just: feed bytes → tick → read
 * That's it. No instincts, no exec, no complexity.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "melvin_simple.h"

// Helper to create brain (exposed from melvin_simple.c)
extern int melvin_simple_create_brain(const char *path);

int main(int argc, char **argv) {
    const char *brain = "melvin_brain.m";  // PERSISTENT - same file for all tests
    bool fresh = false;
    
    if (argc > 1 && strcmp(argv[1], "--fresh") == 0) {
        fresh = true;
        unlink(brain);
    }
    
    printf("=== SIMPLEST TEST ===\n");
    printf("Brain: %s (persistent across all tests)\n\n", brain);
    
    // Create brain only if it doesn't exist (or --fresh flag)
    if (fresh || access(brain, F_OK) != 0) {
        printf("Creating new brain...\n");
        if (melvin_simple_create_brain(brain) < 0) {
            fprintf(stderr, "Failed to create brain\n");
            return 1;
        }
    } else {
        printf("Using existing brain (continuing from previous tests)\n");
    }
    
    // Open brain
    MelvinSimple *m = melvin_open(brain);
    if (!m) {
        fprintf(stderr, "Failed to open brain\n");
        return 1;
    }
    
    printf("Brain opened\n\n");
    
    // Feed pattern: A B A B A B ...
    printf("Feeding pattern: A B A B A B ...\n");
    for (int i = 0; i < 50; i++) {
        melvin_feed(m, 1, 'A');  // Channel 1, byte 'A'
        melvin_tick(m, 50);
        
        melvin_feed(m, 1, 'B');  // Channel 1, byte 'B'
        melvin_tick(m, 50);
    }
    
    // Check: After feeding 'A', does 'B' activate?
    printf("\nTesting: After 'A', does 'B' activate?\n");
    melvin_feed(m, 1, 'A');
    melvin_tick(m, 200);  // Let graph process
    
    float b_activation = melvin_read_byte(m, 'B');
    printf("  B activation: %.4f\n", b_activation);
    
    // Stats
    MelvinStats stats;
    melvin_stats(m, &stats);
    printf("\nGraph stats:\n");
    printf("  Nodes: %llu\n", (unsigned long long)stats.num_nodes);
    printf("  Edges: %llu\n", (unsigned long long)stats.num_edges);
    printf("  Avg activation: %.4f\n", stats.avg_activation);
    
    printf("\n%s\n", b_activation > 0.3f ? "✓ SUCCESS" : "✗ FAILED");
    printf("\n[Brain saved to %s - next test will continue from here]\n", brain);
    
    melvin_close(m);
    return (b_activation > 0.3f) ? 0 : 1;
}


/*
 * TEST 2: Multi-Hop Reasoning (A→B, B→C ⇒ A→C)
 * 
 * Goal: Show that the graph can chain simple relations.
 * 
 * Training: Provide A→B and B→C examples (NOT A→C directly)
 * Probe: Feed A, see if C activates (multi-hop reasoning)
 * 
 * This test respects TESTING_CONTRACT.md
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "melvin_test_api.h"

#define BRAIN_FILE "test_2_brain.m"
#define NUM_AB_EPISODES 100
#define NUM_BC_EPISODES 100

// ========================================================================
// CONTRACT COMPLIANCE CHECKLIST
// ========================================================================
// [✓] Rule 1: melvin.m is the brain
// [✓] Rule 2: Only bytes in/out via API
// [✓] Rule 3: No resets (can reuse test_1 brain)
// [✓] Rule 4: All learning internal
// [✓] Rule 5: Append/evolve only
// ========================================================================

static void train_ab_pattern(MelvinCtx *ctx, int episodes) {
    for (int i = 0; i < episodes; i++) {
        melvin_ingest_byte(ctx, 'A', 1.0f);
        melvin_tick(ctx, 50);
        melvin_ingest_byte(ctx, 'B', 1.0f);
        melvin_tick(ctx, 50);
    }
}

static void train_bc_pattern(MelvinCtx *ctx, int episodes) {
    for (int i = 0; i < episodes; i++) {
        melvin_ingest_byte(ctx, 'B', 1.0f);
        melvin_tick(ctx, 50);
        melvin_ingest_byte(ctx, 'C', 1.0f);
        melvin_tick(ctx, 50);
    }
}

static void probe_multihop(MelvinCtx *ctx, float *b_activation, float *c_activation) {
    // Feed A only
    melvin_ingest_byte(ctx, 'A', 1.0f);
    melvin_tick(ctx, 300);  // Let activation propagate through chain
    
    // Check if B activates (1-hop)
    if (b_activation) {
        *b_activation = melvin_get_data_node_activation(ctx, 'B');
    }
    
    // Check if C activates (2-hop)
    if (c_activation) {
        *c_activation = melvin_get_data_node_activation(ctx, 'C');
    }
}

int main(int argc, char **argv) {
    bool fresh_brain = false;
    if (argc > 1 && strcmp(argv[1], "--fresh") == 0) {
        fresh_brain = true;
        unlink(BRAIN_FILE);
    }
    
    printf("========================================\n");
    printf("TEST 2: Multi-Hop Reasoning\n");
    printf("         (A→B, B→C ⇒ A→C?)\n");
    printf("========================================\n\n");
    
    // Create or reuse brain
    if (fresh_brain || access(BRAIN_FILE, F_OK) != 0) {
        printf("Creating fresh brain...\n");
        
        if (melvin_test_create_brain(BRAIN_FILE) < 0) {
            fprintf(stderr, "FAILED: Cannot create brain\n");
            return 1;
        }
        
        if (melvin_test_inject_instincts(BRAIN_FILE) < 0) {
            fprintf(stderr, "FAILED: Cannot inject instincts\n");
            return 1;
        }
    }
    
    MelvinCtx *ctx;
    if (!melvin_open(BRAIN_FILE, &ctx)) {
        fprintf(stderr, "FAILED: Cannot open brain\n");
        return 1;
    }
    
    MelvinStats stats;
    melvin_get_stats(ctx, &stats);
    printf("Initial state: %llu nodes, %llu edges\n\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    
    // ========================================================================
    // TRAINING PHASE 1: A→B (DO NOT train A→C directly)
    // ========================================================================
    printf("[TRAINING PHASE 1] A→B pattern (%d episodes)...\n", NUM_AB_EPISODES);
    train_ab_pattern(ctx, NUM_AB_EPISODES);
    
    melvin_get_stats(ctx, &stats);
    printf("  After A→B: %llu nodes, %llu edges\n\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    
    // ========================================================================
    // TRAINING PHASE 2: B→C (DO NOT train A→C directly)
    // ========================================================================
    printf("[TRAINING PHASE 2] B→C pattern (%d episodes)...\n", NUM_BC_EPISODES);
    train_bc_pattern(ctx, NUM_BC_EPISODES);
    
    melvin_get_stats(ctx, &stats);
    printf("  After B→C: %llu nodes, %llu edges\n\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    
    // ========================================================================
    // PROBE: Can graph chain A→B→C?
    // ========================================================================
    printf("[PROBE] Testing multi-hop: A → ? → ?\n");
    
    float b_activation, c_activation;
    probe_multihop(ctx, &b_activation, &c_activation);
    
    printf("  After feeding 'A':\n");
    printf("    B activation (1-hop): %.4f\n", b_activation);
    printf("    C activation (2-hop): %.4f\n", c_activation);
    
    // ========================================================================
    // EVALUATION (External to graph)
    // ========================================================================
    float threshold = 0.2f;  // External threshold
    bool one_hop_works = (b_activation > threshold);
    bool multi_hop_works = (c_activation > threshold);
    
    printf("\n[RESULTS]\n");
    printf("  1-hop (A→B): %s (activation %.4f)\n",
           one_hop_works ? "✓ WORKS" : "✗ FAILED",
           b_activation);
    printf("  2-hop (A→C): %s (activation %.4f)\n",
           multi_hop_works ? "✓ WORKS" : "✗ FAILED",
           c_activation);
    
    bool success = multi_hop_works;
    
    printf("\n[CONCLUSION]\n");
    if (success) {
        printf("  ✓ SUCCESS: Graph can chain A→B→C\n");
        printf("    Multi-hop reasoning detected\n");
    } else {
        printf("  ✗ FAILED: Graph cannot reliably chain relations\n");
        printf("    C activation (%.4f) below threshold (%.2f)\n",
               c_activation, threshold);
        if (one_hop_works) {
            printf("    Note: 1-hop (A→B) works, but 2-hop does not\n");
        }
    }
    
    melvin_close(ctx);
    return success ? 0 : 1;
}


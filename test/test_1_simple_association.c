/*
 * TEST 1: Can Melvin Learn a 1-Step Association? (A→B)
 * 
 * Goal: Show that the graph can reduce prediction error on a trivial pattern.
 * 
 * This test respects TESTING_CONTRACT.md:
 * - Only bytes in/out
 * - No graph manipulation
 * - All learning is internal
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "melvin_test_api.h"

#define BRAIN_FILE "test_1_brain.m"
#define NUM_EPISODES 100
#define PROBE_INTERVAL 10

// ========================================================================
// CONTRACT COMPLIANCE CHECKLIST
// ========================================================================
// [✓] Rule 1: melvin.m is the brain
// [✓] Rule 2: Only bytes in/out via API
// [✓] Rule 3: No resets (reuses brain file)
// [✓] Rule 4: All learning internal
// [✓] Rule 5: Append/evolve only
// ========================================================================

typedef struct {
    int episode;
    float b_activation_after_a;
    float ab_edge_weight;
    float prediction_error;
    bool correct_prediction;
} ProbeResult;

static void run_episode(MelvinCtx *ctx, bool training) {
    if (training) {
        // Training: Feed A B sequence
        melvin_ingest_byte(ctx, 'A', 1.0f);
        melvin_tick(ctx, 50);  // Let graph process
        
        melvin_ingest_byte(ctx, 'B', 1.0f);
        melvin_tick(ctx, 50);  // Let graph learn pattern
    } else {
        // Probe: Feed A only, see if B activates
        melvin_ingest_byte(ctx, 'A', 1.0f);
        melvin_tick(ctx, 200);  // Let activation propagate
    }
}

static ProbeResult probe_graph(MelvinCtx *ctx) {
    ProbeResult result = {0};
    
    // Probe: Feed A, check if B activates
    run_episode(ctx, false);
    
    // Read B's activation (this is how we measure "prediction")
    result.b_activation_after_a = melvin_get_data_node_activation(ctx, 'B');
    
    // Get edge weight (if available)
    result.ab_edge_weight = melvin_get_edge_weight(ctx, 'A', 'B');
    
    // Get stats
    MelvinStats stats;
    melvin_get_stats(ctx, &stats);
    result.prediction_error = stats.avg_activation;  // Use activation as proxy for now
    
    // Threshold for "correct prediction" (external evaluation)
    result.correct_prediction = (result.b_activation_after_a > 0.3f);
    
    return result;
}

int main(int argc, char **argv) {
    bool fresh_brain = false;
    if (argc > 1 && strcmp(argv[1], "--fresh") == 0) {
        fresh_brain = true;
        unlink(BRAIN_FILE);
    }
    
    printf("========================================\n");
    printf("TEST 1: Can Melvin Learn A→B?\n");
    printf("========================================\n\n");
    
    // Create brain if needed
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
    
    // Open brain
    MelvinCtx *ctx;
    if (!melvin_open(BRAIN_FILE, &ctx)) {
        fprintf(stderr, "FAILED: Cannot open brain\n");
        return 1;
    }
    
    MelvinStats initial_stats;
    melvin_get_stats(ctx, &initial_stats);
    printf("Initial state: %llu nodes, %llu edges\n",
           (unsigned long long)initial_stats.num_nodes,
           (unsigned long long)initial_stats.num_edges);
    
    printf("\n[TRAINING] Feeding A→B pattern %d times...\n\n", NUM_EPISODES);
    
    ProbeResult probes[NUM_EPISODES / PROBE_INTERVAL + 1];
    int probe_count = 0;
    
    // Training loop
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        // Training episode
        run_episode(ctx, true);
        
        // Probe every PROBE_INTERVAL episodes
        if ((episode + 1) % PROBE_INTERVAL == 0 || episode == 0) {
            ProbeResult probe = probe_graph(ctx);
            probe.episode = episode + 1;
            probes[probe_count++] = probe;
            
            printf("  Episode %3d: B activation=%.4f, correct=%s\n",
                   episode + 1,
                   probe.b_activation_after_a,
                   probe.correct_prediction ? "YES" : "NO");
        }
    }
    
    // Final probe
    ProbeResult final_probe = probe_graph(ctx);
    final_probe.episode = NUM_EPISODES;
    probes[probe_count++] = final_probe;
    
    // Get final stats
    MelvinStats final_stats;
    melvin_get_stats(ctx, &final_stats);
    
    printf("\n[RESULTS]\n");
    printf("  Final state: %llu nodes, %llu edges\n",
           (unsigned long long)final_stats.num_nodes,
           (unsigned long long)final_stats.num_edges);
    printf("  B activation after A: %.4f (started at %.4f)\n",
           probes[probe_count-1].b_activation_after_a,
           probes[0].b_activation_after_a);
    
    // Calculate success metrics
    int correct_count = 0;
    for (int i = 0; i < probe_count; i++) {
        if (probes[i].correct_prediction) correct_count++;
    }
    float accuracy = (float)correct_count / probe_count;
    
    printf("  Prediction accuracy: %.1f%% (%d/%d probes)\n",
           accuracy * 100.0f, correct_count, probe_count);
    
    // Check if learning happened
    float improvement = probes[probe_count-1].b_activation_after_a - probes[0].b_activation_after_a;
    bool learned = (improvement > 0.1f) && (accuracy > 0.5f);
    
    printf("\n[CONCLUSION]\n");
    if (learned) {
        printf("  ✓ SUCCESS: Graph learned A→B association\n");
        printf("    Activation increased by %.4f\n", improvement);
        printf("    Final accuracy: %.1f%%\n", accuracy * 100.0f);
    } else {
        printf("  ✗ FAILED: Graph did not learn A→B reliably\n");
        printf("    Activation change: %.4f (need > 0.1)\n", improvement);
        printf("    Final accuracy: %.1f%% (need > 50%%)\n", accuracy * 100.0f);
    }
    
    printf("\n[Saving brain for next test...]\n");
    melvin_close(ctx);
    
    return learned ? 0 : 1;
}


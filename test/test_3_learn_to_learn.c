/*
 * TEST 3: "Learn to Learn" (Meta-Learning)
 * 
 * Goal: Show that previous training makes future similar learning require fewer examples.
 * 
 * Phase 1: Learn pattern with one symbol set (A→B, B→C, measure N₁ examples)
 * Phase 2: Learn analogous pattern with new symbols (X→Y, Y→Z, measure N₂ examples)
 * 
 * Success: N₂ < N₁ (reusing internal structure to learn faster)
 * 
 * This test respects TESTING_CONTRACT.md
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "melvin_test_api.h"

#define BRAIN_FILE "test_3_brain.m"
#define TARGET_ACCURACY 0.7f  // Target accuracy threshold (70%)
#define MAX_EPISODES 500

// ========================================================================
// CONTRACT COMPLIANCE CHECKLIST
// ========================================================================
// [✓] Rule 1: melvin.m is the brain
// [✓] Rule 2: Only bytes in/out via API
// [✓] Rule 3: No resets (brain persists across phases)
// [✓] Rule 4: All learning internal
// [✓] Rule 5: Append/evolve only
// ========================================================================

typedef struct {
    int episodes_to_learn;
    float final_accuracy;
    bool reached_target;
} LearningResult;

static float probe_accuracy(MelvinCtx *ctx, uint8_t first, uint8_t second) {
    // Feed first symbol
    melvin_ingest_byte(ctx, first, 1.0f);
    melvin_tick(ctx, 200);
    
    // Check if second symbol activates
    float activation = melvin_get_data_node_activation(ctx, second);
    float threshold = 0.3f;
    return (activation > threshold) ? 1.0f : 0.0f;
}

static LearningResult train_pattern(MelvinCtx *ctx, uint8_t first, uint8_t second, uint8_t third) {
    LearningResult result = {0};
    result.reached_target = false;
    
    printf("  Training pattern: %c→%c, %c→%c\n", first, second, second, third);
    
    int probe_interval = 10;
    float accuracy_history[100];
    int history_count = 0;
    
    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        // Train first→second
        melvin_ingest_byte(ctx, first, 1.0f);
        melvin_tick(ctx, 50);
        melvin_ingest_byte(ctx, second, 1.0f);
        melvin_tick(ctx, 50);
        
        // Train second→third
        melvin_ingest_byte(ctx, second, 1.0f);
        melvin_tick(ctx, 50);
        melvin_ingest_byte(ctx, third, 1.0f);
        melvin_tick(ctx, 50);
        
        // Probe accuracy periodically
        if ((episode + 1) % probe_interval == 0) {
            // Test if graph can chain first→second→third
            float acc1 = probe_accuracy(ctx, first, second);
            float acc2 = probe_accuracy(ctx, second, third);
            
            // Multi-hop: can it do first→third?
            melvin_ingest_byte(ctx, first, 1.0f);
            melvin_tick(ctx, 300);
            float third_activation = melvin_get_data_node_activation(ctx, third);
            float acc_multi = (third_activation > 0.2f) ? 1.0f : 0.0f;
            
            // Overall accuracy (average of all three)
            float accuracy = (acc1 + acc2 + acc_multi) / 3.0f;
            accuracy_history[history_count++] = accuracy;
            
            if (episode < 50 || (episode + 1) % 50 == 0) {
                printf("    Episode %3d: accuracy=%.2f\n", episode + 1, accuracy);
            }
            
            // Check if we reached target
            if (accuracy >= TARGET_ACCURACY && !result.reached_target) {
                result.episodes_to_learn = episode + 1;
                result.final_accuracy = accuracy;
                result.reached_target = true;
                printf("    ✓ Target reached at episode %d (accuracy=%.2f)\n",
                       result.episodes_to_learn, result.final_accuracy);
                break;
            }
        }
    }
    
    if (!result.reached_target) {
        result.episodes_to_learn = MAX_EPISODES;
        result.final_accuracy = accuracy_history[history_count - 1];
        printf("    ✗ Did not reach target (final accuracy=%.2f)\n",
               result.final_accuracy);
    }
    
    return result;
}

int main(int argc, char **argv) {
    bool fresh_brain = false;
    if (argc > 1 && strcmp(argv[1], "--fresh") == 0) {
        fresh_brain = true;
        unlink(BRAIN_FILE);
    }
    
    printf("========================================\n");
    printf("TEST 3: Learn to Learn (Meta-Learning)\n");
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
    // PHASE 1: Learn pattern with first symbol set
    // ========================================================================
    printf("[PHASE 1] Learning pattern with symbols A, B, C\n");
    printf("  Target accuracy: %.0f%%\n\n", TARGET_ACCURACY * 100.0f);
    
    LearningResult phase1 = train_pattern(ctx, 'A', 'B', 'C');
    
    melvin_get_stats(ctx, &stats);
    printf("\n  After Phase 1: %llu nodes, %llu edges\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    printf("  Episodes to learn: %d\n", phase1.episodes_to_learn);
    printf("  Final accuracy: %.2f\n\n", phase1.final_accuracy);
    
    if (!phase1.reached_target) {
        printf("[WARNING] Phase 1 did not reach target.\n");
        printf("  Cannot meaningfully test meta-learning.\n");
        melvin_close(ctx);
        return 1;
    }
    
    // ========================================================================
    // PHASE 2: Learn analogous pattern with NEW symbols (same structure)
    // ========================================================================
    printf("[PHASE 2] Learning analogous pattern with symbols X, Y, Z\n");
    printf("  Brain already knows the structure (A→B→C)\n");
    printf("  Can it learn X→Y→Z faster?\n\n");
    
    LearningResult phase2 = train_pattern(ctx, 'X', 'Y', 'Z');
    
    melvin_get_stats(ctx, &stats);
    printf("\n  After Phase 2: %llu nodes, %llu edges\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    printf("  Episodes to learn: %d\n", phase2.episodes_to_learn);
    printf("  Final accuracy: %.2f\n\n", phase2.final_accuracy);
    
    // ========================================================================
    // EVALUATION: Did it learn faster?
    // ========================================================================
    printf("[RESULTS]\n");
    printf("  Phase 1 (A→B→C): %d episodes\n", phase1.episodes_to_learn);
    printf("  Phase 2 (X→Y→Z): %d episodes\n", phase2.episodes_to_learn);
    
    float speedup = (float)phase1.episodes_to_learn / phase2.episodes_to_learn;
    bool meta_learned = (phase2.episodes_to_learn < phase1.episodes_to_learn);
    
    printf("  Speedup: %.2fx\n", speedup);
    
    printf("\n[CONCLUSION]\n");
    if (meta_learned && phase2.reached_target) {
        printf("  ✓ SUCCESS: Graph learned to learn!\n");
        printf("    Phase 2 required %.0f%% fewer episodes than Phase 1\n",
               (1.0f - 1.0f/speedup) * 100.0f);
        printf("    Graph is reusing internal structure\n");
    } else if (phase2.reached_target) {
        printf("  ✗ PARTIAL: Graph learned both patterns, but not faster\n");
        printf("    Phase 2 took %d episodes (vs %d in Phase 1)\n",
               phase2.episodes_to_learn, phase1.episodes_to_learn);
        printf("    No clear meta-learning signal\n");
    } else {
        printf("  ✗ FAILED: Phase 2 did not reach target\n");
        printf("    Cannot evaluate meta-learning\n");
    }
    
    melvin_close(ctx);
    return meta_learned ? 0 : 1;
}


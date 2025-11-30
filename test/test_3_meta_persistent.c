/*
 * TEST 3: Learn to Learn (Meta-Learning) - PERSISTENT BRAIN
 * 
 * Uses SAME melvin_brain.m from Tests 1 & 2
 * Tests if graph learns faster after seeing similar patterns
 * 
 * Design: Train multiple pairs in sequence, measure if later pairs learn faster
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "melvin_simple.h"

#define BRAIN_FILE "melvin_brain.m"  // SAME FILE - all previous learning available
#define TARGET_ACTIVATION 0.3f
#define MAX_EPISODES 150

// Train a single pair for FIXED episodes, return final activation quality
// This measures learning SPEED (higher activation after same training = faster learning)
static float train_pair_fixed_episodes(MelvinSimple *m, uint8_t first, uint8_t second, 
                                       int fixed_episodes) {
    for (int i = 0; i < fixed_episodes; i++) {
        // Train first→second
        melvin_feed(m, 1, first);
        melvin_tick(m, 50);
        melvin_feed(m, 1, second);
        melvin_tick(m, 50);
    }
    
    // Probe: feed first, measure second's activation
    melvin_feed(m, 1, first);
    melvin_tick(m, 100);
    float activation = melvin_read_byte(m, second);
    
    return activation;
}

int main(int argc, char **argv) {
    bool fresh = false;
    if (argc > 1 && strcmp(argv[1], "--fresh") == 0) {
        fresh = true;
        unlink(BRAIN_FILE);
    }
    
    printf("========================================\n");
    printf("TEST 3: Learn to Learn (Meta-Learning)\n");
    printf("Brain: %s (continuing from Tests 1 & 2)\n", BRAIN_FILE);
    printf("========================================\n\n");
    
    printf("Strategy: Train multiple similar pairs in sequence\n");
    printf("Measure if later pairs learn faster (indicates meta-learning)\n\n");
    
    if (fresh || access(BRAIN_FILE, F_OK) != 0) {
        printf("Creating new brain...\n");
        if (melvin_simple_create_brain(BRAIN_FILE) < 0) {
            fprintf(stderr, "FAILED: Cannot create brain\n");
            return 1;
        }
    } else {
        printf("Using existing brain (graph knows A→B→C from previous tests)\n");
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
    
    // Train a family of similar patterns: X→Y, M→N, P→Q, R→S
    // All are structurally identical (single-step associations)
    // If meta-learning works, later pairs should learn faster
    
    struct {
        uint8_t first, second;
        char name[8];
        float final_activation;
    } pairs[] = {
        {'X', 'Y', "X→Y", 0.0f},
        {'M', 'N', "M→N", 0.0f},
        {'P', 'Q', "P→Q", 0.0f},
        {'R', 'S', "R→S", 0.0f},
    };
    
    int num_pairs = sizeof(pairs) / sizeof(pairs[0]);
    int fixed_episodes = 15;  // Train each pair for same number of episodes
    
    printf("[PHASE 1] Training %d pairs sequentially (%d episodes each)...\n", 
           num_pairs, fixed_episodes);
    printf("Measuring activation QUALITY after fixed training\n");
    printf("(Higher activation = faster learning)\n\n");
    printf("Pair      | Final Act | Edge Weight\n");
    printf("----------|-----------|------------\n");
    
    for (int i = 0; i < num_pairs; i++) {
        printf("[Training %s] ", pairs[i].name);
        fflush(stdout);
        
        pairs[i].final_activation = train_pair_fixed_episodes(m, pairs[i].first, 
                                                              pairs[i].second, fixed_episodes);
        float edge_weight = melvin_get_edge_weight(m, pairs[i].first, pairs[i].second);
        
        printf("%s | %9.3f | %11.4f\n", 
               pairs[i].name, 
               pairs[i].final_activation,
               edge_weight);
        
        melvin_stats(m, &stats);
        printf("          | Graph now: %llu nodes, %llu edges\n", 
               (unsigned long long)stats.num_nodes,
               (unsigned long long)stats.num_edges);
    }
    
    printf("\n[RESULTS]\n");
    printf("Activation after %d training episodes:\n", fixed_episodes);
    for (int i = 0; i < num_pairs; i++) {
        printf("  %s: %.3f activation\n", pairs[i].name, pairs[i].final_activation);
    }
    
    // Analyze trend: do later pairs achieve higher activation (faster learning)?
    float first_half_sum = 0.0f, second_half_sum = 0.0f;
    int first_half_count = (num_pairs + 1) / 2;
    
    for (int i = 0; i < first_half_count; i++) {
        first_half_sum += pairs[i].final_activation;
    }
    for (int i = first_half_count; i < num_pairs; i++) {
        second_half_sum += pairs[i].final_activation;
    }
    
    float avg_first = first_half_sum / (float)first_half_count;
    float avg_second = second_half_sum / (float)(num_pairs - first_half_count);
    
    printf("\n[ANALYSIS]\n");
    printf("  First half avg activation: %.3f\n", avg_first);
    printf("  Second half avg activation: %.3f\n", avg_second);
    
    bool meta_learned = (avg_second > avg_first) && (avg_first > 0.01f);
    float improvement = avg_first > 0.01f ? (avg_second / avg_first - 1.0f) * 100.0f : 0.0f;
    
    if (meta_learned) {
        printf("  Improvement: +%.1f%% higher activation in second half!\n", improvement);
        printf("\n✓ SUCCESS: Evidence of meta-learning!\n");
        printf("  Graph learned similar patterns FASTER (higher activation after same training)\n");
        printf("  after exposure to analogous tasks. This suggests structural pattern reuse.\n");
    } else if (avg_second < avg_first * 0.95f) {
        printf("  Decline: %.1f%% lower activation in second half\n", -improvement);
        printf("\n✗ FAILED: No meta-learning signal\n");
        printf("  Later pairs did not learn faster. Graph may be:\n");
        printf("  - Memorizing each pair separately (no pattern reuse)\n");
        printf("  - Interference between similar patterns\n");
    } else {
        printf("  Change: %.1f%% (within noise margin)\n", improvement);
        printf("\n? INCONCLUSIVE: Insufficient difference to conclude meta-learning\n");
        printf("  May need:\n");
        printf("  - More pairs to see trend\n");
        printf("  - Different test design (hierarchical patterns, templates)\n");
    }
    
    melvin_stats(m, &stats);
    printf("\nFinal graph state: %llu nodes, %llu edges\n",
           (unsigned long long)stats.num_nodes,
           (unsigned long long)stats.num_edges);
    printf("\n[Brain saved - all learning preserved in %s]\n", BRAIN_FILE);
    
    melvin_close(m);
    return meta_learned ? 0 : 1;
}

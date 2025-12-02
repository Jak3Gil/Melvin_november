/*
 * Experiment 2: Hierarchical Composition Efficiency
 * 
 * Hypothesis: As patterns compose hierarchically, Melvin's efficiency
 * grows exponentially while LSTM must learn everything from scratch.
 * 
 * Test Design:
 * 1. Phase 1: Teach low-level patterns (AB, CD, EF)
 * 2. Phase 2: Teach mid-level patterns (ABCD, CDEF, EFAB)
 * 3. Phase 3: Teach high-level patterns (ABCDEFAB)
 * 4. Measure: Pattern reuse, memory growth, learning speed
 * 
 * Expected: Melvin reuses patterns exponentially, LSTM scales linearly
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/melvin.h"

/* Test patterns - increasing complexity */
const char *LEVEL1_PATTERNS[] = {"AB", "CD", "EF", "GH"};
const int LEVEL1_COUNT = 4;

const char *LEVEL2_PATTERNS[] = {"ABCD", "CDEF", "EFGH", "GHAB"};
const int LEVEL2_COUNT = 4;

const char *LEVEL3_PATTERNS[] = {"ABCDEFGH", "GHEFCDAB", "ABCDEFGHAB"};
const int LEVEL3_COUNT = 3;

/* Feed pattern and return edge/pattern creation */
typedef struct {
    int new_nodes;
    int new_edges;
    int new_patterns;
} GrowthMetrics;

GrowthMetrics feed_pattern(Graph *g, const char *pattern, uint32_t port, int reps) {
    uint64_t start_nodes = g->node_count;
    uint64_t start_edges = g->edge_count;
    
    /* Count patterns before */
    int patterns_before = 0;
    for (uint64_t i = 840; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) patterns_before++;
    }
    
    /* Feed pattern multiple times */
    for (int r = 0; r < reps; r++) {
        for (int i = 0; pattern[i] != '\0'; i++) {
            melvin_feed_byte(g, port, (uint8_t)pattern[i], 1.0f);
        }
        melvin_feed_byte(g, port, ' ', 0.5f);  /* Separator */
    }
    
    /* Count patterns after */
    int patterns_after = 0;
    for (uint64_t i = 840; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) patterns_after++;
    }
    
    GrowthMetrics metrics;
    metrics.new_nodes = (int)(g->node_count - start_nodes);
    metrics.new_edges = (int)(g->edge_count - start_edges);
    metrics.new_patterns = patterns_after - patterns_before;
    
    return metrics;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    
    printf("==============================================\n");
    printf("EXPERIMENT 2: HIERARCHICAL COMPOSITION\n");
    printf("==============================================\n\n");
    
    printf("Testing: Does Melvin reuse patterns as complexity grows?\n");
    printf("Hypothesis: Pattern reuse → exponential efficiency\n\n");
    
    /* CSV output */
    FILE *csv = fopen("benchmarks/data/experiment2_results.csv", "w");
    if (csv) {
        fprintf(csv, "phase,level,pattern,repetitions,new_nodes,new_edges,new_patterns,total_nodes,total_edges,total_patterns,reuse_ratio\n");
    }
    
    /* Create brain */
    const char *brain_path = "/tmp/hierarchical_brain.m";
    remove(brain_path);
    
    if (melvin_create_v2(brain_path, 3000, 15000, 2048, 0) != 0) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    Graph *g = melvin_open(brain_path, 3000, 15000, 2048);
    if (!g) {
        fprintf(stderr, "Failed to open brain\n");
        return 1;
    }
    
    printf("Brain created: %llu nodes, %llu edges\n\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    
    int total_patterns = 0;
    
    /* PHASE 1: Low-level patterns (2 chars each) */
    printf("=== PHASE 1: Low-Level Patterns ===\n");
    printf("Teaching: AB, CD, EF, GH\n");
    printf("Expected: Each pattern learned independently\n\n");
    
    for (int i = 0; i < LEVEL1_COUNT; i++) {
        const char *pattern = LEVEL1_PATTERNS[i];
        printf("  Teaching \"%s\"... ", pattern);
        fflush(stdout);
        
        GrowthMetrics m = feed_pattern(g, pattern, 0, 3);  /* 3 reps to form pattern */
        total_patterns += m.new_patterns;
        
        printf("→ +%d nodes, +%d edges, +%d patterns\n",
               m.new_nodes, m.new_edges, m.new_patterns);
        
        if (csv) {
            fprintf(csv, "1,low,%s,3,%d,%d,%d,%llu,%llu,%d,0.0\n",
                    pattern, m.new_nodes, m.new_edges, m.new_patterns,
                    (unsigned long long)g->node_count,
                    (unsigned long long)g->edge_count,
                    total_patterns);
        }
    }
    
    int phase1_patterns = total_patterns;
    printf("\nPhase 1 Complete: %d patterns learned\n\n", phase1_patterns);
    
    /* PHASE 2: Mid-level patterns (4 chars - compositions of level 1) */
    printf("=== PHASE 2: Mid-Level Patterns ===\n");
    printf("Teaching: ABCD, CDEF, EFGH, GHAB\n");
    printf("Expected: Reuse level 1 patterns → less growth\n\n");
    
    for (int i = 0; i < LEVEL2_COUNT; i++) {
        const char *pattern = LEVEL2_PATTERNS[i];
        printf("  Teaching \"%s\"... ", pattern);
        fflush(stdout);
        
        GrowthMetrics m = feed_pattern(g, pattern, 0, 3);
        total_patterns += m.new_patterns;
        
        /* Calculate reuse ratio: if reusing patterns, should create fewer new patterns */
        float reuse_ratio = (phase1_patterns > 0) ? 
            (float)m.new_patterns / (float)phase1_patterns : 0.0f;
        
        printf("→ +%d nodes, +%d edges, +%d patterns (reuse: %.2f)\n",
               m.new_nodes, m.new_edges, m.new_patterns, reuse_ratio);
        
        if (csv) {
            fprintf(csv, "2,mid,%s,3,%d,%d,%d,%llu,%llu,%d,%.3f\n",
                    pattern, m.new_nodes, m.new_edges, m.new_patterns,
                    (unsigned long long)g->node_count,
                    (unsigned long long)g->edge_count,
                    total_patterns, reuse_ratio);
        }
    }
    
    int phase2_patterns = total_patterns - phase1_patterns;
    printf("\nPhase 2 Complete: %d new patterns (vs %d in phase 1)\n\n",
           phase2_patterns, phase1_patterns);
    
    /* PHASE 3: High-level patterns (8+ chars - compositions of level 2) */
    printf("=== PHASE 3: High-Level Patterns ===\n");
    printf("Teaching: ABCDEFGH, GHEFCDAB, ABCDEFGHAB\n");
    printf("Expected: Reuse level 2 patterns → minimal growth\n\n");
    
    for (int i = 0; i < LEVEL3_COUNT; i++) {
        const char *pattern = LEVEL3_PATTERNS[i];
        printf("  Teaching \"%s\"... ", pattern);
        fflush(stdout);
        
        GrowthMetrics m = feed_pattern(g, pattern, 0, 3);
        int old_total = total_patterns;
        total_patterns += m.new_patterns;
        
        /* Reuse efficiency: fewer patterns = more reuse */
        float reuse_ratio = (phase1_patterns > 0) ? 
            (float)m.new_patterns / (float)phase1_patterns : 0.0f;
        
        printf("→ +%d nodes, +%d edges, +%d patterns (reuse: %.2f)\n",
               m.new_nodes, m.new_edges, m.new_patterns, reuse_ratio);
        
        if (csv) {
            fprintf(csv, "3,high,%s,3,%d,%d,%d,%llu,%llu,%d,%.3f\n",
                    pattern, m.new_nodes, m.new_edges, m.new_patterns,
                    (unsigned long long)g->node_count,
                    (unsigned long long)g->edge_count,
                    total_patterns, reuse_ratio);
        }
    }
    
    int phase3_patterns = total_patterns - (phase1_patterns + phase2_patterns);
    printf("\nPhase 3 Complete: %d new patterns\n\n", phase3_patterns);
    
    /* ANALYSIS */
    printf("==============================================\n");
    printf("HIERARCHICAL REUSE ANALYSIS\n");
    printf("==============================================\n\n");
    
    printf("Pattern Growth Per Phase:\n");
    printf("  Phase 1 (low):  %d patterns\n", phase1_patterns);
    printf("  Phase 2 (mid):  %d patterns (%.1fx of phase 1)\n",
           phase2_patterns, (float)phase2_patterns / (float)phase1_patterns);
    printf("  Phase 3 (high): %d patterns (%.1fx of phase 1)\n\n",
           phase3_patterns, (float)phase3_patterns / (float)phase1_patterns);
    
    printf("Expected: If patterns reuse hierarchically,\n");
    printf("  each phase should create FEWER patterns than previous\n");
    printf("  (reusing existing patterns as building blocks)\n\n");
    
    printf("Graph Growth:\n");
    printf("  Total nodes:    %llu\n", (unsigned long long)g->node_count);
    printf("  Total edges:    %llu\n", (unsigned long long)g->edge_count);
    printf("  Total patterns: %d\n\n", total_patterns);
    
    /* Efficiency metric: patterns per character taught */
    int total_chars = 0;
    for (int i = 0; i < LEVEL1_COUNT; i++) total_chars += strlen(LEVEL1_PATTERNS[i]);
    for (int i = 0; i < LEVEL2_COUNT; i++) total_chars += strlen(LEVEL2_PATTERNS[i]);
    for (int i = 0; i < LEVEL3_COUNT; i++) total_chars += strlen(LEVEL3_PATTERNS[i]);
    
    float patterns_per_char = (float)total_patterns / (float)total_chars;
    printf("Efficiency: %.2f patterns per character taught\n", patterns_per_char);
    printf("  (Lower = more reuse)\n\n");
    
    if (csv) fclose(csv);
    melvin_close(g);
    remove(brain_path);
    
    printf("==============================================\n");
    printf("Results saved to: benchmarks/data/experiment2_results.csv\n");
    printf("==============================================\n");
    
    return 0;
}


/*
 * Experiment 3: Scaling Efficiency
 * 
 * Question: How does Melvin's efficiency change as pattern complexity grows?
 * 
 * Hypothesis: Traditional ML degrades with complexity (linear/polynomial growth)
 *             Melvin improves with complexity (sublinear growth via reuse)
 * 
 * Test Design:
 * 1. Start with simple 2-char patterns
 * 2. Double complexity each round (2 → 4 → 8 → 16 → 32 → 64 chars)
 * 3. Measure: Memory growth, pattern creation, learning speed
 * 4. Expected: Efficiency ratio improves as complexity increases
 * 
 * Key Metric: "Bytes per new character" should DECREASE as complexity grows
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../src/melvin.h"

/* Generate a complex pattern by composing simpler ones */
void generate_pattern(char *buffer, int length, int seed) {
    /* Create patterns with repeating structure for reuse opportunities */
    const char *base = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int base_len = strlen(base);
    
    for (int i = 0; i < length; i++) {
        /* Use modulo to create repetition: ABCDABCDABCD... */
        buffer[i] = base[(i + seed) % base_len];
    }
    buffer[length] = '\0';
}

typedef struct {
    int pattern_length;
    int repetitions;
    uint64_t nodes_before;
    uint64_t edges_before;
    int patterns_before;
    uint64_t nodes_after;
    uint64_t edges_after;
    int patterns_after;
    double time_seconds;
} ScalingResult;

ScalingResult test_pattern_length(Graph *g, int length, int reps) {
    ScalingResult result = {0};
    result.pattern_length = length;
    result.repetitions = reps;
    
    /* Measure before */
    result.nodes_before = g->node_count;
    result.edges_before = g->edge_count;
    
    result.patterns_before = 0;
    for (uint64_t i = 840; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            result.patterns_before++;
        }
    }
    
    /* Generate and feed pattern */
    char pattern[128];
    if (length >= 128) length = 127;
    generate_pattern(pattern, length, 0);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int r = 0; r < reps; r++) {
        for (int i = 0; pattern[i] != '\0'; i++) {
            melvin_feed_byte(g, 0, (uint8_t)pattern[i], 1.0f);
        }
        melvin_feed_byte(g, 0, ' ', 0.5f);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    result.time_seconds = (end.tv_sec - start.tv_sec) + 
                         (end.tv_nsec - start.tv_nsec) / 1e9;
    
    /* Measure after */
    result.nodes_after = g->node_count;
    result.edges_after = g->edge_count;
    
    result.patterns_after = 0;
    for (uint64_t i = 840; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            result.patterns_after++;
        }
    }
    
    return result;
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    
    printf("==============================================\n");
    printf("EXPERIMENT 3: SCALING EFFICIENCY\n");
    printf("==============================================\n\n");
    
    printf("Question: How does efficiency change with complexity?\n");
    printf("Hypothesis: Melvin gets MORE efficient as complexity grows\n");
    printf("           (due to pattern reuse)\n\n");
    
    /* CSV output */
    FILE *csv = fopen("benchmarks/data/experiment3_results.csv", "w");
    if (csv) {
        fprintf(csv, "length,reps,nodes_growth,edges_growth,patterns_created,");
        fprintf(csv, "bytes_per_char,patterns_per_char,time_per_char,reuse_factor\n");
    }
    
    /* Create brain */
    const char *brain_path = "/tmp/scaling_brain.m";
    remove(brain_path);
    
    if (melvin_create_v2(brain_path, 5000, 25000, 4096, 0) != 0) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    Graph *g = melvin_open(brain_path, 5000, 25000, 4096);
    if (!g) {
        fprintf(stderr, "Failed to open brain\n");
        return 1;
    }
    
    printf("Testing pattern lengths: 2, 4, 8, 16, 32, 64 chars\n");
    printf("Repetitions: 5 per pattern (to form patterns)\n\n");
    
    printf("%-8s %-8s %-12s %-12s %-12s %-15s\n",
           "Length", "Nodes+", "Edges+", "Patterns+", "Bytes/Char", "Reuse");
    printf("--------------------------------------------------------\n");
    
    int lengths[] = {2, 4, 8, 16, 32, 64};
    int num_tests = sizeof(lengths) / sizeof(lengths[0]);
    
    ScalingResult prev_result = {0};
    
    for (int i = 0; i < num_tests; i++) {
        int length = lengths[i];
        
        ScalingResult r = test_pattern_length(g, length, 5);
        
        /* Calculate metrics */
        uint64_t node_growth = r.nodes_after - r.nodes_before;
        uint64_t edge_growth = r.edges_after - r.edges_before;
        int pattern_growth = r.patterns_after - r.patterns_before;
        
        /* Key metric: bytes per character (should decrease with reuse) */
        double bytes_per_char = (double)(node_growth * 64 + edge_growth * 20) / 
                                (double)length;
        
        /* Patterns per character (lower = more reuse) */
        double patterns_per_char = (double)pattern_growth / (double)length;
        
        /* Time per character */
        double time_per_char = r.time_seconds / (double)length;
        
        /* Reuse factor: compare to first iteration (baseline) */
        double reuse_factor = 1.0;
        if (i > 0 && prev_result.pattern_length > 0) {
            double prev_ppc = (double)prev_result.patterns_after / 
                             (double)prev_result.pattern_length;
            if (prev_ppc > 0) {
                reuse_factor = prev_ppc / patterns_per_char;
            }
        }
        
        printf("%-8d +%-7llu +%-11llu +%-11d %-15.1f %.2fx\n",
               length,
               (unsigned long long)node_growth,
               (unsigned long long)edge_growth,
               pattern_growth,
               bytes_per_char,
               reuse_factor);
        
        if (csv) {
            fprintf(csv, "%d,%d,%llu,%llu,%d,%.2f,%.4f,%.6f,%.3f\n",
                    length, r.repetitions,
                    (unsigned long long)node_growth,
                    (unsigned long long)edge_growth,
                    pattern_growth,
                    bytes_per_char,
                    patterns_per_char,
                    time_per_char,
                    reuse_factor);
            fflush(csv);
        }
        
        prev_result = r;
    }
    
    printf("\n==============================================\n");
    printf("ANALYSIS\n");
    printf("==============================================\n\n");
    
    printf("Key Metrics:\n\n");
    
    printf("1. Bytes/Char Trend:\n");
    printf("   - If DECREASING → Melvin reusing patterns (efficient!)\n");
    printf("   - If FLAT → Linear scaling (like traditional ML)\n");
    printf("   - If INCREASING → Getting worse (bad!)\n\n");
    
    printf("2. Reuse Factor:\n");
    printf("   - > 1.0x → Creating FEWER patterns per char (reuse!)\n");
    printf("   - = 1.0x → Same rate (no reuse benefit)\n");
    printf("   - < 1.0x → Creating MORE patterns (worse)\n\n");
    
    printf("Final State:\n");
    printf("  Total nodes:    %llu\n", (unsigned long long)g->node_count);
    printf("  Total edges:    %llu\n", (unsigned long long)g->edge_count);
    
    int total_patterns = 0;
    for (uint64_t i = 840; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) total_patterns++;
    }
    printf("  Total patterns: %d\n\n", total_patterns);
    
    /* Calculate overall efficiency */
    int total_chars = 0;
    for (int i = 0; i < num_tests; i++) {
        total_chars += lengths[i] * 5;  /* 5 reps each */
    }
    
    double avg_patterns_per_char = (double)total_patterns / (double)total_chars;
    printf("Average: %.3f patterns per character\n", avg_patterns_per_char);
    printf("         (Lower = better reuse)\n\n");
    
    if (csv) fclose(csv);
    melvin_close(g);
    remove(brain_path);
    
    printf("==============================================\n");
    printf("Results saved to: benchmarks/data/experiment3_results.csv\n");
    printf("==============================================\n");
    
    return 0;
}


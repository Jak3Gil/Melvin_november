/*
 * Experiment 1: Pattern Discovery Efficiency
 * 
 * Question: How many examples does Melvin need to learn a pattern?
 * 
 * Method:
 * 1. Create fresh brain
 * 2. Feed pattern "HELLO" N times
 * 3. Check if pattern was created
 * 4. Test recognition with variations
 * 5. Record: N examples needed, pattern count, accuracy
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/melvin.h"

/* Test pattern to learn */
const char *TEST_PATTERN = "HELLO";

/* Check if a pattern exists for the test sequence */
int pattern_exists(Graph *g, const char *pattern) {
    if (!g || !g->nodes || !pattern) return 0;
    
    /* Look for pattern nodes (840+) with edges to the pattern bytes */
    /* Simple heuristic: check if any high node has edges to H->E->L->L->O sequence */
    
    for (uint64_t i = 840; i < g->node_count; i++) {
        Node *n = &g->nodes[i];
        if (n->pattern_data_offset > 0) {
            /* This is a pattern node - check if it matches our pattern */
            /* For now, just count any pattern creation as success */
            return 1;
        }
    }
    
    return 0;
}

/* Count total patterns in graph */
int count_patterns(Graph *g) {
    if (!g || !g->nodes) return 0;
    
    int count = 0;
    for (uint64_t i = 840; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            count++;
        }
    }
    return count;
}

/* Feed pattern once */
void feed_pattern_once(Graph *g, const char *pattern, uint32_t port) {
    for (int i = 0; pattern[i] != '\0'; i++) {
        melvin_feed_byte(g, port, (uint8_t)pattern[i], 1.0f);
    }
    /* Feed separator to distinguish repetitions */
    melvin_feed_byte(g, port, ' ', 0.5f);
}

/* Test if pattern is recognized by checking edge strengths */
float test_recognition(Graph *g, const char *pattern) {
    /* Check if sequential edges exist and are strong for H->E->L->L->O */
    /* Since find_edge is not exported, we'll use edge count as proxy */
    float total_strength = 0.0f;
    int edge_count = 0;
    
    /* Check edges from byte nodes */
    for (int i = 0; pattern[i+1] != '\0'; i++) {
        uint32_t src = (uint32_t)pattern[i];
        uint32_t dst = (uint32_t)pattern[i+1];
        
        /* Scan outgoing edges from src node */
        if (src < g->node_count && dst < g->node_count) {
            uint32_t eid = g->nodes[src].first_out;
            while (eid != UINT32_MAX && eid < g->edge_count) {
                if (g->edges[eid].dst == dst) {
                    total_strength += g->edges[eid].w;
                    edge_count++;
                    break;
                }
                eid = g->edges[eid].next_out;
            }
        }
    }
    
    if (edge_count == 0) return 0.0f;
    return total_strength / edge_count;
}

int main(int argc, char *argv[]) {
    printf("==============================================\n");
    printf("EXPERIMENT 1: PATTERN DISCOVERY EFFICIENCY\n");
    printf("==============================================\n\n");
    
    printf("Pattern: \"%s\"\n", TEST_PATTERN);
    printf("Testing: How many repetitions needed to learn?\n\n");
    
    /* CSV header for data collection */
    FILE *csv = fopen("benchmarks/data/experiment1_results.csv", "w");
    if (csv) {
        fprintf(csv, "repetitions,pattern_count,avg_edge_strength,recognition_score,nodes,edges\n");
    }
    
    /* Test with increasing repetitions */
    for (int reps = 1; reps <= 20; reps++) {
        printf("--- Test %d: %d repetitions ---\n", reps, reps);
        
        /* Create fresh brain for this test */
        char brain_path[256];
        snprintf(brain_path, sizeof(brain_path), "/tmp/bench_brain_%d.m", reps);
        
        /* Remove old file if exists */
        remove(brain_path);
        
        /* Create new brain */
        if (melvin_create_v2(brain_path, 2000, 10000, 1024, 0) != 0) {
            fprintf(stderr, "Failed to create brain\n");
            continue;
        }
        
        /* Open brain */
        Graph *g = melvin_open(brain_path, 2000, 10000, 1024);
        if (!g) {
            fprintf(stderr, "Failed to open brain\n");
            continue;
        }
        
        /* Feed pattern N times */
        for (int i = 0; i < reps; i++) {
            feed_pattern_once(g, TEST_PATTERN, 0);
        }
        
        /* Measure results */
        int patterns = count_patterns(g);
        float edge_strength = g->avg_edge_strength;
        float recognition = test_recognition(g, TEST_PATTERN);
        
        printf("  Patterns discovered: %d\n", patterns);
        printf("  Avg edge strength: %.4f\n", edge_strength);
        printf("  Recognition score: %.4f\n", recognition);
        printf("  Nodes: %llu, Edges: %llu\n\n", 
               (unsigned long long)g->node_count, 
               (unsigned long long)g->edge_count);
        
        /* Save to CSV */
        if (csv) {
            fprintf(csv, "%d,%d,%.6f,%.6f,%llu,%llu\n",
                    reps, patterns, edge_strength, recognition,
                    (unsigned long long)g->node_count,
                    (unsigned long long)g->edge_count);
            fflush(csv);
        }
        
        /* Clean up */
        melvin_close(g);
        remove(brain_path);
    }
    
    if (csv) fclose(csv);
    
    printf("\n==============================================\n");
    printf("Results saved to: benchmarks/data/experiment1_results.csv\n");
    printf("==============================================\n");
    
    return 0;
}


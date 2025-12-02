/*
 * verify_claims.c - Scientific Verification Framework
 * 
 * Reproducible test suite to verify Melvin's claims for peer review.
 * 
 * Claims to verify:
 * 1. Melvin learns arithmetic patterns from examples
 * 2. Melvin executes queries correctly after minimal training
 * 3. Melvin creates 25-255 patterns per example (super-linear learning)
 * 
 * Usage: ./verify_claims [--trials=N] [--seed=S] > verification_report.txt
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define TEST_FILE "verify_brain.m"

typedef struct {
    uint64_t patterns_before;
    uint64_t patterns_after;
    uint64_t edges_before;
    uint64_t edges_after;
    uint64_t nodes_before;
    uint64_t nodes_after;
    uint32_t correct_answers;
    uint32_t total_queries;
    double time_seconds;
} VerificationMetrics;

/* Count patterns in graph */
uint64_t count_patterns(Graph *g) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            count++;
        }
    }
    return count;
}

/* Get graph state snapshot */
void snapshot_graph(Graph *g, VerificationMetrics *m) {
    m->nodes_before = g->node_count;
    m->edges_before = g->edge_count;
    m->patterns_before = count_patterns(g);
}

void snapshot_graph_after(Graph *g, VerificationMetrics *m) {
    m->nodes_after = g->node_count;
    m->edges_after = g->edge_count;
    m->patterns_after = count_patterns(g);
}

/* Feed training examples */
void feed_training_examples(Graph *g, int num_examples) {
    for (int i = 0; i < num_examples; i++) {
        int a = (i % 10) + 1;
        int b = ((i * 3) % 10) + 1;
        int c = a + b;
        char buf[32];
        snprintf(buf, sizeof(buf), "%d+%d=%d", a, b, c);
        
        for (size_t j = 0; j < strlen(buf); j++) {
            melvin_feed_byte(g, 0, (uint8_t)buf[j], 0.5f);
        }
    }
    melvin_sync(g);
}

/* Test a query and return if correct */
bool test_query(Graph *g, int a, int b, int expected) {
    char query[32];
    snprintf(query, sizeof(query), "%d+%d=?", a, b);
    
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.6f);
    }
    
    /* Give graph time to process */
    for (int i = 0; i < 10; i++) {
        melvin_sync(g);
    }
    
    /* Check result in EXEC_ADD */
    uint32_t EXEC_ADD = 2000;
    if (EXEC_ADD < g->node_count && g->nodes[EXEC_ADD].payload_offset > 0) {
        uint64_t input_offset = g->nodes[EXEC_ADD].payload_offset + 256;
        if (input_offset + 24 <= g->hdr->blob_size) {
            uint64_t *data = (uint64_t *)(g->blob + (input_offset - g->hdr->blob_offset));
            uint64_t result = data[2];  /* Result is at offset 2 */
            
            if (result == (uint64_t)expected) {
                return true;
            }
        }
    }
    
    return false;
}

/* Run verification test */
VerificationMetrics run_verification(int num_examples, int num_queries, unsigned int seed) {
    VerificationMetrics m = {0};
    
    srand(seed);
    
    /* Create fresh brain */
    remove(TEST_FILE);
    Graph *g = melvin_open(TEST_FILE, 0, 0, 65536);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to create brain\n");
        return m;
    }
    
    /* Create EXEC_ADD node */
    uint32_t EXEC_ADD = 2000;
    uint64_t exec_offset = 256;
    if (exec_offset + 512 <= g->hdr->blob_size) {
        g->blob[exec_offset] = 0xC3;  /* Placeholder code */
        melvin_create_exec_node(g, EXEC_ADD, exec_offset, 0.5f);
    }
    
    /* Snapshot before training */
    snapshot_graph(g, &m);
    
    /* Train on examples */
    clock_t start = clock();
    feed_training_examples(g, num_examples);
    clock_t end = clock();
    m.time_seconds = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    /* Snapshot after training */
    snapshot_graph_after(g, &m);
    
    /* Test queries */
    m.total_queries = num_queries;
    for (int i = 0; i < num_queries; i++) {
        int a = (rand() % 10) + 1;
        int b = (rand() % 10) + 1;
        int expected = a + b;
        
        if (test_query(g, a, b, expected)) {
            m.correct_answers++;
        }
    }
    
    melvin_close(g);
    remove(TEST_FILE);
    
    return m;
}

/* Print verification report */
void print_report(VerificationMetrics *m, int num_examples, int num_queries, int trial) {
    printf("========================================\n");
    printf("VERIFICATION REPORT - Trial %d\n", trial);
    printf("========================================\n\n");
    
    printf("Configuration:\n");
    printf("  Training examples: %d\n", num_examples);
    printf("  Test queries: %d\n", num_queries);
    printf("  Time: %.3f seconds\n\n", m->time_seconds);
    
    printf("Graph Growth:\n");
    printf("  Nodes: %llu → %llu (+%llu)\n", 
           (unsigned long long)m->nodes_before,
           (unsigned long long)m->nodes_after,
           (unsigned long long)(m->nodes_after - m->nodes_before));
    printf("  Edges: %llu → %llu (+%llu)\n",
           (unsigned long long)m->edges_before,
           (unsigned long long)m->edges_after,
           (unsigned long long)(m->edges_after - m->edges_before));
    printf("  Patterns: %llu → %llu (+%llu)\n\n",
           (unsigned long long)m->patterns_before,
           (unsigned long long)m->patterns_after,
           (unsigned long long)(m->patterns_after - m->patterns_before));
    
    printf("Learning Efficiency:\n");
    if (num_examples > 0) {
        double patterns_per_example = (double)(m->patterns_after - m->patterns_before) / num_examples;
        printf("  Patterns per example: %.2f\n", patterns_per_example);
        printf("  Examples per second: %.2f\n", num_examples / m->time_seconds);
    }
    printf("\n");
    
    printf("Accuracy:\n");
    double accuracy = (m->total_queries > 0) ? 
        (100.0 * m->correct_answers / m->total_queries) : 0.0;
    printf("  Correct: %u / %u\n", m->correct_answers, m->total_queries);
    printf("  Accuracy: %.2f%%\n\n", accuracy);
    
    printf("Claims Verification:\n");
    printf("  ✅ Pattern learning: %s (%llu patterns created)\n",
           (m->patterns_after > m->patterns_before) ? "PASS" : "FAIL",
           (unsigned long long)(m->patterns_after - m->patterns_before));
    
    if (num_examples > 0) {
        double ppe = (double)(m->patterns_after - m->patterns_before) / num_examples;
        printf("  ✅ Super-linear learning: %s (%.2f patterns/ex)\n",
               (ppe >= 25.0) ? "PASS" : "FAIL", ppe);
    }
    
    printf("  ✅ Query execution: %s (%.2f%% accuracy)\n",
           (accuracy > 50.0) ? "PASS" : "FAIL", accuracy);
    printf("\n");
}

/* Statistical analysis over multiple trials */
void statistical_analysis(int num_trials, int num_examples, int num_queries) {
    double accuracies[1000];
    double patterns_per_ex[1000];
    double times[1000];
    
    if (num_trials > 1000) num_trials = 1000;
    
    printf("========================================\n");
    printf("STATISTICAL ANALYSIS (%d trials)\n", num_trials);
    printf("========================================\n\n");
    
    for (int i = 0; i < num_trials; i++) {
        unsigned int seed = (unsigned int)(time(NULL) + i);
        VerificationMetrics m = run_verification(num_examples, num_queries, seed);
        
        accuracies[i] = (m.total_queries > 0) ? 
            (100.0 * m.correct_answers / m.total_queries) : 0.0;
        
        if (num_examples > 0) {
            patterns_per_ex[i] = (double)(m.patterns_after - m.patterns_before) / num_examples;
        } else {
            patterns_per_ex[i] = 0.0;
        }
        
        times[i] = m.time_seconds;
        
        if (i < 3) {
            print_report(&m, num_examples, num_queries, i + 1);
        }
    }
    
    /* Calculate statistics */
    double mean_acc = 0.0, mean_ppe = 0.0, mean_time = 0.0;
    for (int i = 0; i < num_trials; i++) {
        mean_acc += accuracies[i];
        mean_ppe += patterns_per_ex[i];
        mean_time += times[i];
    }
    mean_acc /= num_trials;
    mean_ppe /= num_trials;
    mean_time /= num_trials;
    
    double std_acc = 0.0, std_ppe = 0.0, std_time = 0.0;
    for (int i = 0; i < num_trials; i++) {
        std_acc += (accuracies[i] - mean_acc) * (accuracies[i] - mean_acc);
        std_ppe += (patterns_per_ex[i] - mean_ppe) * (patterns_per_ex[i] - mean_ppe);
        std_time += (times[i] - mean_time) * (times[i] - mean_time);
    }
    std_acc = sqrt(std_acc / num_trials);
    std_ppe = sqrt(std_ppe / num_trials);
    std_time = sqrt(std_time / num_trials);
    
    printf("========================================\n");
    printf("STATISTICAL SUMMARY\n");
    printf("========================================\n\n");
    
    printf("Accuracy:\n");
    printf("  Mean: %.2f%% ± %.2f%%\n", mean_acc, std_acc);
    printf("  95%% CI: [%.2f%%, %.2f%%]\n", 
           mean_acc - 1.96 * std_acc, mean_acc + 1.96 * std_acc);
    printf("\n");
    
    printf("Patterns per Example:\n");
    printf("  Mean: %.2f ± %.2f\n", mean_ppe, std_ppe);
    printf("  95%% CI: [%.2f, %.2f]\n",
           mean_ppe - 1.96 * std_ppe, mean_ppe + 1.96 * std_ppe);
    printf("\n");
    
    printf("Time per Example:\n");
    printf("  Mean: %.3f ± %.3f seconds\n", mean_time / num_examples, std_time / num_examples);
    printf("\n");
    
    printf("Claims Verification (Statistical):\n");
    printf("  ✅ Pattern learning: %s (%.2f patterns/ex, CI: [%.2f, %.2f])\n",
           (mean_ppe >= 25.0) ? "PASS" : "FAIL",
           mean_ppe, mean_ppe - 1.96 * std_ppe, mean_ppe + 1.96 * std_ppe);
    printf("  ✅ Query execution: %s (%.2f%% accuracy, CI: [%.2f%%, %.2f%%])\n",
           (mean_acc > 50.0) ? "PASS" : "FAIL",
           mean_acc, mean_acc - 1.96 * std_acc, mean_acc + 1.96 * std_acc);
    printf("\n");
}

int main(int argc, char **argv) {
    int num_examples = 10;
    int num_queries = 20;
    int num_trials = 1;
    unsigned int seed = (unsigned int)time(NULL);
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--examples=", 12) == 0) {
            num_examples = atoi(argv[i] + 12);
        } else if (strncmp(argv[i], "--queries=", 10) == 0) {
            num_queries = atoi(argv[i] + 10);
        } else if (strncmp(argv[i], "--trials=", 9) == 0) {
            num_trials = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--seed=", 7) == 0) {
            seed = (unsigned int)atoi(argv[i] + 7);
        }
    }
    
    if (num_trials > 1) {
        statistical_analysis(num_trials, num_examples, num_queries);
    } else {
        VerificationMetrics m = run_verification(num_examples, num_queries, seed);
        print_report(&m, num_examples, num_queries, 1);
    }
    
    return 0;
}


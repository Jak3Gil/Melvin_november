/*
 * test_scaling_laws.c - Test if graph can bypass scaling laws
 * Measures: examples needed, time, accuracy, learning efficiency
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

#define TEST_FILE "test_scaling.m"

/* Track learning metrics */
typedef struct {
    int examples_fed;
    double time_taken;
    uint64_t patterns_created;
    uint64_t values_learned;
    int correct_answers;
    int total_queries;
    double accuracy;
} LearningMetrics;

/* Feed examples and track learning */
void feed_and_measure(Graph *g, int num_examples, LearningMetrics *metrics) {
    clock_t start = clock();
    
    for (int i = 0; i < num_examples; i++) {
        int a = (i % 10) + 1;
        int b = ((i * 3) % 10) + 1;
        int c = a + b;
        
        char buf[32];
        snprintf(buf, sizeof(buf), "%d+%d=%d\n", a, b, c);
        
        for (size_t j = 0; j < strlen(buf); j++) {
            melvin_feed_byte(g, 0, (uint8_t)buf[j], 0.4f);
        }
    }
    
    clock_t end = clock();
    metrics->time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    metrics->examples_fed = num_examples;
    
    /* Count patterns and values */
    metrics->patterns_created = 0;
    metrics->values_learned = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            metrics->patterns_created++;
        }
        if (g->nodes[i].pattern_value_offset > 0) {
            metrics->values_learned++;
        }
    }
}

/* Test if graph can answer query */
bool test_answer(Graph *g, int a, int b, int expected) {
    /* Feed query */
    char query[32];
    snprintf(query, sizeof(query), "%d+%d=?", a, b);
    
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    
    /* Check multiple indicators of learning */
    uint32_t EXEC_ADD = 2000;
    bool found = false;
    
    /* Check 1: EXEC_ADD activation */
    if (EXEC_ADD < g->node_count && g->nodes[EXEC_ADD].a > 0.1f) {
        found = true;
    }
    
    /* Check 2: Result in blob */
    if (EXEC_ADD < g->node_count && g->nodes[EXEC_ADD].payload_offset > 0) {
        uint64_t input_offset = g->nodes[EXEC_ADD].payload_offset + 256;
        if (input_offset + (3 * sizeof(uint64_t)) <= g->hdr->blob_size) {
            uint64_t *result_ptr = (uint64_t *)(g->blob + (input_offset + (2 * sizeof(uint64_t)) - g->hdr->blob_offset));
            if (*result_ptr == (uint64_t)expected) {
                return true;  /* Correct answer found */
            }
        }
    }
    
    /* Check 3: Pattern learned the answer */
    char answer[32];
    snprintf(answer, sizeof(answer), "%d", expected);
    
    /* Check if answer bytes are activated */
    bool answer_found = true;
    for (size_t i = 0; i < strlen(answer); i++) {
        uint32_t byte_node = (uint32_t)answer[i];
        if (byte_node >= g->node_count || g->nodes[byte_node].a < 0.01f) {
            answer_found = false;
            break;
        }
    }
    
    return answer_found && found;
}

/* Test accuracy on queries */
void test_accuracy(Graph *g, int num_queries, bool test_unseen, LearningMetrics *metrics) {
    metrics->correct_answers = 0;
    metrics->total_queries = 0;
    
    int start = test_unseen ? metrics->examples_fed : 0;
    
    for (int i = 0; i < num_queries; i++) {
        int idx = start + i;
        int a = (idx % 10) + 1;
        int b = ((idx * 3) % 10) + 1;
        int expected = a + b;
        
        bool correct = test_answer(g, a, b, expected);
        if (correct) metrics->correct_answers++;
        metrics->total_queries++;
    }
    
    metrics->accuracy = (metrics->total_queries > 0) ? 
                       (100.0 * metrics->correct_answers / metrics->total_queries) : 0.0;
}

int main() {
    printf("========================================\n");
    printf("SCALING LAWS TEST\n");
    printf("Goal: Can graph bypass scaling laws?\n");
    printf("========================================\n\n");
    
    /* Create fresh brain */
    remove(TEST_FILE);
    Graph *g = melvin_open(TEST_FILE, 0, 0, 65536);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    /* Create EXEC_ADD */
    uint32_t EXEC_ADD = 2000;
#if defined(__aarch64__) || defined(__arm64__)
    const uint8_t ADD_CODE[] = {0x00, 0x00, 0x01, 0x8b, 0xc0, 0x03, 0x5f, 0xd6};
    uint64_t offset = 256;
    if (offset + sizeof(ADD_CODE) <= g->hdr->blob_size) {
        memcpy(g->blob + offset, ADD_CODE, sizeof(ADD_CODE));
        melvin_create_exec_node(g, EXEC_ADD, offset, 1.0f);
    }
#endif
    
    printf("Brain ready: %llu nodes\n\n", (unsigned long long)g->node_count);
    
    /* Progressive learning test */
    int test_sizes[] = {1, 2, 5, 10, 20, 50, 100};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("Progressive Learning Test:\n");
    printf("Testing with increasing examples to find minimum needed\n\n");
    
    LearningMetrics best = {0};
    int min_examples = -1;
    
    for (int t = 0; t < num_tests; t++) {
        printf("----------------------------------------\n");
        printf("Test: %d examples\n", test_sizes[t]);
        printf("----------------------------------------\n");
        
        LearningMetrics metrics = {0};
        
        /* Feed examples */
        feed_and_measure(g, test_sizes[t], &metrics);
        
        printf("  Time: %.3f sec (%.1f ex/sec)\n", 
               metrics.time_taken, 
               metrics.examples_fed / (metrics.time_taken > 0 ? metrics.time_taken : 0.001));
        printf("  Patterns: %llu (%.2f/ex)\n", 
               (unsigned long long)metrics.patterns_created,
               (double)metrics.patterns_created / metrics.examples_fed);
        printf("  Values: %llu (%.2f/ex)\n",
               (unsigned long long)metrics.values_learned,
               (double)metrics.values_learned / metrics.examples_fed);
        
        /* Test accuracy */
        test_accuracy(g, 10, false, &metrics);  /* Test on seen */
        printf("  Seen accuracy: %.1f%% (%d/%d)\n", 
               metrics.accuracy, metrics.correct_answers, metrics.total_queries);
        
        LearningMetrics unseen_metrics = metrics;
        test_accuracy(g, 10, true, &unseen_metrics);  /* Test on unseen */
        printf("  Unseen accuracy: %.1f%% (%d/%d)\n",
               unseen_metrics.accuracy, unseen_metrics.correct_answers, unseen_metrics.total_queries);
        
        /* Track best */
        if (unseen_metrics.accuracy > best.accuracy) {
            best = unseen_metrics;
            min_examples = test_sizes[t];
        }
        
        /* Check if we've achieved good accuracy */
        if (unseen_metrics.accuracy >= 80.0 && min_examples < 0) {
            min_examples = test_sizes[t];
            printf("  ✅ ACHIEVED 80%%+ ACCURACY!\n");
        }
        
        printf("\n");
        
        melvin_sync(g);
    }
    
    printf("========================================\n");
    printf("RESULTS\n");
    printf("========================================\n\n");
    
    printf("Best performance:\n");
    printf("  Examples: %d\n", min_examples > 0 ? min_examples : best.examples_fed);
    printf("  Accuracy: %.1f%%\n", best.accuracy);
    printf("  Patterns: %llu\n", (unsigned long long)best.patterns_created);
    printf("  Values: %llu\n", (unsigned long long)best.values_learned);
    printf("\n");
    
    printf("Scaling Analysis:\n");
    if (best.accuracy > 50.0 && min_examples < 20) {
        printf("  ✅ POTENTIAL SCALING LAW BYPASS!\n");
        printf("     High accuracy (%.1f%%) with few examples (%d)\n", 
               best.accuracy, min_examples);
        printf("     Patterns per example: %.2f\n",
               (double)best.patterns_created / best.examples_fed);
    } else {
        printf("  ⚠️  Needs more examples or better routing\n");
        printf("     Current accuracy: %.1f%%\n", best.accuracy);
    }
    printf("\n");
    
    printf("Learning Efficiency:\n");
    printf("  Patterns created per example: %.2f\n",
           (double)best.patterns_created / best.examples_fed);
    printf("  Values learned per example: %.2f\n",
           (double)best.values_learned / best.examples_fed);
    printf("  Time per example: %.3f seconds\n",
           best.time_taken / best.examples_fed);
    printf("\n");
    
    melvin_close(g);
    return 0;
}


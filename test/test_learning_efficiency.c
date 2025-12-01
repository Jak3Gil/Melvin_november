/*
 * test_learning_efficiency.c - Test learning efficiency: examples needed, time, accuracy
 * Goal: See if graph can bypass scaling laws (learn efficiently with few examples)
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

#define TEST_FILE "test_learning.m"

/* Test configuration */
typedef struct {
    int num_examples;      /* Number of examples to feed */
    int num_queries;        /* Number of test queries */
    bool measure_time;      /* Measure learning time */
    bool measure_accuracy;  /* Measure accuracy */
} TestConfig;

/* Feed examples and measure learning */
void feed_examples(Graph *g, int num_examples, double *time_taken) {
    clock_t start = clock();
    
    /* Generate examples */
    for (int i = 0; i < num_examples; i++) {
        int a = (i % 10) + 1;  /* 1-10 */
        int b = ((i * 3) % 10) + 1;  /* 1-10, different pattern */
        int c = a + b;
        
        char buf[32];
        snprintf(buf, sizeof(buf), "%d+%d=%d\n", a, b, c);
        
        for (size_t j = 0; j < strlen(buf); j++) {
            melvin_feed_byte(g, 0, (uint8_t)buf[j], 0.4f);
        }
        
        if ((i + 1) % 10 == 0) {
            printf("  Fed %d examples...\r", i + 1);
            fflush(stdout);
        }
    }
    
    clock_t end = clock();
    *time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
}

/* Test query and check if graph can answer */
int test_query(Graph *g, int a, int b, int expected) {
    /* Feed query */
    char query[32];
    snprintf(query, sizeof(query), "%d+%d=?", a, b);
    
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    
    /* Give graph time to process (propagation) */
    /* Small delay to let patterns activate and route */
    usleep(10000);  /* 10ms */
    
    /* Check multiple ways the graph might answer */
    uint32_t EXEC_ADD = 2000;
    
    /* Method 1: Check if EXEC_ADD executed with result in blob */
    if (EXEC_ADD < g->node_count && g->nodes[EXEC_ADD].payload_offset > 0) {
        uint64_t input_offset = g->nodes[EXEC_ADD].payload_offset + 256;
        if (input_offset + (3 * sizeof(uint64_t)) <= g->hdr->blob_size) {
            uint64_t *result_ptr = (uint64_t *)(g->blob + (input_offset + (2 * sizeof(uint64_t)) - g->hdr->blob_offset));
            if (*result_ptr == (uint64_t)expected) {
                return 1;  /* Correct - result in blob */
            }
        }
    }
    
    /* Method 2: Check if result appears in output port */
    uint32_t output_port = 100;
    if (output_port < g->node_count && g->nodes[output_port].a > 0.1f) {
        /* Output port activated - check if result bytes are there */
        char expected_str[32];
        snprintf(expected_str, sizeof(expected_str), "%d", expected);
        
        /* Check if expected result bytes are activated */
        bool found = true;
        for (size_t i = 0; i < strlen(expected_str); i++) {
            uint32_t byte_node = (uint32_t)expected_str[i];
            if (byte_node >= g->node_count || g->nodes[byte_node].a < 0.1f) {
                found = false;
                break;
            }
        }
        if (found) {
            return 1;  /* Correct - result in output */
        }
    }
    
    /* Method 3: Check if pattern learned the answer */
    /* Look for pattern that matches "X+Y=Z" where Z=expected */
    char answer_pattern[32];
    snprintf(answer_pattern, sizeof(answer_pattern), "%d+%d=%d", a, b, expected);
    
    /* Check if this pattern exists (simplified check) */
    /* In full implementation, would check pattern matches */
    
    return 0;  /* Not found */
}

/* Run learning efficiency test */
void run_test(Graph *g, TestConfig *config) {
    printf("========================================\n");
    printf("LEARNING EFFICIENCY TEST\n");
    printf("========================================\n\n");
    
    printf("Configuration:\n");
    printf("  Examples: %d\n", config->num_examples);
    printf("  Test queries: %d\n", config->num_queries);
    printf("\n");
    
    /* Step 1: Feed examples */
    printf("[Step 1] Feeding %d examples...\n", config->num_examples);
    double learning_time = 0.0;
    feed_examples(g, config->num_examples, &learning_time);
    printf("\n  Time: %.3f seconds\n", learning_time);
    printf("  Rate: %.1f examples/second\n", config->num_examples / learning_time);
    printf("\n");
    
    /* Step 2: Count what was learned */
    printf("[Step 2] Analyzing what was learned...\n");
    uint64_t pattern_count = 0;
    uint64_t value_count = 0;
    
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_count++;
        }
        if (g->nodes[i].pattern_value_offset > 0) {
            value_count++;
        }
    }
    
    printf("  Patterns created: %llu\n", (unsigned long long)pattern_count);
    printf("  Values learned: %llu\n", (unsigned long long)value_count);
    printf("\n");
    
    /* Step 3: Test accuracy */
    if (config->measure_accuracy) {
        printf("[Step 3] Testing accuracy...\n");
        
        int correct = 0;
        int total = 0;
        
        /* Test on seen examples */
        printf("  Testing on seen examples:\n");
        for (int i = 0; i < config->num_queries && i < config->num_examples; i++) {
            int a = (i % 10) + 1;
            int b = ((i * 3) % 10) + 1;
            int expected = a + b;
            
            int result = test_query(g, a, b, expected);
            if (result) correct++;
            total++;
        }
        
        double seen_accuracy = (total > 0) ? (100.0 * correct / total) : 0.0;
        printf("    Accuracy: %.1f%% (%d/%d)\n", seen_accuracy, correct, total);
        
        /* Test on unseen examples */
        printf("  Testing on unseen examples:\n");
        correct = 0;
        total = 0;
        
        for (int i = config->num_examples; i < config->num_examples + config->num_queries; i++) {
            int a = (i % 20) + 1;  /* Different range */
            int b = ((i * 7) % 20) + 1;
            int expected = a + b;
            
            int result = test_query(g, a, b, expected);
            if (result) correct++;
            total++;
        }
        
        double unseen_accuracy = (total > 0) ? (100.0 * correct / total) : 0.0;
        printf("    Accuracy: %.1f%% (%d/%d)\n", unseen_accuracy, correct, total);
        printf("\n");
        
        /* Scaling analysis */
        printf("[Step 4] Scaling Analysis\n");
        printf("  Examples: %d\n", config->num_examples);
        printf("  Patterns: %llu (%.2f per example)\n", 
               (unsigned long long)pattern_count,
               (double)pattern_count / config->num_examples);
        printf("  Values: %llu (%.2f per example)\n",
               (unsigned long long)value_count,
               (double)value_count / config->num_examples);
        printf("  Seen accuracy: %.1f%%\n", seen_accuracy);
        printf("  Unseen accuracy: %.1f%%\n", unseen_accuracy);
        printf("\n");
        
        /* Check if bypassing scaling laws */
        if (unseen_accuracy > 50.0 && config->num_examples < 100) {
            printf("  ✅ POTENTIAL SCALING LAW BYPASS!\n");
            printf("     High accuracy (%.1f%%) with few examples (%d)\n", 
                   unseen_accuracy, config->num_examples);
        } else if (unseen_accuracy < 50.0) {
            printf("  ⚠️  Needs more examples (accuracy: %.1f%%)\n", unseen_accuracy);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    printf("========================================\n");
    printf("LEARNING EFFICIENCY TEST SUITE\n");
    printf("Testing: Examples needed, time, accuracy, scaling laws\n");
    printf("========================================\n\n");
    
    /* Create fresh brain */
    printf("Creating fresh brain...\n");
    remove(TEST_FILE);  /* Start fresh */
    
    Graph *g = melvin_open(TEST_FILE, 0, 0, 65536);  /* 64KB blob */
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    printf("✅ Brain created: %llu nodes\n\n", (unsigned long long)g->node_count);
    
    /* Create EXEC_ADD */
    uint32_t EXEC_ADD = 2000;
#if defined(__aarch64__) || defined(__arm64__)
    const uint8_t ADD_CODE[] = {0x00, 0x00, 0x01, 0x8b, 0xc0, 0x03, 0x5f, 0xd6};
    uint64_t offset = 256;
    if (offset + sizeof(ADD_CODE) <= g->hdr->blob_size) {
        memcpy(g->blob + offset, ADD_CODE, sizeof(ADD_CODE));
        melvin_create_exec_node(g, EXEC_ADD, offset, 1.0f);
        printf("✅ EXEC_ADD created (node %u)\n\n", EXEC_ADD);
    }
#endif
    
    /* Run tests with different numbers of examples */
    int test_sizes[] = {5, 10, 20, 50, 100};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int t = 0; t < num_tests; t++) {
        printf("\n");
        printf("========================================\n");
        printf("TEST %d/%d: %d Examples\n", t+1, num_tests, test_sizes[t]);
        printf("========================================\n\n");
        
        TestConfig config = {
            .num_examples = test_sizes[t],
            .num_queries = 10,
            .measure_time = true,
            .measure_accuracy = true
        };
        
        run_test(g, &config);
        
        /* Sync after each test */
        melvin_sync(g);
    }
    
    printf("\n========================================\n");
    printf("SUMMARY\n");
    printf("========================================\n\n");
    
    printf("Tests completed!\n");
    printf("Check results above to see:\n");
    printf("  - How many examples needed for accuracy\n");
    printf("  - Learning time per example\n");
    printf("  - Whether scaling laws are bypassed\n");
    printf("\n");
    
    melvin_close(g);
    return 0;
}


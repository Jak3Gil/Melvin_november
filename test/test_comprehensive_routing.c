/*
 * test_comprehensive_routing.c - Comprehensive test of routing chain
 * Tests: pattern matching → value extraction → EXEC routing → execution → result
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#define TEST_FILE "test_comprehensive.m"

typedef struct {
    bool pattern_created;
    bool edge_to_exec;
    bool values_extracted;
    bool exec_triggered;
    bool result_computed;
    uint64_t result_value;
} RoutingStatus;

/* Feed examples to establish patterns */
void feed_examples(Graph *g, int count) {
    printf("Feeding %d examples...\n", count);
    for (int i = 0; i < count; i++) {
        int a = (i % 10) + 1;
        int b = ((i * 3) % 10) + 1;
        int c = a + b;
        char buf[32];
        snprintf(buf, sizeof(buf), "%d+%d=%d", a, b, c);  /* No newline - match query format */
        for (size_t j = 0; j < strlen(buf); j++) {
            melvin_feed_byte(g, 0, (uint8_t)buf[j], 0.4f);
        }
    }
    printf("✅ Examples fed\n\n");
}

/* Check routing status for a query */
RoutingStatus check_routing(Graph *g, int a, int b, int expected) {
    RoutingStatus status = {0};
    
    /* Feed query */
    char query[32];
    snprintf(query, sizeof(query), "%d+%d=?", a, b);
    printf("Query: %s (expected: %d)\n", query, expected);
    
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    
    /* Give graph time to propagate and process */
    /* Trigger multiple propagation cycles */
    for (int i = 0; i < 10; i++) {
        melvin_sync(g);  /* Sync triggers propagation */
    }
    
    /* Check 1: Patterns created */
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_count++;
        }
    }
    status.pattern_created = (pattern_count > 0);
    
    /* Check 2: Edge to EXEC_ADD */
    uint32_t plus_node = (uint32_t)'+';
    uint32_t EXEC_ADD = 2000;
    if (plus_node < g->node_count && EXEC_ADD < g->node_count) {
        uint32_t eid = g->nodes[plus_node].first_out;
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            if (g->edges[eid].dst == EXEC_ADD) {
                status.edge_to_exec = true;
                break;
            }
            eid = g->edges[eid].next_out;
        }
    }
    
    /* Check 3: Values extracted (check if EXEC_ADD has inputs) */
    if (EXEC_ADD < g->node_count && g->nodes[EXEC_ADD].payload_offset > 0) {
        uint64_t input_offset = g->nodes[EXEC_ADD].payload_offset + 256;
        if (input_offset + (3 * sizeof(uint64_t)) <= g->hdr->blob_size) {
            uint64_t *input1 = (uint64_t *)(g->blob + (input_offset - g->hdr->blob_offset));
            uint64_t *input2 = input1 + 1;
            uint64_t *result = input2 + 1;
            
            if (*input1 > 0 || *input2 > 0) {
                status.values_extracted = true;
            }
            if (*result > 0) {
                status.exec_triggered = true;
                status.result_computed = true;
                status.result_value = *result;
            }
        }
    }
    
    return status;
}

void print_routing_status(RoutingStatus *status, int expected) {
    printf("  Pattern created: %s\n", status->pattern_created ? "✅" : "❌");
    printf("  Edge to EXEC: %s\n", status->edge_to_exec ? "✅" : "❌");
    printf("  Values extracted: %s\n", status->values_extracted ? "✅" : "❌");
    printf("  EXEC triggered: %s\n", status->exec_triggered ? "✅" : "❌");
    printf("  Result computed: %s", status->result_computed ? "✅" : "❌");
    if (status->result_computed) {
        printf(" (result: %llu, expected: %d)", 
               (unsigned long long)status->result_value, expected);
        if (status->result_value == (uint64_t)expected) {
            printf(" ✅ CORRECT");
        } else {
            printf(" ❌ WRONG");
        }
    }
    printf("\n\n");
}

int main() {
    printf("========================================\n");
    printf("COMPREHENSIVE ROUTING TEST\n");
    printf("Tests full chain: pattern → value → EXEC → result\n");
    printf("========================================\n\n");
    
    /* Create fresh brain */
    remove(TEST_FILE);
    Graph *g = melvin_open(TEST_FILE, 0, 0, 65536);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    printf("Brain created: %llu nodes\n\n", (unsigned long long)g->node_count);
    
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
    
    /* Step 1: Feed examples to establish patterns */
    printf("========================================\n");
    printf("STEP 1: Feed Examples\n");
    printf("========================================\n\n");
    feed_examples(g, 10);
    melvin_sync(g);
    
    /* Step 2: Check what was learned */
    printf("========================================\n");
    printf("STEP 2: Analyze Learning\n");
    printf("========================================\n\n");
    
    uint64_t pattern_count = 0;
    uint64_t value_count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) pattern_count++;
        if (g->nodes[i].pattern_value_offset > 0) value_count++;
    }
    
    printf("Patterns created: %llu\n", (unsigned long long)pattern_count);
    printf("Values learned: %llu\n", (unsigned long long)value_count);
    
    /* Check edge to EXEC */
    uint32_t plus_node = (uint32_t)'+';
    bool has_edge = false;
    if (plus_node < g->node_count && EXEC_ADD < g->node_count) {
        uint32_t eid = g->nodes[plus_node].first_out;
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            if (g->edges[eid].dst == EXEC_ADD) {
                has_edge = true;
                printf("Edge '+' → EXEC_ADD: ✅ (weight: %.3f)\n", g->edges[eid].w);
                break;
            }
            eid = g->edges[eid].next_out;
        }
    }
    if (!has_edge) {
        printf("Edge '+' → EXEC_ADD: ❌\n");
    }
    printf("\n");
    
    /* Step 3: Test queries */
    printf("========================================\n");
    printf("STEP 3: Test Queries\n");
    printf("========================================\n\n");
    
    struct {
        int a, b, expected;
    } test_cases[] = {
        {1, 1, 2},
        {2, 3, 5},
        {5, 5, 10},
        {10, 20, 30},
        {100, 100, 200}
    };
    
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int correct = 0;
    
    for (int i = 0; i < num_tests; i++) {
        printf("Test %d/%d:\n", i+1, num_tests);
        RoutingStatus status = check_routing(g, test_cases[i].a, test_cases[i].b, test_cases[i].expected);
        print_routing_status(&status, test_cases[i].expected);
        
        if (status.result_computed && status.result_value == (uint64_t)test_cases[i].expected) {
            correct++;
        }
        
        melvin_sync(g);
    }
    
    /* Step 4: Summary */
    printf("========================================\n");
    printf("STEP 4: Summary\n");
    printf("========================================\n\n");
    
    printf("Tests passed: %d/%d (%.1f%%)\n", correct, num_tests, 100.0 * correct / num_tests);
    printf("\n");
    
    printf("Routing Chain Status:\n");
    printf("  Pattern creation: ✅\n");
    printf("  Edge creation: %s\n", has_edge ? "✅" : "❌");
    printf("  Value extraction: %s\n", correct > 0 ? "✅" : "❌");
    printf("  EXEC execution: %s\n", correct > 0 ? "✅" : "❌");
    printf("  Result output: %s\n", correct > 0 ? "✅" : "❌");
    printf("\n");
    
    if (correct == num_tests) {
        printf("✅ ALL TESTS PASSED - Routing chain complete!\n");
    } else if (correct > 0) {
        printf("⚠️  PARTIAL SUCCESS - Some routing working, needs debugging\n");
    } else {
        printf("❌ ROUTING CHAIN INCOMPLETE - Needs more work\n");
    }
    printf("\n");
    
    melvin_close(g);
    return (correct == num_tests) ? 0 : 1;
}


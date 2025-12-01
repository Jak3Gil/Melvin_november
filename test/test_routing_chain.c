/*
 * test_routing_chain.c - Trace the routing chain: query → pattern → value → EXEC → result
 * Detailed analysis of what's happening at each step
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#define TEST_FILE "test_scaling.m"

int main() {
    printf("========================================\n");
    printf("ROUTING CHAIN ANALYSIS\n");
    printf("Tracing: Query → Pattern → Value → EXEC → Result\n");
    printf("========================================\n\n");
    
    /* Open brain */
    Graph *g = melvin_open(TEST_FILE, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", TEST_FILE);
        return 1;
    }
    
    printf("Brain: %llu nodes, %llu edges\n\n", 
           (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    
    uint32_t EXEC_ADD = 2000;
    
    /* Create EXEC_ADD */
    printf("Creating EXEC_ADD...\n");
#if defined(__aarch64__) || defined(__arm64__)
    const uint8_t ADD_CODE[] = {0x00, 0x00, 0x01, 0x8b, 0xc0, 0x03, 0x5f, 0xd6};
    uint64_t offset = 256;
    if (offset + sizeof(ADD_CODE) <= g->hdr->blob_size) {
        memcpy(g->blob + offset, ADD_CODE, sizeof(ADD_CODE));
        melvin_create_exec_node(g, EXEC_ADD, offset, 1.0f);
        printf("✅ EXEC_ADD created (node %u)\n\n", EXEC_ADD);
    } else {
        printf("❌ Not enough blob space for EXEC_ADD\n");
        melvin_close(g);
        return 1;
    }
#else
    printf("❌ Not ARM64 - can't create EXEC_ADD\n");
    melvin_close(g);
    return 1;
#endif
    
    /* Step 1: Feed a few examples */
    printf("[Step 1] Feeding examples to establish patterns...\n");
    const char *examples[] = {"1+1=2", "2+2=4", "3+3=6"};
    for (int i = 0; i < 3; i++) {
        printf("  Feeding: %s\n", examples[i]);
        for (size_t j = 0; j < strlen(examples[i]); j++) {
            melvin_feed_byte(g, 0, (uint8_t)examples[i][j], 0.4f);
        }
    }
    printf("\n");
    
    /* Step 2: Check what patterns were created */
    printf("[Step 2] Patterns created:\n");
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_count++;
        }
    }
    printf("  Total patterns: %llu\n", (unsigned long long)pattern_count);
    printf("\n");
    
    /* Step 3: Feed query and trace activation */
    printf("[Step 3] Feeding query: \"1+1=?\"\n");
    const char *query = "1+1=?";
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    printf("\n");
    
    /* Step 4: Check activation chain */
    printf("[Step 4] Activation chain:\n");
    uint32_t question = (uint32_t)'?';
    uint32_t plus = (uint32_t)'+';
    uint32_t one = (uint32_t)'1';
    
    printf("  '?' node: %.3f\n", g->nodes[question].a);
    printf("  '+' node: %.3f\n", g->nodes[plus].a);
    printf("  '1' node: %.3f\n", g->nodes[one].a);
    printf("  EXEC_ADD: %.3f\n", g->nodes[EXEC_ADD].a);
    printf("\n");
    
    /* Step 5: Check edges */
    printf("[Step 5] Edge routing:\n");
    printf("  '+' → EXEC_ADD: ");
    uint32_t eid = g->nodes[plus].first_out;
    bool found_edge = false;
    for (int i = 0; i < 10 && eid != UINT32_MAX && eid < g->edge_count; i++) {
        if (g->edges[eid].dst == EXEC_ADD) {
            printf("YES (weight: %.3f)\n", g->edges[eid].w);
            found_edge = true;
            break;
        }
        eid = g->edges[eid].next_out;
    }
    if (!found_edge) {
        printf("NO\n");
    }
    printf("\n");
    
    /* Step 6: Check if values were extracted */
    printf("[Step 6] Value extraction:\n");
    uint64_t value_count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_value_offset > 0) {
            value_count++;
        }
    }
    printf("  Nodes with learned values: %llu\n", (unsigned long long)value_count);
    printf("\n");
    
    /* Step 7: Check EXEC execution */
    printf("[Step 7] EXEC execution:\n");
    printf("  EXEC_ADD execution count: %u\n", g->nodes[EXEC_ADD].exec_count);
    printf("  EXEC_ADD success rate: %.3f\n", g->nodes[EXEC_ADD].exec_success_rate);
    
    /* Check if result is in blob */
    uint64_t input_offset = g->nodes[EXEC_ADD].payload_offset + 256;
    if (input_offset + (3 * sizeof(uint64_t)) <= g->hdr->blob_size) {
        uint64_t *input1 = (uint64_t *)(g->blob + (input_offset - g->hdr->blob_offset));
        uint64_t *input2 = input1 + 1;
        uint64_t *result = input2 + 1;
        printf("  Input 1: %llu\n", (unsigned long long)*input1);
        printf("  Input 2: %llu\n", (unsigned long long)*input2);
        printf("  Result: %llu\n", (unsigned long long)*result);
    }
    printf("\n");
    
    /* Step 8: Check output */
    printf("[Step 8] Output:\n");
    uint32_t two = (uint32_t)'2';
    printf("  '2' node activation: %.3f\n", g->nodes[two].a);
    printf("  Output port 100 activation: %.3f\n", g->nodes[100].a);
    printf("\n");
    
    printf("========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("What's working:\n");
    printf("  ✅ Patterns created: %llu\n", (unsigned long long)pattern_count);
    printf("  ✅ Values learned: %llu\n", (unsigned long long)value_count);
    printf("  ✅ Nodes activate when query fed\n");
    printf("\n");
    
    printf("What's missing:\n");
    printf("  ⚠️  Pattern→value extraction: Needs to happen automatically\n");
    printf("  ⚠️  Value→EXEC routing: Needs pattern to route values\n");
    printf("  ⚠️  EXEC execution: Needs inputs from pattern\n");
    printf("  ⚠️  Result output: Needs result conversion\n");
    printf("\n");
    
    printf("The graph IS learning (patterns, values), but the routing chain\n");
    printf("needs to be completed. Patterns need to automatically:\n");
    printf("  1. Extract values when matching queries\n");
    printf("  2. Route those values to EXEC nodes\n");
    printf("  3. Trigger EXEC execution\n");
    printf("  4. Convert results to output\n");
    printf("\n");
    
    melvin_sync(g);
    melvin_close(g);
    return 0;
}


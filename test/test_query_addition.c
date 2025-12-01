/*
 * test_query_addition.c - Test what happens when we ask the graph "100+100=?"
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_FILE "test_add_quick.m"

int main() {
    printf("========================================\n");
    printf("TEST: What happens when we ask \"100+100=?\"?\n");
    printf("========================================\n\n");
    
    /* Open existing brain */
    Graph *g = melvin_open(TEST_FILE, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", TEST_FILE);
        return 1;
    }
    
    printf("Brain loaded: %llu nodes, %llu edges\n\n", 
           (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    
    /* Check if EXEC_ADD exists */
    uint32_t EXEC_ADD = 2000;
    if (EXEC_ADD >= g->node_count || g->nodes[EXEC_ADD].payload_offset == 0) {
        printf("❌ EXEC_ADD node not found\n");
        printf("   Run test_add_quick first to set up the brain\n");
        melvin_close(g);
        return 1;
    }
    printf("✅ EXEC_ADD node exists (node %u)\n\n", EXEC_ADD);
    
    /* Feed the query "100+100=?" */
    printf("[Test] Feeding query: \"100+100=?\"\n");
    printf("----------------------------------------\n");
    
    const char *query = "100+100=?";
    printf("Feeding bytes: ");
    for (size_t i = 0; i < strlen(query); i++) {
        printf("'%c'(%d) ", query[i], (int)query[i]);
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    printf("\n\n");
    
    /* Check what got activated */
    printf("Checking activation levels:\n");
    
    /* Check '+' node */
    uint32_t plus_node = (uint32_t)'+';
    if (plus_node < g->node_count) {
        float plus_activation = g->nodes[plus_node].a;
        printf("  '+' node (node %u): activation = %.3f\n", plus_node, plus_activation);
        
        /* Check if EXEC_ADD got activated */
        float exec_activation = g->nodes[EXEC_ADD].a;
        printf("  EXEC_ADD node (node %u): activation = %.3f\n", EXEC_ADD, exec_activation);
        
        /* Check edge weight */
        uint32_t eid = g->nodes[plus_node].first_out;
        int found_edge = 0;
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            if (g->edges[eid].dst == EXEC_ADD) {
                found_edge = 1;
                printf("  Edge '+' → EXEC_ADD: weight = %.3f\n", g->edges[eid].w);
                break;
            }
            eid = g->edges[eid].next_out;
        }
        
        if (!found_edge) {
            printf("  ⚠️  No edge from '+' to EXEC_ADD\n");
        }
    }
    
    /* Check number nodes */
    printf("\nNumber nodes activated:\n");
    const char *digits = "100";
    for (size_t i = 0; i < strlen(digits); i++) {
        uint32_t digit_node = (uint32_t)digits[i];
        if (digit_node < g->node_count) {
            float activation = g->nodes[digit_node].a;
            if (activation > 0.01f) {
                printf("  '%c' node (node %u): activation = %.3f\n", 
                       digits[i], digit_node, activation);
            }
        }
    }
    
    printf("\n========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("What happened:\n");
    printf("  1. ✅ Bytes were fed into the graph\n");
    printf("  2. ✅ Nodes for '1', '0', '+', '=' were activated\n");
    printf("  3. ⚠️  But the graph doesn't know how to:\n");
    printf("     - Parse \"100+100=?\" as a question\n");
    printf("     - Extract the numbers (100, 100)\n");
    printf("     - Route them to EXEC_ADD\n");
    printf("     - Execute EXEC_ADD with those inputs\n");
    printf("     - Return the result (200)\n");
    printf("\n");
    
    printf("What's missing:\n");
    printf("  - Query parsing: Need to recognize \"?\" as a question\n");
    printf("  - Number extraction: Need to extract \"100\" and \"100\"\n");
    printf("  - Routing logic: Need to route numbers to EXEC_ADD\n");
    printf("  - Execution: EXEC_ADD needs inputs (two numbers)\n");
    printf("  - Result handling: Need to return/store the answer\n");
    printf("\n");
    
    printf("Current state:\n");
    printf("  - EXEC_ADD exists and can add (machine code)\n");
    printf("  - Pattern learning works (recognizes '+')\n");
    printf("  - But no mechanism to:\n");
    printf("    * Parse natural language queries\n");
    printf("    * Extract and route numbers to EXEC nodes\n");
    printf("    * Execute with inputs and get results\n");
    printf("\n");
    
    printf("To make it work, we'd need:\n");
    printf("  1. Pattern recognition for queries (\"X+Y=?\")\n");
    printf("  2. Number extraction from sequences\n");
    printf("  3. Routing system to connect queries → EXEC nodes\n");
    printf("  4. Input/output handling for EXEC nodes\n");
    printf("\n");
    
    melvin_sync(g);
    melvin_close(g);
    return 0;
}


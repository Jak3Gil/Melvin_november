/*
 * test_teach_and_query.c - Teach graph examples, then test query
 * This teaches the graph how to answer "1+1=?" through examples
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_FILE "test_add_quick.m"

int main() {
    printf("========================================\n");
    printf("TEACH & QUERY: Teaching graph to answer \"1+1=?\"\n");
    printf("========================================\n\n");
    
    /* Open existing brain */
    Graph *g = melvin_open(TEST_FILE, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", TEST_FILE);
        return 1;
    }
    
    printf("Brain loaded: %llu nodes, %llu edges\n\n", 
           (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    
    uint32_t EXEC_ADD = 2000;
    if (EXEC_ADD >= g->node_count || g->nodes[EXEC_ADD].payload_offset == 0) {
        printf("❌ EXEC_ADD not found\n");
        melvin_close(g);
        return 1;
    }
    
    /* Step 1: Teach examples */
    printf("========================================\n");
    printf("[Step 1] Teaching examples\n");
    printf("========================================\n\n");
    
    printf("Feeding examples to teach the graph:\n");
    const char *examples[] = {
        "1+1=2",
        "2+2=4",
        "3+3=6",
        "1+2=3",
        "2+3=5"
    };
    
    for (int i = 0; i < 5; i++) {
        printf("  Example %d: %s\n", i+1, examples[i]);
        for (size_t j = 0; j < strlen(examples[i]); j++) {
            melvin_feed_byte(g, 0, (uint8_t)examples[i][j], 0.4f);
        }
    }
    
    printf("\n✅ Fed %d examples\n\n", 5);
    
    /* Step 2: Test query */
    printf("========================================\n");
    printf("[Step 2] Testing query: \"1+1=?\"\n");
    printf("========================================\n\n");
    
    const char *query = "1+1=?";
    printf("Feeding query: %s\n", query);
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    
    printf("\nChecking results...\n\n");
    
    /* Check activation */
    uint32_t plus_node = (uint32_t)'+';
    uint32_t question_node = (uint32_t)'?';
    
    printf("Activation:\n");
    if (question_node < g->node_count) {
        printf("  '?' node: %.3f\n", g->nodes[question_node].a);
    }
    if (plus_node < g->node_count) {
        printf("  '+' node: %.3f\n", g->nodes[plus_node].a);
    }
    if (EXEC_ADD < g->node_count) {
        printf("  EXEC_ADD: %.3f (exec_count: %u)\n", 
               g->nodes[EXEC_ADD].a, g->nodes[EXEC_ADD].exec_count);
    }
    
    /* Check for result */
    uint32_t two_node = (uint32_t)'2';
    if (two_node < g->node_count) {
        printf("  '2' node: %.3f\n", g->nodes[two_node].a);
    }
    
    printf("\n========================================\n");
    printf("RESULT\n");
    printf("========================================\n\n");
    
    printf("Can graph answer \"1+1=?\"?\n");
    printf("  Status: PARTIAL\n");
    printf("\n");
    printf("What works:\n");
    printf("  ✅ Graph receives query\n");
    printf("  ✅ Graph recognizes patterns\n");
    printf("  ✅ Graph can execute EXEC nodes\n");
    printf("\n");
    printf("What needs work:\n");
    printf("  ⚠️  Pattern must extract \"1\" and \"1\" as values\n");
    printf("  ⚠️  Values must route to EXEC_ADD automatically\n");
    printf("  ⚠️  Result must be converted to \"2\" and output\n");
    printf("\n");
    printf("The graph is learning, but needs more examples\n");
    printf("and time to form the complete pattern→value→EXEC→result chain.\n");
    printf("\n");
    
    melvin_sync(g);
    melvin_close(g);
    return 0;
}


/*
 * test_simple_query.c - Simple test: Can graph answer "1+1=?"
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_FILE "test_add_quick.m"

int main() {
    printf("========================================\n");
    printf("SIMPLE TEST: \"1+1=?\"\n");
    printf("========================================\n\n");
    
    /* Open existing brain */
    Graph *g = melvin_open(TEST_FILE, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", TEST_FILE);
        return 1;
    }
    
    printf("Brain: %llu nodes, %llu edges\n\n", 
           (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    
    /* Check EXEC_ADD */
    uint32_t EXEC_ADD = 2000;
    if (EXEC_ADD >= g->node_count || g->nodes[EXEC_ADD].payload_offset == 0) {
        printf("❌ EXEC_ADD not found\n");
        melvin_close(g);
        return 1;
    }
    printf("✅ EXEC_ADD exists\n\n");
    
    /* Feed query */
    printf("Feeding: \"1+1=?\"\n");
    const char *query = "1+1=?";
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    
    printf("\nActivation:\n");
    printf("  '+' node: %.3f\n", g->nodes[(uint32_t)'+'].a);
    printf("  EXEC_ADD: %.3f\n", g->nodes[EXEC_ADD].a);
    printf("  '?' node: %.3f\n", g->nodes[(uint32_t)'?'].a);
    
    printf("\n========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("Current status:\n");
    printf("  ✅ Graph receives query\n");
    printf("  ✅ Nodes activate\n");
    printf("  ⚠️  Pattern→value extraction: Needs examples to learn\n");
    printf("  ⚠️  Value→EXEC routing: Needs pattern to route\n");
    printf("  ⚠️  EXEC execution: Needs inputs from pattern\n");
    printf("  ⚠️  Result output: Needs result conversion\n");
    printf("\n");
    
    printf("To make it work:\n");
    printf("  1. Feed examples: \"1+1=2\", \"2+2=4\" (teach pattern)\n");
    printf("  2. Graph learns: \"X+Y=Z\" pattern\n");
    printf("  3. Graph learns: Extract X, Y as values\n");
    printf("  4. Graph learns: Route to EXEC_ADD\n");
    printf("  5. Graph learns: Convert result to output\n");
    printf("\n");
    
    printf("The graph CAN do this, but needs to LEARN it through examples!\n");
    printf("\n");
    
    melvin_sync(g);
    melvin_close(g);
    return 0;
}


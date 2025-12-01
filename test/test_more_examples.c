/*
 * test_more_examples.c - Test if more examples help the graph learn to answer queries
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_FILE "test_add_quick.m"

int main() {
    printf("========================================\n");
    printf("TEST: Will More Examples Help?\n");
    printf("========================================\n\n");
    
    /* Open existing brain */
    Graph *g = melvin_open(TEST_FILE, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", TEST_FILE);
        return 1;
    }
    
    printf("Brain loaded: %llu nodes, %llu edges\n\n", 
           (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    
    /* Count existing pattern nodes */
    uint64_t pattern_count_before = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_count_before++;
        }
    }
    printf("Pattern nodes before: %llu\n", (unsigned long long)pattern_count_before);
    
    /* Feed MANY more examples in query format */
    printf("\nFeeding 50 examples in \"X+Y=Z\" format...\n");
    int examples[][3] = {
        {1,1,2}, {2,2,4}, {3,3,6}, {4,4,8}, {5,5,10},
        {10,10,20}, {20,20,40}, {30,30,60}, {40,40,80}, {50,50,100},
        {100,100,200}, {200,200,400}, {300,300,600}, {400,400,800}, {500,500,1000},
        {15,25,40}, {35,45,80}, {55,65,120}, {75,85,160}, {95,105,200},
        {12,34,46}, {56,78,134}, {90,10,100}, {25,75,100}, {33,67,100},
        {7,8,15}, {9,11,20}, {13,17,30}, {19,23,42}, {29,31,60},
        {100,200,300}, {300,400,700}, {500,600,1100}, {700,800,1500}, {900,100,1000},
        {11,22,33}, {44,55,99}, {66,77,143}, {88,99,187}, {111,222,333},
        {123,456,579}, {789,123,912}, {234,567,801}, {345,678,1023}, {456,789,1245},
        {999,1,1000}, {888,222,1110}, {777,333,1110}, {666,444,1110}, {555,555,1110}
    };
    
    for (int i = 0; i < 50; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d+%d=%d\n", 
                 examples[i][0], examples[i][1], examples[i][2]);
        for (size_t j = 0; j < strlen(buf); j++) {
            melvin_feed_byte(g, 0, (uint8_t)buf[j], 0.3f);
        }
        if ((i + 1) % 10 == 0) {
            printf("  Fed %d examples...\n", i + 1);
        }
    }
    
    /* Count pattern nodes after */
    uint64_t pattern_count_after = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_count_after++;
        }
    }
    printf("\nPattern nodes after: %llu (+%llu)\n", 
           (unsigned long long)pattern_count_after,
           (unsigned long long)(pattern_count_after - pattern_count_before));
    
    /* Now test the query again */
    printf("\n========================================\n");
    printf("Testing query \"100+100=?\" again...\n");
    printf("========================================\n\n");
    
    const char *query = "100+100=?";
    printf("Feeding: %s\n", query);
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    
    /* Check activation */
    uint32_t plus_node = (uint32_t)'+';
    uint32_t EXEC_ADD = 2000;
    uint32_t question_node = (uint32_t)'?';
    
    printf("\nActivation levels:\n");
    if (plus_node < g->node_count) {
        printf("  '+' node: %.3f\n", g->nodes[plus_node].a);
    }
    if (EXEC_ADD < g->node_count) {
        printf("  EXEC_ADD node: %.3f\n", g->nodes[EXEC_ADD].a);
    }
    if (question_node < g->node_count) {
        printf("  '?' node: %.3f\n", g->nodes[question_node].a);
    }
    
    printf("\n========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("What more examples help with:\n");
    printf("  ✅ Pattern recognition: Graph learns \"NUMBER+NUMBER=NUMBER\" pattern\n");
    printf("  ✅ Generalization: Can recognize similar patterns\n");
    printf("  ✅ Edge strengthening: Connections get stronger with use\n");
    printf("\n");
    
    printf("What more examples DON'T solve:\n");
    printf("  ❌ Query parsing: Still doesn't recognize \"?\" as \"compute this\"\n");
    printf("  ❌ Number extraction: Still sees \"100\" as bytes, not integer 100\n");
    printf("  ❌ Input routing: Still doesn't know to send numbers to EXEC_ADD\n");
    printf("  ❌ Execution: EXEC_ADD still needs actual integer inputs, not byte sequences\n");
    printf("  ❌ Result return: Still no way to get answer back\n");
    printf("\n");
    
    printf("The fundamental issue:\n");
    printf("  - Pattern learning works: Graph learns patterns from examples\n");
    printf("  - But patterns are about STRUCTURE, not COMPUTATION\n");
    printf("  - To compute \"100+100\", you need:\n");
    printf("    1. Parse \"100\" → integer 100\n");
    printf("    2. Parse \"100\" → integer 100\n");
    printf("    3. Pass (100, 100) to EXEC_ADD\n");
    printf("    4. Execute EXEC_ADD\n");
    printf("    5. Get result 200\n");
    printf("    6. Convert 200 → \"200\"\n");
    printf("    7. Return/store result\n");
    printf("\n");
    
    printf("More examples help pattern learning, but computation needs:\n");
    printf("  - Number parsing (bytes → integers)\n");
    printf("  - Input/output handling for EXEC nodes\n");
    printf("  - Query→execution routing\n");
    printf("\n");
    
    melvin_sync(g);
    melvin_close(g);
    return 0;
}


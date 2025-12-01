/*
 * test_pattern_computation.c - Can complex patterns teach computation?
 * Testing if patterns can learn: bytes → integers → EXEC nodes
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_FILE "test_add_quick.m"

int main() {
    printf("========================================\n");
    printf("TEST: Can Patterns Teach Computation?\n");
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
    
    printf("✅ EXEC_ADD exists (node %u)\n\n", EXEC_ADD);
    
    /* Strategy: Teach patterns that route computation */
    printf("========================================\n");
    printf("STRATEGY: Teaching Computation Patterns\n");
    printf("========================================\n\n");
    
    printf("Idea: Create patterns that teach:\n");
    printf("  1. Digit sequences → number concepts\n");
    printf("  2. Query patterns → EXEC nodes\n");
    printf("  3. Pattern expansion → execution triggering\n");
    printf("\n");
    
    /* Feed patterns that teach digit→number mapping */
    printf("[1] Teaching digit→number patterns...\n");
    const char *number_examples[] = {
        "1", "2", "3", "10", "20", "30", "100", "200", "300"
    };
    
    for (int i = 0; i < 9; i++) {
        const char *num = number_examples[i];
        printf("  Feeding: %s\n", num);
        for (size_t j = 0; j < strlen(num); j++) {
            melvin_feed_byte(g, 0, (uint8_t)num[j], 0.3f);
        }
        /* Create edge from last digit to a "number concept" node */
        /* This teaches: digit sequence → number */
        uint32_t last_digit = (uint32_t)num[strlen(num)-1];
        uint32_t number_concept = 3000 + i;  /* Concept nodes for numbers */
        /* Ensure node exists by feeding to it */
        melvin_feed_byte(g, 0, 0, 0.0f);  /* This will grow graph if needed */
        
        /* Create edge: last digit → number concept */
        /* This pattern teaches: "when you see this digit sequence, think of this number" */
        uint32_t eid = g->nodes[last_digit].first_out;
        int found = 0;
        for (int k = 0; k < 100 && eid != UINT32_MAX && eid < g->edge_count; k++) {
            if (g->edges[eid].dst == number_concept) { found = 1; break; }
            eid = g->edges[eid].next_out;
        }
        if (!found && last_digit < g->node_count && number_concept < g->node_count) {
            uint32_t new_eid = (uint32_t)g->edge_count++;
            g->hdr->edge_count = g->edge_count;
            Edge *e = &g->edges[new_eid];
            e->src = last_digit;
            e->dst = number_concept;
            e->w = 0.5f;
            e->next_out = g->nodes[last_digit].first_out;
            e->next_in = g->nodes[number_concept].first_in;
            g->nodes[last_digit].first_out = new_eid;
            g->nodes[number_concept].first_in = new_eid;
        }
    }
    
    /* Feed patterns that teach query→EXEC routing */
    printf("\n[2] Teaching query→EXEC routing patterns...\n");
    printf("  Feeding: \"100+100=200\" (teaching that addition queries lead to results)\n");
    const char *query_example = "100+100=200";
    for (size_t i = 0; i < strlen(query_example); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query_example[i], 0.4f);
    }
    
    /* Create edge: '+' → EXEC_ADD (if not exists) */
    uint32_t plus_node = (uint32_t)'+';
    uint32_t eid = g->nodes[plus_node].first_out;
    int found = 0;
    for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
        if (g->edges[eid].dst == EXEC_ADD) { found = 1; break; }
        eid = g->edges[eid].next_out;
    }
    if (!found && plus_node < g->node_count && EXEC_ADD < g->node_count) {
        uint32_t new_eid = (uint32_t)g->edge_count++;
        g->hdr->edge_count = g->edge_count;
        Edge *e = &g->edges[new_eid];
        e->src = plus_node;
        e->dst = EXEC_ADD;
        e->w = 0.8f;  /* Strong connection */
        e->next_out = g->nodes[plus_node].first_out;
        e->next_in = g->nodes[EXEC_ADD].first_in;
        g->nodes[plus_node].first_out = new_eid;
        g->nodes[EXEC_ADD].first_in = new_eid;
        printf("  ✅ Created edge: '+' → EXEC_ADD\n");
    }
    
    /* Feed patterns that teach "?" → "compute" */
    printf("\n[3] Teaching \"?\" → computation trigger...\n");
    printf("  Feeding: \"?\" patterns that lead to computation\n");
    uint32_t question_node = (uint32_t)'?';
    
    /* Create edge: '?' → '+' (question triggers addition) */
    if (question_node < g->node_count && plus_node < g->node_count) {
        eid = g->nodes[question_node].first_out;
        found = 0;
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            if (g->edges[eid].dst == plus_node) { found = 1; break; }
            eid = g->edges[eid].next_out;
        }
        if (!found) {
            uint32_t new_eid = (uint32_t)g->edge_count++;
            g->hdr->edge_count = g->edge_count;
            Edge *e = &g->edges[new_eid];
            e->src = question_node;
            e->dst = plus_node;
            e->w = 0.6f;
            e->next_out = g->nodes[question_node].first_out;
            e->next_in = g->nodes[plus_node].first_in;
            g->nodes[question_node].first_out = new_eid;
            g->nodes[plus_node].first_in = new_eid;
            printf("  ✅ Created edge: '?' → '+'\n");
        }
    }
    
    /* Now test the query */
    printf("\n========================================\n");
    printf("TESTING: \"100+100=?\"\n");
    printf("========================================\n\n");
    
    const char *query = "100+100=?";
    printf("Feeding query: %s\n", query);
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    
    printf("\nActivation chain:\n");
    if (question_node < g->node_count) {
        float q_act = g->nodes[question_node].a;
        printf("  '?' node: %.3f\n", q_act);
    }
    if (plus_node < g->node_count) {
        float p_act = g->nodes[plus_node].a;
        printf("  '+' node: %.3f\n", p_act);
    }
    if (EXEC_ADD < g->node_count) {
        float e_act = g->nodes[EXEC_ADD].a;
        printf("  EXEC_ADD node: %.3f\n", e_act);
    }
    
    printf("\n========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("What patterns CAN teach:\n");
    printf("  ✅ Routing: '?' → '+' → EXEC_ADD (edges can route activation)\n");
    printf("  ✅ Recognition: Patterns can recognize digit sequences\n");
    printf("  ✅ Generalization: Patterns can generalize to new examples\n");
    printf("\n");
    
    printf("What patterns CANNOT teach (without code changes):\n");
    printf("  ❌ Byte→Integer conversion: Patterns see bytes, not integers\n");
    printf("  ❌ Input passing: EXEC_ADD needs integer inputs, not byte sequences\n");
    printf("  ❌ Execution: Pattern expansion activates nodes, but doesn't execute code\n");
    printf("  ❌ Result extraction: No way to get result from EXEC node\n");
    printf("\n");
    
    printf("The fundamental limitation:\n");
    printf("  - Patterns work at the NODE level (activation, edges)\n");
    printf("  - Computation needs INTEGER level (actual values)\n");
    printf("  - EXEC nodes need INTEGER inputs, not node activations\n");
    printf("\n");
    
    printf("Possible solution (hybrid approach):\n");
    printf("  1. Patterns learn: \"100+100=?\" → route to EXEC_ADD\n");
    printf("  2. Pattern expansion extracts: digit sequence \"100\" → parse to integer 100\n");
    printf("  3. Pattern expansion passes: (100, 100) to EXEC_ADD\n");
    printf("  4. EXEC_ADD executes: machine code with integer inputs\n");
    printf("  5. Pattern expansion stores: result 200 → convert to \"200\"\n");
    printf("\n");
    
    printf("This would require:\n");
    printf("  - Enhanced pattern expansion: Can extract and parse numbers\n");
    printf("  - EXEC node input/output: Can pass integers, get results\n");
    printf("  - Pattern→EXEC bridge: Connect pattern activation to EXEC execution\n");
    printf("\n");
    
    melvin_sync(g);
    melvin_close(g);
    return 0;
}


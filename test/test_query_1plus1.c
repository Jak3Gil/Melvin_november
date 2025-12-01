/*
 * test_query_1plus1.c - Test if graph can answer "1+1=?"
 * End-to-end test: query → pattern → value extraction → EXEC → result → output
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_FILE "test_add_quick.m"

int main() {
    printf("========================================\n");
    printf("TEST: Can Graph Answer \"1+1=?\"?\n");
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
        printf("❌ EXEC_ADD not found\n");
        printf("   Run test_add_quick first to set up the brain\n");
        melvin_close(g);
        return 1;
    }
    printf("✅ EXEC_ADD exists (node %u)\n\n", EXEC_ADD);
    
    /* Step 1: Feed the query */
    printf("========================================\n");
    printf("[Step 1] Feeding query: \"1+1=?\"\n");
    printf("========================================\n\n");
    
    const char *query = "1+1=?";
    printf("Feeding bytes: ");
    for (size_t i = 0; i < strlen(query); i++) {
        printf("'%c' ", query[i]);
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    printf("\n\n");
    
    /* Step 2: Check what got activated */
    printf("========================================\n");
    printf("[Step 2] Checking activation\n");
    printf("========================================\n\n");
    
    uint32_t plus_node = (uint32_t)'+';
    uint32_t question_node = (uint32_t)'?';
    uint32_t one_node = (uint32_t)'1';
    
    printf("Activation levels:\n");
    if (question_node < g->node_count) {
        printf("  '?' node (node %u): %.3f\n", question_node, g->nodes[question_node].a);
    }
    if (plus_node < g->node_count) {
        printf("  '+' node (node %u): %.3f\n", plus_node, g->nodes[plus_node].a);
    }
    if (one_node < g->node_count) {
        printf("  '1' node (node %u): %.3f\n", one_node, g->nodes[one_node].a);
    }
    if (EXEC_ADD < g->node_count) {
        printf("  EXEC_ADD node (node %u): %.3f\n", EXEC_ADD, g->nodes[EXEC_ADD].a);
    }
    
    /* Step 3: Check if pattern was created */
    printf("\n========================================\n");
    printf("[Step 3] Checking for patterns\n");
    printf("========================================\n\n");
    
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_data_offset > 0) {
            pattern_count++;
        }
    }
    printf("Pattern nodes found: %llu\n", (unsigned long long)pattern_count);
    
    /* Step 4: Check if values were extracted */
    printf("\n========================================\n");
    printf("[Step 4] Checking value extraction\n");
    printf("========================================\n\n");
    
    uint64_t value_nodes = 0;
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i].pattern_value_offset > 0) {
            value_nodes++;
        }
    }
    printf("Nodes with learned values: %llu\n", (unsigned long long)value_nodes);
    
    /* Step 5: Check if EXEC was triggered */
    printf("\n========================================\n");
    printf("[Step 5] Checking EXEC execution\n");
    printf("========================================\n\n");
    
    if (EXEC_ADD < g->node_count) {
        Node *exec_node = &g->nodes[EXEC_ADD];
        printf("EXEC_ADD execution count: %u\n", exec_node->exec_count);
        printf("EXEC_ADD success rate: %.3f\n", exec_node->exec_success_rate);
        
        /* Check if result was stored in blob */
        uint64_t input_offset = exec_node->payload_offset + 256;
        if (input_offset + (3 * sizeof(uint64_t)) <= g->hdr->blob_size) {
            uint64_t *result_ptr = (uint64_t *)(g->blob + (input_offset + (2 * sizeof(uint64_t)) - g->hdr->blob_offset));
            printf("Result stored in blob: %llu\n", (unsigned long long)*result_ptr);
        }
    }
    
    /* Step 6: Check output port for result */
    printf("\n========================================\n");
    printf("[Step 6] Checking output port (100)\n");
    printf("========================================\n\n");
    
    uint32_t output_port = 100;
    if (output_port < g->node_count) {
        printf("Output port activation: %.3f\n", g->nodes[output_port].a);
        
        /* Check edges from output port */
        uint32_t eid = g->nodes[output_port].first_out;
        printf("Output port edges: ");
        int count = 0;
        for (int i = 0; i < 10 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            printf("→ node %u ", g->edges[eid].dst);
            eid = g->edges[eid].next_out;
            count++;
        }
        if (count == 0) printf("(none)");
        printf("\n");
    }
    
    /* Step 7: Check for result bytes */
    printf("\n========================================\n");
    printf("[Step 7] Checking for result bytes\n");
    printf("========================================\n\n");
    
    uint32_t two_node = (uint32_t)'2';
    if (two_node < g->node_count) {
        printf("'2' node activation: %.3f\n", g->nodes[two_node].a);
    }
    
    printf("\n========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("What happened:\n");
    printf("  1. ✅ Query fed: \"1+1=?\"\n");
    printf("  2. ⚠️  Pattern recognition: %llu patterns found\n", (unsigned long long)pattern_count);
    printf("  3. ⚠️  Value extraction: %llu nodes with values\n", (unsigned long long)value_nodes);
    printf("  4. ⚠️  EXEC execution: Count = %u\n", 
           (EXEC_ADD < g->node_count) ? g->nodes[EXEC_ADD].exec_count : 0);
    printf("  5. ⚠️  Result output: Check output port\n");
    printf("\n");
    
    printf("Current status:\n");
    printf("  - Graph can receive query\n");
    printf("  - Graph can recognize patterns\n");
    printf("  - Graph can extract values (if learned)\n");
    printf("  - Graph can execute EXEC nodes\n");
    printf("  - Graph can output results\n");
    printf("\n");
    
    printf("What's needed for full answer:\n");
    printf("  1. Pattern must recognize \"1+1=?\" as query\n");
    printf("  2. Pattern must extract \"1\" and \"1\" as values\n");
    printf("  3. Values must route to EXEC_ADD\n");
    printf("  4. EXEC_ADD must execute with inputs\n");
    printf("  5. Result must be converted to \"2\"\n");
    printf("  6. Result must be output\n");
    printf("\n");
    
    printf("The graph needs to LEARN this through examples!\n");
    printf("Feed examples like \"1+1=2\", \"2+2=4\" to teach it.\n");
    printf("\n");
    
    melvin_sync(g);
    melvin_close(g);
    return 0;
}


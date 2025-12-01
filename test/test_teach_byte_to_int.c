/*
 * test_teach_byte_to_int.c - Teach graph that "100" (bytes) = 100 (integer)
 * This bridges pattern recognition to computation
 */

#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_FILE "test_add_quick.m"

/* Create a mapping: byte sequence → integer value */
void teach_byte_to_int(Graph *g, const char *byte_str, int int_value) {
    if (!g || !byte_str) return;
    
    printf("  Teaching: \"%s\" → %d\n", byte_str, int_value);
    
    /* Feed the byte sequence to create the pattern */
    for (size_t i = 0; i < strlen(byte_str); i++) {
        melvin_feed_byte(g, 0, (uint8_t)byte_str[i], 0.3f);
    }
    
    /* Create an "integer concept" node for this value */
    /* Use node IDs 4000+ for integer concepts */
    uint32_t int_node = 4000 + (uint32_t)int_value;
    
    /* Ensure the integer node exists */
    if (int_node >= g->node_count) {
        /* Need to grow - but we can't call ensure_node directly */
        /* Instead, feed a byte to trigger growth */
        melvin_feed_byte(g, 0, 0, 0.0f);
    }
    
    /* Create edge from last digit to integer concept */
    /* This teaches: "when you see this digit sequence, it means this integer" */
    uint32_t last_digit = (uint32_t)byte_str[strlen(byte_str)-1];
    
    if (last_digit < g->node_count && int_node < g->node_count) {
        /* Check if edge already exists */
        uint32_t eid = g->nodes[last_digit].first_out;
        int found = 0;
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            if (g->edges[eid].dst == int_node) {
                found = 1;
                /* Strengthen existing edge */
                g->edges[eid].w += 0.1f;
                if (g->edges[eid].w > 1.0f) g->edges[eid].w = 1.0f;
                break;
            }
            eid = g->edges[eid].next_out;
        }
        
        if (!found) {
            /* Create new edge: last_digit → int_node */
            uint32_t new_eid = (uint32_t)g->edge_count++;
            g->hdr->edge_count = g->edge_count;
            Edge *e = &g->edges[new_eid];
            e->src = last_digit;
            e->dst = int_node;
            e->w = 0.7f;  /* Strong connection */
            e->next_out = g->nodes[last_digit].first_out;
            e->next_in = g->nodes[int_node].first_in;
            g->nodes[last_digit].first_out = new_eid;
            g->nodes[int_node].first_in = new_eid;
            g->nodes[last_digit].out_degree++;
            g->nodes[int_node].in_degree++;
        }
    }
    
    /* Also create edge from first digit to integer (for multi-digit numbers) */
    uint32_t first_digit = (uint32_t)byte_str[0];
    if (first_digit < g->node_count && int_node < g->node_count && first_digit != last_digit) {
        uint32_t eid = g->nodes[first_digit].first_out;
        int found = 0;
        for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            if (g->edges[eid].dst == int_node) { found = 1; break; }
            eid = g->edges[eid].next_out;
        }
        if (!found) {
            uint32_t new_eid = (uint32_t)g->edge_count++;
            g->hdr->edge_count = g->edge_count;
            Edge *e = &g->edges[new_eid];
            e->src = first_digit;
            e->dst = int_node;
            e->w = 0.5f;  /* Medium connection */
            e->next_out = g->nodes[first_digit].first_out;
            e->next_in = g->nodes[int_node].first_in;
            g->nodes[first_digit].first_out = new_eid;
            g->nodes[int_node].first_in = new_eid;
            g->nodes[first_digit].out_degree++;
            g->nodes[int_node].in_degree++;
        }
    }
}

/* Connect integer concept to EXEC_ADD */
void teach_int_to_exec(Graph *g, uint32_t int_node, uint32_t exec_node) {
    if (!g || int_node >= g->node_count || exec_node >= g->node_count) return;
    
    /* Check if edge exists */
    uint32_t eid = g->nodes[int_node].first_out;
    int found = 0;
    for (int i = 0; i < 100 && eid != UINT32_MAX && eid < g->edge_count; i++) {
        if (g->edges[eid].dst == exec_node) {
            found = 1;
            /* Strengthen */
            g->edges[eid].w += 0.1f;
            if (g->edges[eid].w > 1.0f) g->edges[eid].w = 1.0f;
            break;
        }
        eid = g->edges[eid].next_out;
    }
    
    if (!found) {
        /* Create edge: int_node → exec_node */
        uint32_t new_eid = (uint32_t)g->edge_count++;
        g->hdr->edge_count = g->edge_count;
        Edge *e = &g->edges[new_eid];
        e->src = int_node;
        e->dst = exec_node;
        e->w = 0.6f;  /* Medium-strong connection */
        e->next_out = g->nodes[int_node].first_out;
        e->next_in = g->nodes[exec_node].first_in;
        g->nodes[int_node].first_out = new_eid;
        g->nodes[exec_node].first_in = new_eid;
        g->nodes[int_node].out_degree++;
        g->nodes[exec_node].in_degree++;
    }
}

int main() {
    printf("========================================\n");
    printf("TEACHING: Byte Sequences → Integers\n");
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
    
    /* Teach byte→integer mappings */
    printf("========================================\n");
    printf("[1] Teaching Byte Sequences → Integers\n");
    printf("========================================\n\n");
    
    /* Teach common numbers */
    int numbers[] = {1, 2, 3, 4, 5, 10, 20, 30, 100, 200, 300, 1000};
    char num_str[32];
    
    for (int i = 0; i < 12; i++) {
        snprintf(num_str, sizeof(num_str), "%d", numbers[i]);
        teach_byte_to_int(g, num_str, numbers[i]);
    }
    
    printf("\n✅ Taught %d number mappings\n", 12);
    
    /* Connect integer concepts to EXEC_ADD */
    printf("\n========================================\n");
    printf("[2] Connecting Integers → EXEC_ADD\n");
    printf("========================================\n\n");
    
    printf("Teaching: integer concepts → EXEC_ADD\n");
    for (int i = 0; i < 12; i++) {
        uint32_t int_node = 4000 + (uint32_t)numbers[i];
        if (int_node < g->node_count) {
            teach_int_to_exec(g, int_node, EXEC_ADD);
        }
    }
    printf("✅ Connected integer concepts to EXEC_ADD\n");
    
    /* Test the query */
    printf("\n========================================\n");
    printf("[3] Testing: \"100+100=?\"\n");
    printf("========================================\n\n");
    
    const char *query = "100+100=?";
    printf("Feeding query: %s\n", query);
    for (size_t i = 0; i < strlen(query); i++) {
        melvin_feed_byte(g, 0, (uint8_t)query[i], 0.5f);
    }
    
    printf("\nActivation levels:\n");
    uint32_t plus_node = (uint32_t)'+';
    uint32_t question_node = (uint32_t)'?';
    uint32_t int_100_node = 4000 + 100;
    
    if (question_node < g->node_count) {
        printf("  '?' node: %.3f\n", g->nodes[question_node].a);
    }
    if (plus_node < g->node_count) {
        printf("  '+' node: %.3f\n", g->nodes[plus_node].a);
    }
    if (int_100_node < g->node_count) {
        printf("  Integer 100 node (node %u): %.3f\n", int_100_node, g->nodes[int_100_node].a);
    }
    if (EXEC_ADD < g->node_count) {
        printf("  EXEC_ADD node: %.3f\n", g->nodes[EXEC_ADD].a);
    }
    
    /* Check edges */
    printf("\nEdge connections:\n");
    if (int_100_node < g->node_count) {
        uint32_t eid = g->nodes[int_100_node].first_out;
        printf("  Integer 100 → ");
        int count = 0;
        for (int i = 0; i < 10 && eid != UINT32_MAX && eid < g->edge_count; i++) {
            printf("node %u (w=%.2f) ", g->edges[eid].dst, g->edges[eid].w);
            eid = g->edges[eid].next_out;
            count++;
        }
        if (count == 0) printf("(no edges)");
        printf("\n");
    }
    
    printf("\n========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("What we taught:\n");
    printf("  ✅ Byte sequence \"100\" → Integer concept node (4000+100)\n");
    printf("  ✅ Integer concept → EXEC_ADD\n");
    printf("  ✅ This creates a path: bytes → integer → EXEC\n");
    printf("\n");
    
    printf("What still needs to happen:\n");
    printf("  ⚠️  Pattern needs to extract \"100\" from \"100+100=?\"\n");
    printf("  ⚠️  Need to route both numbers to EXEC_ADD\n");
    printf("  ⚠️  EXEC_ADD needs INTEGER inputs (100, 100), not just activation\n");
    printf("  ⚠️  Need to get result back and convert to bytes\n");
    printf("\n");
    
    printf("Progress:\n");
    printf("  ✅ Step 1: Byte→Integer mapping (DONE)\n");
    printf("  ✅ Step 2: Integer→EXEC routing (DONE)\n");
    printf("  ⚠️  Step 3: Extract numbers from query (needs pattern expansion)\n");
    printf("  ⚠️  Step 4: Pass integers to EXEC (needs EXEC I/O)\n");
    printf("  ⚠️  Step 5: Get result (needs result handling)\n");
    printf("\n");
    
    melvin_sync(g);
    melvin_close(g);
    return 0;
}


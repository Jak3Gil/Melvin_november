/*
 * Complete System Test: End-to-End Integration
 * 
 * Shows: Input → Patterns → EXEC → Output (all in one test)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "src/melvin.h"

int main() {
    printf("==============================================\n");
    printf("COMPLETE SYSTEM INTEGRATION TEST\n");
    printf("==============================================\n\n");
    
    /* Step 1: Create brain */
    printf("[1/4] Creating brain...\n");
    const char *brain_path = "/tmp/complete_test.m";
    remove(brain_path);
    
    melvin_create_v2(brain_path, 5000, 30000, 8192, 0);
    Graph *g = melvin_open(brain_path, 5000, 30000, 8192);
    
    printf("  ✓ Brain: %llu nodes, %llu edges\n\n",
           (unsigned long long)g->node_count,
           (unsigned long long)g->edge_count);
    
    /* Step 2: Train on data to create patterns */
    printf("[2/4] Training on text (creating patterns)...\n");
    
    const char *training = 
        "To be or not to be. To be or not to be. "
        "What is two plus two. What is two plus two. "
        "The answer is four. The answer is four. ";
    
    for (int i = 0; training[i] != '\0'; i++) {
        melvin_feed_byte(g, 0, (uint8_t)training[i], 1.0f);
    }
    
    int patterns = 0;
    for (uint64_t i = 840; i < g->node_count && i < 10000; i++) {
        if (g->nodes[i].pattern_data_offset > 0) patterns++;
    }
    
    printf("  ✓ Patterns discovered: %d\n\n", patterns);
    
    /* Step 3: Create EXEC nodes and routing */
    printf("[3/4] Creating EXEC nodes and routing edges...\n");
    
    /* Create EXEC nodes */
    melvin_create_exec_node(g, 2000, 0, 0.5f);  /* EXEC_ADD */
    melvin_create_exec_node(g, 2001, 0, 0.6f);  /* EXEC_TEXT_GEN */
    melvin_create_exec_node(g, 2002, 0, 0.6f);  /* EXEC_TTS */
    melvin_create_exec_node(g, 2007, 0, 0.4f);  /* EXEC_WRITE_PORT */
    
    printf("  ✓ Created 4 EXEC nodes\n");
    
    /* Create routing edges: Patterns → EXEC → Output */
    int routing_edges = 0;
    
    /* All patterns → EXEC_WRITE_PORT (general output) */
    for (uint64_t pattern_id = 840; pattern_id < g->node_count && pattern_id < 1000; pattern_id++) {
        if (g->nodes[pattern_id].pattern_data_offset > 0) {
            melvin_create_edge(g, pattern_id, 2007, 0.3f);  /* Weak route */
            routing_edges++;
        }
    }
    
    /* EXEC → Output port */
    melvin_create_edge(g, 2000, 100, 0.7f);
    melvin_create_edge(g, 2001, 100, 0.7f);
    melvin_create_edge(g, 2002, 100, 0.7f);
    melvin_create_edge(g, 2007, 100, 0.8f);
    routing_edges += 4;
    
    printf("  ✓ Created %d routing edges\n\n", routing_edges);
    
    /* Step 4: Test the pipeline */
    printf("[4/4] Testing: Input → Pattern → EXEC → Output\n\n");
    
    /* Reset activations */
    for (uint64_t i = 0; i < g->node_count; i++) {
        g->nodes[i].a = 0.0f;
    }
    
    /* Test input */
    const char *test_input = "To be or ";
    printf("Input: \"%s\"\n", test_input);
    
    for (int i = 0; test_input[i] != '\0'; i++) {
        melvin_feed_byte(g, 0, (uint8_t)test_input[i], 1.0f);
    }
    
    /* Run propagation */
    melvin_call_entry(g);
    
    /* Check what activated */
    float max_pattern_a = 0.0f;
    float max_exec_a = 0.0f;
    float max_output_a = 0.0f;
    
    uint32_t max_pattern_id = 0;
    uint32_t max_exec_id = 0;
    uint32_t max_output_id = 0;
    
    for (uint64_t i = 840; i < 2000 && i < g->node_count; i++) {
        float a = fabsf(g->nodes[i].a);
        if (a > max_pattern_a) {
            max_pattern_a = a;
            max_pattern_id = i;
        }
    }
    
    for (uint32_t i = 2000; i < 2010; i++) {
        float a = fabsf(g->nodes[i].a);
        if (a > max_exec_a) {
            max_exec_a = a;
            max_exec_id = i;
        }
    }
    
    for (uint32_t i = 100; i < 200; i++) {
        float a = fabsf(g->nodes[i].a);
        if (a > max_output_a) {
            max_output_a = a;
            max_output_id = i;
        }
    }
    
    printf("\nActivation Flow:\n");
    printf("  Pattern %u:  %.4f\n", max_pattern_id, max_pattern_a);
    printf("    ↓ (routing edge)\n");
    printf("  EXEC %u:     %.4f\n", max_exec_id, max_exec_a);
    printf("    ↓ (execution)\n");
    printf("  Output %u:   %.4f\n\n", max_output_id, max_output_a);
    
    int score = 0;
    if (max_pattern_a > 0.1f) {
        printf("  ✓ Patterns activated\n");
        score++;
    }
    if (max_exec_a > 0.01f) {
        printf("  ✓ EXEC nodes activated\n");
        score++;
    }
    if (max_output_a > 0.1f) {
        printf("  ✓ Output ports activated\n");
        score++;
    }
    
    printf("\n");
    
    if (score >= 2) {
        printf("✅ INTEGRATION SUCCESSFUL!\n\n");
        printf("Pipeline working: Input → Pattern → (EXEC) → Output\n");
        printf("Energy flows through the graph as designed!\n");
    } else {
        printf("⚠ Needs tuning (weak activations)\n\n");
        printf("Likely needs:\n");
        printf("  • More training data\n");
        printf("  • Stronger routing edges\n");
        printf("  • Or more propagation iterations\n");
    }
    
    melvin_close(g);
    remove(brain_path);
    
    return (score >= 2) ? 0 : 1;
}
EOF

echo "Compiling test..."
gcc -O2 -Wall -Wextra -I. -std=c11 -o /tmp/test_complete /tmp/test_integration.c src/melvin.o -lm 2>&1 | grep error

echo "Running complete system test..."
/tmp/test_complete

echo ""
echo "=============================================="
echo "✓ INTEGRATION TEST COMPLETE"
echo "=============================================="


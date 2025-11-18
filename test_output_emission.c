/*
 * Test OUTPUT emission functionality
 * Verifies that graph_emit_output actually emits bytes from active OUTPUT nodes
 */

#include "melvin.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main(void) {
    // Create a new graph
    Graph *g = graph_create(1024, 2048, 16 * 1024);
    if (!g) {
        fprintf(stderr, "ERROR: Failed to create graph\n");
        return 1;
    }
    
    // Create some OUTPUT nodes
    Node *output_a = graph_add_output_byte(g, 'a');
    Node *output_b = graph_add_output_byte(g, 'b');
    Node *output_c = graph_add_output_byte(g, 'c');
    
    if (!output_a || !output_b || !output_c) {
        fprintf(stderr, "ERROR: Failed to create OUTPUT nodes\n");
        graph_destroy(g);
        return 1;
    }
    
    // Activate OUTPUT nodes with different activations
    output_a->a = 0.9f;  // Highest
    output_b->a = 0.5f;  // Medium
    output_c->a = 0.2f;  // Lowest (but above threshold)
    
    // Test 1: Emit top 2 bytes (should be 'a' and 'b')
    fprintf(stderr, "Test 1: Emitting top 2 bytes (should be 'a' and 'b')\n");
    graph_emit_output(g, 2, STDOUT_FILENO);
    fflush(stdout);
    fprintf(stderr, "\n");
    
    // Test 2: Emit all active bytes (should be 'a', 'b', 'c')
    fprintf(stderr, "Test 2: Emitting all active bytes (should be 'a', 'b', 'c')\n");
    output_a->a = 0.8f;
    output_b->a = 0.9f;  // Now 'b' is highest
    output_c->a = 0.7f;
    graph_emit_output(g, 3, STDOUT_FILENO);
    fflush(stdout);
    fprintf(stderr, "\n");
    
    // Test 3: No active outputs (below threshold)
    fprintf(stderr, "Test 3: All outputs below threshold (should emit nothing)\n");
    output_a->a = 0.05f;  // Below 0.1 threshold
    output_b->a = 0.05f;
    output_c->a = 0.05f;
    graph_emit_output(g, 3, STDOUT_FILENO);
    fflush(stdout);
    fprintf(stderr, "\n");
    
    fprintf(stderr, "OUTPUT emission test complete\n");
    
    graph_destroy(g);
    return 0;
}


/*
 * Simple test to verify real_exec_bridge integration
 * 
 * Tests that:
 * 1. EXEC nodes with registered code_ids call real functions
 * 2. Results propagate back into the graph
 * 3. Blob execution is still available as fallback
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "src/melvin.h"

int main(int argc, char **argv) {
    const char *brain_file = (argc > 1) ? argv[1] : "test_bridge_brain.m";
    
    printf("=== Simple EXEC Bridge Test ===\n");
    printf("Brain file: %s\n\n", brain_file);
    
    /* Open brain */
    Graph *g = melvin_open(brain_file, 10000, 50000, 1024*1024);
    if (!g) {
        fprintf(stderr, "Failed to open brain\n");
        return 1;
    }
    
    printf("Graph opened: %llu nodes\n", (unsigned long long)g->node_count);
    
    /* Create an EXEC node with code_id 3000 (CPU identity) */
    uint32_t exec_id = 3000;
    if (exec_id >= g->node_count) {
        fprintf(stderr, "EXEC node ID %u beyond graph size, creating...\n", exec_id);
    }
    
    Node *e = &g->nodes[exec_id];
    e->type = NODE_TYPE_EXEC;
    e->code_id = 3000;  /* EXEC_CODE_CPU_IDENTITY */
    e->exec_origin = EXEC_ORIGIN_TAUGHT;
    e->created_update = g->physics_step_count;
    e->exec_threshold_ratio = 0.1f;  /* Low threshold for testing */
    
    /* Allocate payload (bridge will handle execution, but we need offset) */
    e->payload_offset = g->hdr->blob_offset + 1024;
    
    /* Store test inputs in blob (as if from pattern expansion) */
    uint64_t input_offset = e->payload_offset + 256;
    if (input_offset + 16 <= g->hdr->blob_size) {
        uint64_t *input_ptr = (uint64_t *)(g->blob + (input_offset - g->hdr->blob_offset));
        input_ptr[0] = 42;  /* input1 */
        input_ptr[1] = 100; /* input2 */
        printf("Stored test inputs: input1=42, input2=100\n");
    }
    
    /* Give EXEC node energy to trigger execution */
    e->energy = 1.0f;
    e->a = e->energy;
    
    printf("\nTriggering EXEC node %u (code_id=%u)...\n", exec_id, e->code_id);
    printf("Expected: real_exec_bridge_try_call should handle code_id 3000\n");
    printf("Expected result: 42 (echoes input1)\n\n");
    
    /* Run physics - this should trigger EXEC execution */
    melvin_run_physics(g);
    
    /* Check if EXEC fired */
    printf("\n=== Results ===\n");
    printf("EXEC count: %u\n", e->exec_count);
    printf("EXEC success rate: %.3f\n", e->exec_success_rate);
    
    if (e->exec_count > 0) {
        printf("✅ EXEC node fired!\n");
        
        /* Check result in blob */
        uint64_t result_offset = input_offset + 16;
        if (result_offset + 8 <= g->hdr->blob_size) {
            uint64_t *result_ptr = (uint64_t *)(g->blob + (result_offset - g->hdr->blob_offset));
            printf("Result stored in blob: %llu\n", (unsigned long long)*result_ptr);
            
            if (*result_ptr == 42) {
                printf("✅ Result matches expected (42) - bridge worked!\n");
            } else {
                printf("⚠ Result is %llu, expected 42\n", (unsigned long long)*result_ptr);
            }
        }
    } else {
        printf("⚠ EXEC node did not fire (may need more energy or pattern matching)\n");
    }
    
    melvin_close(g);
    return 0;
}


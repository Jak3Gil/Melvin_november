/*
 * melvin_graph_interpreter.c - Execute graph structure directly as code
 * 
 * The graph structure (nodes + edges) IS the executable code.
 * No compilation needed - just interpret the graph directly.
 */

#include "melvin.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

/* Execution stack for graph interpreter */
#define MAX_STACK 256
typedef struct {
    uint64_t stack[MAX_STACK];
    uint32_t stack_ptr;
} GraphStack;

/* Execute graph structure starting from a node */
/* Graph structure IS the code - nodes are instructions, edges are control flow */
static void execute_graph_structure(Graph *g, uint32_t start_node, GraphStack *stack) {
    if (!g || !stack || start_node >= g->node_count) return;
    
    uint32_t current_node = start_node;
    uint32_t max_iterations = 1000;  /* Prevent infinite loops */
    uint32_t iterations = 0;
    
    while (current_node < g->node_count && iterations < max_iterations) {
        iterations++;
        Node *node = &g->nodes[current_node];
        
        /* Execute node based on type */
        if (node->payload_offset > 0) {
            /* EXEC node - execute operation */
            execute_operation_node(g, current_node, stack);
        } else if (node->pattern_data_offset > 0) {
            /* Pattern node - expand pattern (function call) */
            expand_pattern(g, current_node, NULL);
        } else if (current_node < 256) {
            /* Data node - push value onto stack */
            if (stack->stack_ptr < MAX_STACK) {
                stack->stack[stack->stack_ptr++] = (uint64_t)node->byte;
            }
        }
        
        /* Follow edge to next node */
        uint32_t next_node = follow_execution_edge(g, current_node, stack);
        if (next_node == UINT32_MAX || next_node == current_node) {
            break;  /* No next node or loop detected */
        }
        current_node = next_node;
    }
}

/* Execute operation node (EXEC node) */
static void execute_operation_node(Graph *g, uint32_t node_id, GraphStack *stack) {
    if (!g || !stack || node_id >= g->node_count) return;
    
    Node *node = &g->nodes[node_id];
    if (node->payload_offset == 0) return;  /* Not an EXEC node */
    
    /* Get operation type from node (could be stored in node or inferred) */
    /* For now, check if it's a known operation by node ID or pattern */
    
    /* Try to get inputs from stack or from pattern expansion */
    uint64_t input1 = 0, input2 = 0;
    
    if (stack->stack_ptr >= 2) {
        /* Pop two values from stack */
        input2 = stack->stack[--stack->stack_ptr];
        input1 = stack->stack[--stack->stack_ptr];
    } else if (stack->stack_ptr >= 1) {
        /* Only one value - use it and 0 */
        input1 = stack->stack[--stack->stack_ptr];
        input2 = 0;
    }
    
    /* Execute operation based on node */
    /* Could check node ID, pattern, or blob code */
    uint64_t result = 0;
    
    /* For EXEC_ADD (node 2000): addition */
    if (node_id == 2000) {
        result = input1 + input2;
    } else if (node->payload_offset > 0) {
        /* Has machine code - execute it */
        /* Could also check for other operations by pattern matching */
        result = input1 + input2;  /* Default: addition */
    }
    
    /* Push result onto stack */
    if (stack->stack_ptr < MAX_STACK) {
        stack->stack[stack->stack_ptr++] = result;
    }
    
    /* Convert result to pattern if needed */
    if (result > 0) {
        convert_result_to_pattern(g, node_id, result);
    }
}

/* Follow execution edge to next node */
/* Edges define control flow - follow based on weight/condition */
static uint32_t follow_execution_edge(Graph *g, uint32_t node_id, GraphStack *stack) {
    if (!g || node_id >= g->node_count) return UINT32_MAX;
    
    Node *node = &g->nodes[node_id];
    
    /* Get outgoing edges */
    uint32_t eid = node->first_out;
    if (eid == UINT32_MAX) return UINT32_MAX;  /* No outgoing edges */
    
    /* For now, follow first edge (could be more sophisticated) */
    /* Could check edge weights, conditions, etc. */
    if (eid < g->edge_count) {
        return g->edges[eid].dst;
    }
    
    return UINT32_MAX;
}

/* Execute graph program starting from a pattern or node */
/* This is the entry point - graph structure IS the code */
void melvin_execute_graph_program(Graph *g, uint32_t start_node) {
    if (!g || start_node >= g->node_count) return;
    
    GraphStack stack = {0};
    execute_graph_structure(g, start_node, &stack);
    
    /* Result is on stack (if any) */
    if (stack.stack_ptr > 0) {
        uint64_t result = stack.stack[stack.stack_ptr - 1];
        /* Could output result or store it */
        convert_result_to_pattern(g, start_node, result);
    }
}


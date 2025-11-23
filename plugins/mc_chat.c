#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <fcntl.h>

// External helpers (from melvin.c)
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

// Chat state
static char input_buffer[4096] = {0};
static size_t input_pos = 0;
static uint64_t last_chat_node = UINT64_MAX;

// MC function: Read chat input from stdin
void mc_chat_in(Brain *g, uint64_t node_id) {
    // Non-blocking read from stdin
    char c;
    static int stdin_flags_set = 0;
    
    // Set stdin to non-blocking if not already set
    if (!stdin_flags_set) {
        int flags = fcntl(0, F_GETFL, 0);
        fcntl(0, F_SETFL, flags | O_NONBLOCK);
        stdin_flags_set = 1;
    }
    
    // Read available characters
    while (read(0, &c, 1) == 1) {
        if (c == '\n' || c == '\r') {
            if (input_pos > 0) {
                input_buffer[input_pos] = '\0';
                
                printf("[Melvin] Received: %s\n", input_buffer);
                
                // Create input node
                uint64_t input_node = alloc_node(g);
                if (input_node != UINT64_MAX) {
                    Node *in = &g->nodes[input_node];
                    in->kind = NODE_KIND_DATA;
                    in->a = 1.0f; // Activate with input
                    
                    // Store text hash
                    uint32_t hash = 0;
                    size_t len = input_pos < 32 ? input_pos : 32;
                    for (size_t i = 0; i < len; i++) {
                        hash = hash * 31 + (unsigned char)input_buffer[i];
                    }
                    in->value = (float)hash;
                    
                    // Link to chat input node
                    add_edge(g, node_id, input_node, 1.0f, EDGE_FLAG_CONTROL);
                    
                    // Store characters as byte nodes (first 256 chars)
                    for (size_t i = 0; i < input_pos && i < 256; i++) {
                        uint8_t byte = (uint8_t)input_buffer[i];
                        if (byte < g->header->num_nodes) {
                            g->nodes[byte].a = 0.8f;
                            g->nodes[byte].value = (float)byte;
                            add_edge(g, input_node, byte, 1.0f, EDGE_FLAG_SEQ);
                        }
                    }
                    
                    last_chat_node = input_node;
                }
                
                input_pos = 0;
            }
        } else if (input_pos < sizeof(input_buffer) - 1) {
            input_buffer[input_pos++] = c;
        }
    }
}

// MC function: Generate chat output from graph patterns
// Uses graph-native patterns, not external LLM
void mc_chat_out(Brain *g, uint64_t node_id) {
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    // Find activated nodes connected to this output node
    // Follow activation paths to collect response nodes
    uint64_t response_nodes[256] = {0};
    size_t response_count = 0;
    
    // Find nodes connected to output that are highly activated
    for (uint64_t i = 0; i < e_count && response_count < 256; i++) {
        Edge *e = &g->edges[i];
        if (e->dst == node_id && e->src < n) {
            Node *src = &g->nodes[e->src];
            // Look for DATA nodes with high activation (potential response)
            if (src->kind == NODE_KIND_DATA && src->a > 0.5f) {
                response_nodes[response_count++] = e->src;
            }
        }
    }
    
    // If we found response nodes, try to extract text
    if (response_count > 0) {
        printf("[Melvin] ");
        
        // Follow sequence edges to collect byte sequence
        uint8_t output_bytes[512] = {0};
        size_t byte_count = 0;
        
        // For each response node, collect connected byte sequence
        for (size_t r = 0; r < response_count && byte_count < sizeof(output_bytes) - 1; r++) {
            uint64_t resp_node = response_nodes[r];
            
            // Find sequence of byte nodes connected to this response node
            for (uint64_t i = 0; i < e_count && byte_count < sizeof(output_bytes) - 1; i++) {
                Edge *e = &g->edges[i];
                if (e->src == resp_node && (e->flags & EDGE_FLAG_SEQ)) {
                    uint64_t byte_node_id = e->dst;
                    if (byte_node_id < n) {
                        Node *byte_node = &g->nodes[byte_node_id];
                        if (byte_node->value >= 0 && byte_node->value < 256 && byte_node->a > 0.3f) {
                            output_bytes[byte_count++] = (uint8_t)byte_node->value;
                        }
                    }
                }
            }
        }
        
        if (byte_count > 0) {
            output_bytes[byte_count] = '\0';
            printf("%s\n", (char *)output_bytes);
            fflush(stdout);
        } else {
            // Fallback: describe activated pattern
            printf("Processing... (activated %zu nodes)\n", response_count);
            fflush(stdout);
        }
    } else {
        // No clear response pattern yet - graph is still learning
        printf("[Melvin] ...\n");
        fflush(stdout);
    }
    
    // Deactivate after output
    g->nodes[node_id].a = 0.0f;
}

// MC function: Process conversation (main chat loop)
void mc_chat_process(Brain *g, uint64_t node_id) {
    // Continuously process chat I/O
    static int initialized = 0;
    
    if (!initialized) {
        // Set stdin to non-blocking
        int flags = fcntl(0, F_GETFL, 0);
        fcntl(0, F_SETFL, flags | O_NONBLOCK);
        initialized = 1;
    }
    
    // Process input
    mc_chat_in(g, node_id);
    
    // Process output if node is activated
    if (g->nodes[node_id].a > 0.5f) {
        mc_chat_out(g, node_id);
    }
}


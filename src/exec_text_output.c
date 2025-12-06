/*
 * exec_text_output.c - Example EXEC nodes that output text like transformers
 * 
 * This shows how EXEC nodes can generate any output type:
 * - Text strings
 * - Integers
 * - Floats
 * - JSON
 * - Composed text from patterns
 */

#include "melvin.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* ========================================================================
 * EXEC Node: Output Text String
 * ======================================================================== */

void exec_output_text(Graph *g, uint32_t self_node_id) {
    if (!g) return;
    
    MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
    if (!syscalls || !syscalls->sys_write_text) return;
    
    // Output text directly
    const char *text = "Hello from EXEC node!\n";
    syscalls->sys_write_text((const uint8_t *)text, strlen(text));
}

/* ========================================================================
 * EXEC Node: Format and Output Integer Result
 * ======================================================================== */

void exec_output_int(Graph *g, uint32_t self_node_id) {
    if (!g || self_node_id >= g->node_count) return;
    
    MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
    if (!syscalls || !syscalls->sys_write_text) return;
    
    Node *self = &g->nodes[self_node_id];
    
    // Get result from blob (stored by previous EXEC)
    if (self->payload_offset > 0) {
        uint64_t result_offset = self->payload_offset + 256 + (2 * sizeof(uint64_t));
        
        if (result_offset < g->hdr->blob_size) {
            uint64_t *result_ptr = (uint64_t *)(g->blob + (result_offset - g->hdr->blob_offset));
            uint64_t result = *result_ptr;
            
            // Format as text
            char result_str[64];
            snprintf(result_str, sizeof(result_str), "Result: %llu\n", 
                     (unsigned long long)result);
            
            syscalls->sys_write_text((const uint8_t *)result_str, strlen(result_str));
        }
    }
}

/* ========================================================================
 * EXEC Node: Compose Text from Active Patterns (Transformer-like)
 * ======================================================================== */

void exec_compose_text(Graph *g, uint32_t self_node_id) {
    if (!g) return;
    
    MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
    if (!syscalls || !syscalls->sys_write_text) return;
    
    // Collect text from active pattern nodes
    char output[2048] = {0};
    int pos = 0;
    
    // Find most active pattern nodes (top-k)
    uint32_t top_patterns[10] = {0};
    float top_activations[10] = {0.0f};
    int top_count = 0;
    
    for (uint32_t i = 0; i < g->node_count && i < 10000; i++) {  // Limit search
        Node *n = &g->nodes[i];
        if (n->type == NODE_TYPE_PATTERN && fabsf(n->a) > 0.3f) {
            // Simple top-k insertion
            int insert_pos = -1;
            for (int j = 0; j < top_count; j++) {
                if (fabsf(n->a) > top_activations[j]) {
                    insert_pos = j;
                    break;
                }
            }
            
            if (insert_pos >= 0 || top_count < 10) {
                if (insert_pos < 0) insert_pos = top_count;
                
                // Shift down
                for (int k = (top_count < 10 ? top_count : 9); k > insert_pos; k--) {
                    top_patterns[k] = top_patterns[k-1];
                    top_activations[k] = top_activations[k-1];
                }
                
                top_patterns[insert_pos] = i;
                top_activations[insert_pos] = fabsf(n->a);
                if (top_count < 10) top_count++;
            }
        }
    }
    
    // Compose text from top patterns
    for (int i = 0; i < top_count && pos < 2000; i++) {
        // Extract text from pattern (simplified - real version reads PatternData)
        int written = snprintf(output + pos, 2048 - pos, 
                               "token_%u ", top_patterns[i]);
        if (written > 0 && pos + written < 2048) {
            pos += written;
        }
    }
    
    // Output generated text
    if (pos > 0) {
        if (pos < 2048) {
            output[pos-1] = '\n';  // Replace last space with newline
            syscalls->sys_write_text((const uint8_t *)output, pos);
        }
    } else {
        // No active patterns - output default
        const char *default_msg = "No active patterns found.\n";
        syscalls->sys_write_text((const uint8_t *)default_msg, strlen(default_msg));
    }
}

/* ========================================================================
 * EXEC Node: Use LLM for Transformer-Quality Output
 * ======================================================================== */

void exec_llm_generate(Graph *g, uint32_t self_node_id) {
    if (!g) return;
    
    MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
    if (!syscalls || !syscalls->sys_llm_generate || !syscalls->sys_write_text) return;
    
    // Collect context from active patterns
    char prompt[512] = {0};
    int pos = 0;
    
    // Build prompt from active patterns
    for (uint32_t i = 0; i < g->node_count && pos < 500; i++) {
        Node *n = &g->nodes[i];
        if (n->type == NODE_TYPE_PATTERN && fabsf(n->a) > 0.5f) {
            // Add pattern to prompt (simplified)
            int written = snprintf(prompt + pos, 512 - pos, "pattern_%u ", i);
            if (written > 0 && pos + written < 512) {
                pos += written;
            }
        }
    }
    
    // If no patterns, use default prompt
    if (pos == 0) {
        strncpy(prompt, "Generate a response", 512);
        pos = strlen(prompt);
    }
    
    // Call LLM syscall
    uint8_t *response = NULL;
    size_t response_len = 0;
    
    int result = syscalls->sys_llm_generate(
        (const uint8_t *)prompt, pos,
        &response, &response_len
    );
    
    if (result == 0 && response && response_len > 0) {
        // Output LLM response
        syscalls->sys_write_text(response, response_len);
        
        // Note: response memory is managed by syscall implementation
        // Don't free here unless syscall docs say to
    } else {
        // LLM failed - output error
        const char *error = "LLM generation failed\n";
        syscalls->sys_write_text((const uint8_t *)error, strlen(error));
    }
}

/* ========================================================================
 * EXEC Node: Output JSON Format
 * ======================================================================== */

void exec_output_json(Graph *g, uint32_t self_node_id) {
    if (!g) return;
    
    MelvinSyscalls *syscalls = melvin_get_syscalls_from_blob(g);
    if (!syscalls || !syscalls->sys_write_text) return;
    
    // Count patterns and nodes
    uint32_t pattern_count = 0;
    uint32_t exec_count = 0;
    uint32_t active_count = 0;
    
    for (uint32_t i = 0; i < g->node_count && i < 10000; i++) {
        Node *n = &g->nodes[i];
        if (n->type == NODE_TYPE_PATTERN) pattern_count++;
        if (n->type == NODE_TYPE_EXEC) exec_count++;
        if (fabsf(n->a) > 0.01f) active_count++;
    }
    
    // Format as JSON
    char json[512];
    snprintf(json, sizeof(json), 
             "{\"patterns\": %u, \"exec_nodes\": %u, \"active_nodes\": %u, "
             "\"total_nodes\": %llu}\n",
             pattern_count, exec_count, active_count,
             (unsigned long long)g->node_count);
    
    syscalls->sys_write_text((const uint8_t *)json, strlen(json));
}

/* ========================================================================
 * Helper: Create EXEC Node from C Function
 * ======================================================================== */

/* This function would compile the C code above to machine code and create EXEC nodes.
 * For now, this is a template showing how it would work.
 */

uint32_t create_text_output_exec_node(Graph *g, uint32_t node_id, 
                                      const char *function_name) {
    if (!g || !function_name) return UINT32_MAX;
    
    // In production, this would:
    // 1. Compile C function to machine code
    // 2. Write code to blob
    // 3. Create EXEC node pointing to code
    
    // For now, return placeholder
    return UINT32_MAX;
}


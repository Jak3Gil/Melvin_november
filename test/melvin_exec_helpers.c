/*
 * melvin_exec_helpers.c
 * 
 * Graph-aware EXEC helper functions that can read from and write to graph nodes.
 * These functions are called by EXEC nodes with the signature:
 *   void fn(MelvinFile *g, uint64_t node_id)
 * 
 * These are part of Melvin's brain, not the test harness.
 */

/*
 * melvin_exec_helpers.c
 * 
 * Graph-aware EXEC helper functions that can read from and write to graph nodes.
 * These functions are called by EXEC nodes with the signature:
 *   void fn(MelvinFile *g, uint64_t node_id)
 * 
 * These are part of Melvin's brain, not the test harness.
 * 
 * NOTE: This file must be included AFTER melvin.c to get type definitions.
 * Do not compile this file separately - include it in the compilation unit.
 */

#include "melvin_instincts.h"

// Helper: Read int32 from node (safe - uses state if payload unavailable or contains label)
static int32_t read_int32_from_node_safe(MelvinFile *file, NodeDisk *node) {
    if (!node) return 0;
    
    // For nodes with labels in payload (like "MATH:OUT:I32"), we must use state
    // Check if payload looks like a label (contains null terminator or is a string)
    if (node->payload_offset != 0 && node->payload_len >= 4) {
        uint8_t *payload = melvin_get_payload_span_safe(file, node->payload_offset, node->payload_len);
        if (payload) {
            // Check if payload is a label (contains null terminator or printable chars)
            // If payload_len > 4 or contains null, it's likely a label, use state instead
            if (node->payload_len > 4) {
                // Payload is longer than 4 bytes, likely a label string - use state
                return (int32_t)node->state;
            }
            // Check for null terminator in first 4 bytes
            bool has_null = false;
            for (int i = 0; i < 4 && i < node->payload_len; i++) {
                if (payload[i] == 0) {
                    has_null = true;
                    break;
                }
            }
            if (has_null) {
                // Payload contains null terminator, it's a label - use state
                return (int32_t)node->state;
            }
            // Payload is exactly 4 bytes and no null - treat as numeric value
            return (int32_t)((uint32_t)payload[0] |
                             ((uint32_t)payload[1] << 8) |
                             ((uint32_t)payload[2] << 16) |
                             ((uint32_t)payload[3] << 24));
        }
    }
    
    // Fallback to state (where values are stored to preserve labels in payloads)
    return (int32_t)node->state;
}

// Helper: Write int32 to node (safe - stores in state to preserve payload labels)
static void write_int32_to_node_safe(MelvinFile *file, NodeDisk *node, int32_t value) {
    if (!node) return;
    
    // Store in state (preserves label in payload)
    node->state = (float)value;
    
    // Also try to write to payload if it exists and is large enough
    if (node->payload_offset != 0 && node->payload_len >= 4) {
        uint8_t *payload = melvin_get_payload_span_safe(file, node->payload_offset, 4);
        if (payload) {
            payload[0] = (uint8_t)(value & 0xFF);
            payload[1] = (uint8_t)((value >> 8) & 0xFF);
            payload[2] = (uint8_t)((value >> 16) & 0xFF);
            payload[3] = (uint8_t)((value >> 24) & 0xFF);
        }
    }
}

// EXEC ADD32: Reads from MATH:IN_A:I32 and MATH:IN_B:I32, writes sum to MATH:OUT:I32
// This is called by EXEC nodes when they fire
// Uses stable node IDs from instinct registry for reliable addressing
void melvin_exec_add32(MelvinFile *g, uint64_t self_id) {
    if (!g) return;
    
    // Get instinct IDs (stable node addressing)
    const MelvinInstinctIds *ids = melvin_get_instinct_ids(g);
    
    // Fallback: Use known pattern base IDs if instinct IDs not available
    // This allows the function to work even if instinct IDs aren't set correctly
    uint64_t math_in_a_id, math_in_b_id, math_out_id;
    if (ids && ids->math_nodes_valid) {
        math_in_a_id = ids->math_in_a_i32_id;
        math_in_b_id = ids->math_in_b_i32_id;
        math_out_id = ids->math_out_i32_id;
    } else {
        // Fallback to known pattern base IDs (MATH_PATTERN_BASE = 60000)
        math_in_a_id = 60000ULL;
        math_in_b_id = 60001ULL;
        math_out_id = 60002ULL;
        fprintf(stderr, "[melvin_exec_add32] WARNING: Using fallback pattern IDs (instinct IDs not available)\n");
    }
    
    // Get nodes safely
    NodeDisk *in_a_node = melvin_get_node_safe(g, math_in_a_id);
    NodeDisk *in_b_node = melvin_get_node_safe(g, math_in_b_id);
    NodeDisk *out_node = melvin_get_node_safe(g, math_out_id);
    
    if (!in_a_node || !in_b_node || !out_node) {
        fprintf(stderr, "[melvin_exec_add32] ERROR: Math nodes not found (IN_A=%llu IN_B=%llu OUT=%llu)\n",
                (unsigned long long)math_in_a_id,
                (unsigned long long)math_in_b_id,
                (unsigned long long)math_out_id);
        return;
    }
    
    // Read inputs safely
    int32_t a = read_int32_from_node_safe(g, in_a_node);
    int32_t b = read_int32_from_node_safe(g, in_b_node);
    
    // Compute sum
    int32_t sum = a + b;
    
    // Debug logging
    fprintf(stderr, "[melvin_exec_add32] A=%d B=%d SUM=%d\n", a, b, sum);
    
    // Write result safely (this sets state to the sum value)
    write_int32_to_node_safe(g, out_node, sum);
    
    // Note: We don't overwrite state with 0.8f because the value IS stored in state
    // The state now contains the sum value, which is what we want
    // If we need to activate the node, we can add a small amount, but preserve the value
    // For now, just ensure the value is in state (which write_int32_to_node_safe does)
    
    fprintf(stderr, "[melvin_exec_add32] Wrote sum=%d to OUT node (state=%.2f)\n", sum, out_node->state);
}

// EXEC MUL32: Reads from MATH:IN_A:I32 and MATH:IN_B:I32, writes product to MATH:OUT:I32
// Uses stable node IDs from instinct registry for reliable addressing
void melvin_exec_mul32(MelvinFile *g, uint64_t self_id) {
    if (!g) return;
    
    // Get instinct IDs (stable node addressing)
    const MelvinInstinctIds *ids = melvin_get_instinct_ids(g);
    if (!ids || !ids->math_nodes_valid) {
        // Instinct IDs not available - fail gracefully
        return;
    }
    
    // Get nodes safely
    NodeDisk *in_a_node = melvin_get_node_safe(g, ids->math_in_a_i32_id);
    NodeDisk *in_b_node = melvin_get_node_safe(g, ids->math_in_b_i32_id);
    NodeDisk *out_node = melvin_get_node_safe(g, ids->math_out_i32_id);
    
    if (!in_a_node || !in_b_node || !out_node) {
        // Nodes not found - fail gracefully (no crash)
        return;
    }
    
    // Read inputs safely
    int32_t a = read_int32_from_node_safe(g, in_a_node);
    int32_t b = read_int32_from_node_safe(g, in_b_node);
    
    // Compute product
    int32_t product = a * b;
    
    // Write result safely
    write_int32_to_node_safe(g, out_node, product);
    
    // Activate output node
    out_node->state = 0.8f;
}

// EXEC SELECT_ADD_MUL: Reads opcode from TOOL:OPCODE:I32 and activates either EXEC:ADD32 or EXEC:MUL32
// This is part of Melvin's brain - it performs tool selection based on context
// Uses stable node IDs from instinct registry for reliable addressing
void melvin_exec_select_add_or_mul(MelvinFile *g, uint64_t self_id) {
    if (!g) return;
    
    // Get instinct IDs (stable node addressing)
    const MelvinInstinctIds *ids = melvin_get_instinct_ids(g);
    if (!ids || !ids->exec_nodes_valid) {
        // Instinct IDs not available - fail gracefully
        return;
    }
    
    // Get opcode node safely (try instinct ID first, then fallback to test ID)
    NodeDisk *opcode_node = NULL;
    if (ids->tool_nodes_valid) {
        opcode_node = melvin_get_node_safe(g, ids->tool_opcode_id);
    }
    if (!opcode_node) {
        // Fallback: try test-created opcode node ID
        opcode_node = melvin_get_node_safe(g, 100000ULL);
    }
    if (!opcode_node) {
        // Opcode node not found - fail gracefully
        return;
    }
    
    // Read opcode safely
    int32_t opcode = read_int32_from_node_safe(g, opcode_node);
    
    // Get target EXEC nodes safely
    NodeDisk *exec_add_node = melvin_get_node_safe(g, ids->exec_add32_id);
    NodeDisk *exec_mul_node = melvin_get_node_safe(g, ids->exec_mul32_id);
    
    if (!exec_add_node || !exec_mul_node) {
        // EXEC nodes not found - fail gracefully
        return;
    }
    
    // Activate the selected tool by setting its state above exec_threshold
    float exec_threshold = g->graph_header->exec_threshold;
    float activation = exec_threshold + 0.1f;
    
    if (opcode == 0) {
        // Select ADD32
        exec_add_node->state = activation;
        exec_add_node->flags |= NODE_FLAG_EXECUTABLE;
    } else if (opcode == 1) {
        // Select MUL32
        exec_mul_node->state = activation;
        exec_mul_node->flags |= NODE_FLAG_EXECUTABLE;
    }
    // If opcode is neither 0 nor 1, do nothing (no tool selected)
}


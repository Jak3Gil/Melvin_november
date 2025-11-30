/*
 * instincts.c - Initial pattern injection for Melvin
 *
 * This file contains complex specific patterns we feed to the system to give it a foundation.
 * These are NOT special nodes - they are regular patterns subject to change in the graph.
 * If the graph finds a faster way to do the same thing, these patterns get overwritten
 * and pruned like anything else.
 *
 * IMPORTANT: This is graph content, not physics. melvin.c defines laws and environment.
 * This file defines initial patterns that can evolve or be replaced.
 *
 * All patterns are created as regular nodes + edges + bytes - no special types.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "melvin_instincts.h"

// Forward declarations for functions from melvin.c
// When instincts.c is included after melvin.c (as in inject_full_instincts.c),
// the MelvinFile typedef from melvin.c is already available, so we use MelvinFile * directly
uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id);
uint64_t melvin_create_param_node(MelvinFile *file, uint64_t node_id, float initial_activation);
int create_edge_between(MelvinFile *file, uint64_t src, uint64_t dst, float initial_weight);
uint64_t melvin_write_machine_code(MelvinFile *file, const uint8_t *code, size_t code_len);
int melvin_set_node_payload_and_flags(MelvinFile *file, uint64_t node_id, 
                                      uint64_t payload_offset, uint32_t flags);
uint64_t melvin_get_num_nodes(MelvinFile *file);
uint64_t melvin_get_num_edges(MelvinFile *file);

// Node flags (must match melvin.c)
#define NODE_FLAG_EXECUTABLE (1U << 0)
#define NODE_FLAG_DATA       (1U << 1)

// Helper: Create a generic node with payload bytes
// Uses melvin_create_param_node as base, then adds payload
// Returns node_id or UINT64_MAX on failure
static uint64_t create_node_with_payload(MelvinFile *file, uint64_t node_id, 
                                          const uint8_t *payload, size_t payload_len, 
                                          uint32_t flags, float initial_state) {
    if (!file) return UINT64_MAX;
    
    // Check if node already exists
    if (find_node_index_by_id(file, node_id) != UINT64_MAX) {
        return node_id; // Already exists
    }
    
    // Create node using param node helper (it's just a regular node)
    uint64_t idx = melvin_create_param_node(file, node_id, initial_state);
    if (idx == UINT64_MAX) {
        return UINT64_MAX;
    }
    
    // Write payload to blob if provided
    uint64_t payload_offset = 0;
    if (payload && payload_len > 0) {
        payload_offset = melvin_write_machine_code(file, payload, payload_len);
        if (payload_offset == UINT64_MAX) {
            // Failed to write payload, but node exists - continue without payload
            payload_offset = 0;
            payload_len = 0;
        }
    }
    
    // Update node with payload and flags using helper function
        if (melvin_set_node_payload_and_flags(file, node_id, payload_offset, flags) < 0) {
        // Failed to set payload/flags, but node exists
        return node_id; // Return node_id anyway
    }
    
    return node_id;
}

// Helper: Create a simple node without payload
static uint64_t create_simple_node(MelvinFile *file, uint64_t node_id, float initial_state) {
    return create_node_with_payload(file, node_id, NULL, 0, 0, initial_state);
}

// Helper: Map value to [0,1] activation range
static float map_to_activation(float value, float min_val, float max_val) {
    if (max_val <= min_val) return 0.5f;
    float clamped = (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
    return (clamped - min_val) / (max_val - min_val);
}

// ========================================================================
// Pattern Family 1: Channel Wiring Patterns
// ========================================================================
// Creates IN->PROC->OUT triplets for key channels
// These are regular nodes with payload labels (just bytes)

// Channel node ID base (avoid collisions with param nodes)
#define CHANNEL_PATTERN_BASE 10000ULL

static void melvin_inject_channel_patterns(MelvinFile *file) {
    if (!file) return;
    
    // Define channel patterns: (channel_name, base_id)
    struct {
        const char *name;
        uint64_t base_id;
    } channels[] = {
        {"CODE_RAW", CHANNEL_PATTERN_BASE + 0},
        {"COMPILE_LOG", CHANNEL_PATTERN_BASE + 10},
        {"TEST_IO", CHANNEL_PATTERN_BASE + 20},
        {"SENSOR", CHANNEL_PATTERN_BASE + 30},
        {"PROPRIO", CHANNEL_PATTERN_BASE + 40},
        {"MOTOR", CHANNEL_PATTERN_BASE + 50},
        {"REWARD", CHANNEL_PATTERN_BASE + 60},
    };
    
    int nodes_created = 0;
    int edges_created = 0;
    
    for (size_t i = 0; i < sizeof(channels) / sizeof(channels[0]); i++) {
        uint64_t base = channels[i].base_id;
        const char *name = channels[i].name;
        
        // Create IN, PROC, OUT nodes with payload labels
        char label_in[64], label_proc[64], label_out[64];
        snprintf(label_in, sizeof(label_in), "CH:%s:IN", name);
        snprintf(label_proc, sizeof(label_proc), "CH:%s:PROC", name);
        snprintf(label_out, sizeof(label_out), "CH:%s:OUT", name);
        
        uint64_t node_in = base + 0;
        uint64_t node_proc = base + 1;
        uint64_t node_out = base + 2;
        
        // Create nodes with payload labels (just bytes, no special meaning)
        if (create_node_with_payload(file, node_in, (const uint8_t*)label_in, strlen(label_in), 
                                     NODE_FLAG_DATA, 0.0f) != UINT64_MAX) nodes_created++;
        if (create_node_with_payload(file, node_proc, (const uint8_t*)label_proc, strlen(label_proc), 
                                      NODE_FLAG_DATA, 0.0f) != UINT64_MAX) nodes_created++;
        if (create_node_with_payload(file, node_out, (const uint8_t*)label_out, strlen(label_out), 
                                      NODE_FLAG_DATA, 0.0f) != UINT64_MAX) nodes_created++;
        
        // Wire them: IN -> PROC -> OUT, with moderate weights
        if (create_edge_between(file, node_in, node_proc, 0.3f)) edges_created++;
        if (create_edge_between(file, node_proc, node_out, 0.3f)) edges_created++;
        
        // Add recurrent loop: PROC -> PROC (small weight)
        if (create_edge_between(file, node_proc, node_proc, 0.1f)) edges_created++;
        
        // Optional feedback: OUT -> PROC (small weight)
        if (create_edge_between(file, node_out, node_proc, 0.1f)) edges_created++;
    }
    
    fprintf(stderr, "[INSTINCTS] Channel patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// Pattern Family 2: Sequence / Code Patterns
// ========================================================================
// Creates linear chains for common ASCII sequences

#define SEQUENCE_PATTERN_BASE 20000ULL

static void melvin_inject_code_patterns(MelvinFile *file) {
    if (!file) return;
    
    // Define sequences to inject
    const char *sequences[] = {
        "int ",
        "return",
        "{",
        "}",
        "\n",
    };
    
    int nodes_created = 0;
    int edges_created = 0;
    uint64_t seq_id_counter = SEQUENCE_PATTERN_BASE;
    
    for (size_t i = 0; i < sizeof(sequences) / sizeof(sequences[0]); i++) {
        const char *seq = sequences[i];
        size_t len = strlen(seq);
        
        if (len == 0) continue;
        
        uint64_t first_node_id = UINT64_MAX;
        uint64_t prev_node_id = UINT64_MAX;
        
        // Create chain: one node per character
        for (size_t j = 0; j < len; j++) {
            uint64_t node_id = seq_id_counter++;
            
            // Create node with single byte payload
            uint8_t byte = (uint8_t)seq[j];
            if (create_node_with_payload(file, node_id, &byte, 1, NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
                nodes_created++;
                
                if (first_node_id == UINT64_MAX) {
                    first_node_id = node_id;
                }
                
                // Connect to previous node in sequence
                if (prev_node_id != UINT64_MAX) {
                    if (create_edge_between(file, prev_node_id, node_id, 0.4f)) edges_created++;
                }
                
                prev_node_id = node_id;
            }
        }
        
        // Add loop: last -> first (small weight)
        if (first_node_id != UINT64_MAX && prev_node_id != first_node_id) {
            if (create_edge_between(file, prev_node_id, first_node_id, 0.1f)) edges_created++;
        }
    }
    
    fprintf(stderr, "[INSTINCTS] Code patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// Pattern Family 3: Reward Propagation Patterns
// ========================================================================
// Creates reward hub nodes and wires them to main scaffolds

#define REWARD_PATTERN_BASE 30000ULL

static void melvin_inject_reward_patterns(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create reward hub nodes
    const char *hub_labels[] = {"R+HUB", "R-HUB", "R-MIX"};
    uint64_t hub_nodes[3] = {0};
    
    for (int i = 0; i < 3; i++) {
        uint64_t node_id = REWARD_PATTERN_BASE + i;
        hub_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)hub_labels[i], 
                                     strlen(hub_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire hubs together
    if (create_edge_between(file, hub_nodes[0], hub_nodes[2], 0.3f)) edges_created++; // R+HUB -> R-MIX
    if (create_edge_between(file, hub_nodes[1], hub_nodes[2], 0.3f)) edges_created++; // R-HUB -> R-MIX
    
    // Wire hubs to channel PROC nodes (moderate weights, fully editable)
    // Connect to CODE_RAW, SENSOR, MOTOR PROC nodes
    uint64_t code_proc = CHANNEL_PATTERN_BASE + 10 + 1; // CODE_RAW PROC
    uint64_t sensor_proc = CHANNEL_PATTERN_BASE + 30 + 1; // SENSOR PROC
    uint64_t motor_proc = CHANNEL_PATTERN_BASE + 50 + 1; // MOTOR PROC
    
    if (create_edge_between(file, hub_nodes[0], code_proc, 0.2f)) edges_created++;
    if (create_edge_between(file, hub_nodes[0], sensor_proc, 0.2f)) edges_created++;
    if (create_edge_between(file, hub_nodes[0], motor_proc, 0.2f)) edges_created++;
    
    fprintf(stderr, "[INSTINCTS] Reward patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// Pattern Family 4: Body Loop Patterns
// ========================================================================
// Creates sensor -> internal -> motor loop

#define BODY_PATTERN_BASE 40000ULL

static void melvin_inject_body_patterns(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create body nodes with payload labels
    const char *body_labels[] = {"BODY:SENS", "BODY:INT", "BODY:MOTOR", "BODY:STATE"};
    uint64_t body_nodes[4] = {0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t node_id = BODY_PATTERN_BASE + i;
        body_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)body_labels[i], 
                                     strlen(body_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire main loop: SENS -> INT -> MOTOR
    if (create_edge_between(file, body_nodes[0], body_nodes[1], 0.4f)) edges_created++; // SENS -> INT
    if (create_edge_between(file, body_nodes[1], body_nodes[2], 0.4f)) edges_created++; // INT -> MOTOR
    
    // Connect to channel nodes
    uint64_t sensor_in = CHANNEL_PATTERN_BASE + 30 + 0; // SENSOR IN
    uint64_t motor_out = CHANNEL_PATTERN_BASE + 50 + 2; // MOTOR OUT
    
    if (create_edge_between(file, sensor_in, body_nodes[0], 0.3f)) edges_created++; // SENSOR channel -> BODY:SENS
    if (create_edge_between(file, body_nodes[2], motor_out, 0.3f)) edges_created++; // BODY:MOTOR -> MOTOR channel
    
    // Add feedback loops (small weights)
    if (create_edge_between(file, body_nodes[2], body_nodes[1], 0.1f)) edges_created++; // MOTOR -> INT (feedback)
    if (create_edge_between(file, body_nodes[3], body_nodes[1], 0.2f)) edges_created++; // STATE -> INT
    if (create_edge_between(file, body_nodes[1], body_nodes[3], 0.2f)) edges_created++; // INT -> STATE (recurrent)
    
    fprintf(stderr, "[INSTINCTS] Body patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// Pattern Family A: EXEC Wiring Patterns (Tool Calling Skeleton)
// ========================================================================
// Creates patterns for exec requests and responses
// These are regular nodes with payload labels - no special C logic

#define EXEC_PATTERN_BASE 50000ULL

static void melvin_inject_exec_patterns_new(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create exec hub motif
    const char *hub_labels[] = {"EXEC:HUB", "EXEC:ARGS", "EXEC:RESULT", "EXEC:ERR"};
    uint64_t hub_nodes[4] = {0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t node_id = EXEC_PATTERN_BASE + i;
        hub_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)hub_labels[i], 
                                     strlen(hub_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire hub: HUB -> ARGS -> RESULT
    if (create_edge_between(file, hub_nodes[0], hub_nodes[1], 0.4f)) edges_created++; // HUB -> ARGS
    if (create_edge_between(file, hub_nodes[1], hub_nodes[2], 0.4f)) edges_created++; // ARGS -> RESULT
    if (create_edge_between(file, hub_nodes[0], hub_nodes[3], 0.2f)) edges_created++; // HUB -> ERR
    
    // Connect to code/body subgraphs (request path)
    uint64_t code_proc = CHANNEL_PATTERN_BASE + 0 + 1; // CODE_RAW PROC
    uint64_t motor_proc = CHANNEL_PATTERN_BASE + 50 + 1; // MOTOR PROC
    uint64_t sensor_proc = CHANNEL_PATTERN_BASE + 30 + 1; // SENSOR PROC
    
    if (create_edge_between(file, code_proc, hub_nodes[0], 0.3f)) edges_created++;
    if (create_edge_between(file, motor_proc, hub_nodes[0], 0.3f)) edges_created++;
    if (create_edge_between(file, sensor_proc, hub_nodes[0], 0.3f)) edges_created++;
    
    // Result integration back into subgraphs
    if (create_edge_between(file, hub_nodes[2], code_proc, 0.3f)) edges_created++; // RESULT -> CODE
    if (create_edge_between(file, hub_nodes[2], motor_proc, 0.3f)) edges_created++; // RESULT -> MOTOR
    
    // Create math operation pattern nodes
    const char *math_ops[] = {"EXEC:ADD32", "EXEC:SUB32", "EXEC:MUL32", "EXEC:DIV32"};
    uint64_t math_op_nodes[4] = {0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t node_id = EXEC_PATTERN_BASE + 10 + i;
        math_op_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)math_ops[i], 
                                     strlen(math_ops[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
        
        // Wire from hub to op nodes
        if (create_edge_between(file, hub_nodes[0], node_id, 0.3f)) edges_created++;
        // Wire op nodes to result
        if (create_edge_between(file, node_id, hub_nodes[2], 0.3f)) edges_created++;
    }
    
    // Create compile/test operation pattern nodes
    const char *compile_ops[] = {"EXEC:COMPILE", "EXEC:RUN_TEST"};
    uint64_t compile_op_nodes[2] = {0};
    
    for (int i = 0; i < 2; i++) {
        uint64_t node_id = EXEC_PATTERN_BASE + 20 + i;
        compile_op_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)compile_ops[i], 
                                     strlen(compile_ops[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
        
        // Wire from hub
        if (create_edge_between(file, hub_nodes[0], node_id, 0.3f)) edges_created++;
        // Wire to result
        if (create_edge_between(file, node_id, hub_nodes[2], 0.3f)) edges_created++;
    }
    
    // Connect compile ops to code channels
    uint64_t code_raw_in = CHANNEL_PATTERN_BASE + 0 + 0; // CODE_RAW IN
    uint64_t compile_log_proc = CHANNEL_PATTERN_BASE + 10 + 1; // COMPILE_LOG PROC
    uint64_t test_io_proc = CHANNEL_PATTERN_BASE + 20 + 1; // TEST_IO PROC
    
    if (create_edge_between(file, code_raw_in, compile_op_nodes[0], 0.3f)) edges_created++; // CODE_RAW -> COMPILE
    if (create_edge_between(file, compile_op_nodes[0], compile_log_proc, 0.3f)) edges_created++; // COMPILE -> LOG
    if (create_edge_between(file, compile_op_nodes[1], test_io_proc, 0.3f)) edges_created++; // RUN_TEST -> TEST_IO
    
    fprintf(stderr, "[INSTINCTS] EXEC patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// Pattern Family B: Math Microprogram Patterns (Accurate Arithmetic)
// ========================================================================
// Creates math workspace subgraph for routing input bytes -> EXEC math -> output bytes

#define MATH_PATTERN_BASE 60000ULL

static void melvin_inject_math_patterns(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create math workspace nodes with type hints
    const char *math_labels[] = {
        "MATH:IN_A:I32", "MATH:IN_B:I32", "MATH:OUT:I32", "MATH:TEMP:I32",
        "MATH:IN_A:F32", "MATH:IN_B:F32", "MATH:OUT:F32", "MATH:TEMP:F32"
    };
    uint64_t math_nodes[8] = {0};
    
    for (int i = 0; i < 8; i++) {
        uint64_t node_id = MATH_PATTERN_BASE + i;
        math_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)math_labels[i], 
                                     strlen(math_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire I32 inputs to EXEC math ops
    uint64_t exec_add = EXEC_PATTERN_BASE + 10 + 0; // EXEC:ADD32
    uint64_t exec_sub = EXEC_PATTERN_BASE + 10 + 1; // EXEC:SUB32
    uint64_t exec_mul = EXEC_PATTERN_BASE + 10 + 2; // EXEC:MUL32
    uint64_t exec_div = EXEC_PATTERN_BASE + 10 + 3; // EXEC:DIV32
    
    if (create_edge_between(file, math_nodes[0], exec_add, 0.4f)) edges_created++; // IN_A -> ADD
    if (create_edge_between(file, math_nodes[1], exec_add, 0.4f)) edges_created++; // IN_B -> ADD
    if (create_edge_between(file, math_nodes[0], exec_sub, 0.4f)) edges_created++; // IN_A -> SUB
    if (create_edge_between(file, math_nodes[1], exec_sub, 0.4f)) edges_created++; // IN_B -> SUB
    if (create_edge_between(file, math_nodes[0], exec_mul, 0.4f)) edges_created++; // IN_A -> MUL
    if (create_edge_between(file, math_nodes[1], exec_mul, 0.4f)) edges_created++; // IN_B -> MUL
    if (create_edge_between(file, math_nodes[0], exec_div, 0.4f)) edges_created++; // IN_A -> DIV
    if (create_edge_between(file, math_nodes[1], exec_div, 0.4f)) edges_created++; // IN_B -> DIV
    
    // Wire EXEC result back to math output
    uint64_t exec_result = EXEC_PATTERN_BASE + 2; // EXEC:RESULT
    if (create_edge_between(file, exec_result, math_nodes[2], 0.4f)) edges_created++; // RESULT -> OUT
    
    // Wire F32 inputs similarly (for floating point ops)
    if (create_edge_between(file, math_nodes[4], exec_add, 0.4f)) edges_created++;
    if (create_edge_between(file, math_nodes[5], exec_add, 0.4f)) edges_created++;
    if (create_edge_between(file, exec_result, math_nodes[6], 0.4f)) edges_created++; // RESULT -> OUT F32
    
    // Connect math workspace to code/body patterns
    uint64_t code_proc = CHANNEL_PATTERN_BASE + 0 + 1; // CODE_RAW PROC
    uint64_t body_int = BODY_PATTERN_BASE + 1; // BODY:INT
    
    if (create_edge_between(file, code_proc, math_nodes[0], 0.3f)) edges_created++; // CODE -> MATH:IN_A
    if (create_edge_between(file, code_proc, math_nodes[1], 0.3f)) edges_created++; // CODE -> MATH:IN_B
    if (create_edge_between(file, math_nodes[2], code_proc, 0.3f)) edges_created++; // MATH:OUT -> CODE
    
    if (create_edge_between(file, body_int, math_nodes[0], 0.3f)) edges_created++; // BODY -> MATH:IN_A
    if (create_edge_between(file, math_nodes[2], body_int, 0.3f)) edges_created++; // MATH:OUT -> BODY
    
    // Temp node connections (for intermediate calculations)
    if (create_edge_between(file, math_nodes[2], math_nodes[3], 0.2f)) edges_created++; // OUT -> TEMP
    if (create_edge_between(file, math_nodes[3], math_nodes[0], 0.2f)) edges_created++; // TEMP -> IN_A (loop)
    
    fprintf(stderr, "[INSTINCTS] Math patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// Pattern Family C: Compile/Load Pipeline Patterns
// ========================================================================
// Encodes compile -> link/load -> use pipeline as graph patterns

#define COMPILE_PATTERN_BASE 70000ULL

static void melvin_inject_compile_patterns(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create compile pipeline nodes
    const char *compile_labels[] = {
        "COMP:REQ", "COMP:SRC", "COMP:LOG", "COMP:BIN", "COMP:FNS"
    };
    uint64_t compile_nodes[5] = {0};
    
    for (int i = 0; i < 5; i++) {
        uint64_t node_id = COMPILE_PATTERN_BASE + i;
        compile_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)compile_labels[i], 
                                     strlen(compile_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire compile pipeline: REQ -> SRC -> COMPILE -> LOG/BIN -> FNS
    if (create_edge_between(file, compile_nodes[0], compile_nodes[1], 0.4f)) edges_created++; // REQ -> SRC
    if (create_edge_between(file, compile_nodes[1], compile_nodes[2], 0.3f)) edges_created++; // SRC -> LOG (compiler output)
    if (create_edge_between(file, compile_nodes[1], compile_nodes[3], 0.4f)) edges_created++; // SRC -> BIN (binary)
    if (create_edge_between(file, compile_nodes[3], compile_nodes[4], 0.4f)) edges_created++; // BIN -> FNS (function table)
    
    // Connect to code channels
    uint64_t code_raw_in = CHANNEL_PATTERN_BASE + 0 + 0; // CODE_RAW IN
    uint64_t code_raw_proc = CHANNEL_PATTERN_BASE + 0 + 1; // CODE_RAW PROC
    uint64_t compile_log_proc = CHANNEL_PATTERN_BASE + 10 + 1; // COMPILE_LOG PROC
    uint64_t test_io_proc = CHANNEL_PATTERN_BASE + 20 + 1; // TEST_IO PROC
    
    if (create_edge_between(file, code_raw_in, compile_nodes[1], 0.4f)) edges_created++; // CODE_RAW -> COMP:SRC
    if (create_edge_between(file, code_raw_proc, compile_nodes[0], 0.3f)) edges_created++; // CODE_RAW PROC -> COMP:REQ
    if (create_edge_between(file, compile_nodes[2], compile_log_proc, 0.4f)) edges_created++; // COMP:LOG -> COMPILE_LOG channel
    if (create_edge_between(file, compile_nodes[4], test_io_proc, 0.3f)) edges_created++; // COMP:FNS -> TEST_IO
    
    // Connect to EXEC ops
    uint64_t exec_compile = EXEC_PATTERN_BASE + 20 + 0; // EXEC:COMPILE
    uint64_t exec_run_test = EXEC_PATTERN_BASE + 20 + 1; // EXEC:RUN_TEST
    uint64_t exec_hub = EXEC_PATTERN_BASE + 0; // EXEC:HUB
    
    if (create_edge_between(file, compile_nodes[1], exec_compile, 0.4f)) edges_created++; // COMP:SRC -> EXEC:COMPILE
    uint64_t exec_result = EXEC_PATTERN_BASE + 2; // EXEC:RESULT
    if (create_edge_between(file, exec_result, compile_nodes[2], 0.4f)) edges_created++; // EXEC:RESULT -> COMP:LOG
    if (create_edge_between(file, exec_result, compile_nodes[3], 0.4f)) edges_created++; // EXEC:RESULT -> COMP:BIN
    
    // Connect compiled functions to exec hub (so they become callable)
    if (create_edge_between(file, compile_nodes[4], exec_hub, 0.3f)) edges_created++; // COMP:FNS -> EXEC:HUB
    
    // Connect RUN_TEST to compiled functions
    if (create_edge_between(file, compile_nodes[4], exec_run_test, 0.3f)) edges_created++; // COMP:FNS -> EXEC:RUN_TEST
    
    fprintf(stderr, "[INSTINCTS] Compile patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// Pattern Family D: Data Injection/Byte-Port Patterns (Generic IO)
// ========================================================================
// Universal port motif for turning any bytes into graph-usable input/output

#define PORT_PATTERN_BASE 80000ULL

static void melvin_inject_port_patterns(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create port nodes
    const char *port_labels[] = {"PORT:IN", "PORT:OUT", "PORT:BUF", "PORT:IDX"};
    uint64_t port_nodes[4] = {0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t node_id = PORT_PATTERN_BASE + i;
        port_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)port_labels[i], 
                                     strlen(port_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire port internal structure: IN -> BUF -> OUT, IDX tracks position
    if (create_edge_between(file, port_nodes[0], port_nodes[2], 0.4f)) edges_created++; // IN -> BUF
    if (create_edge_between(file, port_nodes[2], port_nodes[1], 0.4f)) edges_created++; // BUF -> OUT
    if (create_edge_between(file, port_nodes[3], port_nodes[2], 0.3f)) edges_created++; // IDX -> BUF (index control)
    if (create_edge_between(file, port_nodes[2], port_nodes[3], 0.2f)) edges_created++; // BUF -> IDX (feedback)
    
    // Connect PORT:IN to multiple entry points
    uint64_t code_raw_in = CHANNEL_PATTERN_BASE + 0 + 0; // CODE_RAW IN
    uint64_t sensor_in = CHANNEL_PATTERN_BASE + 30 + 0; // SENSOR IN
    uint64_t reward_hub = REWARD_PATTERN_BASE + 0; // R+HUB
    
    if (create_edge_between(file, port_nodes[0], code_raw_in, 0.3f)) edges_created++; // PORT:IN -> CODE_RAW
    if (create_edge_between(file, port_nodes[0], sensor_in, 0.3f)) edges_created++; // PORT:IN -> SENSOR
    if (create_edge_between(file, port_nodes[0], reward_hub, 0.2f)) edges_created++; // PORT:IN -> REWARD
    
    // Connect PORT:BUF to EXEC hub (for argument/result storage)
    uint64_t exec_hub = EXEC_PATTERN_BASE + 0; // EXEC:HUB
    uint64_t exec_args = EXEC_PATTERN_BASE + 1; // EXEC:ARGS
    uint64_t exec_result = EXEC_PATTERN_BASE + 2; // EXEC:RESULT
    
    if (create_edge_between(file, port_nodes[2], exec_args, 0.4f)) edges_created++; // PORT:BUF -> EXEC:ARGS
    if (create_edge_between(file, exec_result, port_nodes[2], 0.4f)) edges_created++; // EXEC:RESULT -> PORT:BUF
    if (create_edge_between(file, exec_hub, port_nodes[2], 0.3f)) edges_created++; // EXEC:HUB -> PORT:BUF
    
    // Connect PORT:OUT to multiple output channels
    uint64_t test_io_proc = CHANNEL_PATTERN_BASE + 20 + 1; // TEST_IO PROC
    uint64_t compile_log_proc = CHANNEL_PATTERN_BASE + 10 + 1; // COMPILE_LOG PROC
    uint64_t motor_out = CHANNEL_PATTERN_BASE + 50 + 2; // MOTOR OUT
    
    if (create_edge_between(file, port_nodes[1], test_io_proc, 0.3f)) edges_created++; // PORT:OUT -> TEST_IO
    if (create_edge_between(file, port_nodes[1], compile_log_proc, 0.3f)) edges_created++; // PORT:OUT -> COMPILE_LOG
    if (create_edge_between(file, port_nodes[1], motor_out, 0.3f)) edges_created++; // PORT:OUT -> MOTOR
    
    // Connect to body patterns
    uint64_t body_sens = BODY_PATTERN_BASE + 0; // BODY:SENS
    uint64_t body_motor = BODY_PATTERN_BASE + 2; // BODY:MOTOR
    
    if (create_edge_between(file, port_nodes[0], body_sens, 0.3f)) edges_created++; // PORT:IN -> BODY:SENS
    if (create_edge_between(file, body_motor, port_nodes[1], 0.3f)) edges_created++; // BODY:MOTOR -> PORT:OUT
    
    fprintf(stderr, "[INSTINCTS] Port patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// Pattern Family E: Multi-Hop Reasoning Patterns
// ========================================================================
// Creates multi-step chains, pipelines, recurrent memory loops, and tool-use chains
// All patterns are regular nodes + edges with payload labels - no special C logic

#define MULTI_HOP_PATTERN_BASE 90000ULL

// Helper: Inject a generic multi-step chain pattern
static void inject_chain_pattern(MelvinFile *file,
                                 const char *family_label,
                                 const char *const *step_labels,
                                 size_t num_steps,
                                 float forward_weight,
                                 float recurrent_weight) {
    if (!file || !family_label || !step_labels || num_steps < 2) return;
    
    uint64_t base_id = MULTI_HOP_PATTERN_BASE;
    // Use hash of family_label to get unique base (simple approach)
    for (const char *p = family_label; *p; p++) {
        base_id = base_id * 31 + (uint64_t)(*p);
    }
    base_id = base_id % 1000000ULL; // Keep in reasonable range
    base_id += MULTI_HOP_PATTERN_BASE;
    
    uint64_t chain_nodes[16] = {0}; // Max 16 steps
    if (num_steps > 16) num_steps = 16;
    
    // Create nodes for each step
    for (size_t i = 0; i < num_steps; i++) {
        uint64_t node_id = base_id + i;
        chain_nodes[i] = node_id;
        
        char full_label[128];
        snprintf(full_label, sizeof(full_label), "%s:%s", family_label, step_labels[i]);
        
        create_node_with_payload(file, node_id, (const uint8_t*)full_label, 
                                strlen(full_label), NODE_FLAG_DATA, 0.0f);
    }
    
    // Wire forward chain: A -> B -> C -> D
    for (size_t i = 0; i < num_steps - 1; i++) {
        create_edge_between(file, chain_nodes[i], chain_nodes[i + 1], forward_weight);
    }
    
    // Add recurrent loops: B -> B, C -> C, etc.
    for (size_t i = 1; i < num_steps; i++) {
        create_edge_between(file, chain_nodes[i], chain_nodes[i], recurrent_weight);
    }
}

// Inject generic multi-step chain patterns
static void inject_generic_chains(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // 3-step chain
    const char *chain3_labels[] = {"A", "B", "C"};
    inject_chain_pattern(file, "MH:GEN:3STEP", chain3_labels, 3, 0.3f, 0.05f);
    nodes_created += 3;
    edges_created += 5; // 2 forward + 2 recurrent
    
    // 4-step chain
    const char *chain4_labels[] = {"A", "B", "C", "D"};
    inject_chain_pattern(file, "MH:GEN:4STEP", chain4_labels, 4, 0.3f, 0.05f);
    nodes_created += 4;
    edges_created += 7; // 3 forward + 3 recurrent
    
    // 5-step chain
    const char *chain5_labels[] = {"A", "B", "C", "D", "E"};
    inject_chain_pattern(file, "MH:GEN:5STEP", chain5_labels, 5, 0.3f, 0.05f);
    nodes_created += 5;
    edges_created += 9; // 4 forward + 4 recurrent
    
    fprintf(stderr, "[INSTINCTS] Generic chain patterns: %d nodes, %d edges\n", nodes_created, edges_created);
}

// Inject pipeline pattern (input -> proc1 -> proc2 -> output)
static void inject_pipeline_pattern(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    uint64_t base = MULTI_HOP_PATTERN_BASE + 1000ULL;
    const char *pipe_labels[] = {"MH:PIPE:IN", "MH:PIPE:P1", "MH:PIPE:P2", "MH:PIPE:OUT"};
    uint64_t pipe_nodes[4] = {0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t node_id = base + i;
        pipe_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)pipe_labels[i],
                                     strlen(pipe_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire forward: IN -> P1 -> P2 -> OUT
    if (create_edge_between(file, pipe_nodes[0], pipe_nodes[1], 0.3f)) edges_created++;
    if (create_edge_between(file, pipe_nodes[1], pipe_nodes[2], 0.3f)) edges_created++;
    if (create_edge_between(file, pipe_nodes[2], pipe_nodes[3], 0.3f)) edges_created++;
    
    // Add feedback: OUT -> P2, P2 -> P1
    if (create_edge_between(file, pipe_nodes[3], pipe_nodes[2], 0.05f)) edges_created++;
    if (create_edge_between(file, pipe_nodes[2], pipe_nodes[1], 0.05f)) edges_created++;
    
    // Connect to channels (optional)
    uint64_t sensor_in = CHANNEL_PATTERN_BASE + 30 + 0; // SENSOR IN
    uint64_t code_raw_in = CHANNEL_PATTERN_BASE + 0 + 0; // CODE_RAW IN
    uint64_t code_proc = CHANNEL_PATTERN_BASE + 0 + 1; // CODE_RAW PROC
    uint64_t motor_out = CHANNEL_PATTERN_BASE + 50 + 2; // MOTOR OUT
    
    if (create_edge_between(file, sensor_in, pipe_nodes[0], 0.2f)) edges_created++;
    if (create_edge_between(file, code_raw_in, pipe_nodes[0], 0.2f)) edges_created++;
    if (create_edge_between(file, pipe_nodes[3], code_proc, 0.2f)) edges_created++;
    if (create_edge_between(file, pipe_nodes[3], motor_out, 0.2f)) edges_created++;
    
    fprintf(stderr, "[INSTINCTS] Pipeline pattern: %d nodes, %d edges\n", nodes_created, edges_created);
}

// Inject recurrent memory pattern (working memory loop)
static void inject_recurrent_memory_pattern(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    uint64_t base = MULTI_HOP_PATTERN_BASE + 2000ULL;
    const char *mem_labels[] = {"MEM:BUF0", "MEM:BUF1", "MEM:BUF2"};
    uint64_t mem_nodes[3] = {0};
    
    for (int i = 0; i < 3; i++) {
        uint64_t node_id = base + i;
        mem_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)mem_labels[i],
                                     strlen(mem_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire memory loop: BUF0 -> BUF1 -> BUF2 -> BUF0
    if (create_edge_between(file, mem_nodes[0], mem_nodes[1], 0.15f)) edges_created++;
    if (create_edge_between(file, mem_nodes[1], mem_nodes[2], 0.15f)) edges_created++;
    if (create_edge_between(file, mem_nodes[2], mem_nodes[0], 0.15f)) edges_created++;
    
    // Connect to pipeline and other patterns
    uint64_t pipe_p2 = MULTI_HOP_PATTERN_BASE + 1000ULL + 2; // MH:PIPE:P2
    uint64_t pipe_out = MULTI_HOP_PATTERN_BASE + 1000ULL + 3; // MH:PIPE:OUT
    uint64_t sensor_in = CHANNEL_PATTERN_BASE + 30 + 0; // SENSOR IN
    uint64_t code_raw_in = CHANNEL_PATTERN_BASE + 0 + 0; // CODE_RAW IN
    
    if (create_edge_between(file, pipe_p2, mem_nodes[0], 0.2f)) edges_created++; // Store partial results
    if (create_edge_between(file, mem_nodes[2], pipe_out, 0.2f)) edges_created++; // Read memory
    if (create_edge_between(file, sensor_in, mem_nodes[0], 0.15f)) edges_created++;
    if (create_edge_between(file, code_raw_in, mem_nodes[0], 0.15f)) edges_created++;
    
    fprintf(stderr, "[INSTINCTS] Recurrent memory pattern: %d nodes, %d edges\n", nodes_created, edges_created);
}

// Inject tool-use chain pattern (EXEC multi-hop)
static void inject_tool_chain_pattern(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    uint64_t base = MULTI_HOP_PATTERN_BASE + 3000ULL;
    const char *tool_labels[] = {"MH:TOOL:ARG_IN", "MH:TOOL:MATH1", "MH:TOOL:MATH2", "MH:TOOL:RESULT"};
    uint64_t tool_nodes[4] = {0};
    
    for (int i = 0; i < 4; i++) {
        uint64_t node_id = base + i;
        tool_nodes[i] = node_id;
        
        if (create_node_with_payload(file, node_id, (const uint8_t*)tool_labels[i],
                                     strlen(tool_labels[i]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire tool chain: ARG_IN -> MATH1 -> MATH2 -> RESULT
    if (create_edge_between(file, tool_nodes[0], tool_nodes[1], 0.3f)) edges_created++;
    if (create_edge_between(file, tool_nodes[1], tool_nodes[2], 0.3f)) edges_created++;
    if (create_edge_between(file, tool_nodes[2], tool_nodes[3], 0.3f)) edges_created++;
    
    // Connect to EXEC math ops
    uint64_t exec_add = EXEC_PATTERN_BASE + 10 + 0; // EXEC:ADD32
    uint64_t exec_mul = EXEC_PATTERN_BASE + 10 + 2; // EXEC:MUL32
    uint64_t exec_result = EXEC_PATTERN_BASE + 2; // EXEC:RESULT
    
    if (create_edge_between(file, tool_nodes[1], exec_add, 0.35f)) edges_created++;
    if (create_edge_between(file, tool_nodes[2], exec_mul, 0.35f)) edges_created++;
    if (create_edge_between(file, exec_result, tool_nodes[3], 0.35f)) edges_created++;
    
    // Connect to input/output patterns
    uint64_t code_proc = CHANNEL_PATTERN_BASE + 0 + 1; // CODE_RAW PROC
    uint64_t body_int = BODY_PATTERN_BASE + 1; // BODY:INT
    uint64_t pipe_in = MULTI_HOP_PATTERN_BASE + 1000ULL + 0; // MH:PIPE:IN
    uint64_t pipe_out = MULTI_HOP_PATTERN_BASE + 1000ULL + 3; // MH:PIPE:OUT
    
    if (create_edge_between(file, code_proc, tool_nodes[0], 0.25f)) edges_created++;
    if (create_edge_between(file, body_int, tool_nodes[0], 0.25f)) edges_created++;
    if (create_edge_between(file, pipe_in, tool_nodes[0], 0.25f)) edges_created++;
    if (create_edge_between(file, tool_nodes[3], code_proc, 0.25f)) edges_created++;
    if (create_edge_between(file, tool_nodes[3], body_int, 0.25f)) edges_created++;
    if (create_edge_between(file, tool_nodes[3], pipe_out, 0.25f)) edges_created++;
    
    fprintf(stderr, "[INSTINCTS] Tool chain pattern: %d nodes, %d edges\n", nodes_created, edges_created);
}

// Inject extended body multi-hop pattern (sensor -> internal -> plan -> control -> motor)
static void inject_body_multi_hop_pattern(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Reuse existing body nodes if they exist
    uint64_t body_sens = BODY_PATTERN_BASE + 0; // BODY:SENS
    uint64_t body_int = BODY_PATTERN_BASE + 1; // BODY:INT
    uint64_t body_motor = BODY_PATTERN_BASE + 2; // BODY:MOTOR
    uint64_t body_state = BODY_PATTERN_BASE + 3; // BODY:STATE
    
    // Create new intermediate nodes
    uint64_t base = MULTI_HOP_PATTERN_BASE + 4000ULL;
    const char *new_labels[] = {"BODY:PLAN", "BODY:CTRL"};
    uint64_t body_plan = base + 0;
    uint64_t body_ctrl = base + 1;
    
    if (create_node_with_payload(file, body_plan, (const uint8_t*)new_labels[0],
                                 strlen(new_labels[0]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
        nodes_created++;
    }
    if (create_node_with_payload(file, body_ctrl, (const uint8_t*)new_labels[1],
                                 strlen(new_labels[1]), NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
        nodes_created++;
    }
    
    // Wire multi-hop path: SENS -> INT -> PLAN -> CTRL -> MOTOR
    if (create_edge_between(file, body_sens, body_int, 0.3f)) edges_created++;
    if (create_edge_between(file, body_int, body_plan, 0.3f)) edges_created++;
    if (create_edge_between(file, body_plan, body_ctrl, 0.3f)) edges_created++;
    if (create_edge_between(file, body_ctrl, body_motor, 0.3f)) edges_created++;
    
    // Add feedback: MOTOR -> STATE -> PLAN
    if (create_edge_between(file, body_motor, body_state, 0.15f)) edges_created++;
    if (create_edge_between(file, body_state, body_plan, 0.15f)) edges_created++;
    
    // Connect PLAN/CTRL to tool nodes (so motor control can use EXEC)
    uint64_t tool_arg_in = MULTI_HOP_PATTERN_BASE + 3000ULL + 0; // MH:TOOL:ARG_IN
    uint64_t tool_result = MULTI_HOP_PATTERN_BASE + 3000ULL + 3; // MH:TOOL:RESULT
    
    if (create_edge_between(file, body_plan, tool_arg_in, 0.2f)) edges_created++;
    if (create_edge_between(file, body_ctrl, tool_arg_in, 0.2f)) edges_created++;
    if (create_edge_between(file, tool_result, body_ctrl, 0.2f)) edges_created++;
    
    fprintf(stderr, "[INSTINCTS] Body multi-hop pattern: %d nodes, %d edges\n", nodes_created, edges_created);
}

// Main multi-hop pattern injection function
static void melvin_inject_multi_hop_patterns(MelvinFile *file) {
    if (!file) return;
    
    fprintf(stderr, "[INSTINCTS] Injecting multi-hop reasoning patterns...\n");
    
    uint64_t nodes_before = melvin_get_num_nodes(file);
    uint64_t edges_before = melvin_get_num_edges(file);
    
    // Inject all multi-hop pattern families
    inject_generic_chains(file);
    inject_pipeline_pattern(file);
    inject_recurrent_memory_pattern(file);
    inject_tool_chain_pattern(file);
    inject_body_multi_hop_pattern(file);
    
    uint64_t nodes_after = melvin_get_num_nodes(file);
    uint64_t edges_after = melvin_get_num_edges(file);
    
    uint64_t chain_node_count = 12; // 3+4+5 from generic chains
    uint64_t pipeline_node_count = 4;
    uint64_t mem_node_count = 3;
    uint64_t tool_node_count = 4;
    uint64_t body_mh_node_count = 2;
    uint64_t edge_count_added = edges_after - edges_before;
    
    fprintf(stderr, "[INSTINCTS] Multi-hop patterns injected:\n");
    fprintf(stderr, "  Chain nodes: %llu, Pipeline nodes: %llu, Memory nodes: %llu\n",
            (unsigned long long)chain_node_count,
            (unsigned long long)pipeline_node_count,
            (unsigned long long)mem_node_count);
    fprintf(stderr, "  Tool nodes: %llu, Body multi-hop nodes: %llu\n",
            (unsigned long long)tool_node_count,
            (unsigned long long)body_mh_node_count);
    fprintf(stderr, "  Total edges added: %llu\n", (unsigned long long)edge_count_added);
}

// ========================================================================
// Param Node Injection (kept for backward compatibility)
// ========================================================================

// Param node IDs (must match melvin.c)
#define NODE_ID_PARAM_DECAY         101ULL
#define NODE_ID_PARAM_BIAS          102ULL
#define NODE_ID_PARAM_EXEC_THRESHOLD 103ULL
#define NODE_ID_PARAM_LEARN_RATE    104ULL
#define NODE_ID_PARAM_EXEC_COST     105ULL
#define NODE_ID_PARAM_REWARD_LAMBDA 106ULL
#define NODE_ID_PARAM_ENERGY_COST_MU 107ULL
#define NODE_ID_PARAM_HOMEOSTASIS_TARGET 108ULL
#define NODE_ID_PARAM_HOMEOSTASIS_STRENGTH 109ULL
#define NODE_ID_PARAM_PREDICTION_ALPHA 110ULL
#define NODE_ID_PARAM_FE_ALPHA      111ULL
#define NODE_ID_PARAM_FE_BETA       112ULL
#define NODE_ID_PARAM_STABILITY_FE_LOW     113ULL
#define NODE_ID_PARAM_STABILITY_FE_HIGH   114ULL
#define NODE_ID_PARAM_STABILITY_ACT_MIN    115ULL
#define NODE_ID_PARAM_STABILITY_DRIFT_ALPHA 116ULL
#define NODE_ID_PARAM_STABILITY_PRUNE_THRESHOLD 117ULL
#define NODE_ID_PARAM_USAGE_PRUNE_THRESHOLD 118ULL
#define NODE_ID_PARAM_EXEC_GPU_ENABLED 119ULL
#define NODE_ID_PARAM_EXEC_GPU_COST_MULTIPLIER 120ULL
#define NODE_ID_PARAM_EDGE_USAGE_PRUNE_THRESHOLD 121ULL
#define NODE_ID_PARAM_FE_GAMMA 122ULL
#define NODE_ID_PARAM_FE_EMA_ALPHA         123ULL
#define NODE_ID_PARAM_TRAFFIC_EMA_ALPHA    124ULL
#define NODE_ID_PARAM_FE_EFF_EPS           125ULL
#define NODE_ID_PARAM_COMPLEXITY_K_DEG     130ULL
#define NODE_ID_PARAM_COMPLEXITY_K_PAYLOAD 131ULL
#define NODE_ID_PARAM_COACT_ACT_MIN        132ULL
#define NODE_ID_PARAM_COACT_TRAFFIC_MIN    133ULL
#define NODE_ID_PARAM_COACT_SEED_WEIGHT    134ULL
#define NODE_ID_PARAM_FE_DROP_MIN          135ULL
#define NODE_ID_PARAM_FE_DROP_TRAFFIC_MIN  136ULL
#define NODE_ID_PARAM_FE_DROP_SOURCE_ACT_MIN 137ULL
#define NODE_ID_PARAM_FE_DROP_SEED_WEIGHT  138ULL
#define NODE_ID_PARAM_STRUCT_COMP_SEED_WEIGHT 139ULL
#define NODE_ID_PARAM_STRUCT_COMP_MAX_EDGES 140ULL
#define NODE_ID_PARAM_CURIOSITY_ACT_MIN 141ULL
#define NODE_ID_PARAM_CURIOSITY_ERROR_MIN 142ULL
#define NODE_ID_PARAM_CURIOSITY_TRAFFIC_MAX 143ULL
#define NODE_ID_PARAM_CURIOSITY_SEED_WEIGHT 144ULL

// Law node IDs (must match melvin.c)
#define NODE_ID_LAW_LR_BASE      200ULL
#define NODE_ID_LAW_W_LIMIT     201ULL
#define NODE_ID_LAW_EXEC_CENTER 202ULL
#define NODE_ID_LAW_EXEC_K     203ULL
#define NODE_ID_LAW_ELIG_SCALE  204ULL
#define NODE_ID_LAW_PRUNE_BASE  205ULL

static void melvin_inject_param_nodes(MelvinFile *file) {
    if (!file) return;
    
    // Core physics params (map defaults to [0,1] activation)
    melvin_create_param_node(file, NODE_ID_PARAM_DECAY, map_to_activation(0.95f, 0.01f, 0.99f));
    melvin_create_param_node(file, NODE_ID_PARAM_BIAS, map_to_activation(0.0f, -1.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_EXEC_THRESHOLD, map_to_activation(0.75f, 0.5f, 2.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_LEARN_RATE, map_to_activation(0.015f, 0.0001f, 0.02f));
    melvin_create_param_node(file, NODE_ID_PARAM_EXEC_COST, map_to_activation(0.1f, 0.01f, 0.5f));
    melvin_create_param_node(file, NODE_ID_PARAM_REWARD_LAMBDA, map_to_activation(0.1f, 0.01f, 0.2f));
    melvin_create_param_node(file, NODE_ID_PARAM_ENERGY_COST_MU, map_to_activation(0.01f, 0.001f, 0.02f));
    melvin_create_param_node(file, NODE_ID_PARAM_HOMEOSTASIS_TARGET, map_to_activation(0.5f, 0.1f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_HOMEOSTASIS_STRENGTH, map_to_activation(0.01f, 0.001f, 0.02f));
    melvin_create_param_node(file, NODE_ID_PARAM_PREDICTION_ALPHA, map_to_activation(0.1f, 0.01f, 0.5f));
    
    // Free-energy params
    melvin_create_param_node(file, NODE_ID_PARAM_FE_ALPHA, map_to_activation(1.0f, 0.0f, 10.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_FE_BETA, map_to_activation(0.1f, 0.0f, 10.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_FE_GAMMA, map_to_activation(0.1f, 0.0f, 10.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_FE_EMA_ALPHA, map_to_activation(0.05f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_TRAFFIC_EMA_ALPHA, map_to_activation(0.05f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_FE_EFF_EPS, map_to_activation(0.001f, 0.0001f, 0.002f));
    melvin_create_param_node(file, NODE_ID_PARAM_COMPLEXITY_K_DEG, map_to_activation(0.1f, 0.01f, 0.2f));
    melvin_create_param_node(file, NODE_ID_PARAM_COMPLEXITY_K_PAYLOAD, map_to_activation(0.01f, 0.001f, 0.02f));
    
    // Stability params
    melvin_create_param_node(file, NODE_ID_PARAM_STABILITY_FE_LOW, map_to_activation(0.1f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_STABILITY_FE_HIGH, map_to_activation(0.5f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_STABILITY_ACT_MIN, map_to_activation(0.1f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_STABILITY_DRIFT_ALPHA, map_to_activation(0.1f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_STABILITY_PRUNE_THRESHOLD, map_to_activation(0.5f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_USAGE_PRUNE_THRESHOLD, map_to_activation(0.1f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_EDGE_USAGE_PRUNE_THRESHOLD, map_to_activation(0.1f, 0.0f, 1.0f));
    
    // GPU exec params
    melvin_create_param_node(file, NODE_ID_PARAM_EXEC_GPU_ENABLED, 0.0f);
    melvin_create_param_node(file, NODE_ID_PARAM_EXEC_GPU_COST_MULTIPLIER, map_to_activation(0.5f, 0.1f, 1.0f));
    
    // Edge formation params
    melvin_create_param_node(file, NODE_ID_PARAM_COACT_ACT_MIN, map_to_activation(0.3f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_COACT_TRAFFIC_MIN, map_to_activation(0.1f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_COACT_SEED_WEIGHT, map_to_activation(0.2f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_FE_DROP_MIN, map_to_activation(0.1f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_FE_DROP_TRAFFIC_MIN, map_to_activation(0.1f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_FE_DROP_SOURCE_ACT_MIN, map_to_activation(0.3f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_FE_DROP_SEED_WEIGHT, map_to_activation(0.2f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_STRUCT_COMP_SEED_WEIGHT, map_to_activation(0.2f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_STRUCT_COMP_MAX_EDGES, map_to_activation(10.0f, 5.0f, 50.0f));
    
    // Curiosity params
    melvin_create_param_node(file, NODE_ID_PARAM_CURIOSITY_ACT_MIN, map_to_activation(0.3f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_CURIOSITY_ERROR_MIN, map_to_activation(0.1f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_CURIOSITY_TRAFFIC_MAX, map_to_activation(0.5f, 0.0f, 1.0f));
    melvin_create_param_node(file, NODE_ID_PARAM_CURIOSITY_SEED_WEIGHT, map_to_activation(0.1f, 0.0f, 1.0f));
    
    // Law nodes
    melvin_create_param_node(file, NODE_ID_LAW_LR_BASE, map_to_activation(0.01f, 0.001f, 0.05f));
    melvin_create_param_node(file, NODE_ID_LAW_W_LIMIT, map_to_activation(10.0f, 5.0f, 20.0f));
    melvin_create_param_node(file, NODE_ID_LAW_EXEC_CENTER, map_to_activation(0.5f, 0.3f, 0.7f));
    melvin_create_param_node(file, NODE_ID_LAW_EXEC_K, map_to_activation(5.0f, 2.0f, 10.0f));
    melvin_create_param_node(file, NODE_ID_LAW_ELIG_SCALE, map_to_activation(0.001f, 0.0001f, 0.01f));
    melvin_create_param_node(file, NODE_ID_LAW_PRUNE_BASE, map_to_activation(0.001f, 0.0001f, 0.01f));
}

// ========================================================================
// Main Injection Function
// ========================================================================

// Main injection function - call this after file creation
void melvin_inject_instincts(MelvinFile *file) {
    if (!file) return;
    
    // Detect if graph is fresh (node count below threshold)
    const uint64_t FRESH_THRESHOLD = 50;
    uint64_t num_nodes = melvin_get_num_nodes(file);
    int is_fresh = (num_nodes < FRESH_THRESHOLD);
    
    if (!is_fresh) {
        fprintf(stderr, "[INSTINCTS] Graph not fresh (%llu nodes), skipping pattern injection\n", 
                (unsigned long long)num_nodes);
        return;
    }
    
    fprintf(stderr, "[INSTINCTS] Injecting initial patterns into graph (%llu nodes)...\n", 
            (unsigned long long)num_nodes);
    
    uint64_t nodes_before = melvin_get_num_nodes(file);
    uint64_t edges_before = melvin_get_num_edges(file);
    
    // Inject all pattern families
    melvin_inject_param_nodes(file);
    melvin_inject_channel_patterns(file);
    melvin_inject_code_patterns(file);
    melvin_inject_reward_patterns(file);
    melvin_inject_body_patterns(file);
    melvin_inject_exec_patterns_new(file);  // Pattern Family A: EXEC wiring
    melvin_inject_math_patterns(file);      // Pattern Family B: Math microprograms
    melvin_inject_compile_patterns(file);   // Pattern Family C: Compile/load pipeline
    melvin_inject_port_patterns(file);      // Pattern Family D: Data injection/ports
    melvin_inject_multi_hop_patterns(file); // Pattern Family E: Multi-hop reasoning scaffolds
    
    uint64_t nodes_after = melvin_get_num_nodes(file);
    uint64_t edges_after = melvin_get_num_edges(file);
    
    fprintf(stderr, "[INSTINCTS] Pattern injection complete:\n");
    fprintf(stderr, "  Nodes: %llu -> %llu (+%llu)\n", 
            (unsigned long long)nodes_before, 
            (unsigned long long)nodes_after,
            (unsigned long long)(nodes_after - nodes_before));
    fprintf(stderr, "  Edges: %llu -> %llu (+%llu)\n", 
            (unsigned long long)edges_before, 
            (unsigned long long)edges_after,
            (unsigned long long)(edges_after - edges_before));
    fprintf(stderr, "[INSTINCTS] Graph can now evolve or replace these patterns.\n");
    
    // Populate and register instinct IDs for stable node addressing
    MelvinInstinctIds ids = {0};
    
    // Math workspace nodes (MATH_PATTERN_BASE = 60000)
    ids.math_in_a_i32_id = MATH_PATTERN_BASE + 0;      // 60000
    ids.math_in_b_i32_id = MATH_PATTERN_BASE + 1;      // 60001
    ids.math_out_i32_id = MATH_PATTERN_BASE + 2;       // 60002
    ids.math_temp_i32_id = MATH_PATTERN_BASE + 3;      // 60003
    ids.math_nodes_valid = 1;
    
    // EXEC math operation nodes (EXEC_PATTERN_BASE + 10 = 50010)
    ids.exec_add32_id = EXEC_PATTERN_BASE + 10 + 0;    // 50010
    ids.exec_sub32_id = EXEC_PATTERN_BASE + 10 + 1;    // 50011
    ids.exec_mul32_id = EXEC_PATTERN_BASE + 10 + 2;    // 50012
    ids.exec_div32_id = EXEC_PATTERN_BASE + 10 + 3;    // 50013
    ids.exec_nodes_valid = 1;
    
    // EXEC hub nodes (EXEC_PATTERN_BASE = 50000)
    ids.exec_hub_id = EXEC_PATTERN_BASE + 0;           // 50000
    ids.exec_args_id = EXEC_PATTERN_BASE + 1;          // 50001
    ids.exec_result_id = EXEC_PATTERN_BASE + 2;        // 50002
    ids.exec_err_id = EXEC_PATTERN_BASE + 3;           // 50003
    
    // Tool selection nodes (created by tests, will be set by tests)
    ids.tool_opcode_id = 100000ULL;                     // Default test ID
    ids.exec_select_add_mul_id = 100001ULL;            // Default test ID
    ids.tool_nodes_valid = 0;  // Will be set by tests when they create these nodes
    
    // Register IDs with file
    melvin_set_instinct_ids(file, &ids);
    
    fprintf(stderr, "[INSTINCTS] Instinct IDs registered for stable node addressing\n");
}

// Individual injection functions (for testing/debugging)
void melvin_inject_param_nodes_public(MelvinFile *file) {
    melvin_inject_param_nodes(file);
}

void melvin_inject_exec_patterns(MelvinFile *file) {
    // Legacy function - redirects to new exec patterns
    // Keep for backward compatibility
    melvin_inject_exec_patterns_new(file);
}

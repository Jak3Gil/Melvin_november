/*
 * code_instincts.c - Code Compilation/Execution Instincts
 * 
 * Implements instincts for code compilation and execution as pure graph structure:
 * - Port/layout instincts (SRC_IN, BIN_IN, CMD_IN, OUT_LOG)
 * - Source↔binary mapping patterns
 * - EXEC instincts (compile, link/load, run)
 * 
 * All instincts are nodes, edges, and patterns - no external logic.
 * The graph learns to use these through energy flow and pattern matching.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

// Forward declarations (from melvin.c)
#ifndef _MELVIN_FILE_DEFINED
typedef struct MelvinFile MelvinFile;
#endif
uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id);
uint64_t melvin_create_param_node(MelvinFile *file, uint64_t node_id, float initial_activation);
int create_edge_between(MelvinFile *file, uint64_t src, uint64_t dst, float initial_weight);
uint64_t melvin_write_machine_code(MelvinFile *file, const uint8_t *code, size_t code_len);
int melvin_set_node_payload_and_flags(MelvinFile *file, uint64_t node_id, 
                                      uint64_t payload_offset, uint32_t flags);

#define NODE_FLAG_EXECUTABLE (1U << 0)
#define NODE_FLAG_DATA       (1U << 1)

// ========================================================================
// CODE INSTINCT NODE IDS (Stable IDs for addressing)
// ========================================================================

#define CODE_INSTINCT_BASE 100000ULL

// Port nodes
#define NODE_ID_PORT_SRC_IN   (CODE_INSTINCT_BASE + 0)   // 100000
#define NODE_ID_PORT_BIN_IN   (CODE_INSTINCT_BASE + 1)   // 100001
#define NODE_ID_PORT_CMD_IN   (CODE_INSTINCT_BASE + 2)   // 100002
#define NODE_ID_PORT_OUT_LOG  (CODE_INSTINCT_BASE + 3)   // 100003

// Region metadata nodes (describe where code/data lives in blob)
#define NODE_ID_REGION_SRC    (CODE_INSTINCT_BASE + 10)  // 100010
#define NODE_ID_REGION_BIN    (CODE_INSTINCT_BASE + 11)  // 100011
#define NODE_ID_REGION_BLOB   (CODE_INSTINCT_BASE + 12)  // 100012

// Code block metadata nodes
#define NODE_ID_BLOCK_ENTRY   (CODE_INSTINCT_BASE + 20)  // 100020
#define NODE_ID_BLOCK_SIZE    (CODE_INSTINCT_BASE + 21)  // 100021
#define NODE_ID_BLOCK_SYMBOL  (CODE_INSTINCT_BASE + 22)  // 100022

// EXEC nodes (EXECUTABLE - will contain machine code or function pointers)
#define NODE_ID_EXEC_COMPILE  (CODE_INSTINCT_BASE + 30)  // 100030
#define NODE_ID_EXEC_LINK     (CODE_INSTINCT_BASE + 31)  // 100031
#define NODE_ID_EXEC_RUN      (CODE_INSTINCT_BASE + 32)  // 100032

// Command nodes (trigger EXEC nodes)
#define NODE_ID_CMD_COMPILE   (CODE_INSTINCT_BASE + 40)  // 100040
#define NODE_ID_CMD_RUN        (CODE_INSTINCT_BASE + 41)  // 100041

// Pattern nodes (for source↔binary mapping)
#define NODE_ID_PATTERN_SRC_BIN (CODE_INSTINCT_BASE + 50) // 100050

// Helper: Create node with payload
static uint64_t create_node_with_payload(MelvinFile *file, uint64_t node_id, 
                                          const uint8_t *payload, size_t payload_len, 
                                          uint32_t flags, float initial_state) {
    if (!file) return UINT64_MAX;
    
    if (find_node_index_by_id(file, node_id) != UINT64_MAX) {
        return node_id; // Already exists
    }
    
    uint64_t idx = melvin_create_param_node(file, node_id, initial_state);
    if (idx == UINT64_MAX) return UINT64_MAX;
    
    uint64_t payload_offset = 0;
    if (payload && payload_len > 0) {
        payload_offset = melvin_write_machine_code(file, payload, payload_len);
        if (payload_offset == UINT64_MAX) {
            payload_offset = 0;
            payload_len = 0;
        }
    }
    
    if (melvin_set_node_payload_and_flags(file, node_id, payload_offset, flags) < 0) {
        return node_id; // Return anyway
    }
    
    return node_id;
}

// ========================================================================
// (A) Port/Layout Instincts
// ========================================================================
// Creates port nodes and region metadata nodes

static void inject_port_layout_instincts(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create port nodes (DATA nodes with labels)
    const char *port_labels[] = {
        "PORT:SRC_IN",
        "PORT:BIN_IN", 
        "PORT:CMD_IN",
        "PORT:OUT_LOG"
    };
    uint64_t port_ids[] = {
        NODE_ID_PORT_SRC_IN,
        NODE_ID_PORT_BIN_IN,
        NODE_ID_PORT_CMD_IN,
        NODE_ID_PORT_OUT_LOG
    };
    
    for (int i = 0; i < 4; i++) {
        if (create_node_with_payload(file, port_ids[i], 
                                     (const uint8_t*)port_labels[i], 
                                     strlen(port_labels[i]),
                                     NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Create region metadata nodes (describe blob regions)
    const char *region_labels[] = {
        "REGION:SRC",   // Source code region
        "REGION:BIN",   // Binary/machine code region
        "REGION:BLOB"   // Entire blob
    };
    uint64_t region_ids[] = {
        NODE_ID_REGION_SRC,
        NODE_ID_REGION_BIN,
        NODE_ID_REGION_BLOB
    };
    
    for (int i = 0; i < 3; i++) {
        if (create_node_with_payload(file, region_ids[i],
                                     (const uint8_t*)region_labels[i],
                                     strlen(region_labels[i]),
                                     NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire ports to regions
    if (create_edge_between(file, NODE_ID_PORT_SRC_IN, NODE_ID_REGION_SRC, 0.5f)) edges_created++;
    if (create_edge_between(file, NODE_ID_PORT_BIN_IN, NODE_ID_REGION_BIN, 0.5f)) edges_created++;
    if (create_edge_between(file, NODE_ID_REGION_SRC, NODE_ID_REGION_BLOB, 0.3f)) edges_created++;
    if (create_edge_between(file, NODE_ID_REGION_BIN, NODE_ID_REGION_BLOB, 0.3f)) edges_created++;
    
    // Create code block metadata nodes
    const char *block_labels[] = {
        "BLOCK:ENTRY",  // Entry point pointer
        "BLOCK:SIZE",   // Block size
        "BLOCK:SYMBOL"  // Symbol name
    };
    uint64_t block_ids[] = {
        NODE_ID_BLOCK_ENTRY,
        NODE_ID_BLOCK_SIZE,
        NODE_ID_BLOCK_SYMBOL
    };
    
    for (int i = 0; i < 3; i++) {
        if (create_node_with_payload(file, block_ids[i],
                                     (const uint8_t*)block_labels[i],
                                     strlen(block_labels[i]),
                                     NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire block metadata to regions
    if (create_edge_between(file, NODE_ID_REGION_BIN, NODE_ID_BLOCK_ENTRY, 0.4f)) edges_created++;
    if (create_edge_between(file, NODE_ID_REGION_BIN, NODE_ID_BLOCK_SIZE, 0.4f)) edges_created++;
    if (create_edge_between(file, NODE_ID_BLOCK_ENTRY, NODE_ID_BLOCK_SYMBOL, 0.3f)) edges_created++;
    
    fprintf(stderr, "[CODE_INSTINCTS] Port/layout: %d nodes, %d edges\n", nodes_created, edges_created);
}

// ========================================================================
// (B) Source↔Binary Mapping Instincts
// ========================================================================
// Creates patterns that link source code bytes to compiled binary bytes

static void inject_source_binary_mapping_instincts(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create pattern node for source↔binary mapping
    const char *pattern_label = "PATTERN:SRC_BIN";
    if (create_node_with_payload(file, NODE_ID_PATTERN_SRC_BIN,
                                 (const uint8_t*)pattern_label,
                                 strlen(pattern_label),
                                 NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
        nodes_created++;
    }
    
    // Wire pattern to ports and regions
    if (create_edge_between(file, NODE_ID_PORT_SRC_IN, NODE_ID_PATTERN_SRC_BIN, 0.4f)) edges_created++;
    if (create_edge_between(file, NODE_ID_PATTERN_SRC_BIN, NODE_ID_PORT_BIN_IN, 0.4f)) edges_created++;
    if (create_edge_between(file, NODE_ID_REGION_SRC, NODE_ID_PATTERN_SRC_BIN, 0.3f)) edges_created++;
    if (create_edge_between(file, NODE_ID_PATTERN_SRC_BIN, NODE_ID_REGION_BIN, 0.3f)) edges_created++;
    
    // Note: Actual pattern slots (SRC:[...], BIN:[...]) would be created
    // by the pattern induction system when it sees source↔binary pairs.
    // This instinct just creates the scaffolding.
    
    fprintf(stderr, "[CODE_INSTINCTS] Source↔binary mapping: %d nodes, %d edges\n", 
            nodes_created, edges_created);
}

// ========================================================================
// (C) EXEC Instincts (Compile, Link, Run)
// ========================================================================
// Creates EXECUTABLE nodes that can call compilers and run code

static void inject_exec_instincts(MelvinFile *file) {
    if (!file) return;
    
    int nodes_created = 0;
    int edges_created = 0;
    
    // Create EXEC nodes (EXECUTABLE - will be bound to C functions)
    // These are marked EXECUTABLE but don't have machine code yet.
    // The actual function pointers are set up separately.
    const char *exec_labels[] = {
        "EXEC:COMPILE",  // Compiles source → binary
        "EXEC:LINK",      // Links/loads binary into executable memory
        "EXEC:RUN"        // Runs compiled code block
    };
    uint64_t exec_ids[] = {
        NODE_ID_EXEC_COMPILE,
        NODE_ID_EXEC_LINK,
        NODE_ID_EXEC_RUN
    };
    
    for (int i = 0; i < 3; i++) {
        // Create as EXECUTABLE node (but payload is just label for now)
        // Actual machine code binding happens separately
        if (create_node_with_payload(file, exec_ids[i],
                                     (const uint8_t*)exec_labels[i],
                                     strlen(exec_labels[i]),
                                     NODE_FLAG_EXECUTABLE | NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Create command nodes (trigger EXEC nodes)
    const char *cmd_labels[] = {
        "CMD:COMPILE",
        "CMD:RUN"
    };
    uint64_t cmd_ids[] = {
        NODE_ID_CMD_COMPILE,
        NODE_ID_CMD_RUN
    };
    
    for (int i = 0; i < 2; i++) {
        if (create_node_with_payload(file, cmd_ids[i],
                                     (const uint8_t*)cmd_labels[i],
                                     strlen(cmd_labels[i]),
                                     NODE_FLAG_DATA, 0.0f) != UINT64_MAX) {
            nodes_created++;
        }
    }
    
    // Wire command → EXEC
    if (create_edge_between(file, NODE_ID_CMD_COMPILE, NODE_ID_EXEC_COMPILE, 0.6f)) edges_created++;
    if (create_edge_between(file, NODE_ID_CMD_RUN, NODE_ID_EXEC_RUN, 0.6f)) edges_created++;
    
    // Wire ports → EXEC
    if (create_edge_between(file, NODE_ID_PORT_SRC_IN, NODE_ID_EXEC_COMPILE, 0.5f)) edges_created++;
    if (create_edge_between(file, NODE_ID_PORT_CMD_IN, NODE_ID_CMD_COMPILE, 0.5f)) edges_created++;
    if (create_edge_between(file, NODE_ID_PORT_CMD_IN, NODE_ID_CMD_RUN, 0.5f)) edges_created++;
    
    // Wire EXEC → output ports
    if (create_edge_between(file, NODE_ID_EXEC_COMPILE, NODE_ID_PORT_BIN_IN, 0.5f)) edges_created++;
    if (create_edge_between(file, NODE_ID_EXEC_COMPILE, NODE_ID_PORT_OUT_LOG, 0.4f)) edges_created++;
    if (create_edge_between(file, NODE_ID_EXEC_LINK, NODE_ID_BLOCK_ENTRY, 0.5f)) edges_created++;
    if (create_edge_between(file, NODE_ID_EXEC_RUN, NODE_ID_PORT_OUT_LOG, 0.4f)) edges_created++;
    
    // Wire regions → EXEC
    if (create_edge_between(file, NODE_ID_REGION_SRC, NODE_ID_EXEC_COMPILE, 0.4f)) edges_created++;
    if (create_edge_between(file, NODE_ID_REGION_BIN, NODE_ID_EXEC_LINK, 0.4f)) edges_created++;
    if (create_edge_between(file, NODE_ID_BLOCK_ENTRY, NODE_ID_EXEC_RUN, 0.4f)) edges_created++;
    
    fprintf(stderr, "[CODE_INSTINCTS] EXEC instincts: %d nodes, %d edges\n", 
            nodes_created, edges_created);
}

// ========================================================================
// Main Injection Function
// ========================================================================

void melvin_inject_code_instincts(MelvinFile *file) {
    if (!file) return;
    
    fprintf(stderr, "[CODE_INSTINCTS] Injecting code compilation/execution instincts...\n");
    
    uint64_t nodes_before = 0;
    uint64_t edges_before = 0;
    
    // Get current counts (if function exists)
    // For now, just inject
    inject_port_layout_instincts(file);
    inject_source_binary_mapping_instincts(file);
    inject_exec_instincts(file);
    
    fprintf(stderr, "[CODE_INSTINCTS] Code instincts injection complete.\n");
    fprintf(stderr, "[CODE_INSTINCTS] Graph can now learn to use these for compilation/execution.\n");
}


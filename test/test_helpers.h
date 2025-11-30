/*
 * test_helpers.h
 * 
 * Shared test helper functions for graph-driven tests.
 * 
 * POLICY: These helpers only manipulate node payloads/states via Graph + blob.
 * They do NOT call any EXEC functions directly.
 * 
 * Tests must follow the "graph-driven" rule:
 * - Test harness provides inputs, ticks the graph, and reads outputs.
 * - No direct calls to melvin_exec_*.
 * - No core task computation is done in the harness.
 */

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include "melvin_instincts.h"

// Note: This header must be included AFTER melvin.c to get type definitions
// The types are forward-declared here for reference, but actual definitions
// come from melvin.c when included

// ========================================================================
// Node Finding Helpers
// ========================================================================

// Find a unique node by payload label
// Returns node index (UINT64_MAX if not found)
static inline uint64_t find_node_index_by_label(MelvinFile *file, const char *label) {
    if (!file || !label) return UINT64_MAX;
    
    GraphHeaderDisk *gh = file->graph_header;
    size_t label_len = strlen(label);
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *node = &file->nodes[i];
        if (node->id == UINT64_MAX) continue;
        if (node->payload_offset == 0 || node->payload_len == 0) continue;
        if (node->payload_len < label_len) continue;
        
        const char *payload = (const char *)(file->blob + node->payload_offset);
        if (strncmp(payload, label, label_len) == 0) {
            return i;
        }
    }
    
    return UINT64_MAX;
}

// Find node by ID and return pointer (or NULL)
static inline NodeDisk* find_singleton_node_by_id(MelvinFile *file, uint64_t node_id) {
    uint64_t idx = find_node_index_by_id(file, node_id);
    if (idx == UINT64_MAX) return NULL;
    return &file->nodes[idx];
}

// Find node by label and return pointer (or NULL)
static inline NodeDisk* find_singleton_node_by_label(MelvinFile *file, const char *label) {
    uint64_t idx = find_node_index_by_label(file, label);
    if (idx == UINT64_MAX) return NULL;
    return &file->nodes[idx];
}

// ========================================================================
// Integer Read/Write Helpers
// ========================================================================

// Encode int32 to bytes (little-endian)
static inline void encode_int32_to_bytes(int32_t value, uint8_t *buf) {
    buf[0] = (uint8_t)(value & 0xFF);
    buf[1] = (uint8_t)((value >> 8) & 0xFF);
    buf[2] = (uint8_t)((value >> 16) & 0xFF);
    buf[3] = (uint8_t)((value >> 24) & 0xFF);
}

// Decode int32 from bytes (little-endian)
static inline int32_t decode_int32_from_bytes(const uint8_t *buf) {
    return (int32_t)((uint32_t)buf[0] |
                     ((uint32_t)buf[1] << 8) |
                     ((uint32_t)buf[2] << 16) |
                     ((uint32_t)buf[3] << 24));
}

// Read int32 value from a labeled node
// Reads from node state (where values are stored to preserve labels in payloads)
static inline int32_t read_int32_from_labeled_node(MelvinFile *file, const char *label) {
    NodeDisk *node = find_singleton_node_by_label(file, label);
    if (!node) {
        fprintf(stderr, "ERROR: Cannot read from node '%s'\n", label);
        return 0;
    }
    
    // Read from state (where values are stored to preserve label in payload)
    return (int32_t)node->state;
}

// Write int32 value to a labeled node
// Stores in node state (preserves label in payload)
static inline void write_int32_to_labeled_node(MelvinFile *file, const char *label, int32_t value) {
    NodeDisk *node = find_singleton_node_by_label(file, label);
    if (!node) {
        fprintf(stderr, "ERROR: Cannot write to node '%s'\n", label);
        return;
    }
    
    // Store value in node state (preserves label in payload)
    node->state = (float)value;
}

// ========================================================================
// Node State Management
// ========================================================================

// Reset state for a labeled node
static inline void reset_node_state_by_label(MelvinFile *file, const char *label) {
    NodeDisk *node = find_singleton_node_by_label(file, label);
    if (node) {
        node->state = 0.0f;
    }
}

// Activate a node by label (set state above threshold)
static inline void activate_node_by_label(MelvinFile *file, const char *label, float activation) {
    NodeDisk *node = find_singleton_node_by_label(file, label);
    if (node) {
        node->state = activation;
    }
}

// Activate EXEC node to trigger execution (sets state above exec_threshold)
static inline void activate_exec_node_by_label(MelvinFile *file, const char *label) {
    NodeDisk *node = find_singleton_node_by_label(file, label);
    if (node && file->graph_header) {
        // Set activation above exec_threshold to trigger execution
        float threshold = file->graph_header->exec_threshold;
        node->state = threshold + 0.1f;
        // Mark as executable if not already
        node->flags |= NODE_FLAG_EXECUTABLE;
    }
}

// ========================================================================
// ID-Based Helpers (Preferred - Uses Stable Instinct IDs)
// ========================================================================

// Read int32 from node by ID (uses instinct IDs when available)
static inline int32_t read_int32_from_node_by_id(MelvinFile *file, uint64_t node_id) {
    NodeDisk *node = melvin_get_node_safe(file, node_id);
    if (!node) {
        fprintf(stderr, "ERROR: Cannot read from node ID %llu\n", (unsigned long long)node_id);
        return 0;
    }
    return (int32_t)node->state;
}

// Write int32 to node by ID (uses instinct IDs when available)
static inline void write_int32_to_node_by_id(MelvinFile *file, uint64_t node_id, int32_t value) {
    NodeDisk *node = melvin_get_node_safe(file, node_id);
    if (!node) {
        fprintf(stderr, "ERROR: Cannot write to node ID %llu\n", (unsigned long long)node_id);
        return;
    }
    node->state = (float)value;
}

// Reset state for node by ID
static inline void reset_node_state_by_id(MelvinFile *file, uint64_t node_id) {
    NodeDisk *node = melvin_get_node_safe(file, node_id);
    if (node) {
        node->state = 0.0f;
    }
}

// Activate node by ID (set state above threshold)
static inline void activate_node_by_id(MelvinFile *file, uint64_t node_id, float activation) {
    NodeDisk *node = melvin_get_node_safe(file, node_id);
    if (node) {
        node->state = activation;
    }
}

// Activate EXEC node by ID to trigger execution
static inline void activate_exec_node_by_id(MelvinFile *file, uint64_t node_id) {
    NodeDisk *node = melvin_get_node_safe(file, node_id);
    if (node && file->graph_header) {
        float threshold = file->graph_header->exec_threshold;
        node->state = threshold + 0.1f;
        node->flags |= NODE_FLAG_EXECUTABLE;
    }
}

#endif // TEST_HELPERS_H


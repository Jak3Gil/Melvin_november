/*
 * melvin_instincts.h
 * 
 * Stable node ID registry for instinct patterns.
 * 
 * This provides a reliable way to address critical nodes without
 * relying on fragile label-based payload searches.
 */

#ifndef MELVIN_INSTINCTS_H
#define MELVIN_INSTINCTS_H

#include <stdint.h>

// Forward declaration (only if not already defined by melvin.c)
#ifndef _MELVIN_FILE_DEFINED
typedef struct MelvinFile MelvinFile;
#endif

// Stable node IDs for instinct patterns
// These IDs are known at compile time from instincts.c pattern bases
typedef struct MelvinInstinctIds {
    // Math workspace nodes (MATH_PATTERN_BASE = 60000)
    uint64_t math_in_a_i32_id;      // 60000
    uint64_t math_in_b_i32_id;      // 60001
    uint64_t math_out_i32_id;       // 60002
    uint64_t math_temp_i32_id;      // 60003
    
    // EXEC math operation nodes (EXEC_PATTERN_BASE + 10 = 50010)
    uint64_t exec_add32_id;         // 50010
    uint64_t exec_sub32_id;         // 50011
    uint64_t exec_mul32_id;         // 50012
    uint64_t exec_div32_id;         // 50013
    
    // EXEC hub nodes (EXEC_PATTERN_BASE = 50000)
    uint64_t exec_hub_id;           // 50000
    uint64_t exec_args_id;          // 50001
    uint64_t exec_result_id;        // 50002
    uint64_t exec_err_id;           // 50003
    
    // Tool selection nodes (created by tests, IDs vary)
    uint64_t tool_opcode_id;        // 100000 (test-created)
    uint64_t exec_select_add_mul_id; // 100001 (test-created)
    
    // Flags to indicate which IDs are valid
    uint8_t math_nodes_valid;
    uint8_t exec_nodes_valid;
    uint8_t tool_nodes_valid;
} MelvinInstinctIds;

// Get instinct IDs from a MelvinFile
// Returns NULL if instincts haven't been injected yet
const MelvinInstinctIds *melvin_get_instinct_ids(MelvinFile *file);

// Initialize instinct IDs (called by instincts.c after injection)
void melvin_set_instinct_ids(MelvinFile *file, const MelvinInstinctIds *ids);

#endif // MELVIN_INSTINCTS_H


#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/mman.h>

// Include the implementation
#include "melvin.c"

// ========================================================================
// TEST: Code Evolution
// 
// Tests if EXEC nodes can modify other EXEC nodes' code.
// This tests mutation and recombination in code evolution.
// ========================================================================

// Target function: Simple function that returns 0x10
// We'll try to modify it to return 0x20
__attribute__((noinline))
static void target_function(MelvinFile *g, uint64_t self_id) {
    // This function will be modified by the mutator
    if (g && g->blob) {
        // Just a placeholder - actual code will be machine code
    }
}

// Mutator function: Modifies another EXEC node's code
__attribute__((noinline))
static void exec_mutator(MelvinFile *g, uint64_t self_id) {
    if (!g || !g->blob || !g->nodes || !g->graph_header) {
        return;
    }
    
    GraphHeaderDisk *gh = g->graph_header;
    
    // Find a target EXEC node (not ourselves)
    uint64_t target_node_id = UINT64_MAX;
    uint64_t target_idx = UINT64_MAX;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *node = &g->nodes[i];
        if (node->id != UINT64_MAX && 
            node->id != self_id &&
            (node->flags & NODE_FLAG_EXECUTABLE) &&
            node->payload_len > 0) {
            target_node_id = node->id;
            target_idx = i;
            break;
        }
    }
    
    if (target_node_id == UINT64_MAX) {
        return;  // No target found
    }
    
    NodeDisk *target_node = &g->nodes[target_idx];
    
    // Check bounds
    if (target_node->payload_offset + target_node->payload_len > g->blob_size) {
        return;
    }
    
    // Modify the target's code (simple mutation: change one byte)
    // For x86-64: change return value from 0x10 to 0x20
    // mov $0x10, %rax -> mov $0x20, %rax
    // Byte 3 changes from 0x10 to 0x20
#if defined(__x86_64__) || defined(_M_X64)
    uint8_t *code_ptr = g->blob + target_node->payload_offset;
    if (target_node->payload_len >= 8 && code_ptr[0] == 0x48 && code_ptr[1] == 0xC7) {
        // Make writable
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t code_start = (uintptr_t)code_ptr;
        uintptr_t page_start = code_start & ~(page_size - 1);
        size_t protect_size = target_node->payload_len;
        
        if (mprotect((void*)page_start, protect_size, PROT_READ | PROT_WRITE) == 0) {
            // Mutate: change 0x10 to 0x20
            if (code_ptr[3] == 0x10) {
                code_ptr[3] = 0x20;
            }
            // Restore executable
            mprotect((void*)page_start, protect_size, PROT_READ | PROT_EXEC);
        }
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM64: modify immediate value
    uint8_t *code_ptr = g->blob + target_node->payload_offset;
    if (target_node->payload_len >= 4) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t code_start = (uintptr_t)code_ptr;
        uintptr_t page_start = code_start & ~(page_size - 1);
        size_t protect_size = target_node->payload_len;
        
        if (mprotect((void*)page_start, protect_size, PROT_READ | PROT_WRITE) == 0) {
            // Simple mutation: flip a bit in the immediate
            code_ptr[0] ^= 0x01;
            mprotect((void*)page_start, protect_size, PROT_READ | PROT_EXEC);
        }
    }
#endif
    
    // Mark that we mutated (store in blob[100] as marker)
    if (g->blob_capacity > 100) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)g->blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            g->blob[100] = 0xFF;  // Mutation marker
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
}

int main(int argc, char **argv) {
    const char *file_path = "test_code_evolution.m";
    
    printf("========================================\n");
    printf("CODE EVOLUTION TEST\n");
    printf("========================================\n\n");
    printf("Goal: Test if EXEC nodes can modify other EXEC nodes' code\n");
    printf("This tests mutation in code evolution.\n\n");
    
    // Create new file
    GraphParams params;
    params.decay_rate = 0.1f;
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.exec_threshold = 0.8f;
    params.learning_rate = 0.001f;
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    unlink(file_path);
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return 1;
    }
    printf("✓ Created %s\n", file_path);
    
    // Map file
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    printf("✓ Mapped file\n");
    
    // Write target function (returns 0x10)
    printf("\nStep 1: Writing target function...\n");
#if defined(__x86_64__) || defined(_M_X64)
    uint8_t target_code[] = {
        0x48, 0xC7, 0xC0, 0x10, 0x00, 0x00, 0x00,  // mov $0x10, %rax
        0xC3                                        // ret
    };
#elif defined(__aarch64__) || defined(_M_ARM64)
    uint8_t target_code[] = {
        0x40, 0x02, 0x80, 0xD2,   // mov x0, #0x10
        0xC0, 0x03, 0x5F, 0xD6    // ret
    };
#else
    uint8_t target_code[] = {0x10, 0x10, 0x10, 0x10};
#endif
    
    uint64_t target_offset = melvin_write_machine_code(&file, target_code, sizeof(target_code));
    if (target_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write target code\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Target code written at offset %llu\n", (unsigned long long)target_offset);
    
    // Create target EXEC node
    uint64_t target_node_id = melvin_create_executable_node(&file, target_offset, sizeof(target_code));
    if (target_node_id == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to create target EXEC node\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Created target EXEC node ID: %llu\n", (unsigned long long)target_node_id);
    
    // Write mutator function
    printf("\nStep 2: Writing mutator function...\n");
    void *mutator_ptr = (void*)exec_mutator;
    size_t mutator_size = 256;
    
    uint64_t mutator_offset = melvin_write_machine_code(&file, (uint8_t*)mutator_ptr, mutator_size);
    if (mutator_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write mutator code\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Mutator code written at offset %llu\n", (unsigned long long)mutator_offset);
    
    // Create mutator EXEC node
    uint64_t mutator_node_id = melvin_create_executable_node(&file, mutator_offset, mutator_size);
    if (mutator_node_id == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to create mutator EXEC node\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Created mutator EXEC node ID: %llu\n", (unsigned long long)mutator_node_id);
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n");
    
    // Save original target code for comparison
    uint8_t original_code[16];
    memcpy(original_code, target_code, sizeof(target_code));
    
    // Activate mutator to modify target
    printf("\nStep 3: Activating mutator to modify target code...\n");
    MelvinEvent ev = {
        .type = EV_NODE_DELTA,
        .node_id = mutator_node_id,
        .value = 1.5f
    };
    melvin_event_enqueue(&rt.evq, &ev);
    
    printf("Processing events (mutator should execute and modify target)...\n");
    melvin_process_n_events(&rt, 50);
    
    // Check if mutation occurred
    printf("\nStep 4: Checking if mutation occurred...\n");
    
    // Read target code from blob
    uint8_t modified_code[16];
    if (target_offset < file.blob_size) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)file.blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            memcpy(modified_code, file.blob + target_offset, 
                   sizeof(modified_code) < (file.blob_size - target_offset) ? 
                   sizeof(modified_code) : (file.blob_size - target_offset));
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
    
    // Check mutation marker
    uint8_t mutation_marker = 0;
    if (file.blob && file.blob_capacity > 100) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)file.blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            mutation_marker = file.blob[100];
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
    
    // Compare codes
    int code_changed = memcmp(original_code, modified_code, sizeof(original_code)) != 0;
    
    // Evaluate results
    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    
    int success = 1;
    
    if (mutation_marker == 0xFF) {
        printf("✓ PASS: Mutation marker set (mutator executed)\n");
    } else {
        printf("⚠ WARNING: Mutation marker not set (mutator may not have executed)\n");
        printf("  Marker value: 0x%02X\n", mutation_marker);
    }
    
    if (code_changed) {
        printf("✓ PASS: Target code was modified\n");
        printf("  Original: ");
        for (size_t i = 0; i < sizeof(original_code); i++) {
            printf("%02X ", original_code[i]);
        }
        printf("\n  Modified: ");
        for (size_t i = 0; i < sizeof(modified_code); i++) {
            printf("%02X ", modified_code[i]);
        }
        printf("\n");
    } else {
        printf("✗ FAIL: Target code was not modified\n");
        printf("  Code unchanged (evolution did not occur)\n");
        success = 0;
    }
    
    // Final evaluation
    printf("\n========================================\n");
    if (success && code_changed) {
        printf("✅ TEST PASSED: Code evolution works!\n");
        printf("EXEC nodes can modify other EXEC nodes' code.\n");
        printf("This enables mutation in code evolution.\n");
    } else {
        printf("❌ TEST FAILED: Code evolution did not work\n");
        printf("EXEC nodes may not be able to modify other nodes.\n");
    }
    printf("========================================\n");
    
    // Sync and close
    melvin_m_sync(&file);
    runtime_cleanup(&rt);
    close_file(&file);
    
    return success ? 0 : 1;
}


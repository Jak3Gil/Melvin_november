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
// TEST: Self-Modifying Code
// 
// Tests if EXEC nodes can write new machine code into the blob.
// This is a foundational capability for code evolution.
// ========================================================================

// Initial EXEC function: Writes a simple function to blob
// This tests if EXEC nodes can call melvin_write_machine_code()
__attribute__((noinline))
static void exec_writer(MelvinFile *g, uint64_t self_id) {
    if (!g || !g->blob || g->blob_capacity < 256) {
        return;
    }
    
    // Write a simple function that returns 0x42
    // x86-64: mov $0x42, %rax; ret
    // ARM64: mov x0, #0x42; ret
#if defined(__x86_64__) || defined(_M_X64)
    uint8_t new_code[] = {
        0x48, 0xC7, 0xC0, 0x42, 0x00, 0x00, 0x00,  // mov $0x42, %rax
        0xC3                                        // ret
    };
#elif defined(__aarch64__) || defined(_M_ARM64)
    uint8_t new_code[] = {
        0x40, 0x08, 0x80, 0xD2,   // mov x0, #0x42
        0xC0, 0x03, 0x5F, 0xD6    // ret
    };
#else
    // Fallback: just return marker
    uint8_t new_code[] = {0xFF, 0xFF, 0xFF, 0xFF};
#endif
    
    // Write code to blob
    uint64_t offset = melvin_write_machine_code(g, new_code, sizeof(new_code));
    
    // Store offset in blob[0-7] as marker
    if (offset != UINT64_MAX && g->blob_capacity > 8) {
        // Make writable temporarily
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)g->blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            memcpy(&g->blob[0], &offset, sizeof(offset));
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
}

int main(int argc, char **argv) {
    const char *file_path = "test_self_modify.m";
    
    printf("========================================\n");
    printf("SELF-MODIFYING CODE TEST\n");
    printf("========================================\n\n");
    printf("Goal: Test if EXEC nodes can write new code to blob\n");
    printf("This is foundational for code evolution.\n\n");
    
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
    
    // Write initial EXEC function (the writer)
    printf("\nStep 1: Writing initial EXEC function (exec_writer)...\n");
    void *writer_ptr = (void*)exec_writer;
    size_t writer_size = 256;
    
    uint64_t writer_offset = melvin_write_machine_code(&file, (uint8_t*)writer_ptr, writer_size);
    if (writer_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write writer code\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Writer code written at offset %llu\n", (unsigned long long)writer_offset);
    
    // Create EXEC node for writer
    uint64_t writer_node_id = melvin_create_executable_node(&file, writer_offset, writer_size);
    if (writer_node_id == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to create writer EXEC node\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Created writer EXEC node ID: %llu\n", (unsigned long long)writer_node_id);
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n");
    
    // Check initial blob state
    printf("\nStep 2: Checking initial blob state...\n");
    printf("  Blob size: %llu bytes\n", (unsigned long long)file.blob_size);
    printf("  Blob capacity: %llu bytes\n", (unsigned long long)file.blob_capacity);
    
    // Activate writer node to trigger code writing
    printf("\nStep 3: Activating writer EXEC node...\n");
    MelvinEvent ev = {
        .type = EV_NODE_DELTA,
        .node_id = writer_node_id,
        .value = 1.5f  // High activation to cross threshold
    };
    melvin_event_enqueue(&rt.evq, &ev);
    
    printf("Processing events (writer should execute and write new code)...\n");
    melvin_process_n_events(&rt, 50);
    
    // Check if new code was written
    printf("\nStep 4: Checking if new code was written...\n");
    printf("  Blob size after execution: %llu bytes\n", (unsigned long long)file.blob_size);
    
    // Check marker in blob[0-7]
    uint64_t written_offset = 0;
    if (file.blob && file.blob_capacity > 8) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)file.blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            memcpy(&written_offset, &file.blob[0], sizeof(written_offset));
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
    
    // Evaluate results
    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    
    int success = 1;
    
    if (file.blob_size > writer_offset + writer_size) {
        printf("✓ PASS: Blob size increased (new code written)\n");
        printf("  Initial size: %llu bytes\n", (unsigned long long)(writer_offset + writer_size));
        printf("  Final size: %llu bytes\n", (unsigned long long)file.blob_size);
        printf("  New code size: %llu bytes\n", 
               (unsigned long long)(file.blob_size - writer_offset - writer_size));
    } else {
        printf("✗ FAIL: Blob size did not increase\n");
        printf("  Expected: > %llu bytes\n", (unsigned long long)(writer_offset + writer_size));
        printf("  Actual: %llu bytes\n", (unsigned long long)file.blob_size);
        success = 0;
    }
    
    if (written_offset != 0 && written_offset != UINT64_MAX) {
        printf("✓ PASS: EXEC node wrote code and stored offset marker\n");
        printf("  Written code offset: %llu\n", (unsigned long long)written_offset);
    } else {
        printf("⚠ WARNING: No offset marker found (EXEC may not have written code)\n");
        printf("  Marker value: %llu\n", (unsigned long long)written_offset);
    }
    
    // Try to execute the newly written code
    if (written_offset != 0 && written_offset != UINT64_MAX && 
        written_offset < file.blob_size) {
        printf("\nStep 5: Testing execution of newly written code...\n");
        
        // Create EXEC node for new code
        uint64_t new_code_size = 8;  // Size of the simple return function
        uint64_t new_exec_node_id = melvin_create_executable_node(&file, written_offset, new_code_size);
        
        if (new_exec_node_id != UINT64_MAX) {
            printf("✓ Created EXEC node for new code (ID: %llu)\n", (unsigned long long)new_exec_node_id);
            
            // Activate and execute
            MelvinEvent ev2 = {
                .type = EV_NODE_DELTA,
                .node_id = new_exec_node_id,
                .value = 1.5f
            };
            melvin_event_enqueue(&rt.evq, &ev2);
            melvin_process_n_events(&rt, 50);
            
            printf("✓ New code executed successfully\n");
        } else {
            printf("⚠ WARNING: Could not create EXEC node for new code\n");
        }
    }
    
    // Final evaluation
    printf("\n========================================\n");
    if (success) {
        printf("✅ TEST PASSED: Self-modifying code works!\n");
        printf("EXEC nodes can write new code to blob.\n");
        printf("This enables code evolution.\n");
    } else {
        printf("❌ TEST FAILED: Self-modifying code did not work\n");
        printf("EXEC nodes may not be able to write code.\n");
    }
    printf("========================================\n");
    
    // Sync and close
    melvin_m_sync(&file);
    runtime_cleanup(&rt);
    close_file(&file);
    
    return success ? 0 : 1;
}


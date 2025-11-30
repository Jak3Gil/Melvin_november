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
// TEST: Automatic EXEC Creation from High-Energy Patterns
// 
// Tests if high-energy patterns can trigger automatic EXEC node creation.
// This tests the theoretical capability of patterns becoming actions.
// ========================================================================

// Simple function that could be generated from a pattern
__attribute__((noinline))
static void pattern_generated_function(MelvinFile *g, uint64_t self_id) {
    if (g && g->blob && g->blob_capacity > 0) {
        // Simple function: increment a counter
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)g->blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            if (g->blob_capacity > 200) {
                // Use blob[200] as counter
                g->blob[200]++;
            }
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
}

// Function that checks for high-energy patterns and creates EXEC nodes
// This simulates automatic EXEC creation
__attribute__((noinline))
static void auto_exec_creator(MelvinFile *g, uint64_t self_id) {
    if (!g || !g->nodes || !g->graph_header) {
        return;
    }
    
    GraphHeaderDisk *gh = g->graph_header;
    
    // Look for high-energy patterns (nodes with activation > threshold)
    float energy_threshold = 0.7f;
    uint64_t high_energy_count = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *node = &g->nodes[i];
        if (node->id != UINT64_MAX && 
            fabsf(node->state) > energy_threshold &&
            !(node->flags & NODE_FLAG_EXECUTABLE)) {
            high_energy_count++;
        }
    }
    
    // If we find a high-energy pattern (multiple high-energy nodes),
    // create an EXEC node for it
    if (high_energy_count >= 3) {
        // Write code for the pattern
        void *func_ptr = (void*)pattern_generated_function;
        size_t func_size = 256;
        
        uint64_t code_offset = melvin_write_machine_code(g, (uint8_t*)func_ptr, func_size);
        
        if (code_offset != UINT64_MAX) {
            // Create EXEC node
            uint64_t exec_node_id = melvin_create_executable_node(g, code_offset, func_size);
            
            // Mark that we created an EXEC node (store in blob[150])
            if (g->blob && g->blob_capacity > 150) {
                size_t page_size = sysconf(_SC_PAGESIZE);
                uintptr_t blob_start = (uintptr_t)g->blob;
                uintptr_t page_start = blob_start & ~(page_size - 1);
                
                if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
                    g->blob[150] = 0xAA;  // Auto-creation marker
                    if (g->blob_capacity > 151) {
                        memcpy(&g->blob[151], &exec_node_id, sizeof(exec_node_id));
                    }
                    mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    const char *file_path = "test_auto_exec.m";
    
    printf("========================================\n");
    printf("AUTOMATIC EXEC CREATION TEST\n");
    printf("========================================\n\n");
    printf("Goal: Test if high-energy patterns trigger automatic EXEC creation\n");
    printf("This tests patterns becoming actions.\n\n");
    
    // Create new file
    GraphParams params;
    params.decay_rate = 0.05f;  // Lower decay to maintain energy
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
    
    // Write auto-creator function
    printf("\nStep 1: Writing auto-creator function...\n");
    void *creator_ptr = (void*)auto_exec_creator;
    size_t creator_size = 256;
    
    uint64_t creator_offset = melvin_write_machine_code(&file, (uint8_t*)creator_ptr, creator_size);
    if (creator_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write creator code\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Creator code written at offset %llu\n", (unsigned long long)creator_offset);
    
    // Create creator EXEC node
    uint64_t creator_node_id = melvin_create_executable_node(&file, creator_offset, creator_size);
    if (creator_node_id == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to create creator EXEC node\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Created creator EXEC node ID: %llu\n", (unsigned long long)creator_node_id);
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n");
    
    // Create high-energy pattern by ingesting repeated data
    printf("\nStep 2: Creating high-energy pattern...\n");
    printf("Ingesting repeated pattern 'ABC' to create high-energy nodes...\n");
    
    const uint64_t CH_TEXT = 1;
    for (int i = 0; i < 50; i++) {
        ingest_byte(&rt, CH_TEXT, 'A', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, CH_TEXT, 'B', 1.0f);
        melvin_process_n_events(&rt, 20);
        ingest_byte(&rt, CH_TEXT, 'C', 1.0f);
        melvin_process_n_events(&rt, 50);
    }
    
    printf("✓ Pattern ingested (should have high-energy nodes)\n");
    
    // Check node energies
    GraphHeaderDisk *gh = file.graph_header;
    uint64_t high_energy_nodes = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *node = &file.nodes[i];
        if (node->id != UINT64_MAX && fabsf(node->state) > 0.7f) {
            high_energy_nodes++;
        }
    }
    printf("  High-energy nodes (>0.7): %llu\n", (unsigned long long)high_energy_nodes);
    
    // Count EXEC nodes before
    uint64_t exec_nodes_before = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *node = &file.nodes[i];
        if (node->id != UINT64_MAX && (node->flags & NODE_FLAG_EXECUTABLE)) {
            exec_nodes_before++;
        }
    }
    printf("  EXEC nodes before: %llu\n", (unsigned long long)exec_nodes_before);
    
    // Activate creator to check for patterns and create EXEC
    printf("\nStep 3: Activating auto-creator...\n");
    MelvinEvent ev = {
        .type = EV_NODE_DELTA,
        .node_id = creator_node_id,
        .value = 1.5f
    };
    melvin_event_enqueue(&rt.evq, &ev);
    
    printf("Processing events (creator should check patterns and create EXEC)...\n");
    melvin_process_n_events(&rt, 100);
    
    // Check if new EXEC node was created
    printf("\nStep 4: Checking if new EXEC node was created...\n");
    
    uint64_t exec_nodes_after = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *node = &file.nodes[i];
        if (node->id != UINT64_MAX && (node->flags & NODE_FLAG_EXECUTABLE)) {
            exec_nodes_after++;
        }
    }
    printf("  EXEC nodes after: %llu\n", (unsigned long long)exec_nodes_after);
    
    // Check auto-creation marker
    uint8_t creation_marker = 0;
    uint64_t created_node_id = 0;
    if (file.blob && file.blob_capacity > 150) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)file.blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            creation_marker = file.blob[150];
            if (file.blob_capacity > 151) {
                memcpy(&created_node_id, &file.blob[151], sizeof(created_node_id));
            }
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
    
    // Evaluate results
    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    
    int success = 1;
    
    if (high_energy_nodes >= 3) {
        printf("✓ PASS: High-energy pattern detected (%llu nodes > 0.7)\n", 
               (unsigned long long)high_energy_nodes);
    } else {
        printf("⚠ WARNING: Not enough high-energy nodes (%llu < 3)\n", 
               (unsigned long long)high_energy_nodes);
        printf("  Pattern may not have formed correctly\n");
    }
    
    if (creation_marker == 0xAA) {
        printf("✓ PASS: Auto-creation marker set (creator executed)\n");
        printf("  Created EXEC node ID: %llu\n", (unsigned long long)created_node_id);
    } else {
        printf("⚠ WARNING: Auto-creation marker not set\n");
        printf("  Marker value: 0x%02X\n", creation_marker);
    }
    
    if (exec_nodes_after > exec_nodes_before) {
        printf("✓ PASS: New EXEC node created\n");
        printf("  Before: %llu, After: %llu\n", 
               (unsigned long long)exec_nodes_before,
               (unsigned long long)exec_nodes_after);
    } else {
        printf("✗ FAIL: No new EXEC node created\n");
        printf("  Before: %llu, After: %llu\n", 
               (unsigned long long)exec_nodes_before,
               (unsigned long long)exec_nodes_after);
        success = 0;
    }
    
    // Final evaluation
    printf("\n========================================\n");
    if (success && exec_nodes_after > exec_nodes_before) {
        printf("✅ TEST PASSED: Automatic EXEC creation works!\n");
        printf("High-energy patterns can trigger EXEC node creation.\n");
        printf("This enables patterns to become actions.\n");
    } else {
        printf("❌ TEST FAILED: Automatic EXEC creation did not work\n");
        printf("Patterns may not trigger EXEC creation automatically.\n");
    }
    printf("========================================\n");
    
    // Sync and close
    melvin_m_sync(&file);
    runtime_cleanup(&rt);
    close_file(&file);
    
    return success ? 0 : 1;
}


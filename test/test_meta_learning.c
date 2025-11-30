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
// TEST: Meta-Learning (Self-Optimization)
// 
// Tests if EXEC nodes can modify physics parameters (decay_rate, 
// exec_threshold, learning_rate) to optimize system performance.
// ========================================================================

// Meta-optimizer function: Modifies physics parameters based on performance
__attribute__((noinline))
static void meta_optimizer(MelvinFile *g, uint64_t self_id) {
    if (!g || !g->graph_header) {
        return;
    }
    
    GraphHeaderDisk *gh = g->graph_header;
    
    // Measure current performance (average prediction error)
    float total_error = 0.0f;
    uint64_t error_count = 0;
    
    if (g->nodes) {
        for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
            NodeDisk *node = &g->nodes[i];
            if (node->id != UINT64_MAX && node->prediction_error > 0.0f) {
                total_error += node->prediction_error;
                error_count++;
            }
        }
    }
    
    float avg_error = error_count > 0 ? total_error / error_count : 0.0f;
    
    // Optimize parameters based on performance
    // If error is high, increase learning rate and decrease decay
    // If error is low, decrease learning rate (fine-tuning)
    
    float original_learning_rate = gh->learning_rate;
    float original_decay_rate = gh->decay_rate;
    float original_exec_threshold = gh->exec_threshold;
    
    // Modify parameters (simple optimization heuristic)
    if (avg_error > 0.5f) {
        // High error: increase learning rate, decrease decay
        gh->learning_rate = fminf(original_learning_rate * 1.1f, 0.01f);
        gh->decay_rate = fmaxf(original_decay_rate * 0.9f, 0.01f);
    } else if (avg_error < 0.1f) {
        // Low error: decrease learning rate (fine-tuning)
        gh->learning_rate = fmaxf(original_learning_rate * 0.9f, 0.0001f);
    }
    
    // Adjust exec threshold based on activation levels
    if (gh->avg_activation > 0.8f) {
        // High activation: increase threshold to reduce EXEC frequency
        gh->exec_threshold = fminf(original_exec_threshold * 1.1f, 2.0f);
    } else if (gh->avg_activation < 0.2f) {
        // Low activation: decrease threshold to increase EXEC frequency
        gh->exec_threshold = fmaxf(original_exec_threshold * 0.9f, 0.5f);
    }
    
    // Store modification marker
    if (g->blob && g->blob_capacity > 250) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)g->blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            g->blob[250] = 0xBB;  // Meta-optimization marker
            // Store original and new values for comparison
            memcpy(&g->blob[251], &original_learning_rate, sizeof(float));
            memcpy(&g->blob[255], &gh->learning_rate, sizeof(float));
            memcpy(&g->blob[259], &original_decay_rate, sizeof(float));
            memcpy(&g->blob[263], &gh->decay_rate, sizeof(float));
            memcpy(&g->blob[267], &original_exec_threshold, sizeof(float));
            memcpy(&g->blob[271], &gh->exec_threshold, sizeof(float));
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
}

int main(int argc, char **argv) {
    const char *file_path = "test_meta_learning.m";
    
    printf("========================================\n");
    printf("META-LEARNING TEST\n");
    printf("========================================\n\n");
    printf("Goal: Test if EXEC nodes can modify physics parameters\n");
    printf("This tests self-optimization capabilities.\n\n");
    
    // Create new file with initial parameters
    GraphParams params;
    params.decay_rate = 0.1f;
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.exec_threshold = 1.0f;
    params.learning_rate = 0.001f;
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    unlink(file_path);
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return 1;
    }
    printf("✓ Created %s\n", file_path);
    printf("  Initial parameters:\n");
    printf("    decay_rate: %.4f\n", params.decay_rate);
    printf("    learning_rate: %.4f\n", params.learning_rate);
    printf("    exec_threshold: %.4f\n", params.exec_threshold);
    
    // Map file
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    printf("✓ Mapped file\n");
    
    // Read initial parameters from graph header
    GraphHeaderDisk *gh = file.graph_header;
    float initial_decay = gh->decay_rate;
    float initial_learning = gh->learning_rate;
    float initial_threshold = gh->exec_threshold;
    
    printf("\n  Parameters from graph header:\n");
    printf("    decay_rate: %.6f\n", initial_decay);
    printf("    learning_rate: %.6f\n", initial_learning);
    printf("    exec_threshold: %.6f\n", initial_threshold);
    
    // Write meta-optimizer function
    printf("\nStep 1: Writing meta-optimizer function...\n");
    void *optimizer_ptr = (void*)meta_optimizer;
    size_t optimizer_size = 256;
    
    uint64_t optimizer_offset = melvin_write_machine_code(&file, (uint8_t*)optimizer_ptr, optimizer_size);
    if (optimizer_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write optimizer code\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Optimizer code written at offset %llu\n", (unsigned long long)optimizer_offset);
    
    // Create optimizer EXEC node
    uint64_t optimizer_node_id = melvin_create_executable_node(&file, optimizer_offset, optimizer_size);
    if (optimizer_node_id == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to create optimizer EXEC node\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Created optimizer EXEC node ID: %llu\n", (unsigned long long)optimizer_node_id);
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n");
    
    // Create some data to generate prediction errors
    printf("\nStep 2: Creating data patterns to generate prediction errors...\n");
    const uint64_t CH_TEXT = 1;
    
    // Ingest some data to create nodes and edges
    for (int i = 0; i < 20; i++) {
        ingest_byte(&rt, CH_TEXT, 'A' + (i % 26), 1.0f);
        melvin_process_n_events(&rt, 30);
    }
    
    printf("✓ Data ingested\n");
    
    // Check initial prediction error
    float initial_avg_error = 0.0f;
    uint64_t error_count = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *node = &file.nodes[i];
        if (node->id != UINT64_MAX && node->prediction_error > 0.0f) {
            initial_avg_error += node->prediction_error;
            error_count++;
        }
    }
    if (error_count > 0) {
        initial_avg_error /= error_count;
    }
    printf("  Initial average prediction error: %.6f\n", initial_avg_error);
    
    // Activate optimizer to modify parameters
    printf("\nStep 3: Activating meta-optimizer...\n");
    MelvinEvent ev = {
        .type = EV_NODE_DELTA,
        .node_id = optimizer_node_id,
        .value = 1.5f
    };
    melvin_event_enqueue(&rt.evq, &ev);
    
    printf("Processing events (optimizer should modify parameters)...\n");
    melvin_process_n_events(&rt, 50);
    
    // Re-read parameters from graph header
    printf("\nStep 4: Checking if parameters were modified...\n");
    
    // Need to re-map or sync to see changes
    melvin_m_sync(&file);
    
    float final_decay = gh->decay_rate;
    float final_learning = gh->learning_rate;
    float final_threshold = gh->exec_threshold;
    
    printf("  Parameters after optimization:\n");
    printf("    decay_rate: %.6f (was %.6f)\n", final_decay, initial_decay);
    printf("    learning_rate: %.6f (was %.6f)\n", final_learning, initial_learning);
    printf("    exec_threshold: %.6f (was %.6f)\n", final_threshold, initial_threshold);
    
    // Check modification marker
    uint8_t mod_marker = 0;
    float stored_orig_learning = 0, stored_new_learning = 0;
    float stored_orig_decay = 0, stored_new_decay = 0;
    float stored_orig_threshold = 0, stored_new_threshold = 0;
    
    if (file.blob && file.blob_capacity > 250) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t blob_start = (uintptr_t)file.blob;
        uintptr_t page_start = blob_start & ~(page_size - 1);
        
        if (mprotect((void*)page_start, page_size, PROT_READ | PROT_WRITE) == 0) {
            mod_marker = file.blob[250];
            if (file.blob_capacity > 275) {
                memcpy(&stored_orig_learning, &file.blob[251], sizeof(float));
                memcpy(&stored_new_learning, &file.blob[255], sizeof(float));
                memcpy(&stored_orig_decay, &file.blob[259], sizeof(float));
                memcpy(&stored_new_decay, &file.blob[263], sizeof(float));
                memcpy(&stored_orig_threshold, &file.blob[267], sizeof(float));
                memcpy(&stored_new_threshold, &file.blob[271], sizeof(float));
            }
            mprotect((void*)page_start, page_size, PROT_READ | PROT_EXEC);
        }
    }
    
    // Evaluate results
    printf("\n========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    
    int success = 1;
    int params_changed = 0;
    
    if (mod_marker == 0xBB) {
        printf("✓ PASS: Meta-optimization marker set (optimizer executed)\n");
    } else {
        printf("⚠ WARNING: Meta-optimization marker not set\n");
        printf("  Marker value: 0x%02X\n", mod_marker);
    }
    
    if (fabsf(final_decay - initial_decay) > 0.0001f) {
        printf("✓ PASS: decay_rate modified (%.6f -> %.6f)\n", initial_decay, final_decay);
        params_changed = 1;
    } else {
        printf("⚠ decay_rate unchanged (%.6f)\n", final_decay);
    }
    
    if (fabsf(final_learning - initial_learning) > 0.0001f) {
        printf("✓ PASS: learning_rate modified (%.6f -> %.6f)\n", initial_learning, final_learning);
        params_changed = 1;
    } else {
        printf("⚠ learning_rate unchanged (%.6f)\n", final_learning);
    }
    
    if (fabsf(final_threshold - initial_threshold) > 0.0001f) {
        printf("✓ PASS: exec_threshold modified (%.6f -> %.6f)\n", initial_threshold, final_threshold);
        params_changed = 1;
    } else {
        printf("⚠ exec_threshold unchanged (%.6f)\n", final_threshold);
    }
    
    if (!params_changed) {
        printf("\n✗ FAIL: No parameters were modified\n");
        success = 0;
    }
    
    // Final evaluation
    printf("\n========================================\n");
    if (success && params_changed) {
        printf("✅ TEST PASSED: Meta-learning works!\n");
        printf("EXEC nodes can modify physics parameters.\n");
        printf("This enables self-optimization.\n");
    } else {
        printf("❌ TEST FAILED: Meta-learning did not work\n");
        printf("EXEC nodes may not be able to modify parameters.\n");
    }
    printf("========================================\n");
    
    // Sync and close
    melvin_m_sync(&file);
    runtime_cleanup(&rt);
    close_file(&file);
    
    return success ? 0 : 1;
}


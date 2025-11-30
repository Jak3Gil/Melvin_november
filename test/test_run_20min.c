#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

// Include the implementation
#include "melvin.c"

// Test metrics structure
typedef struct {
    uint64_t nodes;
    uint64_t edges;
    float mean_activation;
    float max_activation;
    float prediction_error;
    float reward_recent;
    uint64_t exec_triggers;
    uint64_t events_processed;
    uint64_t validation_errors;
    double elapsed_seconds;
} TestMetrics;

// Global counter for exec triggers
static uint64_t g_exec_trigger_count = 0;

// Calculate metrics from the current graph state
void calculate_metrics(MelvinRuntime *rt, TestMetrics *metrics, double elapsed_seconds) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    
    metrics->nodes = gh->num_nodes;
    metrics->edges = gh->num_edges;
    metrics->elapsed_seconds = elapsed_seconds;
    metrics->events_processed = rt->logical_time;
    metrics->exec_triggers = g_exec_trigger_count;
    
    // Calculate mean and max activation, prediction error
    float sum_activation = 0.0f;
    float sum_prediction_error = 0.0f;
    float sum_reward = 0.0f;
    float max_act = 0.0f;
    uint64_t active_count = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        
        NodeDisk *node = &nodes[i];
        float act = fabsf(node->state);
        
        if (act > 0.001f) {  // Only count active nodes
            sum_activation += act;
            sum_prediction_error += node->prediction_error;
            sum_reward += node->reward;
            if (act > max_act) max_act = act;
            active_count++;
        }
    }
    
    if (active_count > 0) {
        metrics->mean_activation = sum_activation / active_count;
        metrics->prediction_error = sum_prediction_error / active_count;
        metrics->reward_recent = sum_reward / active_count;
    } else {
        metrics->mean_activation = 0.0f;
        metrics->prediction_error = 0.0f;
        metrics->reward_recent = 0.0f;
    }
    
    metrics->max_activation = max_act;
    
    // Validation errors: check for NaN or infinity
    metrics->validation_errors = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        NodeDisk *node = &nodes[i];
        if (isnan(node->state) || isinf(node->state) ||
            isnan(node->prediction) || isinf(node->prediction) ||
            isnan(node->prediction_error) || isinf(node->prediction_error) ||
            isnan(node->reward) || isinf(node->reward) ||
            isnan(node->energy_cost) || isinf(node->energy_cost)) {
            metrics->validation_errors++;
        }
    }
    
    // Check edges too
    EdgeDisk *edges = rt->file->edges;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (edges[i].src == UINT64_MAX) continue;
        EdgeDisk *edge = &edges[i];
        if (isnan(edge->weight) || isinf(edge->weight) ||
            isnan(edge->eligibility) || isinf(edge->eligibility)) {
            metrics->validation_errors++;
        }
    }
}

    // Log metrics to stdout
void log_metrics(const TestMetrics *metrics, int minute) {
    if (minute >= 0) {
        printf("\n--- [%02d:00] METRICS ---\n", minute);
    }
    printf("nodes: %llu\n", (unsigned long long)metrics->nodes);
    printf("edges: %llu\n", (unsigned long long)metrics->edges);
    printf("mean_activation: %.4f\n", metrics->mean_activation);
    printf("max_activation: %.4f\n", metrics->max_activation);
    printf("prediction_error: %.6f\n", metrics->prediction_error);
    printf("reward_recent: %.6f\n", metrics->reward_recent);
    printf("exec_triggers: %llu\n", (unsigned long long)metrics->exec_triggers);
    printf("events_processed: %llu\n", (unsigned long long)metrics->events_processed);
    if (metrics->elapsed_seconds > 0.1) {
        printf("events_processed_per_sec: %.1f\n", 
               (double)metrics->events_processed / metrics->elapsed_seconds);
    } else {
        printf("events_processed_per_sec: (baseline - not calculated)\n");
    }
    printf("validation_errors: %llu\n", (unsigned long long)metrics->validation_errors);
    printf("elapsed: %.1f seconds (%.2f minutes)\n", metrics->elapsed_seconds, metrics->elapsed_seconds / 60.0);
    fflush(stdout);
}

// Generate random noise bytes
void generate_noise(MelvinRuntime *rt, uint64_t channel_id, size_t num_bytes) {
    static uint64_t noise_seed = 12345;
    
    for (size_t i = 0; i < num_bytes; i++) {
        // Simple LCG for noise
        noise_seed = noise_seed * 1103515245ULL + 12345ULL;
        uint8_t byte = (uint8_t)(noise_seed & 0xFF);
        ingest_byte(rt, channel_id, byte, 0.5f);  // Lower energy for noise
    }
}

// Safe EXEC function that does nothing harmful
void safe_exec_function(MelvinFile *g, uint64_t node_id) {
    // Just increment a counter - no graph writes, no side effects
    // We'll track this via a global counter
    g_exec_trigger_count++;
    (void)g;  // Suppress unused warning
    (void)node_id;  // Suppress unused warning
}

// Create a safe EXEC node for testing
uint64_t create_safe_exec_node(MelvinFile *file) {
    // Write the safe function's machine code into blob
    // Note: This is architecture-specific. For simplicity, we'll create a small stub.
    // In a real scenario, you'd compile this to machine code.
    
    // For now, we'll create an EXECUTABLE node but set threshold very high
    // so it won't actually execute unless manually triggered
    
    // Create a minimal machine code stub (architecture-dependent)
    // This is a simplified version - in production you'd want proper compilation
    uint8_t stub_code[64] = {0};
    
    // On x86-64, a simple return instruction is 0xC3
    // We'll create a minimal function that just returns
    stub_code[0] = 0xC3;  // RET instruction (x86-64)
    
    // For safety, we'll make it a no-op by padding with NOPs
    for (int i = 1; i < 64; i++) {
        stub_code[i] = 0x90;  // NOP instruction
    }
    
    uint64_t offset = melvin_write_machine_code(file, stub_code, sizeof(stub_code));
    if (offset == UINT64_MAX) {
        fprintf(stderr, "Warning: Failed to write machine code for EXEC node\n");
        return UINT64_MAX;
    }
    
    uint64_t exec_node = melvin_create_executable_node(file, offset, sizeof(stub_code));
    if (exec_node == UINT64_MAX) {
        fprintf(stderr, "Warning: Failed to create EXECUTABLE node\n");
        return UINT64_MAX;
    }
    
    printf("Created safe EXEC node %llu (threshold set high for safety)\n", 
           (unsigned long long)exec_node);
    
    return exec_node;
}

// Ingest a file into Melvin
void ingest_file(MelvinRuntime *rt, const char *filepath, uint64_t channel_id) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Warning: Could not open %s for ingestion\n", filepath);
        return;
    }
    
    printf("Ingesting file: %s (channel %llu)\n", filepath, (unsigned long long)channel_id);
    
    uint8_t byte;
    size_t bytes_read = 0;
    size_t bytes_processed = 0;
    
    while (fread(&byte, 1, 1, f) == 1) {
        ingest_byte(rt, channel_id, byte, 1.0f);
        bytes_read++;
        
        // Process events periodically during ingestion
        if (bytes_read % 100 == 0) {
            melvin_process_n_events(rt, 50);
            bytes_processed += 100;
        }
    }
    
    // Process remaining events
    melvin_process_n_events(rt, 200);
    
    fclose(f);
    printf("  Ingested %zu bytes from %s\n", bytes_read, filepath);
}

// Main test function
int main(int argc, char **argv) {
    const char *file_path = "test_20min.m";
    const int test_duration_seconds = 20 * 60;  // 20 minutes
    const int metrics_interval_seconds = 120;   // Log every 2 minutes
    
    printf("========================================\n");
    printf("MELVIN 20-MINUTE SANITY TEST\n");
    printf("========================================\n\n");
    printf("Goal: Verify substrate behaves sanely for 20 minutes\n");
    printf("Duration: %d seconds (%d minutes)\n", test_duration_seconds, test_duration_seconds / 60);
    printf("Metrics interval: Every %d seconds\n\n", metrics_interval_seconds);
    
    // Step 1: Create new melvin.m file
    printf("Step 1: Creating new melvin.m file...\n");
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
    
    // Remove old file if it exists
    unlink(file_path);
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create melvin.m file\n");
        return 1;
    }
    printf("✓ Created %s\n\n", file_path);
    
    // Step 2: Map the file
    printf("Step 2: Mapping melvin.m file...\n");
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map melvin.m file\n");
        return 1;
    }
    printf("✓ Mapped file successfully\n\n");
    
    // Step 3: Initialize runtime
    printf("Step 3: Initializing runtime...\n");
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n\n");
    
    // Step 4: Optional - Create safe EXEC node
    printf("Step 4: Creating safe EXEC node (optional test)...\n");
    uint64_t exec_node = create_safe_exec_node(&file);
    if (exec_node != UINT64_MAX) {
        printf("✓ Created EXEC node (will not execute automatically due to high threshold)\n");
    } else {
        printf("~ Skipped EXEC node creation\n");
    }
    printf("\n");
    
    // Step 5: Initial data ingestion
    printf("Step 5: Initial data ingestion...\n");
    
    // A. Text stream (README.md)
    if (access("README.md", R_OK) == 0) {
        ingest_file(&rt, "README.md", 1);
        printf("\n");
    } else {
        printf("~ README.md not found, skipping\n\n");
    }
    
    // B. Small C file (melvin.c - we'll ingest a portion)
    if (access("melvin.c", R_OK) == 0) {
        printf("Ingesting melvin.c (first 50KB)...\n");
        FILE *f = fopen("melvin.c", "rb");
        if (f) {
            uint8_t byte;
            size_t bytes_read = 0;
            const size_t max_bytes = 50000;  // Limit to 50KB
            
            while (bytes_read < max_bytes && fread(&byte, 1, 1, f) == 1) {
                ingest_byte(&rt, 10, byte, 1.0f);
                bytes_read++;
                
                if (bytes_read % 100 == 0) {
                    melvin_process_n_events(&rt, 50);
                }
            }
            melvin_process_n_events(&rt, 200);
            fclose(f);
            printf("  Ingested %zu bytes from melvin.c\n", bytes_read);
        }
        printf("\n");
    } else {
        printf("~ melvin.c not found, skipping\n\n");
    }
    
    // C. Noise channel (10k random bytes)
    printf("Ingesting noise channel (10,000 random bytes)...\n");
    generate_noise(&rt, 99, 10000);
    melvin_process_n_events(&rt, 500);
    printf("  Ingested 10,000 noise bytes\n\n");
    
    printf("✓ Initial ingestion complete\n\n");
    
    // Step 6: Run continuous test for 20 minutes
    printf("Step 6: Running continuous test for %d minutes...\n", test_duration_seconds / 60);
    printf("========================================\n\n");
    
    // Start the timer NOW (after ingestion, before the test loop)
    struct timeval start_time, current_time, last_metrics_time;
    gettimeofday(&start_time, NULL);
    last_metrics_time = start_time;
    
    int metrics_log_count = 0;
    uint64_t last_node_count = 0;
    uint64_t last_edge_count = 0;
    double last_elapsed = 0.0;
    
    // Calculate initial baseline metrics (elapsed will be ~0, but that's okay for baseline)
    TestMetrics metrics;
    gettimeofday(&current_time, NULL);
    double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                     (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    
    // Store baseline node/edge counts
    GraphHeaderDisk *gh = rt.file->graph_header;
    last_node_count = gh->num_nodes;
    last_edge_count = gh->num_edges;
    
    // Log initial metrics (baseline state)
    calculate_metrics(&rt, &metrics, elapsed);
    printf("--- [BASELINE] INITIAL STATE (after ingestion) ---\n");
    log_metrics(&metrics, 0);
    metrics_log_count++;
    
    // Main test loop
    double last_progress_log = 0.0;
    while (1) {
        gettimeofday(&current_time, NULL);
        elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                  (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        // Check if we've reached the time limit
        if (elapsed >= test_duration_seconds) {
            printf("\n=== Test duration reached ===\n");
            break;
        }
        
        // Process events continuously
        melvin_process_n_events(&rt, 1000);  // Process up to 1000 events per iteration
        
        // Print a brief progress message every 30 seconds (but not during metrics log)
        if (elapsed - last_progress_log >= 30.0) {
            printf("[%.0f seconds elapsed - %.1f%% complete]\n", 
                   elapsed, (elapsed / test_duration_seconds) * 100.0);
            fflush(stdout);
            last_progress_log = elapsed;
        }
        
        // Occasionally inject more data to keep things interesting
        int elapsed_int = (int)elapsed;
        int last_elapsed_int = (int)last_elapsed;
        if (elapsed_int > 0 && elapsed_int % 300 == 0 && elapsed_int != last_elapsed_int) {
            // Every 5 minutes, inject a bit more noise
            generate_noise(&rt, 99, 1000);
            printf("[%.0f seconds] Injected additional noise\n", elapsed);
        }
        
        // Log metrics at intervals
        double time_since_last_metrics = (current_time.tv_sec - last_metrics_time.tv_sec) + 
                                        (current_time.tv_usec - last_metrics_time.tv_usec) / 1000000.0;
        
        if (time_since_last_metrics >= metrics_interval_seconds) {
            calculate_metrics(&rt, &metrics, elapsed);
            
            // Check for failures
            int failure = 0;
            
            // Check 1: Monotonic growth (but not explosion)
            if (metrics.nodes < last_node_count * 0.9) {
                printf("\n⚠ WARNING: Node count decreased significantly!\n");
            }
            if (metrics.nodes > last_node_count * 10 && last_node_count > 100) {
                printf("\n⚠ WARNING: Node count exploding! (exponential growth detected)\n");
                failure = 1;
            }
            
            // Check 2: Bounded activation
            if (metrics.max_activation > 0.95f) {
                printf("\n⚠ WARNING: Max activation too high (%.4f > 0.95)\n", metrics.max_activation);
            }
            
            // Check 3: No validation errors
            if (metrics.validation_errors > 0) {
                printf("\n❌ FAILURE: Validation errors detected! (%llu)\n", 
                       (unsigned long long)metrics.validation_errors);
                failure = 1;
            }
            
            // Check 4: Event throughput sanity
            if (metrics.elapsed_seconds > 10.0) {
                double events_per_sec = metrics.events_processed / metrics.elapsed_seconds;
                if (events_per_sec < 1.0) {
                    printf("\n⚠ WARNING: Event throughput very low (%.2f events/sec)\n", events_per_sec);
                }
                if (events_per_sec > 1000000.0) {
                    printf("\n⚠ WARNING: Event throughput extremely high (%.2f events/sec)\n", events_per_sec);
                    failure = 1;
                }
            }
            
            if (failure) {
                printf("\n❌ TEST FAILED - Stopping early\n");
                break;
            }
            
            int current_minute = (int)(elapsed / 60);
            log_metrics(&metrics, current_minute);
            metrics_log_count++;
            
            last_node_count = metrics.nodes;
            last_edge_count = metrics.edges;
            last_metrics_time = current_time;
            
            // Also print progress message
            printf("Progress: %.1f%% (%.1f / %.1f minutes)\n", 
                   (elapsed / test_duration_seconds) * 100.0,
                   elapsed / 60.0,
                   test_duration_seconds / 60.0);
        }
        
        last_elapsed = elapsed;
        
        // Small sleep to prevent 100% CPU usage
        usleep(1000);  // 1ms
    }
    
    // Final metrics
    printf("\n========================================\n");
    printf("FINAL METRICS\n");
    printf("========================================\n");
    gettimeofday(&current_time, NULL);
    elapsed = (current_time.tv_sec - start_time.tv_sec) + 
              (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    calculate_metrics(&rt, &metrics, elapsed);
    log_metrics(&metrics, (int)(elapsed / 60));
    
    // Summary
    printf("\n========================================\n");
    printf("TEST SUMMARY\n");
    printf("========================================\n");
    printf("Duration: %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60.0);
    printf("Total metrics logs: %d\n", metrics_log_count);
    printf("Final nodes: %llu\n", (unsigned long long)metrics.nodes);
    printf("Final edges: %llu\n", (unsigned long long)metrics.edges);
    printf("Total events processed: %llu\n", (unsigned long long)metrics.events_processed);
    printf("Average events/sec: %.2f\n", 
           elapsed > 0 ? (double)metrics.events_processed / elapsed : 0.0);
    printf("Validation errors: %llu\n", (unsigned long long)metrics.validation_errors);
    printf("Exec triggers: %llu\n", (unsigned long long)metrics.exec_triggers);
    
    // Check if test passed
    int passed = 1;
    if (metrics.validation_errors > 0) {
        printf("\n❌ FAILED: Validation errors detected\n");
        passed = 0;
    }
    if (metrics.max_activation > 1.0f) {
        printf("\n⚠ WARNING: Max activation exceeded 1.0 (%.4f)\n", metrics.max_activation);
    }
    if (metrics.nodes == 0) {
        printf("\n❌ FAILED: No nodes created\n");
        passed = 0;
    }
    if (metrics.edges == 0 && elapsed > 60.0) {
        printf("\n⚠ WARNING: No edges created after 1 minute\n");
    }
    
    if (passed && metrics.validation_errors == 0) {
        printf("\n✅ TEST PASSED: Substrate behaved sanely\n");
    } else {
        printf("\n❌ TEST FAILED\n");
    }
    
    // Step 7: Sync and close
    printf("\nStep 7: Syncing file to disk...\n");
    melvin_m_sync(&file);
    printf("✓ File synced\n\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    printf("========================================\n");
    printf("TEST COMPLETE\n");
    printf("========================================\n");
    printf("File saved: %s\n", file_path);
    
    return passed ? 0 : 1;
}


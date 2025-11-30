/*
 * TEST: EXEC Node Efficiency - Connect EXEC to patterns and see if it gains stability
 * 
 * This test:
 * 1. Creates an EXEC node
 * 2. Connects it to patterns (so it receives traffic)
 * 3. Activates patterns to feed traffic to EXEC
 * 4. Measures if EXEC gains stability when it's efficient
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "melvin.c"

#define TEST_FILE "test_exec_efficiency.m"
#define TEST_DURATION_SECONDS 180  // 3 minutes
#define PROGRESS_INTERVAL_SECONDS 10
#define INGEST_CHANNEL 0

// Print progress bar
static void print_progress(int current, int total, const char *label) {
    int bar_width = 50;
    float progress = (float)current / (float)total;
    int pos = (int)(bar_width * progress);
    
    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d%% - %s", (int)(progress * 100), label);
    fflush(stdout);
}

// Get current time in seconds
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Helper to create edge if it doesn't exist
static void ensure_edge(MelvinFile *file, uint64_t from_id, uint64_t to_id, float weight) {
    if (edge_exists_between(file, from_id, to_id)) return;
    create_edge_between(file, from_id, to_id, weight);
}

int main() {
    printf("========================================\n");
    printf("EXEC NODE EFFICIENCY TEST\n");
    printf("========================================\n\n");
    
    printf("Goal: Connect EXEC to patterns and see if it gains stability\n");
    printf("      when it's efficient (low complexity, low FE per traffic)\n\n");
    
    srand(time(NULL));
    unlink(TEST_FILE);
    
    // Initialize
    printf("Initializing...\n");
    GraphParams params;
    init_default_params(&params);
    params.decay_rate = 0.90f;
    params.learning_rate = 0.02f;
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        return 1;
    }
    printf("  ✓ Initialized\n\n");
    
    // Step 1: Create EXEC node
    printf("Step 1: Creating EXEC node...\n");
    const uint8_t ARM64_ADD[] = {
        0x00, 0x00, 0x01, 0x8b,  // add x0, x0, x1
        0xc0, 0x03, 0x5f, 0xd6   // ret
    };
    uint64_t add_offset = melvin_write_machine_code(&file, ARM64_ADD, sizeof(ARM64_ADD));
    uint64_t exec_id = melvin_create_executable_node(&file, add_offset, sizeof(ARM64_ADD));
    printf("  ✓ EXEC node created: %llu\n", (unsigned long long)exec_id);
    
    // Step 2: Feed patterns to create pattern nodes
    printf("\nStep 2: Feeding patterns to create pattern nodes...\n");
    const char *patterns[] = {"ABC", "DEF", "GHI", "JKL", "MNO"};
    for (int i = 0; i < 50; i++) {
        const char *pattern = patterns[i % (sizeof(patterns) / sizeof(patterns[0]))];
        for (int j = 0; pattern[j] != '\0'; j++) {
            ingest_byte(&rt, INGEST_CHANNEL, pattern[j], 1.0f);
            melvin_process_n_events(&rt, 5);
        }
    }
    printf("  ✓ Patterns created\n");
    
    // Step 3: Connect EXEC to patterns (so it receives traffic)
    printf("\nStep 3: Connecting EXEC to patterns...\n");
    GraphHeaderDisk *gh = file.graph_header;
    EdgeDisk *edges = file.edges;
    uint64_t connections_made = 0;
    
    // Find pattern nodes and connect them to EXEC
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file.nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            // This is a pattern node - connect it to EXEC
            ensure_edge(&file, n->id, exec_id, 0.3f);
            connections_made++;
            if (connections_made >= 10) break;  // Connect to first 10 patterns
        }
    }
    printf("  ✓ Connected EXEC to %llu pattern nodes\n", (unsigned long long)connections_made);
    
    // Step 4: Measure initial state
    printf("\nStep 4: Initial state...\n");
    uint64_t exec_idx = find_node_index_by_id(&file, exec_id);
    if (exec_idx != UINT64_MAX) {
        NodeDisk *exec = &file.nodes[exec_idx];
        printf("  EXEC node:\n");
        printf("    Complexity: %.4f\n", 0.1f * (float)exec->out_degree + 0.01f * (float)exec->payload_len);
        printf("    Traffic EMA: %.6f\n", exec->traffic_ema);
        printf("    FE EMA: %.6f\n", exec->fe_ema);
        printf("    Efficiency: %.6f\n", exec->fe_ema / (exec->traffic_ema + 0.001f));
        printf("    Stability: %.6f\n", exec->stability);
    }
    
    // Step 5: Main loop - activate patterns to feed traffic to EXEC
    printf("\nStep 5: Running main loop (%d seconds)...\n", TEST_DURATION_SECONDS);
    double start_time = get_time();
    double last_progress_time = start_time;
    int progress_counter = 0;
    int total_progress_steps = TEST_DURATION_SECONDS / PROGRESS_INTERVAL_SECONDS;
    
    uint64_t activation_count = 0;
    
    while (1) {
        double current_time = get_time();
        double elapsed = current_time - start_time;
        
        if (elapsed >= TEST_DURATION_SECONDS) break;
        
        // Print progress
        if (current_time - last_progress_time >= PROGRESS_INTERVAL_SECONDS) {
            progress_counter++;
            uint64_t exec_idx = find_node_index_by_id(&file, exec_id);
            if (exec_idx != UINT64_MAX) {
                NodeDisk *exec = &file.nodes[exec_idx];
                char label[128];
                snprintf(label, sizeof(label), "EXEC: eff=%.3f, stab=%.3f", 
                        exec->fe_ema / (exec->traffic_ema + 0.001f), exec->stability);
                print_progress(progress_counter, total_progress_steps, label);
            } else {
                print_progress(progress_counter, total_progress_steps, "Processing...");
            }
            last_progress_time = current_time;
        }
        
        // Activate patterns to feed traffic to EXEC
        if (activation_count % 20 == 0) {
            // Ingest a pattern to activate it
            const char *pattern = patterns[activation_count % (sizeof(patterns) / sizeof(patterns[0]))];
            for (int j = 0; pattern[j] != '\0'; j++) {
                ingest_byte(&rt, INGEST_CHANNEL, pattern[j], 1.0f);
                melvin_process_n_events(&rt, 10);
            }
        }
        activation_count++;
        
        // Process events
        melvin_process_n_events(&rt, 20);
        
        // Small sleep
        usleep(10000);
    }
    
    printf("\n\n"); // New line after progress bar
    
    // Step 6: Final state
    printf("Step 6: Final state...\n");
    exec_idx = find_node_index_by_id(&file, exec_id);
    if (exec_idx != UINT64_MAX) {
        NodeDisk *exec = &file.nodes[exec_idx];
        
        // Count edges
        uint64_t deg_in = 0;
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            if (edges[e].src == UINT64_MAX) continue;
            if (edges[e].dst == exec_id) deg_in++;
        }
        float complexity = 0.1f * (float)(deg_in + exec->out_degree) + 0.01f * (float)exec->payload_len;
        float efficiency = exec->fe_ema / (exec->traffic_ema + 0.001f);
        
        printf("  EXEC node:\n");
        printf("    Degree: %llu (in: %llu, out: %llu)\n", 
               (unsigned long long)(deg_in + exec->out_degree), 
               (unsigned long long)deg_in, (unsigned long long)exec->out_degree);
        printf("    Payload: %u bytes\n", exec->payload_len);
        printf("    Complexity: %.6f\n", complexity);
        printf("    Activation: %.6f\n", exec->state);
        printf("    Traffic EMA: %.6f\n", exec->traffic_ema);
        printf("    FE EMA: %.6f\n", exec->fe_ema);
        printf("    Efficiency: %.6f (lower = better)\n", efficiency);
        printf("    Stability: %.6f (higher = better)\n", exec->stability);
    }
    
    // Compare with pattern nodes
    printf("\nStep 7: Comparison with pattern nodes...\n");
    uint64_t pattern_count = 0;
    float total_pattern_complexity = 0.0f;
    float total_pattern_efficiency = 0.0f;
    float total_pattern_stability = 0.0f;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file.nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            pattern_count++;
            
            uint64_t deg_in = 0;
            for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
                if (edges[e].src == UINT64_MAX) continue;
                if (edges[e].dst == n->id) deg_in++;
            }
            float complexity = 0.1f * (float)(deg_in + n->out_degree) + 0.01f * (float)n->payload_len;
            float efficiency = n->fe_ema / (n->traffic_ema + 0.001f);
            
            total_pattern_complexity += complexity;
            total_pattern_efficiency += efficiency;
            total_pattern_stability += n->stability;
        }
    }
    
    if (pattern_count > 0) {
        printf("  Pattern nodes (avg of %llu):\n", (unsigned long long)pattern_count);
        printf("    Avg complexity: %.6f\n", total_pattern_complexity / pattern_count);
        printf("    Avg efficiency: %.6f\n", total_pattern_efficiency / pattern_count);
        printf("    Avg stability: %.6f\n", total_pattern_stability / pattern_count);
    }
    
    printf("\n========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    if (exec_idx != UINT64_MAX) {
        NodeDisk *exec = &file.nodes[exec_idx];
        float exec_efficiency = exec->fe_ema / (exec->traffic_ema + 0.001f);
        float pattern_avg_efficiency = pattern_count > 0 ? total_pattern_efficiency / pattern_count : 0.0f;
        
        printf("Key question: Does EXEC gain stability when it's efficient?\n\n");
        
        if (exec->traffic_ema > 0.0f) {
            printf("✓ EXEC received traffic: %.6f\n", exec->traffic_ema);
            if (exec_efficiency < pattern_avg_efficiency) {
                printf("✓ EXEC is MORE efficient than patterns (%.6f vs %.6f)\n", 
                       exec_efficiency, pattern_avg_efficiency);
                if (exec->stability > 0.1f) {
                    printf("✓ EXEC gained stability: %.6f\n", exec->stability);
                    printf("\n✅ CONCLUSION: EXEC can gain stability when efficient!\n");
                } else {
                    printf("⚠️  EXEC stability is low: %.6f (may need more time)\n", exec->stability);
                }
            } else {
                printf("⚠️  EXEC efficiency (%.6f) is similar to patterns (%.6f)\n",
                       exec_efficiency, pattern_avg_efficiency);
            }
        } else {
            printf("❌ EXEC received no traffic (not connected properly)\n");
        }
    }
    
    printf("\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    return 0;
}


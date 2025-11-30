/*
 * TEST: Long-running efficiency test with progress bar
 * 
 * This test runs for a longer duration to observe:
 * 1. Stability building up over time
 * 2. Complexity penalty affecting large structures
 * 3. Efficient structures gaining stability
 * 
 * Goal: Unnecessary complexity is penalized, not that EXEC always wins
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

#define TEST_FILE "test_efficiency_long.m"
#define TEST_DURATION_SECONDS 300  // 5 minutes
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

// Measure summary statistics
static void print_stats(MelvinFile *file, const char *label) {
    GraphHeaderDisk *gh = file->graph_header;
    EdgeDisk *edges = file->edges;
    
    uint64_t pattern_count = 0;
    uint64_t exec_count = 0;
    float total_pattern_complexity = 0.0f;
    float total_pattern_efficiency = 0.0f;
    float total_pattern_stability = 0.0f;
    float total_exec_complexity = 0.0f;
    float total_exec_efficiency = 0.0f;
    float total_exec_stability = 0.0f;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        
        // Count edges
        uint64_t deg_in = 0;
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            if (edges[e].src == UINT64_MAX) continue;
            if (edges[e].dst == n->id) deg_in++;
        }
        float complexity = 0.1f * (float)(deg_in + n->out_degree) + 0.01f * (float)n->payload_len;
        float eps_eff = 0.001f;
        float efficiency = (n->traffic_ema > 0.0f || n->fe_ema > 0.0f) ? 
                           n->fe_ema / (n->traffic_ema + eps_eff) : 0.0f;
        
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            // Pattern node
            pattern_count++;
            total_pattern_complexity += complexity;
            total_pattern_efficiency += efficiency;
            total_pattern_stability += n->stability;
        } else if (n->flags & NODE_FLAG_EXECUTABLE) {
            // EXEC node
            exec_count++;
            total_exec_complexity += complexity;
            total_exec_efficiency += efficiency;
            total_exec_stability += n->stability;
        }
    }
    
    printf("\n%s:\n", label);
    printf("  Total nodes: %llu, Edges: %llu\n", 
           (unsigned long long)gh->num_nodes, (unsigned long long)gh->num_edges);
    printf("  Pattern nodes: %llu\n", (unsigned long long)pattern_count);
    if (pattern_count > 0) {
        printf("    Avg complexity: %.4f\n", total_pattern_complexity / pattern_count);
        printf("    Avg efficiency: %.4f\n", total_pattern_efficiency / pattern_count);
        printf("    Avg stability: %.4f\n", total_pattern_stability / pattern_count);
    }
    printf("  EXEC nodes: %llu\n", (unsigned long long)exec_count);
    if (exec_count > 0) {
        printf("    Avg complexity: %.4f\n", total_exec_complexity / exec_count);
        printf("    Avg efficiency: %.4f\n", total_exec_efficiency / exec_count);
        printf("    Avg stability: %.4f\n", total_exec_stability / exec_count);
    }
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("LONG-RUNNING EFFICIENCY TEST\n");
    printf("========================================\n\n");
    
    printf("Duration: %d seconds (%.1f minutes)\n", TEST_DURATION_SECONDS, TEST_DURATION_SECONDS / 60.0);
    printf("Goal: Observe complexity penalty and stability building\n");
    printf("      Unnecessary complexity is penalized\n\n");
    
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
    
    // Create EXEC node
    printf("Creating EXEC node...\n");
    const uint8_t ARM64_ADD[] = {
        0x00, 0x00, 0x01, 0x8b,  // add x0, x0, x1
        0xc0, 0x03, 0x5f, 0xd6   // ret
    };
    uint64_t add_offset = melvin_write_machine_code(&file, ARM64_ADD, sizeof(ARM64_ADD));
    uint64_t exec_id = melvin_create_executable_node(&file, add_offset, sizeof(ARM64_ADD));
    printf("  ✓ EXEC node created: %llu\n\n", (unsigned long long)exec_id);
    
    // Initial stats
    print_stats(&file, "Initial State");
    
    // Main loop
    printf("Starting main loop...\n");
    double start_time = get_time();
    double last_progress_time = start_time;
    int progress_counter = 0;
    int total_progress_steps = TEST_DURATION_SECONDS / PROGRESS_INTERVAL_SECONDS;
    
    uint64_t ingest_counter = 0;
    const char *patterns[] = {"ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX", "YZ0", "123"};
    int pattern_idx = 0;
    
    while (1) {
        double current_time = get_time();
        double elapsed = current_time - start_time;
        
        if (elapsed >= TEST_DURATION_SECONDS) break;
        
        // Print progress every PROGRESS_INTERVAL_SECONDS
        if (current_time - last_progress_time >= PROGRESS_INTERVAL_SECONDS) {
            progress_counter++;
            print_progress(progress_counter, total_progress_steps, 
                          "Processing...");
            last_progress_time = current_time;
        }
        
        // Ingest some data
        if (ingest_counter % 10 == 0) {
            const char *pattern = patterns[pattern_idx % (sizeof(patterns) / sizeof(patterns[0]))];
            pattern_idx++;
            for (int i = 0; pattern[i] != '\0'; i++) {
                ingest_byte(&rt, INGEST_CHANNEL, pattern[i], 1.0f);
                melvin_process_n_events(&rt, 5);
            }
        }
        ingest_counter++;
        
        // Process events
        melvin_process_n_events(&rt, 20);
        
        // Small sleep to avoid burning CPU
        usleep(10000); // 10ms
    }
    
    printf("\n\n"); // New line after progress bar
    
    // Final stats
    print_stats(&file, "Final State");
    
    // Analysis
    printf("========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    printf("Key observations:\n");
    printf("  1. Complexity penalty: Large structures (many edges) should have higher complexity\n");
    printf("  2. Efficiency: Nodes with low FE per traffic should have lower efficiency scores\n");
    printf("  3. Stability: Efficient nodes should gain stability over time\n");
    printf("  4. Unnecessary complexity: Nodes with high complexity but low value should lose stability\n");
    printf("\n");
    
    printf("Expected behavior:\n");
    printf("  - Pattern nodes with many edges: Higher complexity, may lose stability if inefficient\n");
    printf("  - EXEC nodes with few edges: Lower complexity, may gain stability if efficient\n");
    printf("  - System penalizes unnecessary complexity, not pattern nodes in general\n");
    printf("\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    return 0;
}


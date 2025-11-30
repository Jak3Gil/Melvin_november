/*
 * TEST: Automatic Connection Discovery
 * 
 * This test verifies that the system automatically connects nodes
 * without manual intervention:
 * 1. Create EXEC node (not manually connected)
 * 2. Feed patterns
 * 3. Let system discover connections via co-activation
 * 4. Verify EXEC receives traffic automatically
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

#define TEST_FILE "test_auto_connection.m"
#define TEST_DURATION_SECONDS 180
#define PROGRESS_INTERVAL_SECONDS 10
#define INGEST_CHANNEL 0

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

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    printf("========================================\n");
    printf("AUTOMATIC CONNECTION DISCOVERY TEST\n");
    printf("========================================\n\n");
    
    printf("Goal: Verify system automatically connects EXEC to patterns\n");
    printf("      via co-activation (NO manual connections)\n\n");
    
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
    
    // Step 1: Create EXEC node (NOT manually connected)
    printf("Step 1: Creating EXEC node (NOT manually connected)...\n");
    const uint8_t ARM64_ADD[] = {
        0x00, 0x00, 0x01, 0x8b,  // add x0, x0, x1
        0xc0, 0x03, 0x5f, 0xd6   // ret
    };
    uint64_t add_offset = melvin_write_machine_code(&file, ARM64_ADD, sizeof(ARM64_ADD));
    uint64_t exec_id = melvin_create_executable_node(&file, add_offset, sizeof(ARM64_ADD));
    printf("  ✓ EXEC node created: %llu\n", (unsigned long long)exec_id);
    
    // Check initial connections
    GraphHeaderDisk *gh = file.graph_header;
    EdgeDisk *edges = file.edges;
    uint64_t initial_connections = 0;
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        if (edges[e].src == UINT64_MAX) continue;
        if (edges[e].dst == exec_id || edges[e].src == exec_id) {
            initial_connections++;
        }
    }
    printf("  Initial connections to EXEC: %llu\n", (unsigned long long)initial_connections);
    printf("\n");
    
    // Step 2: Feed patterns (this will create pattern nodes)
    printf("Step 2: Feeding patterns to create pattern nodes...\n");
    const char *patterns[] = {"ABC", "DEF", "GHI", "JKL", "MNO"};
    for (int i = 0; i < 50; i++) {
        const char *pattern = patterns[i % (sizeof(patterns) / sizeof(patterns[0]))];
        for (int j = 0; pattern[j] != '\0'; j++) {
            ingest_byte(&rt, INGEST_CHANNEL, pattern[j], 1.0f);
            melvin_process_n_events(&rt, 5);
        }
    }
    printf("  ✓ Patterns created\n");
    printf("  Pattern nodes: ");
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file.nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            pattern_count++;
        }
    }
    printf("%llu\n", (unsigned long long)pattern_count);
    printf("\n");
    
    // Step 3: Main loop - let system discover connections
    printf("Step 3: Running main loop - system will discover connections...\n");
    printf("  (No manual connections - purely co-activation based)\n\n");
    
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
            
            // Count connections to EXEC
            uint64_t connections = 0;
            for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
                if (edges[e].src == UINT64_MAX) continue;
                if (edges[e].dst == exec_id) connections++;
            }
            
            uint64_t exec_idx = find_node_index_by_id(&file, exec_id);
            if (exec_idx != UINT64_MAX) {
                NodeDisk *exec = &file.nodes[exec_idx];
                char label[128];
                snprintf(label, sizeof(label), "Connections: %llu, Traffic: %.3f, Stab: %.3f", 
                        (unsigned long long)connections, exec->traffic_ema, exec->stability);
                print_progress(progress_counter, total_progress_steps, label);
            } else {
                print_progress(progress_counter, total_progress_steps, "Processing...");
            }
            last_progress_time = current_time;
        }
        
        // Activate patterns to feed traffic (co-activation will discover connections)
        if (activation_count % 20 == 0) {
            const char *pattern = patterns[activation_count % (sizeof(patterns) / sizeof(patterns[0]))];
            for (int j = 0; pattern[j] != '\0'; j++) {
                ingest_byte(&rt, INGEST_CHANNEL, pattern[j], 1.0f);
                melvin_process_n_events(&rt, 10);
            }
        }
        activation_count++;
        
        // Process events (homeostasis sweep will trigger connection discovery)
        melvin_process_n_events(&rt, 20);
        
        usleep(10000);
    }
    
    printf("\n\n");
    
    // Step 4: Final analysis
    printf("Step 4: Final analysis...\n\n");
    
    uint64_t final_connections = 0;
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        if (edges[e].src == UINT64_MAX) continue;
        if (edges[e].dst == exec_id) final_connections++;
    }
    
    uint64_t exec_idx = find_node_index_by_id(&file, exec_id);
    if (exec_idx != UINT64_MAX) {
        NodeDisk *exec = &file.nodes[exec_idx];
        
        uint64_t deg_in = 0;
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            if (edges[e].src == UINT64_MAX) continue;
            if (edges[e].dst == exec_id) deg_in++;
        }
        
        printf("  EXEC node:\n");
        printf("    Initial connections: %llu\n", (unsigned long long)initial_connections);
        printf("    Final connections: %llu\n", (unsigned long long)final_connections);
        printf("    Connections discovered: %llu\n", 
               (unsigned long long)(final_connections - initial_connections));
        printf("    Traffic EMA: %.6f\n", exec->traffic_ema);
        printf("    FE EMA: %.6f\n", exec->fe_ema);
        printf("    Efficiency: %.6f\n", exec->fe_ema / (exec->traffic_ema + 0.001f));
        printf("    Stability: %.6f\n", exec->stability);
    }
    
    printf("\n========================================\n");
    printf("ANALYSIS\n");
    printf("========================================\n\n");
    
    if (final_connections > initial_connections) {
        printf("✅ SUCCESS: System automatically discovered %llu connections!\n",
               (unsigned long long)(final_connections - initial_connections));
        printf("   The graph is staying connected via co-activation.\n");
    } else {
        printf("⚠️  No connections discovered automatically.\n");
        printf("   May need more time or higher activation levels.\n");
    }
    
    if (exec_idx != UINT64_MAX) {
        NodeDisk *exec = &file.nodes[exec_idx];
        if (exec->traffic_ema > 0.0f) {
            printf("✅ EXEC received traffic: %.6f\n", exec->traffic_ema);
            printf("   Automatic connection discovery is working!\n");
        } else {
            printf("❌ EXEC received no traffic.\n");
            printf("   Connection discovery may need tuning.\n");
        }
    }
    
    printf("\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    return 0;
}


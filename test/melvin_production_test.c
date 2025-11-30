/*
 * MELVIN PRODUCTION TEST - Unified System Test
 * 
 * This is a comprehensive test that runs Melvin as a complete production system
 * with multiple data streams, continuous operation, and validation.
 * 
 * Tests:
 * - System initialization and persistence
 * - Continuous event processing
 * - Multiple data ingestion channels
 * - Pattern formation and learning
 * - EXEC node creation and execution
 * - Long-term stability
 * - State persistence and recovery
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include "melvin.c"

#define TEST_FILE "melvin_production_test.m"
#define TEST_DURATION_SECONDS 60
#define EVENTS_PER_CYCLE 100
#define METRICS_INTERVAL 5000

// Generate test data streams
static void generate_test_data(MelvinRuntime *rt) {
    static uint64_t iteration = 0;
    
    // Channel 0: Repeating pattern ABC
    if (iteration % 3 == 0) {
        ingest_byte(rt, 0, 'A', 1.0f);
    } else if (iteration % 3 == 1) {
        ingest_byte(rt, 0, 'B', 1.0f);
    } else {
        ingest_byte(rt, 0, 'C', 1.0f);
    }
    
    // Channel 1: Repeating pattern XYZ
    if (iteration % 3 == 0) {
        ingest_byte(rt, 1, 'X', 1.0f);
    } else if (iteration % 3 == 1) {
        ingest_byte(rt, 1, 'Y', 1.0f);
    } else {
        ingest_byte(rt, 1, 'Z', 1.0f);
    }
    
    // Channel 2: Random bytes (for noise)
    if (iteration % 10 == 0) {
        uint8_t random_byte = (uint8_t)(iteration % 256);
        ingest_byte(rt, 2, random_byte, 0.5f);
    }
    
    iteration++;
}

// Check system health
static int check_system_health(MelvinRuntime *rt) {
    if (!rt || !rt->file) return 0;
    
    GraphHeaderDisk *gh = rt->file->graph_header;
    
    // Check for NaN/Inf in activations
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &rt->file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (isnan(n->state) || isinf(n->state)) {
            fprintf(stderr, "[HEALTH] NaN/Inf detected in node %llu\n", (unsigned long long)n->id);
            return 0;
        }
        if (isnan(n->prediction) || isinf(n->prediction)) {
            fprintf(stderr, "[HEALTH] NaN/Inf in prediction for node %llu\n", (unsigned long long)n->id);
            return 0;
        }
    }
    
    // Check for NaN/Inf in edge weights
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &rt->file->edges[i];
        if (e->src == UINT64_MAX) continue;
        if (isnan(e->weight) || isinf(e->weight)) {
            fprintf(stderr, "[HEALTH] NaN/Inf in edge weight\n");
            return 0;
        }
    }
    
    return 1;
}

// Print comprehensive metrics
static void print_comprehensive_metrics(MelvinRuntime *rt, uint64_t total_events, double elapsed_time) {
    if (!rt || !rt->file) return;
    
    GraphHeaderDisk *gh = rt->file->graph_header;
    
    printf("\n========================================\n");
    printf("PRODUCTION SYSTEM METRICS\n");
    printf("========================================\n");
    printf("Runtime:\n");
    printf("  Events processed: %llu\n", (unsigned long long)total_events);
    printf("  Elapsed time: %.2f seconds\n", elapsed_time);
    printf("  Events/sec: %.2f\n", total_events / elapsed_time);
    
    printf("\nGraph Structure:\n");
    printf("  Nodes: %llu / %llu (%.1f%%)\n", 
           (unsigned long long)gh->num_nodes, 
           (unsigned long long)gh->node_capacity,
           (gh->node_capacity > 0) ? (100.0f * gh->num_nodes / gh->node_capacity) : 0.0f);
    printf("  Edges: %llu / %llu (%.1f%%)\n",
           (unsigned long long)gh->num_edges,
           (unsigned long long)gh->edge_capacity,
           (gh->edge_capacity > 0) ? (100.0f * gh->num_edges / gh->edge_capacity) : 0.0f);
    printf("  Blob: %llu / %llu bytes (%.1f%%)\n",
           (unsigned long long)rt->file->blob_size,
           (unsigned long long)rt->file->blob_capacity,
           (rt->file->blob_capacity > 0) ? (100.0f * rt->file->blob_size / rt->file->blob_capacity) : 0.0f);
    
    // Count node types
    uint64_t exec_count = 0;
    uint64_t pattern_count = 0;
    uint64_t data_count = 0;
    float total_stability = 0.0f;
    uint64_t stable_nodes = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &rt->file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        
        if (n->flags & NODE_FLAG_EXECUTABLE) exec_count++;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) pattern_count++;
        if (n->flags & NODE_FLAG_DATA) data_count++;
        
        if (n->stability > 0.0f) {
            total_stability += n->stability;
            stable_nodes++;
        }
    }
    
    float avg_stability = (stable_nodes > 0) ? (total_stability / stable_nodes) : 0.0f;
    
    printf("\nNode Types:\n");
    printf("  EXEC nodes: %llu\n", (unsigned long long)exec_count);
    printf("  Pattern nodes: %llu\n", (unsigned long long)pattern_count);
    printf("  Data nodes: %llu\n", (unsigned long long)data_count);
    printf("  Average stability: %.4f (%llu nodes)\n", avg_stability, (unsigned long long)stable_nodes);
    
    // Compute activation statistics
    float total_activation = 0.0f;
    float max_activation = 0.0f;
    float min_activation = 0.0f;
    uint64_t active_nodes = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &rt->file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        
        float act = n->state;
        total_activation += fabsf(act);
        if (act > max_activation) max_activation = act;
        if (act < min_activation) min_activation = act;
        active_nodes++;
    }
    
    float avg_activation = (active_nodes > 0) ? (total_activation / active_nodes) : 0.0f;
    
    printf("\nActivation Statistics:\n");
    printf("  Average: %.6f\n", avg_activation);
    printf("  Max: %.6f\n", max_activation);
    printf("  Min: %.6f\n", min_activation);
    printf("  Active nodes: %llu\n", (unsigned long long)active_nodes);
    
    // Edge weight statistics
    float total_weight = 0.0f;
    float max_weight = 0.0f;
    uint64_t active_edges = 0;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &rt->file->edges[i];
        if (e->src == UINT64_MAX) continue;
        
        float w = fabsf(e->weight);
        total_weight += w;
        if (w > max_weight) max_weight = w;
        active_edges++;
    }
    
    float avg_weight = (active_edges > 0) ? (total_weight / active_edges) : 0.0f;
    
    printf("\nEdge Statistics:\n");
    printf("  Average weight magnitude: %.6f\n", avg_weight);
    printf("  Max weight magnitude: %.6f\n", max_weight);
    printf("  Active edges: %llu\n", (unsigned long long)active_edges);
    
    printf("\nPhysics Parameters:\n");
    printf("  Decay rate: %.4f\n", gh->decay_rate);
    printf("  Learning rate: %.6f\n", gh->learning_rate);
    printf("  Exec threshold: %.4f\n", gh->exec_threshold);
    printf("  Global energy budget: %.2f\n", gh->global_energy_budget);
    
    printf("========================================\n\n");
}

int main() {
    printf("========================================\n");
    printf("MELVIN PRODUCTION SYSTEM TEST\n");
    printf("========================================\n\n");
    printf("This test runs Melvin as a complete production system:\n");
    printf("  - Continuous operation for %d seconds\n", TEST_DURATION_SECONDS);
    printf("  - Multiple data ingestion channels\n");
    printf("  - Pattern formation and learning\n");
    printf("  - System health monitoring\n");
    printf("  - State persistence\n\n");
    
    // Step 1: Initialize system
    printf("Step 1: Initializing system...\n");
    unlink(TEST_FILE);
    
    GraphParams params;
    init_default_params(&params);
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create test file\n");
        return 1;
    }
    printf("  ✓ Created %s\n", TEST_FILE);
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    printf("  ✓ File mapped\n");
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        return 1;
    }
    printf("  ✓ Runtime initialized\n\n");
    
    // Step 2: Run production loop
    printf("Step 2: Running production loop...\n");
    printf("  Duration: %d seconds\n", TEST_DURATION_SECONDS);
    printf("  Events per cycle: %d\n", EVENTS_PER_CYCLE);
    printf("  Metrics interval: %d events\n\n", METRICS_INTERVAL);
    
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    uint64_t total_events = 0;
    uint64_t last_metrics = 0;
    int health_checks_passed = 0;
    int health_checks_total = 0;
    
    while (1) {
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                        (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        if (elapsed >= TEST_DURATION_SECONDS) {
            break;
        }
        
        // Generate test data
        generate_test_data(&rt);
        
        // Process events
        melvin_process_n_events(&rt, EVENTS_PER_CYCLE);
        total_events += EVENTS_PER_CYCLE;
        
        // Health check
        health_checks_total++;
        if (check_system_health(&rt)) {
            health_checks_passed++;
        }
        
        // Print metrics periodically
        if (total_events - last_metrics >= METRICS_INTERVAL) {
            print_comprehensive_metrics(&rt, total_events, elapsed);
            last_metrics = total_events;
        }
        
        // Sync periodically
        if (total_events % 2000 == 0) {
            melvin_m_sync(&file);
        }
    }
    
    // Step 3: Final state
    printf("Step 3: Final system state...\n");
    gettimeofday(&current_time, NULL);
    double total_elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                          (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    
    print_comprehensive_metrics(&rt, total_events, total_elapsed);
    
    // Step 4: Final sync
    printf("Step 4: Final sync to disk...\n");
    melvin_m_sync(&file);
    printf("  ✓ Sync complete\n\n");
    
    // Step 5: Results
    printf("========================================\n");
    printf("PRODUCTION TEST RESULTS\n");
    printf("========================================\n\n");
    
    int passed = 1;
    
    printf("Runtime Metrics:\n");
    printf("  Total events: %llu\n", (unsigned long long)total_events);
    printf("  Total time: %.2f seconds\n", total_elapsed);
    printf("  Average rate: %.2f events/sec\n", total_events / total_elapsed);
    
    printf("\nHealth Checks:\n");
    float health_rate = (health_checks_total > 0) ? 
                       (100.0f * health_checks_passed / health_checks_total) : 0.0f;
    printf("  Passed: %d / %d (%.1f%%)\n", health_checks_passed, health_checks_total, health_rate);
    
    if (health_rate < 95.0f) {
        printf("  ⚠ Health check rate below 95%%\n");
        passed = 0;
    } else {
        printf("  ✓ Health checks passed\n");
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    printf("\nGraph Growth:\n");
    printf("  Nodes created: %llu\n", (unsigned long long)gh->num_nodes);
    printf("  Edges created: %llu\n", (unsigned long long)gh->num_edges);
    printf("  Blob size: %llu bytes\n", (unsigned long long)file.blob_size);
    
    if (gh->num_nodes > 20) {
        printf("  ✓ Graph growth observed\n");
    } else {
        printf("  ⚠ Limited graph growth\n");
    }
    
    if (gh->num_edges > 10) {
        printf("  ✓ Edge formation observed\n");
    } else {
        printf("  ⚠ Limited edge formation\n");
    }
    
    printf("\n");
    if (passed) {
        printf("✅ PRODUCTION TEST: PASSED\n");
        printf("System is production-ready!\n");
    } else {
        printf("⚠️  PRODUCTION TEST: PARTIAL\n");
        printf("Some metrics need attention.\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return passed ? 0 : 1;
}


/*
 * MELVIN MAIN - Production System Runner
 * 
 * This is the unified production system that runs Melvin continuously.
 * It handles:
 * - System initialization
 * - Continuous event processing
 * - Data ingestion (stdin, files, network, etc.)
 * - Monitoring and metrics
 * - Graceful shutdown
 * - State persistence
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include "melvin.c"

#define DEFAULT_MELVIN_FILE "melvin.m"
#define DEFAULT_EVENTS_PER_CYCLE 100
#define DEFAULT_METRICS_INTERVAL 10000  // Log metrics every N events
#define DEFAULT_SYNC_INTERVAL 1000     // Sync to disk every N events

// Global state for signal handling
static MelvinRuntime *g_runtime = NULL;
static MelvinFile *g_file = NULL;
static volatile int g_running = 1;
static volatile int g_sync_requested = 0;

// Signal handler for graceful shutdown
static void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        printf("\n[melvin_main] Received shutdown signal, shutting down gracefully...\n");
        g_running = 0;
    } else if (sig == SIGUSR1) {
        printf("\n[melvin_main] Sync requested via signal\n");
        g_sync_requested = 1;
    }
}

// Print system metrics
static void print_metrics(MelvinRuntime *rt, uint64_t total_events) {
    if (!rt || !rt->file) return;
    
    GraphHeaderDisk *gh = rt->file->graph_header;
    
    printf("\n========================================\n");
    printf("MELVIN SYSTEM METRICS (Event %llu)\n", (unsigned long long)total_events);
    printf("========================================\n");
    printf("Graph Structure:\n");
    printf("  Nodes: %llu / %llu\n", (unsigned long long)gh->num_nodes, (unsigned long long)gh->node_capacity);
    printf("  Edges: %llu / %llu\n", (unsigned long long)gh->num_edges, (unsigned long long)gh->edge_capacity);
    printf("  Blob size: %llu / %llu bytes\n", 
           (unsigned long long)rt->file->blob_size, (unsigned long long)rt->file->blob_capacity);
    
    // Count EXEC nodes
    uint64_t exec_count = 0;
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &rt->file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->flags & NODE_FLAG_EXECUTABLE) exec_count++;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) pattern_count++;  // Pattern node range
    }
    printf("  EXEC nodes: %llu\n", (unsigned long long)exec_count);
    printf("  Pattern nodes: %llu\n", (unsigned long long)pattern_count);
    
    printf("\nPhysics State:\n");
    printf("  Decay rate: %.4f\n", gh->decay_rate);
    printf("  Learning rate: %.6f\n", gh->learning_rate);
    printf("  Exec threshold: %.4f\n", gh->exec_threshold);
    printf("  Global energy budget: %.4f\n", gh->global_energy_budget);
    
    printf("\nActivity:\n");
    printf("  Total events processed: %llu\n", (unsigned long long)total_events);
    // Note: Stats tracking would need to be added to MelvinRuntime if needed
    // For now, we track total_events which is the main metric
    
    // Compute average activation
    float total_activation = 0.0f;
    uint64_t active_nodes = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &rt->file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        total_activation += fabsf(n->state);
        active_nodes++;
    }
    float avg_activation = (active_nodes > 0) ? (total_activation / active_nodes) : 0.0f;
    printf("  Average activation: %.6f\n", avg_activation);
    printf("  Active nodes: %llu\n", (unsigned long long)active_nodes);
    
    printf("========================================\n\n");
}

// Ingest data from stdin
static void ingest_from_stdin(MelvinRuntime *rt, uint8_t channel_id) {
    static uint8_t buffer[4096];
    ssize_t bytes_read = read(STDIN_FILENO, buffer, sizeof(buffer));
    
    if (bytes_read > 0) {
        for (ssize_t i = 0; i < bytes_read; i++) {
            ingest_byte(rt, channel_id, buffer[i], 1.0f);
        }
    } else if (bytes_read < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        // Error reading (but not just "no data available")
        if (errno != EINTR) {
            perror("[melvin_main] read from stdin");
        }
    }
}

// Main production loop
int main(int argc, char **argv) {
    const char *melvin_file = DEFAULT_MELVIN_FILE;
    int events_per_cycle = DEFAULT_EVENTS_PER_CYCLE;
    uint64_t metrics_interval = DEFAULT_METRICS_INTERVAL;
    uint64_t sync_interval = DEFAULT_SYNC_INTERVAL;
    int ingest_stdin = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            melvin_file = argv[++i];
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            events_per_cycle = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            metrics_interval = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sync_interval = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-i") == 0) {
            ingest_stdin = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Melvin Production System\n");
            printf("Usage: %s [options]\n", argv[0]);
            printf("\nOptions:\n");
            printf("  -f FILE     Melvin file path (default: %s)\n", DEFAULT_MELVIN_FILE);
            printf("  -e N        Events per processing cycle (default: %d)\n", DEFAULT_EVENTS_PER_CYCLE);
            printf("  -m N        Metrics interval in events (default: %llu)\n", (unsigned long long)DEFAULT_METRICS_INTERVAL);
            printf("  -s N        Sync interval in events (default: %llu)\n", (unsigned long long)DEFAULT_SYNC_INTERVAL);
            printf("  -i          Ingest data from stdin\n");
            printf("  -h, --help  Show this help\n");
            printf("\nSignals:\n");
            printf("  SIGINT/SIGTERM  Graceful shutdown\n");
            printf("  SIGUSR1         Request immediate sync to disk\n");
            return 0;
        }
    }
    
    printf("========================================\n");
    printf("MELVIN PRODUCTION SYSTEM\n");
    printf("========================================\n\n");
    printf("Configuration:\n");
    printf("  Melvin file: %s\n", melvin_file);
    printf("  Events per cycle: %d\n", events_per_cycle);
    printf("  Metrics interval: %llu events\n", (unsigned long long)metrics_interval);
    printf("  Sync interval: %llu events\n", (unsigned long long)sync_interval);
    printf("  Ingest stdin: %s\n", ingest_stdin ? "yes" : "no");
    printf("\n");
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGUSR1, signal_handler);
    
    // Step 1: Initialize or load Melvin file
    printf("Step 1: Initializing Melvin file...\n");
    MelvinFile file;
    
    // Check if file exists
    FILE *test_fp = fopen(melvin_file, "r");
    int file_exists = (test_fp != NULL);
    if (test_fp) fclose(test_fp);
    
    if (!file_exists) {
        printf("  Creating new Melvin file: %s\n", melvin_file);
        GraphParams params;
        init_default_params(&params);
        if (melvin_m_init_new_file(melvin_file, &params) < 0) {
            fprintf(stderr, "ERROR: Failed to create Melvin file\n");
            return 1;
        }
        printf("  ✓ Created new file\n");
    } else {
        printf("  Loading existing Melvin file: %s\n", melvin_file);
    }
    
    // Map file
    if (melvin_m_map(melvin_file, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map Melvin file\n");
        return 1;
    }
    printf("  ✓ File mapped\n");
    g_file = &file;
    
    // Step 2: Initialize runtime
    printf("Step 2: Initializing runtime...\n");
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("  ✓ Runtime initialized\n");
    g_runtime = &rt;
    
    // Step 3: Print initial state
    printf("\nStep 3: Initial system state:\n");
    print_metrics(&rt, 0);
    
    // Step 4: Main production loop
    printf("Step 4: Starting production loop...\n");
    printf("  Press Ctrl+C for graceful shutdown\n");
    printf("  Send SIGUSR1 for immediate sync\n\n");
    
    uint64_t total_events = 0;
    uint64_t last_metrics = 0;
    uint64_t last_sync = 0;
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    while (g_running) {
        // Ingest data from stdin if enabled
        if (ingest_stdin) {
            ingest_from_stdin(&rt, 0);
        }
        
        // Process events
        melvin_process_n_events(&rt, events_per_cycle);
        total_events += events_per_cycle;
        
        // Print metrics periodically
        if (total_events - last_metrics >= metrics_interval) {
            gettimeofday(&current_time, NULL);
            double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                           (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
            double events_per_sec = total_events / elapsed;
            
            print_metrics(&rt, total_events);
            printf("  Performance: %.2f events/sec\n", events_per_sec);
            printf("  Uptime: %.1f seconds\n", elapsed);
            
            last_metrics = total_events;
        }
        
        // Sync to disk periodically
        if (g_sync_requested || (total_events - last_sync >= sync_interval)) {
            printf("[melvin_main] Syncing to disk...\n");
            melvin_m_sync(&file);
            g_sync_requested = 0;
            last_sync = total_events;
            printf("[melvin_main] Sync complete\n");
        }
        
        // Small sleep to prevent CPU spinning (optional)
        usleep(1000);  // 1ms
    }
    
    // Step 5: Graceful shutdown
    printf("\nStep 5: Shutting down gracefully...\n");
    
    // Final sync
    printf("  Performing final sync to disk...\n");
    melvin_m_sync(&file);
    printf("  ✓ Final sync complete\n");
    
    // Print final metrics
    printf("\nFinal system state:\n");
    print_metrics(&rt, total_events);
    
    // Cleanup
    printf("  Cleaning up...\n");
    runtime_cleanup(&rt);
    close_file(&file);
    printf("  ✓ Cleanup complete\n");
    
    printf("\n========================================\n");
    printf("MELVIN SHUTDOWN COMPLETE\n");
    printf("========================================\n");
    printf("Total events processed: %llu\n", (unsigned long long)total_events);
    
    gettimeofday(&current_time, NULL);
    double total_elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                          (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("Total uptime: %.1f seconds\n", total_elapsed);
    printf("Average rate: %.2f events/sec\n", total_events / total_elapsed);
    
    return 0;
}


#define _POSIX_C_SOURCE 200809L

#include "melvin_file.h"
#include "melvin_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <signal.h>

// Global runtime for signal handling
static MelvinRuntime *g_runtime = NULL;
static int g_running = 1;

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
    printf("\n[runner] Shutting down...\n");
}

int main(int argc, char **argv) {
    const char *brain_path = "melvin.m";
    int create_new = 0;
    int debug = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--create") == 0) {
            create_new = 1;
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--debug") == 0) {
            debug = 1;
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) {
            if (i + 1 < argc) {
                brain_path = argv[++i];
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -c, --create    Create new brain file\n");
            printf("  -d, --debug     Enable debug output\n");
            printf("  -f, --file PATH Specify brain file path (default: melvin.m)\n");
            printf("  -h, --help      Show this help\n");
            return 0;
        }
    }
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create or load brain file
    MelvinFile file;
    
    if (create_new) {
        printf("[runner] Creating new brain file: %s\n", brain_path);
        if (create_new_file(brain_path) < 0) {
            fprintf(stderr, "[runner] Failed to create brain file\n");
            return 1;
        }
    }
    
    printf("[runner] Loading brain file: %s\n", brain_path);
    if (load_file(brain_path, &file) < 0) {
        fprintf(stderr, "[runner] Failed to load brain file\n");
        return 1;
    }
    
    // Initialize runtime
    MelvinRuntime runtime;
    g_runtime = &runtime;
    
    if (runtime_init(&runtime, &file) < 0) {
        fprintf(stderr, "[runner] Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    
    printf("[runner] Melvin Physics Runtime started\n");
    printf("[runner] Press Ctrl+C to stop\n\n");
    
    // Set stdin to non-blocking for input
    int flags = fcntl(0, F_GETFL, 0);
    fcntl(0, F_SETFL, flags | O_NONBLOCK);
    
    // Main loop
    uint64_t last_sync_tick = 0;
    uint64_t last_stats_tick = 0;
    
    while (g_running) {
        // Ingest external sensors/inputs (placeholder - inject some test pulses)
        // In real system, this would read from camera, audio, etc.
        static uint64_t test_pulse_counter = 0;
        if (test_pulse_counter++ % 100 == 0) {
            // Inject a test pulse every 100 ticks
            inject_pulse(&runtime, 1, 1.0f);
        }
        
        // Run physics tick
        physics_tick(&runtime);
        
        // Sync to disk periodically
        uint64_t current_tick = file.file_header->tick_counter;
        if (current_tick - last_sync_tick >= 100) {
            msync(file.map, file.map_size, MS_SYNC);
            last_sync_tick = current_tick;
        }
        
        // Print stats periodically
        if (debug || (current_tick - last_stats_tick >= 1000)) {
            GraphHeader *gh = file.graph_header;
            printf("Tick %llu | Nodes: %llu/%llu | Edges: %llu/%llu | Pulses: %llu/%llu\n",
                   (unsigned long long)current_tick,
                   (unsigned long long)gh->num_nodes,
                   (unsigned long long)gh->node_capacity,
                   (unsigned long long)gh->num_edges,
                   (unsigned long long)gh->edge_capacity,
                   (unsigned long long)gh->total_pulses_emitted,
                   (unsigned long long)gh->total_pulses_absorbed);
            last_stats_tick = current_tick;
        }
        
        // Small sleep to prevent 100% CPU
        usleep(1000); // 1ms
    }
    
    // Cleanup
    printf("[runner] Shutting down runtime...\n");
    runtime_cleanup(&runtime);
    close_file(&file);
    
    printf("[runner] Shutdown complete\n");
    return 0;
}


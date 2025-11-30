#define _POSIX_C_SOURCE 200809L
#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <signal.h>

// Forward declarations from melvin.c
extern void melvin_tick(Brain *g);
extern uint32_t g_mc_count;
extern uint64_t feedback_channel_node;
extern uint64_t feedback_output_node;

static volatile int g_running = 1;

void signal_handler(int sig) {
    g_running = 0;
    printf("\n[test] Received signal %d, shutting down gracefully...\n", sig);
}

int main(int argc, char **argv) {
    const char *db_path = "melvin.m";
    uint64_t max_ticks = 0; // 0 = run indefinitely
    int quiet = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-ticks") == 0 && i + 1 < argc) {
            max_ticks = strtoull(argv[i + 1], NULL, 10);
            i++;
        } else if (strcmp(argv[i], "--quiet") == 0 || strcmp(argv[i], "-q") == 0) {
            quiet = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--max-ticks N] [--quiet] [brain_file]\n", argv[0]);
            printf("  --max-ticks N: Run for N ticks then exit (default: run indefinitely)\n");
            printf("  --quiet: Reduce output (only summary every 1000 ticks)\n");
            return 0;
        } else {
            db_path = argv[i];
        }
    }
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("========================================\n");
    printf("MELVIN CONTINUOUS TEST\n");
    printf("========================================\n");
    printf("Brain file: %s\n", db_path);
    if (max_ticks > 0) {
        printf("Max ticks: %llu\n", (unsigned long long)max_ticks);
    } else {
        printf("Running indefinitely (Ctrl+C to stop)\n");
    }
    printf("Quiet mode: %s\n", quiet ? "ON" : "OFF");
    printf("\n");
    
    // Open brain file
    int fd = open(db_path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Could not open %s\n", db_path);
        fprintf(stderr, "Run init_melvin_simple first to initialize the brain.\n");
        return 1;
    }
    
    struct stat st;
    fstat(fd, &st);
    size_t filesize = st.st_size;
    
    void *map = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    
    Brain g;
    g.header = (BrainHeader *)map;
    g.nodes = (Node *)((char *)map + g.header->node_region_offset);
    g.edges = (Edge *)((char *)map + g.header->edge_region_offset);
    
    uint64_t start_tick = g.header->tick;
    uint64_t start_nodes = g.header->num_nodes;
    uint64_t start_edges = g.header->num_edges;
    
    time_t start_time = time(NULL);
    uint64_t tick_count = 0;
    
    // Track metrics
    uint64_t last_pattern_count = 0;
    uint64_t last_parse_activation = 0;
    uint64_t last_compile_activation = 0;
    
    printf("Starting at tick %llu\n", (unsigned long long)start_tick);
    printf("Initial state: %llu nodes, %llu edges\n", 
           (unsigned long long)start_nodes, (unsigned long long)start_edges);
    printf("\n");
    
    if (!quiet) {
        printf("Tick | Nodes | Edges | Patterns | FB_IN | FB_OUT | Parse | Compile\n");
        printf("-----|-------|-------|----------|-------|--------|-------|--------\n");
    }
    
    while (g_running && (max_ticks == 0 || tick_count < max_ticks)) {
        melvin_tick(&g);
        tick_count++;
        
        // Count patterns (PATTERN_ROOT nodes)
        uint64_t pattern_count = 0;
        for (uint64_t i = 0; i < g.header->num_nodes; i++) {
            if (g.nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
                pattern_count++;
            }
        }
        
        // Check for parse/compile node activations
        uint64_t parse_active = 0;
        uint64_t compile_active = 0;
        for (uint64_t i = 0; i < g.header->num_nodes; i++) {
            if (g.nodes[i].mc_id > 0 && g.nodes[i].a > 0.5f) {
                // Would need to check MC table to see which function, but for now just track activation
                if (g.nodes[i].a > 0.5f) {
                    // This is a heuristic - in real system would check MC name
                    if (g.nodes[i].kind == NODE_KIND_CONTROL) {
                        parse_active = i;
                    }
                }
            }
        }
        
        float fb_in = 0.0f;
        float fb_out = 0.0f;
        if (feedback_channel_node < g.header->num_nodes) {
            fb_in = g.nodes[feedback_channel_node].a;
        }
        if (feedback_output_node < g.header->num_nodes) {
            fb_out = g.nodes[feedback_output_node].a;
        }
        
        // Print status
        if (!quiet) {
            if (tick_count % 100 == 0 || pattern_count != last_pattern_count || 
                parse_active != last_parse_activation || compile_active != last_compile_activation) {
                printf("%4llu | %5llu | %5llu | %8llu | %.3f | %.3f | %s | %s\n",
                       (unsigned long long)tick_count,
                       (unsigned long long)g.header->num_nodes,
                       (unsigned long long)g.header->num_edges,
                       (unsigned long long)pattern_count,
                       fb_in, fb_out,
                       parse_active ? "YES" : "no",
                       compile_active ? "YES" : "no");
            }
        } else {
            // Quiet mode: only print every 1000 ticks
            if (tick_count % 1000 == 0) {
                time_t now = time(NULL);
                double elapsed = difftime(now, start_time);
                double ticks_per_sec = (elapsed > 0) ? tick_count / elapsed : 0;
                printf("[%llu] Nodes=%llu Edges=%llu Patterns=%llu FB_IN=%.3f FB_OUT=%.3f (%.1f ticks/sec)\n",
                       (unsigned long long)tick_count,
                       (unsigned long long)g.header->num_nodes,
                       (unsigned long long)g.header->num_edges,
                       (unsigned long long)pattern_count,
                       fb_in, fb_out,
                       ticks_per_sec);
            }
        }
        
        last_pattern_count = pattern_count;
        last_parse_activation = parse_active;
        last_compile_activation = compile_active;
    }
    
    time_t end_time = time(NULL);
    double elapsed = difftime(end_time, start_time);
    double ticks_per_sec = (elapsed > 0) ? tick_count / elapsed : 0;
    
    printf("\n");
    printf("========================================\n");
    printf("FINAL STATE\n");
    printf("========================================\n");
    printf("Ticks completed: %llu\n", (unsigned long long)tick_count);
    printf("Time elapsed: %.1f seconds\n", elapsed);
    printf("Ticks per second: %.2f\n", ticks_per_sec);
    printf("\n");
    printf("Graph growth:\n");
    printf("  Nodes: %llu → %llu (+%llu, +%.1f%%)\n",
           (unsigned long long)start_nodes,
           (unsigned long long)g.header->num_nodes,
           (unsigned long long)(g.header->num_nodes - start_nodes),
           start_nodes > 0 ? 100.0 * (g.header->num_nodes - start_nodes) / start_nodes : 0.0);
    printf("  Edges: %llu → %llu (+%llu, +%.1f%%)\n",
           (unsigned long long)start_edges,
           (unsigned long long)g.header->num_edges,
           (unsigned long long)(g.header->num_edges - start_edges),
           start_edges > 0 ? 100.0 * (g.header->num_edges - start_edges) / start_edges : 0.0);
    
    // Count final patterns
    uint64_t final_pattern_count = 0;
    for (uint64_t i = 0; i < g.header->num_nodes; i++) {
        if (g.nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
            final_pattern_count++;
        }
    }
    printf("  Patterns: %llu\n", (unsigned long long)final_pattern_count);
    
    float final_fb_in = 0.0f;
    float final_fb_out = 0.0f;
    if (feedback_channel_node < g.header->num_nodes) {
        final_fb_in = g.nodes[feedback_channel_node].a;
    }
    if (feedback_output_node < g.header->num_nodes) {
        final_fb_out = g.nodes[feedback_output_node].a;
    }
    printf("\n");
    printf("Feedback loop:\n");
    printf("  FB_IN: %.3f\n", final_fb_in);
    printf("  FB_OUT: %.3f\n", final_fb_out);
    printf("  Loop active: %s\n", (final_fb_in > 0.01f && final_fb_out > 0.01f) ? "YES" : "NO");
    printf("========================================\n");
    
    munmap(map, filesize);
    close(fd);
    
    return 0;
}


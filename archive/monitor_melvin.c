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
#include <unistd.h>

static volatile int g_running = 1;

void signal_handler(int sig) {
    g_running = 0;
}

int main(int argc, char **argv) {
    const char *db_path = "melvin.m";
    double refresh_rate = 0.5; // seconds
    
    if (argc > 1) {
        db_path = argv[1];
    }
    if (argc > 2) {
        refresh_rate = atof(argv[2]);
        if (refresh_rate < 0.1) refresh_rate = 0.1;
        if (refresh_rate > 5.0) refresh_rate = 5.0;
    }
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    uint64_t last_tick = 0;
    uint64_t last_nodes = 0;
    uint64_t last_edges = 0;
    time_t start_time = time(NULL);
    
    printf("\033[2J\033[H"); // Clear screen and move to top
    
    while (g_running) {
        // Open and read brain file
        int fd = open(db_path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "ERROR: Could not open %s\n", db_path);
            sleep(1);
            continue;
        }
        
        struct stat st;
        if (fstat(fd, &st) < 0) {
            close(fd);
            sleep(1);
            continue;
        }
        
        size_t filesize = st.st_size;
        
        void *map = mmap(NULL, filesize, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            close(fd);
            sleep(1);
            continue;
        }
        
        BrainHeader *header = (BrainHeader *)map;
        
        // Validate header
        if (header->num_nodes > header->node_capacity ||
            header->num_edges > header->edge_capacity) {
            munmap(map, filesize);
            close(fd);
            sleep(1);
            continue;
        }
        
        Node *nodes = (Node *)((char *)map + header->node_region_offset);
        
        // Count patterns
        uint64_t pattern_count = 0;
        uint64_t data_count = 0;
        uint64_t control_count = 0;
        uint64_t mc_count = 0;
        for (uint64_t i = 0; i < header->num_nodes; i++) {
            if (nodes[i].kind == NODE_KIND_PATTERN_ROOT) pattern_count++;
            if (nodes[i].kind == NODE_KIND_DATA) data_count++;
            if (nodes[i].kind == NODE_KIND_CONTROL) control_count++;
            if (nodes[i].mc_id > 0) mc_count++;
        }
        
        // Find feedback nodes
        float fb_in = 0.0f;
        float fb_out = 0.0f;
        for (uint64_t i = 0; i < header->num_nodes; i++) {
            if (nodes[i].kind == NODE_KIND_META) {
                if (nodes[i].value == 0x4642494E) { // "FBIN"
                    fb_in = nodes[i].a;
                } else if (nodes[i].value == 0x46424F55) { // "FBOUT"
                    fb_out = nodes[i].a;
                }
            }
        }
        
        // Calculate rates
        uint64_t tick_delta = (header->tick > last_tick) ? header->tick - last_tick : 0;
        uint64_t node_delta = (header->num_nodes > last_nodes) ? header->num_nodes - last_nodes : 0;
        uint64_t edge_delta = (header->num_edges > last_edges) ? header->num_edges - last_edges : 0;
        
        time_t now = time(NULL);
        double elapsed = difftime(now, start_time);
        double ticks_per_sec = (elapsed > 0 && header->tick > 0) ? header->tick / elapsed : 0;
        
        // Clear screen and print stats
        printf("\033[2J\033[H"); // Clear screen, move to top
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║                    MELVIN LIVE MONITOR                     ║\n");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║ Brain File: %-45s ║\n", db_path);
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║ TICKS                                                       ║\n");
        printf("║   Current:  %-50llu ║\n", (unsigned long long)header->tick);
        printf("║   Rate:     %-50.1f ║\n", ticks_per_sec);
        printf("║   Delta:    %-50llu ║\n", (unsigned long long)tick_delta);
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║ NODES                                                       ║\n");
        printf("║   Total:    %-50llu ║\n", (unsigned long long)header->num_nodes);
        printf("║   Capacity: %-50llu ║\n", (unsigned long long)header->node_capacity);
        printf("║   Used:     %-50.1f%% ║\n", 
               header->node_capacity > 0 ? 100.0 * header->num_nodes / header->node_capacity : 0.0);
        printf("║   Delta:    %-50llu ║\n", (unsigned long long)node_delta);
        printf("║   Data:     %-50llu ║\n", (unsigned long long)data_count);
        printf("║   Control:  %-50llu ║\n", (unsigned long long)control_count);
        printf("║   MC:       %-50llu ║\n", (unsigned long long)mc_count);
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║ EDGES                                                       ║\n");
        printf("║   Total:    %-50llu ║\n", (unsigned long long)header->num_edges);
        printf("║   Capacity: %-50llu ║\n", (unsigned long long)header->edge_capacity);
        printf("║   Used:     %-50.1f%% ║\n",
               header->edge_capacity > 0 ? 100.0 * header->num_edges / header->edge_capacity : 0.0);
        printf("║   Delta:    %-50llu ║\n", (unsigned long long)edge_delta);
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║ PATTERNS                                                    ║\n");
        printf("║   Count:    %-50llu ║\n", (unsigned long long)pattern_count);
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║ FEEDBACK LOOP                                               ║\n");
        printf("║   FB_IN:    %-50.3f ║\n", fb_in);
        printf("║   FB_OUT:   %-50.3f ║\n", fb_out);
        printf("║   Active:   %-50s ║\n", (fb_in > 0.01f && fb_out > 0.01f) ? "YES" : "NO");
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║ PARAMETERS                                                  ║\n");
        printf("║   Edge Threshold:    %-40.3f ║\n", header->edge_activation_threshold);
        printf("║   MC Threshold:      %-40.3f ║\n", header->mc_execution_threshold);
        printf("║   Decay Factor:       %-40.3f ║\n", header->decay_factor);
        printf("║   Learning Rate:     %-40.3f ║\n", header->learning_rate);
        printf("╠════════════════════════════════════════════════════════════╣\n");
        printf("║ Runtime: %-50.1fs ║\n", elapsed);
        printf("╚════════════════════════════════════════════════════════════╝\n");
        printf("\nPress Ctrl+C to stop\n");
        
        last_tick = header->tick;
        last_nodes = header->num_nodes;
        last_edges = header->num_edges;
        
        munmap(map, filesize);
        close(fd);
        
        usleep((useconds_t)(refresh_rate * 1000000));
    }
    
    printf("\nMonitor stopped.\n");
    return 0;
}


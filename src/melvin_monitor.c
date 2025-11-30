/*
 * melvin_monitor.c - Live monitoring tool for Melvin brain
 * 
 * Connects to Jetson and displays real-time graph statistics:
 * - Node/edge counts
 * - Average chaos, activation, edge strength
 * - Drive mechanism states
 * - Recent activity
 */

#include "melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

static void print_stats(Graph *g, int iteration) {
    if (!g || !g->hdr) return;
    
    time_t now = time(NULL);
    char *time_str = ctime(&now);
    time_str[strlen(time_str) - 1] = '\0';  /* Remove newline */
    
    printf("\n=== Melvin Brain Stats [%s] (Iteration %d) ===\n", time_str, iteration);
    printf("File: %s\n", g->hdr->magic);
    printf("Version: %u\n", g->hdr->version);
    printf("File Size: %.2f GB\n", g->hdr->file_size / (1024.0 * 1024.0 * 1024.0));
    
    printf("\n--- Graph Structure ---\n");
    printf("Hot Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("Hot Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("Blob Size: %.2f MB\n", g->blob_size / (1024.0 * 1024.0));
    printf("Cold Data: %.2f GB\n", g->cold_data_size / (1024.0 * 1024.0 * 1024.0));
    
    printf("\n--- UEL State (Relative Measures) ---\n");
    printf("Avg Chaos: %.6f\n", g->avg_chaos);
    printf("Avg Activation: %.6f\n", g->avg_activation);
    printf("Avg Edge Strength: %.6f\n", g->avg_edge_strength);
    
    printf("\n--- Drive Mechanisms ---\n");
    printf("Avg Output Activity: %.6f\n", g->avg_output_activity);
    printf("Avg Feedback Correlation: %.6f\n", g->avg_feedback_correlation);
    printf("Avg Prediction Accuracy: %.6f\n", g->avg_prediction_accuracy);
    
    /* Count active nodes (activation > threshold) */
    uint64_t active_nodes = 0;
    float active_threshold = g->avg_activation * 0.1f;
    if (active_threshold < 0.001f) active_threshold = 0.001f;
    
    for (uint64_t i = 0; i < g->node_count; i++) {
        if (fabsf(g->nodes[i].a) > active_threshold) {
            active_nodes++;
        }
    }
    
    printf("\n--- Activity ---\n");
    printf("Active Nodes: %llu (%.2f%%)\n", 
           (unsigned long long)active_nodes,
           (g->node_count > 0) ? (100.0 * active_nodes / g->node_count) : 0.0);
    
    /* Queue status (simplified - just show size) */
    if (g->prop_queue) {
        printf("Propagation Queue Size: %llu\n", (unsigned long long)g->prop_queue_size);
    }
    
    printf("========================================\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <melvin.m file> [update_interval_seconds]\n", argv[0]);
        fprintf(stderr, "  Default update interval: 5 seconds\n");
        return 1;
    }
    
    const char *path = argv[1];
    int interval = (argc > 2) ? atoi(argv[2]) : 5;
    if (interval < 1) interval = 5;
    
    printf("Monitoring Melvin brain: %s\n", path);
    printf("Update interval: %d seconds\n", interval);
    printf("Press Ctrl+C to stop\n\n");
    
    Graph *g = melvin_open(path, 0, 0, 0);
    if (!g) {
        fprintf(stderr, "Failed to open %s\n", path);
        return 1;
    }
    
    int iteration = 0;
    while (1) {
        /* Reopen to get fresh stats (if file is being modified) */
        melvin_close(g);
        g = melvin_open(path, 0, 0, 0);
        if (!g) {
            fprintf(stderr, "Failed to reopen %s\n", path);
            break;
        }
        
        print_stats(g, iteration++);
        sleep(interval);
    }
    
    melvin_close(g);
    return 0;
}


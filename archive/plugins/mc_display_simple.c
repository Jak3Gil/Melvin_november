#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

// Melvin graph structures (from melvin.h)
typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t tick;
    uint64_t node_cap;
    uint64_t edge_cap;
} BrainHeader;

typedef struct {
    float a;
    float bias;
    float decay;
    uint32_t kind;
    uint32_t flags;
    float reliability;
    uint32_t success_count;
    uint32_t failure_count;
    uint32_t mc_id;
    uint16_t mc_flags;
    uint16_t mc_role;
    float value;
} Node;

typedef struct {
    uint64_t src;
    uint64_t dst;
    float w;
    uint32_t flags;
    float elig;
    uint32_t usage_count;
} Edge;

typedef struct {
    BrainHeader *header;
    Node *nodes;
    Edge *edges;
    size_t mmap_size;
    int fd;
} Brain;

// Display state
static int display_initialized = 0;
static uint64_t last_display_tick = 0;

// Render graph visualization to console
static void render_graph_console(Brain *g) {
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    // Clear screen (ANSI escape codes)
    printf("\033[2J\033[H"); // Clear and home cursor
    
    // Header
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║           MELVIN GRAPH VISUALIZATION                   ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Stats
    printf("Tick: %llu\n", (unsigned long long)g->header->tick);
    printf("Nodes: %llu / %llu  (%.1f%%)\n", 
           (unsigned long long)n, 
           (unsigned long long)g->header->node_cap,
           (double)n / (double)g->header->node_cap * 100.0);
    printf("Edges: %llu / %llu  (%.1f%%)\n",
           (unsigned long long)e_count,
           (unsigned long long)g->header->edge_cap,
           (double)e_count / (double)g->header->edge_cap * 100.0);
    printf("\n");
    
    // Graph representation (ASCII art)
    printf("Graph Structure:\n");
    for (int i = 0; i < 50; i++) printf("─");
    printf("\n");
    
    // Show active nodes (top 20)
    uint64_t max_display = (n < 20) ? n : 20;
    printf("Top Active Nodes:\n");
    
    // Sort nodes by activation (simple bubble for small N)
    uint64_t *active_nodes = (uint64_t *)malloc(max_display * sizeof(uint64_t));
    if (active_nodes) {
        uint64_t count = 0;
        for (uint64_t i = 0; i < n && count < max_display; i++) {
            if (g->nodes[i].a > 0.1f) {
                active_nodes[count++] = i;
            }
        }
        
        // Simple sort by activation
        for (uint64_t i = 0; i < count; i++) {
            for (uint64_t j = i + 1; j < count; j++) {
                if (g->nodes[active_nodes[j]].a > g->nodes[active_nodes[i]].a) {
                    uint64_t tmp = active_nodes[i];
                    active_nodes[i] = active_nodes[j];
                    active_nodes[j] = tmp;
                }
            }
        }
        
        // Display top active nodes
        for (uint64_t i = 0; i < count && i < 10; i++) {
            Node *node = &g->nodes[active_nodes[i]];
            float bar_length = node->a * 30.0f;
            
            printf("  Node %4llu: [", (unsigned long long)active_nodes[i]);
            for (int j = 0; j < (int)bar_length; j++) {
                printf("█");
            }
            for (int j = (int)bar_length; j < 30; j++) {
                printf("░");
            }
            printf("] %.3f (kind=%u)\n", node->a, node->kind);
        }
        
        free(active_nodes);
    }
    
    printf("\n");
    
    // Show edge statistics
    uint64_t strong_edges = 0;
    uint64_t weak_edges = 0;
    float total_weight = 0.0f;
    
    for (uint64_t i = 0; i < e_count; i++) {
        float w = fabsf(g->edges[i].w);
        total_weight += w;
        if (w > 1.0f) strong_edges++;
        else if (w > 0.01f) weak_edges++;
    }
    
    printf("Edge Statistics:\n");
    printf("  Strong edges (>1.0): %llu\n", (unsigned long long)strong_edges);
    printf("  Weak edges (0.01-1.0): %llu\n", (unsigned long long)weak_edges);
    printf("  Average weight: %.3f\n", e_count > 0 ? total_weight / (float)e_count : 0.0f);
    printf("\n");
    
    // Node type distribution
    uint64_t kind_counts[16] = {0};
    for (uint64_t i = 0; i < n; i++) {
        if (g->nodes[i].kind < 16) {
            kind_counts[g->nodes[i].kind]++;
        }
    }
    
    printf("Node Types:\n");
    const char *kind_names[] = {"BLANK", "DATA", "PATTERN", "CONTROL", "TAG", "META"};
    for (int k = 0; k < 6; k++) {
        if (kind_counts[k] > 0) {
            printf("  %s: %llu\n", kind_names[k], (unsigned long long)kind_counts[k]);
        }
    }
    
    printf("\n");
    printf("Press Ctrl+C to stop\n");
    fflush(stdout);
}

// MC Function: Display graph on console
void mc_display_graph(Brain *g, uint64_t node_id) {
    // Only update display every N ticks for performance
    if (g->header->tick < last_display_tick + 10) {
        return;
    }
    
    last_display_tick = g->header->tick;
    
    // Render to console
    render_graph_console(g);
}

// MC Function: Initialize display
void mc_display_init(Brain *g, uint64_t node_id) {
    if (!display_initialized) {
        printf("[mc_display] Console visualization initialized\n");
        display_initialized = 1;
        
        // Set node activation to trigger continuous display
        if (node_id < g->header->node_cap) {
            g->nodes[node_id].a = 1.0f;
            g->nodes[node_id].bias = 5.0f; // High bias to keep active
        }
    }
}


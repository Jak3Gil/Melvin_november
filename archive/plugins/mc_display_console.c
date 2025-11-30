#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <math.h>
#include <time.h>

// Simple console/TTY display for Jetson
// Writes directly to console output that shows on display port

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
static int console_fd = -1;
static int display_initialized = 0;
static uint64_t last_display_tick = 0;

// Initialize console display
static int init_console_display(void) {
    if (display_initialized) return 1;
    
    // Try to open console/TTY
    const char *console_devices[] = {
        "/dev/tty0",      // Virtual console
        "/dev/console",   // System console
        "/dev/tty1",      // First TTY
        NULL
    };
    
    for (int i = 0; console_devices[i]; i++) {
        console_fd = open(console_devices[i], O_WRONLY);
        if (console_fd >= 0) {
            printf("[mc_display] Opened console: %s\n", console_devices[i]);
            display_initialized = 1;
            return 1;
        }
    }
    
    // Fallback to stdout/stderr
    printf("[mc_display] No console device found, using stdout\n");
    console_fd = 1; // stdout
    display_initialized = 1;
    return 1;
}

// Write to console
static void write_console(const char *text, size_t len) {
    if (console_fd >= 0) {
        write(console_fd, text, len);
    } else {
        write(1, text, len); // stdout
    }
}

// Render graph visualization to console
static void render_graph_console(Brain *g) {
    char buffer[4096];
    int pos = 0;
    
    // Clear screen (ANSI escape codes)
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "\033[2J\033[H");
    
    // Header
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "╔════════════════════════════════════════════════════╗\n"
                    "║        MELVIN GRAPH VISUALIZATION                   ║\n"
                    "╚════════════════════════════════════════════════════╝\n"
                    "\n");
    
    // Stats
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "Tick: %llu\n",
                    (unsigned long long)g->header->tick);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "Nodes: %llu / %llu  (%.1f%%)\n",
                    (unsigned long long)g->header->num_nodes,
                    (unsigned long long)g->header->node_cap,
                    (double)g->header->num_nodes / (double)g->header->node_cap * 100.0);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "Edges: %llu / %llu  (%.1f%%)\n",
                    (unsigned long long)g->header->num_edges,
                    (unsigned long long)g->header->edge_cap,
                    (double)g->header->num_edges / (double)g->header->edge_cap * 100.0);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "\n");
    
    // Graph representation
    for (int i = 0; i < 50; i++) {
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "─");
    }
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "\n\n");
    
    // Top active nodes
    uint64_t n = g->header->num_nodes;
    uint64_t max_display = (n < 15) ? n : 15;
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "Top Active Nodes:\n");
    
    // Find active nodes
    uint64_t active_nodes[15];
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
    
    // Display top nodes
    for (uint64_t i = 0; i < count && i < 10; i++) {
        Node *node = &g->nodes[active_nodes[i]];
        float bar_length = node->a * 30.0f;
        
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "  Node %4llu: [",
                        (unsigned long long)active_nodes[i]);
        
        for (int j = 0; j < (int)bar_length && j < 30; j++) {
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, "█");
        }
        for (int j = (int)bar_length; j < 30; j++) {
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, "░");
        }
        
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "] %.3f\n",
                        node->a);
    }
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "\n");
    
    // Edge statistics
    uint64_t e_count = g->header->num_edges;
    uint64_t strong_edges = 0;
    float total_weight = 0.0f;
    
    uint64_t max_check = (e_count < 5000) ? e_count : 5000;
    for (uint64_t i = 0; i < max_check; i++) {
        float w = fabsf(g->edges[i].w);
        total_weight += w;
        if (w > 1.0f) strong_edges++;
    }
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "Edge Statistics:\n");
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "  Strong edges (>1.0): %llu\n",
                    (unsigned long long)strong_edges);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
                    "  Average weight: %.3f\n",
                    max_check > 0 ? total_weight / (float)max_check : 0.0f);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "\n");
    
    // Node type distribution
    uint64_t kind_counts[6] = {0};
    for (uint64_t i = 0; i < n; i++) {
        if (g->nodes[i].kind < 6) {
            kind_counts[g->nodes[i].kind]++;
        }
    }
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "Node Types:\n");
    const char *kind_names[] = {"BLANK", "DATA", "PATTERN", "CONTROL", "TAG", "META"};
    for (int k = 0; k < 6; k++) {
        if (kind_counts[k] > 0) {
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, "  %s: %llu\n",
                            kind_names[k], (unsigned long long)kind_counts[k]);
        }
    }
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "\n");
    
    // Write to console
    write_console(buffer, pos);
    fsync(console_fd >= 0 ? console_fd : 1);
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
        init_console_display();
        printf("[mc_display] Console visualization initialized\n");
        
        // Set node activation to trigger continuous display
        if (node_id < g->header->node_cap) {
            g->nodes[node_id].a = 1.0f;
            g->nodes[node_id].bias = 5.0f; // High bias to keep active
        }
    }
}



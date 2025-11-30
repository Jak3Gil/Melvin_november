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

// Forward declarations from melvin.c
extern void melvin_tick(Brain *g);
extern uint32_t g_mc_count;
extern uint64_t feedback_channel_node;
extern uint64_t feedback_output_node;

int main(int argc, char **argv) {
    const char *db_path = "melvin.m";
    if (argc > 1) {
        db_path = argv[1];
    }
    
    printf("========================================\n");
    printf("MELVIN 100 TICK TEST - FEEDBACK LOOP\n");
    printf("========================================\n");
    printf("Brain file: %s\n", db_path);
    printf("\n");
    
    // Open brain file
    int fd = open(db_path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Could not open %s\n", db_path);
        fprintf(stderr, "Run melvin_minit first to initialize the brain.\n");
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
    g.fd = fd;
    g.mmap_size = filesize;
    g.header = (BrainHeader*)map;
    
    size_t header_size = sizeof(BrainHeader);
    size_t node_size = sizeof(Node);
    size_t edge_size = sizeof(Edge);
    
    // Initialize if needed
    int needs_init = 0;
    if (g.header->node_capacity == 0 || g.header->edge_capacity == 0) {
        needs_init = 1;
    }
    
    if (needs_init) {
        g.header->node_capacity = 10000;
        g.header->edge_capacity = 50000;
        g.header->node_region_offset = header_size;
        g.header->edge_region_offset = g.header->node_region_offset + 
                                       g.header->node_capacity * node_size;
        
        size_t required_size = g.header->edge_region_offset + 
                              g.header->edge_capacity * edge_size;
        
        if (filesize < required_size) {
            if (ftruncate(fd, required_size) < 0) {
                perror("ftruncate");
                return 1;
            }
            munmap(map, filesize);
            map = mmap(NULL, required_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (map == MAP_FAILED) {
                perror("mmap after grow");
                return 1;
            }
            g.header = (BrainHeader*)map;
            g.mmap_size = required_size;
        }
    }
    
    // Set pointers
    uint8_t *base = (uint8_t*)map;
    g.nodes = (Node*)(base + g.header->node_region_offset);
    g.edges = (Edge*)(base + g.header->edge_region_offset);
    
    // Initialize parameters if needed
    if (g.header->edge_activation_threshold == 0.0f) {
        g.header->edge_activation_threshold = 0.3f;
    }
    if (g.header->mc_execution_threshold == 0.0f) {
        g.header->mc_execution_threshold = 0.3f;
    }
    if (g.header->decay_factor == 0.0f) {
        g.header->decay_factor = 0.99f;
    }
    if (g.header->edge_creation_score_threshold == 0.0f) {
        g.header->edge_creation_score_threshold = 0.15f;
    }
    if (g.header->learning_rate == 0.0f) {
        g.header->learning_rate = 0.01f;
    }
    if (g.header->alpha_blend == 0.0f) {
        g.header->alpha_blend = 0.3f;
    }
    
    // Initial state
    uint64_t start_tick = g.header->tick;
    uint64_t start_nodes = g.header->num_nodes;
    uint64_t start_edges = g.header->num_edges;
    
    printf("INITIAL STATE:\n");
    printf("  Tick: %llu\n", (unsigned long long)start_tick);
    printf("  Nodes: %llu\n", (unsigned long long)start_nodes);
    printf("  Edges: %llu\n", (unsigned long long)start_edges);
    printf("\n");
    
    // Count initial patterns and MC nodes
    uint64_t start_patterns = 0;
    uint64_t start_mc_nodes = 0;
    for (uint64_t i = 0; i < start_nodes; i++) {
        if (g.nodes[i].kind == NODE_KIND_PATTERN_ROOT) start_patterns++;
        if (g.nodes[i].mc_id > 0) start_mc_nodes++;
    }
    printf("  Patterns: %llu\n", (unsigned long long)start_patterns);
    printf("  MC nodes: %llu\n", (unsigned long long)start_mc_nodes);
    printf("\n");
    
    // Run 100 ticks
    printf("RUNNING 100 TICKS...\n");
    printf("(Showing feedback loop activity)\n");
    printf("\n");
    
    time_t start_time = time(NULL);
    uint64_t target_ticks = start_tick + 100;
    
    uint64_t feedback_activations = 0;
    uint64_t pattern_activations = 0;
    uint64_t mc_executions = 0;
    
    while (g.header->tick < target_ticks) {
        uint64_t tick_before = g.header->tick;
        uint64_t nodes_before = g.header->num_nodes;
        uint64_t edges_before = g.header->num_edges;
        
        melvin_tick(&g);
        
        uint64_t tick_after = g.header->tick;
        uint64_t nodes_after = g.header->num_nodes;
        uint64_t edges_after = g.header->num_edges;
        
        // Check feedback loop activity
        if (feedback_channel_node != UINT64_MAX && feedback_channel_node < g.header->num_nodes) {
            if (g.nodes[feedback_channel_node].a > 0.1f) {
                feedback_activations++;
            }
        }
        if (feedback_output_node != UINT64_MAX && feedback_output_node < g.header->num_nodes) {
            if (g.nodes[feedback_output_node].a > 0.1f) {
                // Feedback output is active
            }
        }
        
        // Count pattern activations
        for (uint64_t i = 0; i < g.header->num_nodes && i < 10000; i++) {
            if (g.nodes[i].kind == NODE_KIND_PATTERN_ROOT && g.nodes[i].a > 0.3f) {
                pattern_activations++;
            }
        }
        
        // Count MC executions (nodes with success_count increased)
        for (uint64_t i = 0; i < g.header->num_nodes && i < 10000; i++) {
            if (g.nodes[i].mc_id > 0 && g.nodes[i].success_count > 0) {
                mc_executions++;
            }
        }
        
        // Show progress every 10 ticks or when graph changes
        if (tick_after % 10 == 0 || nodes_after != nodes_before || edges_after != edges_before) {
            float fb_in_activation = 0.0f;
            float fb_out_activation = 0.0f;
            if (feedback_channel_node != UINT64_MAX && feedback_channel_node < g.header->num_nodes) {
                fb_in_activation = g.nodes[feedback_channel_node].a;
            }
            if (feedback_output_node != UINT64_MAX && feedback_output_node < g.header->num_nodes) {
                fb_out_activation = g.nodes[feedback_output_node].a;
            }
            
            printf("Tick %llu: Nodes=%llu Edges=%llu FB_IN=%.3f FB_OUT=%.3f\n",
                   (unsigned long long)tick_after,
                   (unsigned long long)nodes_after,
                   (unsigned long long)edges_after,
                   fb_in_activation,
                   fb_out_activation);
            fflush(stdout);
        }
    }
    
    time_t end_time = time(NULL);
    double elapsed = difftime(end_time, start_time);
    
    // Final state
    uint64_t final_tick = g.header->tick;
    uint64_t final_nodes = g.header->num_nodes;
    uint64_t final_edges = g.header->num_edges;
    
    // Count final patterns
    uint64_t final_patterns = 0;
    uint64_t final_mc_nodes = 0;
    for (uint64_t i = 0; i < final_nodes; i++) {
        if (g.nodes[i].kind == NODE_KIND_PATTERN_ROOT) final_patterns++;
        if (g.nodes[i].mc_id > 0) final_mc_nodes++;
    }
    
    // Check feedback nodes
    float final_fb_in = 0.0f;
    float final_fb_out = 0.0f;
    if (feedback_channel_node != UINT64_MAX && feedback_channel_node < final_nodes) {
        final_fb_in = g.nodes[feedback_channel_node].a;
    }
    if (feedback_output_node != UINT64_MAX && feedback_output_node < final_nodes) {
        final_fb_out = g.nodes[feedback_output_node].a;
    }
    
    printf("\n");
    printf("========================================\n");
    printf("FINAL STATE (After 100 ticks)\n");
    printf("========================================\n");
    printf("Tick: %llu (started at %llu)\n", 
           (unsigned long long)final_tick, (unsigned long long)start_tick);
    printf("Nodes: %llu (started with %llu, +%lld)\n",
           (unsigned long long)final_nodes,
           (unsigned long long)start_nodes,
           (long long)(final_nodes - start_nodes));
    printf("Edges: %llu (started with %llu, +%lld)\n",
           (unsigned long long)final_edges,
           (unsigned long long)start_edges,
           (long long)(final_edges - start_edges));
    printf("Patterns: %llu (started with %llu, +%lld)\n",
           (unsigned long long)final_patterns,
           (unsigned long long)start_patterns,
           (long long)(final_patterns - start_patterns));
    printf("\n");
    printf("FEEDBACK LOOP STATUS:\n");
    printf("  Feedback Input Node: activation=%.3f\n", final_fb_in);
    printf("  Feedback Output Node: activation=%.3f\n", final_fb_out);
    printf("  Feedback loop active: %s\n", 
           (final_fb_in > 0.1f || final_fb_out > 0.1f) ? "YES ✓" : "NO");
    printf("\n");
    printf("ACTIVITY SUMMARY:\n");
    printf("  Pattern activations detected: %llu\n", (unsigned long long)pattern_activations);
    printf("  MC node executions: %llu\n", (unsigned long long)mc_executions);
    printf("  Feedback activations: %llu\n", (unsigned long long)feedback_activations);
    printf("\n");
    printf("Time elapsed: %.2f seconds\n", elapsed);
    printf("Ticks per second: %.1f\n", 100.0 / elapsed);
    printf("\n");
    
    // Check if feedback edge exists
    int feedback_edge_exists = 0;
    if (feedback_channel_node != UINT64_MAX && feedback_output_node != UINT64_MAX) {
        uint64_t e_count = g.header->num_edges;
        Node *fb_in = &g.nodes[feedback_channel_node];
        uint64_t edge_id = fb_in->first_in;
        
        while (edge_id != UINT64_MAX && edge_id < e_count) {
            Edge *e = &g.edges[edge_id];
            if (e->src == feedback_output_node) {
                feedback_edge_exists = 1;
                printf("✓ Feedback edge exists: Output → Input (weight=%.3f)\n", e->w);
                break;
            }
            edge_id = e->next_in;
        }
    }
    
    if (!feedback_edge_exists && (final_fb_in > 0.0f || final_fb_out > 0.0f)) {
        printf("⚠ Feedback nodes active but edge not found (may be created next tick)\n");
    }
    
    printf("========================================\n");
    
    // Sync to disk
    msync(map, g.mmap_size, MS_SYNC);
    
    munmap(map, g.mmap_size);
    close(fd);
    
    return 0;
}


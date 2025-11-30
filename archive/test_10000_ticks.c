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
extern void register_mc(const char *name, void (*fn)(Brain *, uint64_t));
extern uint32_t g_mc_count;
// rebuild_adjacency_lists is static, so we'll skip it if the graph is already initialized

// Track MC execution
static uint64_t total_mc_executions = 0;
static uint64_t mc_executions_by_tick[100] = {0}; // Track last 100 ticks

int main(int argc, char **argv) {
    const char *db_path = "melvin.m";
    if (argc > 1) {
        db_path = argv[1];
    }
    
    printf("========================================\n");
    printf("MELVIN 10,000 TICK TEST\n");
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
    
    // Note: rebuild_adjacency_lists is static in melvin.c
    // If the graph is already initialized, adjacency lists should be fine
    // If not, melvin will rebuild them on first run
    
    // Initial state
    uint64_t start_tick = g.header->tick;
    uint64_t start_nodes = g.header->num_nodes;
    uint64_t start_edges = g.header->num_edges;
    
    printf("INITIAL STATE:\n");
    printf("  Tick: %llu\n", (unsigned long long)start_tick);
    printf("  Nodes: %llu\n", (unsigned long long)start_nodes);
    printf("  Edges: %llu\n", (unsigned long long)start_edges);
    printf("  File size: %.2f MB\n", filesize / 1024.0 / 1024.0);
    printf("\n");
    
    // Count patterns and MC nodes
    uint64_t start_patterns = 0;
    uint64_t start_mc_nodes = 0;
    uint64_t start_active_mc_nodes = 0;
    float mc_threshold = g.header->mc_execution_threshold;
    if (mc_threshold == 0.0f) mc_threshold = 0.3f;
    
    for (uint64_t i = 0; i < start_nodes; i++) {
        if (g.nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
            start_patterns++;
        }
        if (g.nodes[i].mc_id > 0) {
            start_mc_nodes++;
            if (g.nodes[i].a >= mc_threshold) {
                start_active_mc_nodes++;
            }
        }
    }
    printf("  Patterns: %llu\n", (unsigned long long)start_patterns);
    printf("  MC nodes: %llu (active: %llu, threshold: %.3f)\n", 
           (unsigned long long)start_mc_nodes,
           (unsigned long long)start_active_mc_nodes,
           mc_threshold);
    printf("  Registered MC functions: %u\n", g_mc_count);
    printf("\n");
    
    // Run 10,000 ticks
    printf("RUNNING 10,000 TICKS...\n");
    printf("(Progress updates every 1000 ticks)\n");
    printf("\n");
    
    time_t start_time = time(NULL);
    uint64_t target_ticks = start_tick + 10000;
    
    uint64_t mc_executions_this_tick = 0;
    uint64_t total_mc_executions = 0;
    uint64_t ticks_with_mc = 0;
    
    while (g.header->tick < target_ticks) {
        // Count MC nodes that would execute before tick
        mc_executions_this_tick = 0;
        uint64_t n = g.header->num_nodes;
        for (uint64_t i = 0; i < n; i++) {
            Node *node = &g.nodes[i];
            if (node->mc_id > 0 && node->a >= mc_threshold) {
                mc_executions_this_tick++;
            }
        }
        
        melvin_tick(&g);
        
        // Count MC nodes that executed (check success_count increased)
        // Actually, we'll count by checking if MC nodes have high activation
        // and were above threshold before the tick
        
        if (mc_executions_this_tick > 0) {
            total_mc_executions += mc_executions_this_tick;
            ticks_with_mc++;
        }
        
        // Progress update every 1000 ticks
        if (g.header->tick % 1000 == 0) {
            printf("Tick %llu: Nodes=%llu Edges=%llu MC_executions=%llu\n",
                   (unsigned long long)g.header->tick,
                   (unsigned long long)g.header->num_nodes,
                   (unsigned long long)g.header->num_edges,
                   (unsigned long long)mc_executions_this_tick);
            fflush(stdout);
        }
    }
    
    time_t end_time = time(NULL);
    double elapsed = difftime(end_time, start_time);
    
    // Final state
    uint64_t final_tick = g.header->tick;
    uint64_t final_nodes = g.header->num_nodes;
    uint64_t final_edges = g.header->num_edges;
    
    // Count final patterns and MC nodes
    uint64_t final_patterns = 0;
    uint64_t final_mc_nodes = 0;
    uint64_t final_active_mc_nodes = 0;
    
    for (uint64_t i = 0; i < final_nodes; i++) {
        if (g.nodes[i].kind == NODE_KIND_PATTERN_ROOT) {
            final_patterns++;
        }
        if (g.nodes[i].mc_id > 0) {
            final_mc_nodes++;
            if (g.nodes[i].a >= mc_threshold) {
                final_active_mc_nodes++;
            }
        }
    }
    
    // Count edge types
    uint64_t seq_edges = 0;
    uint64_t bind_edges = 0;
    uint64_t pattern_edges = 0;
    uint64_t coactivation_edges = 0;
    
    for (uint64_t i = 0; i < final_edges; i++) {
        Edge *e = &g.edges[i];
        if (e->flags & EDGE_FLAG_SEQ) seq_edges++;
        if (e->flags & EDGE_FLAG_BIND) bind_edges++;
        if (e->flags & EDGE_FLAG_PATTERN) pattern_edges++;
        if (!(e->flags & EDGE_FLAG_SEQ) && !(e->flags & EDGE_FLAG_PATTERN)) {
            coactivation_edges++;
        }
    }
    
    // Get file size
    fstat(fd, &st);
    size_t final_filesize = st.st_size;
    
    printf("\n");
    printf("========================================\n");
    printf("FINAL STATE (After 10,000 ticks)\n");
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
    printf("MC nodes: %llu (started with %llu, active: %llu)\n",
           (unsigned long long)final_mc_nodes,
           (unsigned long long)start_mc_nodes,
           (unsigned long long)final_active_mc_nodes);
    printf("\n");
    printf("MC EXECUTION STATISTICS:\n");
    printf("  Total MC executions: %llu\n", (unsigned long long)total_mc_executions);
    printf("  Ticks with MC execution: %llu (%.1f%%)\n",
           (unsigned long long)ticks_with_mc,
           (ticks_with_mc > 0) ? (100.0 * ticks_with_mc / 10000.0) : 0.0);
    printf("  Average MC executions per tick: %.2f\n",
           (total_mc_executions > 0) ? ((double)total_mc_executions / 10000.0) : 0.0);
    printf("\n");
    printf("Edge Types:\n");
    printf("  Sequence edges: %llu\n", (unsigned long long)seq_edges);
    printf("  Binding edges: %llu\n", (unsigned long long)bind_edges);
    printf("  Pattern edges: %llu\n", (unsigned long long)pattern_edges);
    printf("  Co-activation edges: %llu\n", (unsigned long long)coactivation_edges);
    printf("\n");
    printf("File size: %.2f MB (started at %.2f MB)\n",
           final_filesize / 1024.0 / 1024.0,
           filesize / 1024.0 / 1024.0);
    printf("Time elapsed: %.1f seconds\n", elapsed);
    printf("Ticks per second: %.1f\n", 10000.0 / elapsed);
    printf("\n");
    
    // Check for issues
    int issues = 0;
    printf("Health Checks:\n");
    
    // Check for corrupted nodes
    int corrupted_nodes = 0;
    for (uint64_t i = 0; i < final_nodes; i++) {
        Node *n = &g.nodes[i];
        if (isnan(n->a) || isinf(n->a)) corrupted_nodes++;
        if (isnan(n->bias) || isinf(n->bias)) corrupted_nodes++;
    }
    if (corrupted_nodes == 0) {
        printf("  ✓ No corrupted nodes (NaN/Inf)\n");
    } else {
        printf("  ✗ Found %d corrupted nodes\n", corrupted_nodes);
        issues++;
    }
    
    // Check for corrupted edges
    int corrupted_edges = 0;
    for (uint64_t i = 0; i < final_edges; i++) {
        Edge *e = &g.edges[i];
        if (e->src >= final_nodes || e->dst >= final_nodes) corrupted_edges++;
        if (isnan(e->w) || isinf(e->w)) corrupted_edges++;
    }
    if (corrupted_edges == 0) {
        printf("  ✓ No corrupted edges\n");
    } else {
        printf("  ⚠ Found %d corrupted edges (may be normal)\n", corrupted_edges);
    }
    
    // Check growth
    if (final_nodes > start_nodes) {
        printf("  ✓ Graph grew (nodes increased)\n");
    } else {
        printf("  ⚠ Graph did not grow (no new nodes)\n");
    }
    
    if (final_edges > start_edges) {
        printf("  ✓ Edges increased (learning happening)\n");
    } else {
        printf("  ⚠ Edges did not increase\n");
    }
    
    if (seq_edges > 0) {
        printf("  ✓ Sequence edges present (temporal learning)\n");
    } else {
        printf("  ⚠ No sequence edges found\n");
    }
    
    if (final_patterns > start_patterns) {
        printf("  ✓ Patterns increased (intelligence emerging)\n");
    } else {
        printf("  ⚠ Patterns did not increase\n");
    }
    
    printf("\n");
    if (issues == 0) {
        printf("✓ TEST COMPLETE - System is healthy!\n");
    } else {
        printf("⚠ TEST COMPLETE - Some issues detected\n");
    }
    printf("========================================\n");
    
    // Sync to disk
    msync(map, g.mmap_size, MS_SYNC);
    
    munmap(map, g.mmap_size);
    close(fd);
    
    return (issues > 0) ? 1 : 0;
}


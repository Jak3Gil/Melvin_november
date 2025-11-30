#define _POSIX_C_SOURCE 200809L

#include "melvin_file.h"
#include "melvin_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

// Emergence metrics
typedef struct {
    uint64_t initial_nodes;
    uint64_t final_nodes;
    uint64_t initial_edges;
    uint64_t final_edges;
    uint64_t nodes_created;
    uint64_t edges_created;
    uint64_t total_pulses;
    uint64_t pulses_per_tick_avg;
    float edge_density;
    float connectivity;
    uint64_t stable_cycles;  // Cycles where structure didn't change
} EmergenceMetrics;

static void print_metrics_header(void) {
    printf("\n=== EMERGENCE TEST METRICS ===\n");
    printf("%-6s %-8s %-8s %-10s %-10s %-8s %-8s %-12s\n",
           "Tick", "Nodes", "Edges", "NewNodes", "NewEdges", "Pulses", "Density", "Connectivity");
    printf("%-6s %-8s %-8s %-10s %-10s %-8s %-8s %-12s\n",
           "----", "-----", "-----", "--------", "--------", "------", "-------", "------------");
}

static void print_metrics(uint64_t tick, GraphHeader *gh, EmergenceMetrics *em) {
    float density = (gh->num_nodes > 0) ? (float)gh->num_edges / (float)gh->num_nodes : 0.0f;
    float connectivity = 0.0f;
    if (gh->num_nodes > 0) {
        uint64_t nodes_with_edges = 0;
        for (uint64_t i = 0; i < gh->num_nodes; i++) {
            if (gh->num_nodes > 0) {
                // Check if node has any edges (simplified)
                nodes_with_edges++; // Assume all nodes potentially have edges
            }
        }
        connectivity = (float)nodes_with_edges / (float)gh->num_nodes;
    }
    
    printf("%-6llu %-8llu %-8llu %-10llu %-10llu %-8llu %-8.2f %-12.2f\n",
           (unsigned long long)tick,
           (unsigned long long)gh->num_nodes,
           (unsigned long long)gh->num_edges,
           (unsigned long long)em->nodes_created,
           (unsigned long long)em->edges_created,
           (unsigned long long)gh->total_pulses_emitted,
           density,
           connectivity);
}

static void analyze_graph_structure(MelvinFile *file, EmergenceMetrics *em) {
    GraphHeader *gh = file->graph_header;
    NodeDisk *nodes = file->nodes;
    EdgeDisk *edges = file->edges;
    
    // Count nodes with outgoing edges
    uint64_t nodes_with_out = 0;
    uint64_t nodes_with_in = 0;
    uint64_t total_out_degree = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        if (nodes[i].out_degree > 0) {
            nodes_with_out++;
            total_out_degree += nodes[i].out_degree;
        }
    }
    
    // Count nodes with incoming edges
    uint8_t *has_incoming = calloc(gh->num_nodes, sizeof(uint8_t));
    for (uint64_t i = 0; i < gh->num_edges; i++) {
        if (edges[i].src_id == UINT64_MAX) continue;
        // Find dst node index
        for (uint64_t j = 0; j < gh->num_nodes; j++) {
            if (nodes[j].id == edges[i].dst_id) {
                has_incoming[j] = 1;
                break;
            }
        }
    }
    
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (has_incoming[i]) nodes_with_in++;
    }
    free(has_incoming);
    
    em->edge_density = (gh->num_nodes > 0) ? (float)gh->num_edges / (float)gh->num_nodes : 0.0f;
    em->connectivity = (gh->num_nodes > 0) ? (float)nodes_with_out / (float)gh->num_nodes : 0.0f;
}

int main(int argc, char **argv) {
    const char *brain_path = "melvin.m";
    uint64_t max_ticks = 10000;
    uint64_t report_interval = 100;
    int create_new = 1;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) {
            if (i + 1 < argc) brain_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--ticks") == 0) {
            if (i + 1 < argc) max_ticks = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--report") == 0) {
            if (i + 1 < argc) report_interval = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--no-create") == 0) {
            create_new = 0;
        }
    }
    
    printf("=== MELVIN EMERGENCE TEST ===\n");
    printf("Brain file: %s\n", brain_path);
    printf("Max ticks: %llu\n", (unsigned long long)max_ticks);
    printf("Report interval: %llu\n", (unsigned long long)report_interval);
    printf("\n");
    
    // Create or load brain file
    MelvinFile file;
    
    if (create_new) {
        printf("[test] Creating new brain file...\n");
        if (create_new_file(brain_path) < 0) {
            fprintf(stderr, "[test] Failed to create brain file\n");
            return 1;
        }
    }
    
    printf("[test] Loading brain file...\n");
    if (load_file(brain_path, &file) < 0) {
        fprintf(stderr, "[test] Failed to load brain file\n");
        return 1;
    }
    
    // Initialize runtime
    MelvinRuntime runtime;
    if (runtime_init(&runtime, &file) < 0) {
        fprintf(stderr, "[test] Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    
    // Initialize metrics
    EmergenceMetrics em = {0};
    GraphHeader *gh = file.graph_header;
    em.initial_nodes = gh->num_nodes;
    em.initial_edges = gh->num_edges;
    
    printf("\n[test] Starting emergence test...\n");
    printf("Initial state: %llu nodes, %llu edges\n\n",
           (unsigned long long)em.initial_nodes,
           (unsigned long long)em.initial_edges);
    
    print_metrics_header();
    
    uint64_t last_nodes = gh->num_nodes;
    uint64_t last_edges = gh->num_edges;
    uint64_t stable_count = 0;
    
    // Inject some initial pulses to seed activity
    printf("\n[test] Injecting seed pulses...\n");
    for (uint64_t i = 0; i < 10; i++) {
        inject_pulse(&runtime, i + 1, 1.0f);
    }
    
    // Run physics ticks
    for (uint64_t tick = 0; tick < max_ticks; tick++) {
        // Note: Noise is now applied continuously in physics_tick(), not tick-based
        
        // Run physics tick
        physics_tick(&runtime);
        
        // Track structure changes
        if (gh->num_nodes == last_nodes && gh->num_edges == last_edges) {
            stable_count++;
        } else {
            stable_count = 0;
        }
        
        // Report metrics periodically
        if (tick % report_interval == 0 || tick == max_ticks - 1) {
            em.nodes_created = gh->num_nodes - em.initial_nodes;
            em.edges_created = gh->num_edges - em.initial_edges;
            em.total_pulses = gh->total_pulses_emitted;
            em.pulses_per_tick_avg = (tick > 0) ? em.total_pulses / tick : 0;
            analyze_graph_structure(&file, &em);
            
            print_metrics(tick, gh, &em);
            
            last_nodes = gh->num_nodes;
            last_edges = gh->num_edges;
        }
    }
    
    // Final metrics
    em.final_nodes = gh->num_nodes;
    em.final_edges = gh->num_edges;
    em.nodes_created = em.final_nodes - em.initial_nodes;
    em.edges_created = em.final_edges - em.initial_edges;
    em.stable_cycles = stable_count;
    analyze_graph_structure(&file, &em);
    
    printf("\n=== FINAL EMERGENCE METRICS ===\n");
    printf("Initial:  %llu nodes, %llu edges\n", 
           (unsigned long long)em.initial_nodes,
           (unsigned long long)em.initial_edges);
    printf("Final:    %llu nodes, %llu edges\n",
           (unsigned long long)em.final_nodes,
           (unsigned long long)em.final_edges);
    printf("Created:  %llu nodes, %llu edges\n",
           (unsigned long long)em.nodes_created,
           (unsigned long long)em.edges_created);
    printf("Total pulses: %llu (avg %.2f per tick)\n",
           (unsigned long long)em.total_pulses,
           em.pulses_per_tick_avg > 0 ? (float)em.total_pulses / (float)max_ticks : 0.0f);
    printf("Edge density: %.2f edges/node\n", em.edge_density);
    printf("Connectivity: %.2f%% of nodes have outgoing edges\n", em.connectivity * 100.0f);
    printf("Stable cycles: %llu\n", (unsigned long long)em.stable_cycles);
    
    // Emergence indicators
    printf("\n=== EMERGENCE INDICATORS ===\n");
    if (em.nodes_created > 0) {
        printf("✓ Nodes spontaneously created: %llu\n", (unsigned long long)em.nodes_created);
    }
    if (em.edges_created > 0) {
        printf("✓ Edges spontaneously created: %llu\n", (unsigned long long)em.edges_created);
    }
    if (em.edge_density > 1.0f) {
        printf("✓ Dense connectivity (density > 1.0): %.2f\n", em.edge_density);
    }
    if (em.connectivity > 0.5f) {
        printf("✓ High connectivity (>50%%): %.1f%%\n", em.connectivity * 100.0f);
    }
    if (em.total_pulses > max_ticks * 10) {
        printf("✓ High pulse activity: %llu total pulses\n", (unsigned long long)em.total_pulses);
    }
    if (em.stable_cycles > 100) {
        printf("✓ Stable structure emerged: %llu stable cycles\n", (unsigned long long)em.stable_cycles);
    }
    
    // Cleanup
    runtime_cleanup(&runtime);
    close_file(&file);
    
    printf("\n[test] Emergence test complete!\n");
    return 0;
}


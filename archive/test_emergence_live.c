#define _POSIX_C_SOURCE 200809L

#include "melvin_file.h"
#include "melvin_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

// Live visualization with real-time updates
static void print_live_stats(uint64_t tick, GraphHeader *gh, MelvinRuntime *rt) {
    float density = (gh->num_nodes > 0) ? (float)gh->num_edges / (float)gh->num_nodes : 0.0f;
    
    // Get active pulse count
    uint64_t active_pulses = rt->current_pulse_count;
    
    // Calculate mean weight
    float mean_weight = 0.0f;
    uint64_t edges_with_weight = 0;
    if (rt->file && rt->file->edges) {
        for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
            EdgeDisk *e = &rt->file->edges[i];
            if (e->src_id != UINT64_MAX && e->weight > 0.001f) {
                mean_weight += e->weight;
                edges_with_weight++;
            }
        }
    }
    if (edges_with_weight > 0) {
        mean_weight /= edges_with_weight;
    }
    
    // Use runtime-stored counts (updated event-driven by detect_formations)
    // If bonds are dirty, trigger detection now (for display)
    if (rt->bonds_dirty) {
        uint64_t bond_count, molecule_count;
        detect_formations(rt, &bond_count, &molecule_count);
        rt->bonds_dirty = 0;
    }
    
    // Print on same line with carriage return
    printf("\r[TICK %6llu] Nodes: %5llu | Edges: %5llu | Bonds: %4llu | Molecules: %3llu | Pulses: %4llu | Density: %5.2f | MeanW: %5.3f",
           (unsigned long long)tick,
           (unsigned long long)gh->num_nodes,
           (unsigned long long)gh->num_edges,
           (unsigned long long)rt->bond_edge_count,
           (unsigned long long)rt->molecule_count,
           (unsigned long long)active_pulses,
           density,
           mean_weight);
    fflush(stdout);
}

// ASCII bar graph for pulse activity
static void print_pulse_bar(uint64_t pulse_count, uint64_t max_pulses) {
    int bar_width = 40;
    int filled = (int)((float)pulse_count / (float)max_pulses * bar_width);
    if (filled > bar_width) filled = bar_width;
    
    printf(" [");
    for (int i = 0; i < filled; i++) printf("█");
    for (int i = filled; i < bar_width; i++) printf(" ");
    printf("]");
}

int main(int argc, char **argv) {
    const char *brain_path = "melvin.m";
    uint64_t max_ticks = 10000;
    uint64_t update_interval = 1; // Update every tick for live view
    int create_new = 1;
    int show_bar = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--file") == 0) {
            if (i + 1 < argc) brain_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--ticks") == 0) {
            if (i + 1 < argc) max_ticks = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "--update") == 0) {
            if (i + 1 < argc) update_interval = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--bar") == 0) {
            show_bar = 1;
        } else if (strcmp(argv[i], "--no-create") == 0) {
            create_new = 0;
        }
    }
    
    printf("=== MELVIN EMERGENCE TEST (LIVE VIEW) ===\n");
    printf("Brain file: %s | Max ticks: %llu\n", brain_path, (unsigned long long)max_ticks);
    printf("Press Ctrl+C to stop\n\n");
    
    // Create or load brain file
    MelvinFile file;
    
    if (create_new) {
        if (create_new_file(brain_path) < 0) {
            fprintf(stderr, "Failed to create brain file\n");
            return 1;
        }
    }
    
    if (load_file(brain_path, &file) < 0) {
        fprintf(stderr, "Failed to load brain file\n");
        return 1;
    }
    
    // Initialize runtime
    MelvinRuntime runtime;
    if (runtime_init(&runtime, &file) < 0) {
        fprintf(stderr, "Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    
    GraphHeader *gh = file.graph_header;
    
    printf("Initial: %llu nodes, %llu edges\n\n", 
           (unsigned long long)gh->num_nodes,
           (unsigned long long)gh->num_edges);
    
    // Inject seed pulses
    for (uint64_t i = 0; i < 10; i++) {
        inject_pulse(&runtime, i + 1, 1.0f);
    }
    
    printf("Seed pulses injected. Starting physics...\n\n");
    
    // Print header
    printf("Tick      Nodes   Edges   Bonds   Molecules Pulses  Density  MeanW\n");
    printf("----------------------------------------------------------------------------\n");
    
    uint64_t last_edge_count = gh->num_edges;
    uint64_t last_node_count = gh->num_nodes;
    uint64_t edges_created_this_tick = 0;
    uint64_t nodes_created_this_tick = 0;
    
    // Run physics ticks
    for (uint64_t tick = 0; tick < max_ticks; tick++) {
        // Note: Noise is now applied continuously in physics_tick(), not tick-based
        
        // Track creation
        uint64_t nodes_before = gh->num_nodes;
        uint64_t edges_before = gh->num_edges;
        
        // Run physics tick
        physics_tick(&runtime);
        
        // Calculate changes
        nodes_created_this_tick = (gh->num_nodes > nodes_before) ? gh->num_nodes - nodes_before : 0;
        edges_created_this_tick = (gh->num_edges > edges_before) ? gh->num_edges - edges_before : 0;
        
        last_node_count = gh->num_nodes;
        last_edge_count = gh->num_edges;
        
        // Update display
        if (tick % update_interval == 0) {
            print_live_stats(tick, gh, &runtime);
            if (show_bar) {
                print_pulse_bar(runtime.current_pulse_count, 100);
            }
            
            // Show creation events
            if (nodes_created_this_tick > 0 || edges_created_this_tick > 0) {
                printf("  [NEW: +%llu nodes, +%llu edges]",
                       (unsigned long long)nodes_created_this_tick,
                       (unsigned long long)edges_created_this_tick);
            }
        }
        
        // Decay instrumentation (test/visualization only - every 200 ticks)
        if (tick % 200 == 0 && tick > 0) {
            float edge_decay = gh->weight_decay;
            float t_half_energy = 0.693f / 0.5f;  // NODE_DECAY_RATE = 0.5f
            float t_half_bond = 0.693f / edge_decay;
            printf("\n[DECAY] node_energy_decay=0.5000 edge_weight_decay=%.4f\n",
                   edge_decay);
            printf("[HALF-LIFE] energy=%.2f ticks, bonds=%.2f ticks\n",
                   t_half_energy, t_half_bond);
        }
        
        // Formation logging (test/visualization only - event-driven)
        // Check if formations were just detected (bonds_dirty was set and cleared)
        static uint64_t last_molecule_count = 0;
        if (runtime.molecule_count != last_molecule_count) {
            printf("\n[FORMATION] molecules=%llu bonds=%llu (detected at tick %llu)\n",
                   (unsigned long long)runtime.molecule_count,
                   (unsigned long long)runtime.bond_edge_count,
                   (unsigned long long)tick);
            last_molecule_count = runtime.molecule_count;
        }
        
        // Small sleep to make it watchable (optional)
        if (update_interval == 1) {
            usleep(10000); // 10ms = 100 ticks per second
        }
    }
    
    // Final newline
    printf("\n\n=== FINAL STATS ===\n");
    printf("Nodes: %llu | Edges: %llu | Density: %.2f\n",
           (unsigned long long)gh->num_nodes,
           (unsigned long long)gh->num_edges,
           (gh->num_nodes > 0) ? (float)gh->num_edges / (float)gh->num_nodes : 0.0f);
    printf("Total pulses emitted: %llu\n", (unsigned long long)gh->total_pulses_emitted);
    printf("Total pulses absorbed: %llu\n", (unsigned long long)gh->total_pulses_absorbed);
    
    // Cleanup
    runtime_cleanup(&runtime);
    close_file(&file);
    
    printf("\nTest complete!\n");
    return 0;
}


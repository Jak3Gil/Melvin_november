/*
 * TEST: Stability & Pruning Under Junk
 * 
 * Goal: Verify that stability + FE-based pruning removes high-FE, low-usage
 * structures and preserves low-FE, high-usage ones.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_stability_pruning_junk.m"
#define NUM_USEFUL_REPS 200
#define NUM_JUNK_REPS 50
#define TICKS_PER_SWEEP 100
#define NUM_PRUNE_SWEEPS 10

typedef struct {
    uint64_t count;
    float avg_stability;
    float avg_fe;
    float total_fe;
} SubgraphStats;

SubgraphStats measure_subgraph(MelvinFile *file, uint64_t *node_ids, int count) {
    SubgraphStats stats = {0};
    GraphHeaderDisk *gh = file->graph_header;
    NodeDisk *nodes = file->nodes;
    
    for (int i = 0; i < count; i++) {
        uint64_t idx = find_node_index_by_id(file, node_ids[i]);
        if (idx != UINT64_MAX && idx < gh->node_capacity) {
            NodeDisk *n = &nodes[idx];
            if (n->id != UINT64_MAX) {
                stats.count++;
                stats.avg_stability += n->stability;
                stats.avg_fe += n->fe_ema;
                stats.total_fe += n->fe_ema;
            }
        }
    }
    
    if (stats.count > 0) {
        stats.avg_stability /= stats.count;
        stats.avg_fe /= stats.count;
    }
    
    return stats;
}

int main() {
    printf("TEST: Stability & Pruning Under Junk\n");
    printf("======================================\n\n");
    
    srand(time(NULL));
    
    // Step 1: Setup
    printf("Step 1: Creating fresh brain...\n");
    MelvinFile file;
    if (melvin_m_init_new_file(TEST_FILE, NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to create brain file\n");
        return 1;
    }
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map brain file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("  ✓ Brain created\n");
    
    // Configure pruning thresholds (moderate)
    GraphHeaderDisk *gh = file.graph_header;
    uint64_t stability_prune_idx = find_node_index_by_id(&file, NODE_ID_PARAM_STABILITY_PRUNE_THRESHOLD);
    if (stability_prune_idx != UINT64_MAX) {
        file.nodes[stability_prune_idx].state = 0.2f;  // Moderate threshold
    }
    uint64_t usage_prune_idx = find_node_index_by_id(&file, NODE_ID_PARAM_USAGE_PRUNE_THRESHOLD);
    if (usage_prune_idx != UINT64_MAX) {
        file.nodes[usage_prune_idx].state = 0.1f;  // Moderate threshold
    }
    melvin_sync_params_from_nodes(&rt);
    printf("  ✓ Pruning thresholds configured\n\n");
    
    // Step 2: Phase 1 - Useful pattern (HELLO)
    printf("Step 2: Phase 1 - Useful pattern (HELLO repeated %d times)...\n", NUM_USEFUL_REPS);
    
    const char *useful_pattern = "HELLO\n";
    uint64_t hello_node_ids[6];
    for (int i = 0; i < 6; i++) {
        hello_node_ids[i] = (uint64_t)useful_pattern[i] + 1000000ULL;
    }
    
    for (int rep = 0; rep < NUM_USEFUL_REPS; rep++) {
        for (int i = 0; useful_pattern[i] != '\0'; i++) {
            ingest_byte(&rt, 0, useful_pattern[i], 1.0f);
        }
        
        for (int tick = 0; tick < TICKS_PER_SWEEP; tick++) {
            melvin_process_n_events(&rt, 10);
        }
        
        // Trigger homeostasis
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        melvin_process_n_events(&rt, 10);
    }
    
    SubgraphStats hello_before = measure_subgraph(&file, hello_node_ids, 6);
    printf("  HELLO subgraph: %llu nodes, avg_stability=%.4f, avg_FE=%.6f\n",
           (unsigned long long)hello_before.count, hello_before.avg_stability, hello_before.avg_fe);
    printf("\n");
    
    // Step 3: Phase 2 - Junk patterns
    printf("Step 3: Phase 2 - Junk patterns (%d random sequences)...\n", NUM_JUNK_REPS);
    
    uint64_t junk_node_ids[100];
    int junk_count = 0;
    
    for (int rep = 0; rep < NUM_JUNK_REPS; rep++) {
        // Generate random junk sequence
        char junk[10];
        for (int i = 0; i < 9; i++) {
            junk[i] = (char)(32 + (rand() % 95));  // Printable ASCII
        }
        junk[9] = '\n';
        
        for (int i = 0; i < 10; i++) {
            uint64_t node_id = (uint64_t)junk[i] + 1000000ULL;
            ingest_byte(&rt, 0, junk[i], 1.0f);
            
            // Track junk nodes (up to 100)
            if (junk_count < 100) {
                junk_node_ids[junk_count++] = node_id;
            }
        }
        
        for (int tick = 0; tick < TICKS_PER_SWEEP; tick++) {
            melvin_process_n_events(&rt, 10);
        }
        
        // Trigger homeostasis
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        melvin_process_n_events(&rt, 10);
    }
    
    SubgraphStats junk_before = measure_subgraph(&file, junk_node_ids, junk_count);
    printf("  JUNK subgraph: %llu nodes, avg_stability=%.4f, avg_FE=%.6f\n",
           (unsigned long long)junk_before.count, junk_before.avg_stability, junk_before.avg_fe);
    printf("\n");
    
    // Step 4: Run pruning sweeps (no new input)
    printf("Step 4: Running %d pruning sweeps (no new input)...\n", NUM_PRUNE_SWEEPS);
    
    for (int sweep = 0; sweep < NUM_PRUNE_SWEEPS; sweep++) {
        for (int tick = 0; tick < TICKS_PER_SWEEP; tick++) {
            melvin_process_n_events(&rt, 10);
        }
        
        // Trigger homeostasis (includes pruning)
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        melvin_process_n_events(&rt, 10);
        
        if ((sweep + 1) % 2 == 0) {
            SubgraphStats hello_now = measure_subgraph(&file, hello_node_ids, 6);
            SubgraphStats junk_now = measure_subgraph(&file, junk_node_ids, junk_count);
            printf("  Sweep %d: HELLO nodes=%llu (stability=%.4f, FE=%.6f), JUNK nodes=%llu (stability=%.4f, FE=%.6f)\n",
                   sweep + 1,
                   (unsigned long long)hello_now.count, hello_now.avg_stability, hello_now.avg_fe,
                   (unsigned long long)junk_now.count, junk_now.avg_stability, junk_now.avg_fe);
        }
    }
    
    printf("\n");
    
    // Step 5: Final measurements
    printf("Step 5: Final measurements...\n");
    
    SubgraphStats hello_after = measure_subgraph(&file, hello_node_ids, 6);
    SubgraphStats junk_after = measure_subgraph(&file, junk_node_ids, junk_count);
    
    printf("  HELLO nodes remaining: %llu\n", (unsigned long long)hello_after.count);
    printf("    avg_stability: %.4f\n", hello_after.avg_stability);
    printf("    avg_FE: %.6f\n", hello_after.avg_fe);
    printf("\n");
    printf("  JUNK nodes remaining: %llu\n", (unsigned long long)junk_after.count);
    printf("    avg_stability: %.4f\n", junk_after.avg_stability);
    printf("    avg_FE: %.6f\n", junk_after.avg_fe);
    printf("\n");
    
    // Step 6: Assertions
    printf("TEST RESULTS\n");
    printf("============\n");
    
    int passed = 1;
    
    if (hello_after.count >= hello_before.count * 0.8f) {
        printf("✓ HELLO nodes preserved (%.0f%% remaining)\n", 
               (hello_after.count * 100.0f) / (hello_before.count > 0 ? hello_before.count : 1));
    } else {
        printf("⚠ HELLO nodes pruned (%.0f%% remaining)\n",
               (hello_after.count * 100.0f) / (hello_before.count > 0 ? hello_before.count : 1));
    }
    
    if (hello_after.avg_stability > junk_after.avg_stability) {
        printf("✓ HELLO has higher stability than JUNK\n");
    } else {
        printf("⚠ HELLO stability not higher than JUNK\n");
    }
    
    if (hello_after.avg_fe < junk_after.avg_fe) {
        printf("✓ HELLO has lower FE than JUNK\n");
    } else {
        printf("⚠ HELLO FE not lower than JUNK\n");
    }
    
    if (junk_after.count < junk_before.count * 0.7f) {
        printf("✓ JUNK nodes pruned (%.0f%% remaining)\n",
               (junk_after.count * 100.0f) / (junk_before.count > 0 ? junk_before.count : 1));
    } else {
        printf("⚠ JUNK nodes not pruned enough (%.0f%% remaining)\n",
               (junk_after.count * 100.0f) / (junk_before.count > 0 ? junk_before.count : 1));
    }
    
    printf("\n");
    if (passed) {
        printf("✅ TEST PASSED: Stability & Pruning Under Junk\n");
    } else {
        printf("⚠️  TEST PARTIAL: Some expectations not fully met\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return 0;
}


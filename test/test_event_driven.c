#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

// Include the implementation
#include "melvin.c"

// Test event-driven behavior with different data types
void test_event_driven_behavior() {
    const char *file_path = "test_event_driven.m";
    
    printf("========================================\n");
    printf("EVENT-DRIVEN MELVIN TEST\n");
    printf("========================================\n\n");
    
    // Create new file
    GraphParams params;
    params.decay_rate = 0.1f;
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.exec_threshold = 1.0f;
    params.learning_rate = 0.001f;
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "Failed to create file\n");
        return;
    }
    
    // Map file
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "Failed to map file\n");
        return;
    }
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "Failed to initialize runtime\n");
        close_file(&file);
        return;
    }
    
    printf("✓ Initialized event-driven runtime\n\n");
    
    // Test 1: Text data - shows how events propagate
    printf("=== TEST 1: Text Data Event Propagation ===\n");
    const char *text = "ABC";
    printf("Ingesting text: \"%s\"\n", text);
    
    for (size_t i = 0; i < strlen(text); i++) {
        ingest_byte(&rt, 1, (uint8_t)text[i], 1.0f);
        printf("  Enqueued event for byte '%c'\n", text[i]);
    }
    
    printf("\nProcessing events...\n");
    melvin_process_n_events(&rt, 50);  // Process up to 50 events
    
    GraphHeaderDisk *gh = rt.file->graph_header;
    printf("After processing: Nodes=%llu, Edges=%llu\n\n",
           (unsigned long long)gh->num_nodes,
           (unsigned long long)gh->num_edges);
    
    // Test 2: Binary data
    printf("=== TEST 2: Binary Data ===\n");
    uint8_t binary[] = {0xFF, 0x00, 0xAA};
    printf("Ingesting binary: ");
    for (size_t i = 0; i < sizeof(binary); i++) {
        printf("0x%02x ", binary[i]);
        ingest_byte(&rt, 2, binary[i], 0.8f);
    }
    printf("\n");
    
    melvin_process_n_events(&rt, 30);
    printf("After processing: Nodes=%llu, Edges=%llu\n\n",
           (unsigned long long)gh->num_nodes,
           (unsigned long long)gh->num_edges);
    
    // Test 3: Reward injection
    printf("=== TEST 3: Reward Injection ===\n");
    printf("Injecting reward events...\n");
    
    // Find a DATA node to reward
    for (uint64_t i = 0; i < gh->num_nodes && i < 5; i++) {
        if (rt.file->nodes[i].id != UINT64_MAX && 
            (rt.file->nodes[i].flags & NODE_FLAG_DATA)) {
            uint64_t node_id = rt.file->nodes[i].id;
            inject_reward(&rt, node_id, 0.5f);
            printf("  Enqueued reward event for node %llu\n", 
                   (unsigned long long)node_id);
        }
    }
    
    melvin_process_n_events(&rt, 10);
    printf("Rewards processed\n\n");
    
    // Test 4: Show how events create structure
    printf("=== TEST 4: Event-Driven Structure Formation ===\n");
    printf("Processing more events to see structure evolve...\n");
    
    // Inject some more data to create more events
    ingest_byte(&rt, 1, 'D', 1.0f);
    ingest_byte(&rt, 1, 'E', 1.0f);
    
    melvin_process_n_events(&rt, 20);
    
    printf("Final state:\n");
    printf("  Nodes: %llu\n", (unsigned long long)gh->num_nodes);
    printf("  Edges: %llu\n", (unsigned long long)gh->num_edges);
    printf("  Avg activation: %.4f\n", gh->avg_activation);
    printf("  Total events processed: %llu\n", 
           (unsigned long long)rt.logical_time);
    
    // Show some active nodes
    printf("\nActive nodes (activation > 0.05):\n");
    int active_count = 0;
    for (uint64_t i = 0; i < gh->num_nodes && active_count < 5; i++) {
        if (rt.file->nodes[i].id != UINT64_MAX) {
            NodeDisk *node = &rt.file->nodes[i];
            if (fabsf(node->state) > 0.05f) {
                printf("  Node %llu: activation=%.4f, prediction=%.4f, error=%.4f\n",
                       (unsigned long long)node->id,
                       node->state,
                       node->prediction,
                       node->prediction_error);
                active_count++;
            }
        }
    }
    
    // Sync and close
    melvin_m_sync(&file);
    runtime_cleanup(&rt);
    close_file(&file);
    
    printf("\n========================================\n");
    printf("TEST COMPLETE\n");
    printf("========================================\n");
    printf("File: %s\n", file_path);
}

int main(int argc, char **argv) {
    test_event_driven_behavior();
    return 0;
}


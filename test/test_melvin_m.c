#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
// Include the implementation to get all types and functions
#include "melvin.c"

// Test different types of data ingestion
void test_text_data(MelvinRuntime *rt, const char *text) {
    printf("\n=== Testing TEXT DATA: \"%s\" ===\n", text);
    
    uint64_t channel_id = 1; // Text channel
    size_t len = strlen(text);
    
    for (size_t i = 0; i < len; i++) {
        ingest_byte(rt, channel_id, (uint8_t)text[i], 1.0f);
        printf("  Ingested byte: '%c' (0x%02x)\n", text[i], (uint8_t)text[i]);
    }
    
    printf("  Text data ingested: %zu bytes\n", len);
}

void test_binary_data(MelvinRuntime *rt, const uint8_t *data, size_t len) {
    printf("\n=== Testing BINARY DATA (%zu bytes) ===\n", len);
    
    uint64_t channel_id = 2; // Binary channel
    
    for (size_t i = 0; i < len; i++) {
        ingest_byte(rt, channel_id, data[i], 0.8f);
    }
    
    printf("  Binary data ingested: %zu bytes\n", len);
    printf("  First few bytes: ");
    for (size_t i = 0; i < (len > 8 ? 8 : len); i++) {
        printf("0x%02x ", data[i]);
    }
    printf("\n");
}

void test_structured_data(MelvinRuntime *rt) {
    printf("\n=== Testing STRUCTURED DATA ===\n");
    
    // Simulate structured data: temperature readings
    uint64_t channel_id = 3; // Sensor channel
    float temperatures[] = {72.5f, 73.1f, 72.8f, 73.5f};
    
    for (size_t i = 0; i < sizeof(temperatures)/sizeof(temperatures[0]); i++) {
        // Ingest as bytes (little-endian float)
        uint8_t *bytes = (uint8_t*)&temperatures[i];
        for (size_t j = 0; j < sizeof(float); j++) {
            ingest_byte(rt, channel_id, bytes[j], 0.5f);
        }
        printf("  Ingested temperature: %.1f°F\n", temperatures[i]);
    }
}

void print_graph_stats(MelvinRuntime *rt) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    
    printf("\n=== GRAPH STATISTICS ===\n");
    printf("  Nodes: %llu / %llu (capacity)\n",
           (unsigned long long)gh->num_nodes,
           (unsigned long long)gh->node_capacity);
    printf("  Edges: %llu / %llu (capacity)\n",
           (unsigned long long)gh->num_edges,
           (unsigned long long)gh->edge_capacity);
    printf("  Blob size: %llu / %llu bytes\n",
           (unsigned long long)rt->file->blob_size,
           (unsigned long long)rt->file->blob_capacity);
    printf("  Average activation: %.4f\n", gh->avg_activation);
    printf("  Total pulses emitted: %llu\n",
           (unsigned long long)gh->total_pulses_emitted);
    printf("  Total pulses absorbed: %llu\n",
           (unsigned long long)gh->total_pulses_absorbed);
    printf("  Tick counter: %llu\n",
           (unsigned long long)rt->file->file_header->tick_counter);
}

void print_node_info(MelvinRuntime *rt, uint64_t node_id) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (nodes[i].id == node_id) {
            NodeDisk *node = &nodes[i];
            printf("\n  Node ID: %llu\n", (unsigned long long)node_id);
            printf("    State (activation): %.4f\n", node->state);
            printf("    Prediction: %.4f\n", node->prediction);
            printf("    Prediction error: %.4f\n", node->prediction_error);
            printf("    Reward: %.4f\n", node->reward);
            printf("    Energy cost: %.4f\n", node->energy_cost);
            printf("    Flags: 0x%x", node->flags);
            if (node->flags & NODE_FLAG_DATA) printf(" (DATA)");
            if (node->flags & NODE_FLAG_EXECUTABLE) printf(" (EXECUTABLE)");
            printf("\n");
            printf("    Out degree: %u\n", node->out_degree);
            printf("    Firing count: %u\n", node->firing_count);
            if (node->payload_len > 0) {
                printf("    Payload: offset=%llu, len=%u\n",
                       (unsigned long long)node->payload_offset,
                       node->payload_len);
            }
            return;
        }
    }
    printf("  Node %llu not found\n", (unsigned long long)node_id);
}

void print_edge_info(MelvinRuntime *rt, uint64_t src_id, uint64_t dst_id) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    EdgeDisk *edges = rt->file->edges;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (edges[i].src == UINT64_MAX) continue;
        if (edges[i].src == src_id && edges[i].dst == dst_id) {
            EdgeDisk *edge = &edges[i];
            printf("\n  Edge: %llu -> %llu\n",
                   (unsigned long long)src_id,
                   (unsigned long long)dst_id);
            printf("    Weight: %.4f\n", edge->weight);
            printf("    Eligibility: %.4f\n", edge->eligibility);
            printf("    Usage: %.4f\n", edge->usage);
            printf("    Pulse count: %u\n", edge->pulse_count);
            printf("    Flags: 0x%x", edge->flags);
            if (edge->flags & EDGE_FLAG_SEQ) printf(" (SEQ)");
            if (edge->flags & EDGE_FLAG_CHAN) printf(" (CHAN)");
            printf("\n");
            printf("    Is bond: %u\n", edge->is_bond);
            return;
        }
    }
    printf("  Edge %llu -> %llu not found\n",
           (unsigned long long)src_id,
           (unsigned long long)dst_id);
}

int main(int argc, char **argv) {
    const char *file_path = "test_melvin.m";
    
    printf("========================================\n");
    printf("MELVIN.M FILE FORMAT TEST\n");
    printf("========================================\n\n");
    
    // Step 1: Create new melvin.m file
    printf("Step 1: Creating new melvin.m file...\n");
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
        fprintf(stderr, "Failed to create melvin.m file\n");
        return 1;
    }
    printf("✓ Created %s\n\n", file_path);
    
    // Step 2: Map the file
    printf("Step 2: Mapping melvin.m file...\n");
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "Failed to map melvin.m file\n");
        return 1;
    }
    printf("✓ Mapped file successfully\n\n");
    
    // Step 3: Initialize runtime
    printf("Step 3: Initializing runtime...\n");
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n\n");
    
    // Step 4: Test different types of data ingestion
    printf("Step 4: Testing data ingestion...\n");
    
    // Test 1: Text data
    test_text_data(&rt, "Hello Melvin!");
    
    // Test 2: Binary data
    uint8_t binary_data[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
    test_binary_data(&rt, binary_data, sizeof(binary_data));
    
    // Test 3: Structured data
    test_structured_data(&rt);
    
    printf("\n✓ Data ingestion complete\n\n");
    
    // Step 5: Run physics ticks to see what happens
    printf("Step 5: Running physics ticks...\n");
    int num_ticks = 10;
    
    for (int tick = 0; tick < num_ticks; tick++) {
        printf("\n--- Tick %d ---\n", tick + 1);
        physics_tick(&rt);
        
        // Print some stats every few ticks
        if ((tick + 1) % 3 == 0) {
            GraphHeaderDisk *gh = rt.file->graph_header;
            printf("  Nodes: %llu, Edges: %llu, Avg activation: %.4f\n",
                   (unsigned long long)gh->num_nodes,
                   (unsigned long long)gh->num_edges,
                   gh->avg_activation);
        }
    }
    
    printf("\n✓ Physics ticks complete\n\n");
    
    // Step 6: Print final statistics
    print_graph_stats(&rt);
    
    // Step 7: Show some example nodes and edges
    printf("\n=== EXAMPLE NODES ===\n");
    
    // Show a few DATA nodes
    GraphHeaderDisk *gh = rt.file->graph_header;
    int nodes_shown = 0;
    for (uint64_t i = 0; i < gh->num_nodes && nodes_shown < 5; i++) {
        if (rt.file->nodes[i].id != UINT64_MAX) {
            print_node_info(&rt, rt.file->nodes[i].id);
            nodes_shown++;
        }
    }
    
    // Show some edges
    printf("\n=== EXAMPLE EDGES ===\n");
    int edges_shown = 0;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity && edges_shown < 5; i++) {
        if (rt.file->edges[i].src != UINT64_MAX) {
            EdgeDisk *e = &rt.file->edges[i];
            print_edge_info(&rt, e->src, e->dst);
            edges_shown++;
        }
    }
    
    // Step 8: Sync and close
    printf("\nStep 8: Syncing file to disk...\n");
    melvin_m_sync(&file);
    printf("✓ File synced\n\n");
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    printf("========================================\n");
    printf("TEST COMPLETE\n");
    printf("========================================\n");
    printf("\nFile saved: %s\n", file_path);
    printf("You can inspect it with: hexdump -C %s | head -50\n", file_path);
    
    return 0;
}


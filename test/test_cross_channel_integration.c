/*
 * TEST: Cross-Channel Integration
 * 
 * Goal: Demonstrate that when two pseudo-modalities (channels) carry
 * correlated patterns, the graph learns cross-channel structure.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "melvin.c"

#define TEST_FILE "test_cross_channel_integration.m"
#define NUM_EPISODES 100
#define TICKS_PER_EPISODE 100

typedef struct {
    uint64_t src_node;
    uint64_t dst_node;
    float weight;
    uint64_t src_channel;
    uint64_t dst_channel;
} CrossChannelEdge;

int main() {
    printf("TEST: Cross-Channel Integration\n");
    printf("================================\n\n");
    
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
    printf("  ✓ Brain created\n\n");
    
    // Step 2: Define channels
    printf("Step 2: Defining channels...\n");
    uint64_t channel1_id = 0;  // "text-like"
    uint64_t channel2_id = 1;  // "motor-like"
    printf("  Channel 1 (text): %llu\n", (unsigned long long)channel1_id);
    printf("  Channel 2 (motor): %llu\n", (unsigned long long)channel2_id);
    printf("\n");
    
    // Step 3: Feed correlated patterns
    printf("Step 3: Feeding correlated patterns (%d episodes)...\n", NUM_EPISODES);
    printf("  Channel 1: \"GO\\n\"\n");
    printf("  Channel 2: \"FORWARD\\n\"\n");
    printf("  (Correlated: GO → FORWARD)\n\n");
    
    const char *channel1_pattern = "GO\n";
    const char *channel2_pattern = "FORWARD\n";
    
    uint64_t go_node_ids[3];
    uint64_t forward_node_ids[8];
    
    for (int i = 0; i < 3; i++) {
        go_node_ids[i] = (uint64_t)channel1_pattern[i] + 1000000ULL;
    }
    for (int i = 0; i < 8; i++) {
        forward_node_ids[i] = (uint64_t)channel2_pattern[i] + 1000000ULL;
    }
    
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        // Feed correlated patterns
        for (int i = 0; channel1_pattern[i] != '\0'; i++) {
            ingest_byte(&rt, channel1_id, channel1_pattern[i], 1.0f);
        }
        for (int i = 0; channel2_pattern[i] != '\0'; i++) {
            ingest_byte(&rt, channel2_id, channel2_pattern[i], 1.0f);
        }
        
        // Occasionally feed noise (uncorrelated)
        if (episode % 10 == 0) {
            ingest_byte(&rt, channel1_id, 'X', 1.0f);
            ingest_byte(&rt, channel2_id, 'Y', 1.0f);
        }
        
        for (int tick = 0; tick < TICKS_PER_EPISODE; tick++) {
            melvin_process_n_events(&rt, 10);
        }
        
        // Trigger homeostasis (triggers edge formation)
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        melvin_process_n_events(&rt, 10);
    }
    
    printf("  ✓ Patterns fed\n\n");
    
    // Step 4: Analyze cross-channel edges
    printf("Step 4: Analyzing cross-channel edges...\n");
    
    GraphHeaderDisk *gh = file.graph_header;
    EdgeDisk *edges = file.edges;
    NodeDisk *nodes = file.nodes;
    
    CrossChannelEdge cross_edges[1000];
    int cross_edge_count = 0;
    
    // Find edges between channel 1 and channel 2 nodes
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity && cross_edge_count < 1000; e++) {
        EdgeDisk *edge = &edges[e];
        if (edge->src == UINT64_MAX) continue;
        
        // Determine which channel each node belongs to
        // Channel 1 nodes: ingested on channel 0
        // Channel 2 nodes: ingested on channel 1
        // We can identify by checking if node is in GO or FORWARD sets
        
        int src_is_ch1 = 0, dst_is_ch1 = 0;
        int src_is_ch2 = 0, dst_is_ch2 = 0;
        
        // Check if src is in channel 1 (GO pattern)
        for (int i = 0; i < 3; i++) {
            if (edge->src == go_node_ids[i]) {
                src_is_ch1 = 1;
                break;
            }
        }
        
        // Check if src is in channel 2 (FORWARD pattern)
        for (int i = 0; i < 8; i++) {
            if (edge->src == forward_node_ids[i]) {
                src_is_ch2 = 1;
                break;
            }
        }
        
        // Check if dst is in channel 1
        for (int i = 0; i < 3; i++) {
            if (edge->dst == go_node_ids[i]) {
                dst_is_ch1 = 1;
                break;
            }
        }
        
        // Check if dst is in channel 2
        for (int i = 0; i < 8; i++) {
            if (edge->dst == forward_node_ids[i]) {
                dst_is_ch2 = 1;
                break;
            }
        }
        
        // Cross-channel edge: (ch1→ch2) or (ch2→ch1)
        if ((src_is_ch1 && dst_is_ch2) || (src_is_ch2 && dst_is_ch1)) {
            cross_edges[cross_edge_count].src_node = edge->src;
            cross_edges[cross_edge_count].dst_node = edge->dst;
            cross_edges[cross_edge_count].weight = edge->weight;
            cross_edges[cross_edge_count].src_channel = src_is_ch1 ? 1 : 2;
            cross_edges[cross_edge_count].dst_channel = dst_is_ch1 ? 1 : 2;
            cross_edge_count++;
        }
    }
    
    printf("  Found %d cross-channel edges\n", cross_edge_count);
    
    // Sort by weight
    for (int i = 0; i < cross_edge_count - 1; i++) {
        for (int j = i + 1; j < cross_edge_count; j++) {
            if (fabsf(cross_edges[i].weight) < fabsf(cross_edges[j].weight)) {
                CrossChannelEdge temp = cross_edges[i];
                cross_edges[i] = cross_edges[j];
                cross_edges[j] = temp;
            }
        }
    }
    
    // Show top cross-channel edges
    printf("\n  Top cross-channel edges:\n");
    int shown = 0;
    for (int i = 0; i < cross_edge_count && shown < 10; i++) {
        if (fabsf(cross_edges[i].weight) > 0.01f) {
            printf("    Ch%llu → Ch%llu: weight=%.4f\n",
                   (unsigned long long)cross_edges[i].src_channel,
                   (unsigned long long)cross_edges[i].dst_channel,
                   cross_edges[i].weight);
            shown++;
        }
    }
    
    // Step 5: Measure FE for correlated vs uncorrelated pairs
    printf("\nStep 5: Measuring FE for correlated vs uncorrelated pairs...\n");
    
    float fe_correlated = 0.0f;
    int fe_correlated_count = 0;
    
    // FE for GO-FORWARD region (correlated)
    for (int i = 0; i < 3; i++) {
        uint64_t idx = find_node_index_by_id(&file, go_node_ids[i]);
        if (idx != UINT64_MAX) {
            fe_correlated += nodes[idx].fe_ema;
            fe_correlated_count++;
        }
    }
    for (int i = 0; i < 8; i++) {
        uint64_t idx = find_node_index_by_id(&file, forward_node_ids[i]);
        if (idx != UINT64_MAX) {
            fe_correlated += nodes[idx].fe_ema;
            fe_correlated_count++;
        }
    }
    
    float fe_correlated_avg = (fe_correlated_count > 0) ? (fe_correlated / fe_correlated_count) : 0.0f;
    
    // FE for noise nodes (uncorrelated)
    uint64_t noise_x = (uint64_t)'X' + 1000000ULL;
    uint64_t noise_y = (uint64_t)'Y' + 1000000ULL;
    float fe_uncorrelated = 0.0f;
    int fe_uncorrelated_count = 0;
    
    uint64_t x_idx = find_node_index_by_id(&file, noise_x);
    if (x_idx != UINT64_MAX) {
        fe_uncorrelated += nodes[x_idx].fe_ema;
        fe_uncorrelated_count++;
    }
    uint64_t y_idx = find_node_index_by_id(&file, noise_y);
    if (y_idx != UINT64_MAX) {
        fe_uncorrelated += nodes[y_idx].fe_ema;
        fe_uncorrelated_count++;
    }
    
    float fe_uncorrelated_avg = (fe_uncorrelated_count > 0) ? (fe_uncorrelated / fe_uncorrelated_count) : 0.0f;
    
    printf("  Correlated (GO-FORWARD) avg FE: %.6f\n", fe_correlated_avg);
    printf("  Uncorrelated (X-Y) avg FE: %.6f\n", fe_uncorrelated_avg);
    printf("\n");
    
    // Step 6: Assertions
    printf("TEST RESULTS\n");
    printf("============\n");
    
    int passed = 1;
    
    if (cross_edge_count > 0) {
        printf("✓ Cross-channel edges found: %d\n", cross_edge_count);
    } else {
        printf("✗ No cross-channel edges found\n");
        passed = 0;
    }
    
    if (cross_edge_count > 0) {
        float max_weight = 0.0f;
        for (int i = 0; i < cross_edge_count; i++) {
            if (fabsf(cross_edges[i].weight) > max_weight) {
                max_weight = fabsf(cross_edges[i].weight);
            }
        }
        if (max_weight > 0.1f) {
            printf("✓ Strong cross-channel edges (max weight: %.4f)\n", max_weight);
        } else {
            printf("⚠ Cross-channel edges are weak (max weight: %.4f)\n", max_weight);
        }
    }
    
    if (fe_correlated_avg < fe_uncorrelated_avg) {
        printf("✓ Correlated pairs have lower FE than uncorrelated\n");
    } else {
        printf("⚠ Correlated FE not lower than uncorrelated\n");
    }
    
    printf("\n");
    if (passed) {
        printf("✅ TEST PASSED: Cross-Channel Integration\n");
    } else {
        printf("✗ TEST FAILED: Cross-channel integration not detected\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return (passed ? 0 : 1);
}


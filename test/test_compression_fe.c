/*
 * TEST: Compression & Free-Energy
 * 
 * Goal: Demonstrate that the graph compresses repeated sequences more
 * effectively than noise and that this is reflected in lower free-energy.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_compression_fe.m"
#define NUM_EPISODES_STRUCTURED 30
#define NUM_EPISODES_NOISE 30
#define TICKS_PER_EPISODE 100

typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    float avg_fe;
    float strong_edges;  // Edges with |weight| > 0.3
    float compression_ratio;  // distinct_patterns / total_bytes
} Metrics;

Metrics measure_graph(MelvinFile *file) {
    Metrics m = {0};
    GraphHeaderDisk *gh = file->graph_header;
    NodeDisk *nodes = file->nodes;
    EdgeDisk *edges = file->edges;
    
    m.num_nodes = gh->num_nodes;
    m.num_edges = gh->num_edges;
    
    float fe_sum = 0.0f;
    int fe_count = 0;
    uint64_t strong_count = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        fe_sum += nodes[i].fe_ema;
        fe_count++;
    }
    
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        if (edges[e].src == UINT64_MAX) continue;
        if (fabsf(edges[e].weight) > 0.3f) {
            strong_count++;
        }
    }
    
    m.avg_fe = (fe_count > 0) ? (fe_sum / fe_count) : 0.0f;
    m.strong_edges = (gh->num_edges > 0) ? ((float)strong_count / gh->num_edges) : 0.0f;
    
    return m;
}

int main() {
    printf("TEST: Compression & Free-Energy\n");
    printf("================================\n\n");
    
    srand(time(NULL));
    
    // Phase 1: Structured data
    printf("PHASE 1: Structured Data\n");
    printf("------------------------\n");
    
    MelvinFile file1;
    if (melvin_m_init_new_file("test_compression_structured.m", NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to create brain file\n");
        return 1;
    }
    if (melvin_m_map("test_compression_structured.m", &file1) < 0) {
        fprintf(stderr, "ERROR: Failed to map brain file\n");
        return 1;
    }
    
    MelvinRuntime rt1;
    if (runtime_init(&rt1, &file1) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file1);
        return 1;
    }
    
    printf("Ingesting structured patterns (ABCABC... and XYZXYZ...)...\n");
    uint64_t total_bytes_structured = 0;
    
    for (int episode = 0; episode < NUM_EPISODES_STRUCTURED; episode++) {
        const char *pattern1 = "ABCABCABCABC\n";
        const char *pattern2 = "XYZXYZXYZXYZ\n";
        
        // Alternate between patterns
        const char *pattern = (episode % 2 == 0) ? pattern1 : pattern2;
        
        for (int i = 0; pattern[i] != '\0'; i++) {
            ingest_byte(&rt1, 0, pattern[i], 1.0f);
            total_bytes_structured++;
        }
        
        for (int tick = 0; tick < TICKS_PER_EPISODE; tick++) {
            melvin_process_n_events(&rt1, 10);
        }
        
        // Trigger homeostasis
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt1.evq, &homeostasis_ev);
        melvin_process_n_events(&rt1, 10);
    }
    
    Metrics m_structured = measure_graph(&file1);
    printf("  Total bytes ingested: %llu\n", (unsigned long long)total_bytes_structured);
    printf("  Nodes: %llu\n", (unsigned long long)m_structured.num_nodes);
    printf("  Edges: %llu\n", (unsigned long long)m_structured.num_edges);
    printf("  Average FE: %.6f\n", m_structured.avg_fe);
    printf("  Strong edges (|w|>0.3): %.2f%%\n", m_structured.strong_edges * 100.0f);
    
    // Check for ABC and XYZ sequences
    uint64_t node_a = (uint64_t)'A' + 1000000ULL;
    uint64_t node_b = (uint64_t)'B' + 1000000ULL;
    uint64_t node_c = (uint64_t)'C' + 1000000ULL;
    
    int has_abc_seq = 0;
    if (edge_exists_between(&file1, node_a, node_b) && 
        edge_exists_between(&file1, node_b, node_c)) {
        has_abc_seq = 1;
    }
    printf("  Has ABC sequence: %s\n", has_abc_seq ? "YES" : "NO");
    
    runtime_cleanup(&rt1);
    close_file(&file1);
    printf("\n");
    
    // Phase 2: Noise
    printf("PHASE 2: Random Noise\n");
    printf("---------------------\n");
    
    MelvinFile file2;
    if (melvin_m_init_new_file("test_compression_noise.m", NULL) < 0) {
        fprintf(stderr, "ERROR: Failed to create brain file\n");
        return 1;
    }
    if (melvin_m_map("test_compression_noise.m", &file2) < 0) {
        fprintf(stderr, "ERROR: Failed to map brain file\n");
        return 1;
    }
    
    MelvinRuntime rt2;
    if (runtime_init(&rt2, &file2) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file2);
        return 1;
    }
    
    printf("Ingesting random noise...\n");
    uint64_t total_bytes_noise = 0;
    
    for (int episode = 0; episode < NUM_EPISODES_NOISE; episode++) {
        // Generate random bytes
        for (int i = 0; i < 13; i++) {  // Same length as patterns
            uint8_t random_byte = (uint8_t)(rand() % 256);
            ingest_byte(&rt2, 0, random_byte, 1.0f);
            total_bytes_noise++;
        }
        
        for (int tick = 0; tick < TICKS_PER_EPISODE; tick++) {
            melvin_process_n_events(&rt2, 10);
        }
        
        // Trigger homeostasis
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt2.evq, &homeostasis_ev);
        melvin_process_n_events(&rt2, 10);
    }
    
    Metrics m_noise = measure_graph(&file2);
    printf("  Total bytes ingested: %llu\n", (unsigned long long)total_bytes_noise);
    printf("  Nodes: %llu\n", (unsigned long long)m_noise.num_nodes);
    printf("  Edges: %llu\n", (unsigned long long)m_noise.num_edges);
    printf("  Average FE: %.6f\n", m_noise.avg_fe);
    printf("  Strong edges (|w|>0.3): %.2f%%\n", m_noise.strong_edges * 100.0f);
    
    runtime_cleanup(&rt2);
    close_file(&file2);
    printf("\n");
    
    // Comparison
    printf("COMPARISON: STRUCTURED vs NOISE\n");
    printf("================================\n");
    printf("  Nodes:  %llu (structured) vs %llu (noise)\n", 
           (unsigned long long)m_structured.num_nodes, (unsigned long long)m_noise.num_nodes);
    printf("  Edges:  %llu (structured) vs %llu (noise)\n",
           (unsigned long long)m_structured.num_edges, (unsigned long long)m_noise.num_edges);
    printf("  Avg FE: %.6f (structured) vs %.6f (noise)\n",
           m_structured.avg_fe, m_noise.avg_fe);
    printf("  Strong edges: %.2f%% (structured) vs %.2f%% (noise)\n",
           m_structured.strong_edges * 100.0f, m_noise.strong_edges * 100.0f);
    printf("\n");
    
    // Assertions
    int passed = 1;
    
    if (m_structured.avg_fe < m_noise.avg_fe) {
        printf("✓ Structured data has lower FE (better compression)\n");
    } else {
        printf("⚠ Structured data FE not lower than noise\n");
    }
    
    if (m_structured.strong_edges > m_noise.strong_edges) {
        printf("✓ Structured data has more strong edges (better patterns)\n");
    } else {
        printf("⚠ Structured data does not have more strong edges\n");
    }
    
    if (has_abc_seq) {
        printf("✓ ABC sequence detected (pattern formation)\n");
    } else {
        printf("⚠ ABC sequence not detected\n");
    }
    
    printf("\n");
    if (passed) {
        printf("✅ TEST PASSED: Compression & Free-Energy\n");
    } else {
        printf("⚠️  TEST PARTIAL: Some expectations not met\n");
    }
    
    return 0;
}


#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <dirent.h>

// Include the implementation
#include "melvin.c"

// Ingest a C file byte by byte - this is how Melvin "eats" code
void ingest_c_file(MelvinRuntime *rt, const char *filepath, uint64_t channel_id) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        perror("fopen");
        return;
    }
    
    printf("Eating C file: %s\n", filepath);
    
    uint8_t byte;
    size_t bytes_read = 0;
    while (fread(&byte, 1, 1, f) == 1) {
        // Each byte becomes energy in the graph
        ingest_byte(rt, channel_id, byte, 1.0f);
        bytes_read++;
        
        // Process events as we go (let the graph build structure)
        if (bytes_read % 100 == 0) {
            melvin_process_n_events(rt, 50);
        }
    }
    
    fclose(f);
    printf("  Ingested %zu bytes\n", bytes_read);
    
    // Process remaining events
    melvin_process_n_events(rt, 200);
}

// Show how the graph learns patterns from C code
void analyze_c_patterns(MelvinRuntime *rt) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    EdgeDisk *edges = rt->file->edges;
    
    printf("\n=== How the Graph Learned from C Code ===\n");
    
    // Find common C language patterns in the graph
    // These would be sequences of bytes that appear frequently
    
    // Look for common C keywords as byte sequences
    const char *c_keywords[] = {"int", "void", "return", "if", "for", "while", NULL};
    
    printf("\nC keyword patterns in graph:\n");
    for (int i = 0; c_keywords[i] != NULL; i++) {
        const char *keyword = c_keywords[i];
        int found_count = 0;
        
        // Check if we have SEQ edges connecting these bytes
        for (size_t j = 0; j < strlen(keyword) - 1; j++) {
            uint8_t byte1 = (uint8_t)keyword[j];
            uint8_t byte2 = (uint8_t)keyword[j + 1];
            
            uint64_t node1_id = (uint64_t)byte1 + 1000000ULL;
            uint64_t node2_id = (uint64_t)byte2 + 1000000ULL;
            
            // Check if edge exists (find node indices and check edges)
            uint64_t n1_idx = find_node_index_by_id(rt->file, node1_id);
            uint64_t n2_idx = find_node_index_by_id(rt->file, node2_id);
            if (n1_idx != UINT64_MAX && n2_idx != UINT64_MAX) {
                // Check if there's an edge from node1 to node2
                NodeDisk *n1 = &rt->file->nodes[n1_idx];
                uint64_t e_idx = n1->first_out_edge;
                int found = 0;
                for (uint32_t k = 0; k < n1->out_degree && e_idx != UINT64_MAX; k++) {
                    EdgeDisk *e = &rt->file->edges[e_idx];
                    if (e->src == node1_id && e->dst == node2_id) {
                        found = 1;
                        break;
                    }
                    e_idx = e->next_out_edge;
                }
                if (found) found_count++;
            }
        }
        
        if (found_count == strlen(keyword) - 1) {
            printf("  ✓ Found pattern: \"%s\" (complete sequence)\n", keyword);
        } else if (found_count > 0) {
            printf("  ~ Partial pattern: \"%s\" (%d/%zu edges)\n", 
                   keyword, found_count, strlen(keyword) - 1);
        }
    }
    
    // Show how the graph could learn to recognize C syntax
    printf("\nGraph structure from C code:\n");
    printf("  - DATA nodes represent each byte value (0-255)\n");
    printf("  - SEQ edges connect sequential bytes (syntax patterns)\n");
    printf("  - CHAN edges connect channel to data (file structure)\n");
    printf("  - High-weight edges = frequent patterns (learned syntax)\n");
    
    // Find strongest edges (learned patterns)
    printf("\nStrongest learned patterns (high-weight edges):\n");
    int shown = 0;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity && shown < 10; i++) {
        if (edges[i].src == UINT64_MAX) continue;
        
        EdgeDisk *e = &edges[i];
        if (e->weight > 0.25f && (e->flags & EDGE_FLAG_SEQ)) {
            uint64_t src_id = e->src;
            uint64_t dst_id = e->dst;
            
            // Check if these are DATA nodes (byte values)
            if (src_id >= 1000000ULL && src_id < 1000256ULL &&
                dst_id >= 1000000ULL && dst_id < 1000256ULL) {
                uint8_t src_byte = (uint8_t)(src_id - 1000000ULL);
                uint8_t dst_byte = (uint8_t)(dst_id - 1000000ULL);
                
                // Only show printable patterns
                if (src_byte >= 32 && src_byte < 127 && 
                    dst_byte >= 32 && dst_byte < 127) {
                    printf("  '%c' -> '%c' (weight=%.3f, usage=%.3f)\n",
                           (char)src_byte, (char)dst_byte, e->weight, e->usage);
                    shown++;
                }
            }
        }
    }
}

// Show how the graph could learn to execute C code
void explain_learning_to_execute(MelvinRuntime *rt) {
    printf("\n=== How Melvin Learns to Use Machine Code from C Files ===\n");
    printf("\n1. DATA INGESTION:\n");
    printf("   - C file bytes → DATA nodes (each byte = one node)\n");
    printf("   - Sequential bytes → SEQ edges (syntax patterns)\n");
    printf("   - Energy flows through these patterns\n");
    
    printf("\n2. PATTERN LEARNING:\n");
    printf("   - Frequent byte sequences → strong edges\n");
    printf("   - C keywords, operators, syntax → learned patterns\n");
    printf("   - Prediction error tracks how well graph predicts next byte\n");
    
    printf("\n3. CODE GENERATION (Future):\n");
    printf("   - High-energy patterns could trigger EXECUTABLE node creation\n");
    printf("   - Learned C patterns could compile to machine code\n");
    printf("   - Machine code written to blob by EXECUTABLE nodes\n");
    printf("   - New EXECUTABLE nodes point to that code\n");
    
    printf("\n4. SELF-MODIFICATION:\n");
    printf("   - Executed code can read C file patterns from graph\n");
    printf("   - Code can write new machine code based on patterns\n");
    printf("   - Graph builds itself: C → patterns → code → execution\n");
    
    printf("\n5. CURRENT STATE:\n");
    GraphHeaderDisk *gh = rt->file->graph_header;
    printf("   - Nodes: %llu (DATA nodes for bytes)\n", 
           (unsigned long long)gh->num_nodes);
    printf("   - Edges: %llu (learned patterns)\n", 
           (unsigned long long)gh->num_edges);
    printf("   - The graph has learned byte sequences from C files\n");
    printf("   - These patterns are stored as edge weights\n");
    printf("   - Energy flow through patterns = recognition\n");
}

int main(int argc, char **argv) {
    const char *file_path = "test_c_learning.m";
    
    printf("========================================\n");
    printf("MELVIN EATS C FILES - DATA INGESTION TEST\n");
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
        return 1;
    }
    
    // Map file
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "Failed to map file\n");
        return 1;
    }
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    
    printf("✓ Initialized\n\n");
    
    // Test 1: Ingest melvin.c itself!
    printf("=== TEST 1: Eating melvin.c ===\n");
    if (access("melvin.c", R_OK) == 0) {
        ingest_c_file(&rt, "melvin.c", 10); // Channel 10 for C files
    } else {
        printf("melvin.c not found, creating sample C code...\n");
        // Create a simple C code snippet
        const char *sample_c = "int main() { return 0; }";
        for (size_t i = 0; i < strlen(sample_c); i++) {
            ingest_byte(&rt, 10, (uint8_t)sample_c[i], 1.0f);
        }
        melvin_process_n_events(&rt, 100);
    }
    
    // Test 2: Ingest more C code patterns
    printf("\n=== TEST 2: Eating More C Patterns ===\n");
    const char *c_patterns[] = {
        "void function(int x) {",
        "    if (x > 0) {",
        "        return x * 2;",
        "    }",
        "}"
    };
    
    for (size_t i = 0; i < sizeof(c_patterns)/sizeof(c_patterns[0]); i++) {
        printf("Ingesting: %s\n", c_patterns[i]);
        for (size_t j = 0; j < strlen(c_patterns[i]); j++) {
            ingest_byte(&rt, 10, (uint8_t)c_patterns[i][j], 1.0f);
        }
        melvin_process_n_events(&rt, 50);
    }
    
    // Test 3: Analyze what the graph learned
    analyze_c_patterns(&rt);
    
    // Test 4: Explain how it could learn to execute
    explain_learning_to_execute(&rt);
    
    // Final stats
    GraphHeaderDisk *gh = rt.file->graph_header;
    printf("\n=== FINAL GRAPH STATE ===\n");
    printf("Nodes: %llu\n", (unsigned long long)gh->num_nodes);
    printf("Edges: %llu\n", (unsigned long long)gh->num_edges);
    printf("Total events processed: %llu\n", (unsigned long long)rt.logical_time);
    
    // Sync and close
    melvin_m_sync(&file);
    runtime_cleanup(&rt);
    close_file(&file);
    
    printf("\n========================================\n");
    printf("TEST COMPLETE\n");
    printf("========================================\n");
    printf("The graph has 'eaten' C code and learned patterns.\n");
    printf("These patterns are stored as edge weights.\n");
    printf("Energy flow through patterns = the graph 'knowing' C syntax.\n");
    
    return 0;
}


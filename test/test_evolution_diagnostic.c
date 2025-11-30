#define _POSIX_C_SOURCE 200809L

/*
 * test_evolution_diagnostic.c
 * 
 * Deep diagnostic tests that use PERSISTENT .m files to track
 * learning compounding and evolution over time.
 * 
 * Key insight: .m files should BUILD and EMERGE, not start fresh each time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/stat.h>

#include "melvin.c"

// Forward declarations for functions we'll call directly
extern void strengthen_edges_with_prediction_and_reward(MelvinRuntime *rt);
extern uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id);

// ========================================================================
// DIAGNOSTIC FUNCTIONS
// ========================================================================

static void diagnose_graph_state(MelvinFile *file, const char *label) {
    GraphHeaderDisk *gh = file->graph_header;
    
    printf("\n=== DIAGNOSTIC: %s ===\n", label);
    printf("Nodes: %llu/%llu\n", (unsigned long long)gh->num_nodes, 
           (unsigned long long)gh->node_capacity);
    printf("Edges: %llu/%llu\n", (unsigned long long)gh->num_edges,
           (unsigned long long)gh->edge_capacity);
    printf("Decay rate: %.4f\n", gh->decay_rate);
    printf("Learning rate: %.6f\n", gh->learning_rate);
    
    // Count active nodes
    uint64_t active_nodes = 0;
    float total_activation = 0.0f;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (fabsf(file->nodes[i].state) > 0.01f) {
            active_nodes++;
            total_activation += fabsf(file->nodes[i].state);
        }
    }
    printf("Active nodes: %llu (avg activation: %.4f)\n",
           (unsigned long long)active_nodes,
           active_nodes > 0 ? total_activation / active_nodes : 0.0f);
    
    // Count strong edges
    uint64_t strong_edges = 0;
    float total_weight = 0.0f;
    float max_weight = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file->edges[i].weight > 0.3f) {
            strong_edges++;
            total_weight += file->edges[i].weight;
            if (file->edges[i].weight > max_weight) {
                max_weight = file->edges[i].weight;
            }
        }
    }
    printf("Strong edges (>0.3): %llu (avg: %.4f, max: %.4f)\n",
           (unsigned long long)strong_edges,
           strong_edges > 0 ? total_weight / strong_edges : 0.0f,
           max_weight);
}

static void diagnose_node(MelvinFile *file, uint64_t node_id, const char *label) {
    GraphHeaderDisk *gh = file->graph_header;
    
    uint64_t node_idx = find_node_index_by_id(file, node_id);
    if (node_idx == UINT64_MAX) {
        printf("\n=== NODE %llu (%s): NOT FOUND ===\n",
               (unsigned long long)node_id, label);
        return;
    }
    
    NodeDisk *node = &file->nodes[node_idx];
    
    printf("\n=== NODE %llu (%s) ===\n", (unsigned long long)node_id, label);
    printf("State (activation): %.6f\n", node->state);
    printf("Prediction: %.6f\n", node->prediction);
    printf("Prediction error: %.6f\n", node->prediction_error);
    printf("Reward: %.6f\n", node->reward);
    printf("Stability: %.6f\n", node->stability);
    printf("Trace: %.6f\n", node->trace);
    printf("Bias: %.6f\n", node->bias);
    printf("Flags: 0x%x\n", node->flags);
    printf("Out degree: %u\n", node->out_degree);
    printf("Firing count: %llu\n", (unsigned long long)node->firing_count);
    
    // List outgoing edges
    if (node->out_degree > 0) {
        printf("Outgoing edges:\n");
        uint64_t e_idx = node->first_out_edge;
        for (uint32_t i = 0; i < node->out_degree && e_idx != UINT64_MAX && 
             e_idx < gh->edge_capacity; i++) {
            EdgeDisk *e = &file->edges[e_idx];
            printf("  -> Node %llu: weight=%.6f, trace=%.6f, usage=%llu\n",
                   (unsigned long long)e->dst, e->weight, e->trace,
                   (unsigned long long)e->usage);
            e_idx = e->next_out_edge;
        }
    }
}

static void diagnose_edge(MelvinFile *file, uint64_t src_id, uint64_t dst_id, const char *label) {
    GraphHeaderDisk *gh = file->graph_header;
    
    // Find edge
    bool found = false;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file->edges[i].src == src_id && file->edges[i].dst == dst_id) {
            EdgeDisk *e = &file->edges[i];
            printf("\n=== EDGE %llu -> %llu (%s) ===\n",
                   (unsigned long long)src_id, (unsigned long long)dst_id, label);
            printf("Weight: %.6f\n", e->weight);
            printf("Trace: %.6f\n", e->trace);
            printf("Eligibility: %.6f\n", e->eligibility);
            printf("Usage: %llu\n", (unsigned long long)e->usage);
            printf("Age: %llu\n", (unsigned long long)e->age);
            found = true;
            break;
        }
    }
    
    if (!found) {
        printf("\n=== EDGE %llu -> %llu (%s): NOT FOUND ===\n",
               (unsigned long long)src_id, (unsigned long long)dst_id, label);
    }
}

// Helper to log edge state for evolution tracking
static void log_edge_state(const char *label, MelvinFile *file, uint64_t src_id, uint64_t dst_id) {
    GraphHeaderDisk *gh = file->graph_header;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file->edges[i].src == src_id && file->edges[i].dst == dst_id) {
            EdgeDisk *e = &file->edges[i];
            printf("[EDGE_EVOLUTION] %s: src=%llu dst=%llu weight=%.6f trace=%.2f eligibility=%.6f usage=%llu\n",
                   label,
                   (unsigned long long)src_id, (unsigned long long)dst_id,
                   e->weight, e->trace, e->eligibility, (unsigned long long)e->usage);
            return;
        }
    }
    printf("[EDGE_EVOLUTION] %s: src=%llu dst=%llu NOT FOUND\n",
           label, (unsigned long long)src_id, (unsigned long long)dst_id);
}

static void diagnose_learning_progress(MelvinFile *file, const char *pattern) {
    printf("\n=== LEARNING PROGRESS: %s ===\n", pattern);
    
    GraphHeaderDisk *gh = file->graph_header;
    
    // Node IDs for bytes are: byte_value + 1000000ULL
    uint64_t pattern_nodes[10] = {0};
    uint64_t pattern_count = 0;
    
    for (int j = 0; pattern[j] != '\0' && j < 10; j++) {
        uint64_t expected_id = (uint64_t)(unsigned char)pattern[j] + 1000000ULL;
        uint64_t node_idx = find_node_index_by_id(file, expected_id);
        if (node_idx != UINT64_MAX) {
            pattern_nodes[pattern_count++] = expected_id;
        }
    }
    
    printf("Found %llu pattern nodes\n", (unsigned long long)pattern_count);
    
    // Check edges between pattern nodes
    for (uint64_t i = 0; i < pattern_count - 1; i++) {
        uint64_t src = pattern_nodes[i];
        uint64_t dst = pattern_nodes[i + 1];
        
        diagnose_edge(file, src, dst, "Pattern edge");
    }
}

// ========================================================================
// PERSISTENT EVOLUTION TESTS
// ========================================================================

static void test_evolution_1_compounding_learning(const char *file_path, int run_number) {
    printf("\n========================================\n");
    printf("EVOLUTION TEST 1: Compounding Learning\n");
    printf("Run #%d (building on previous runs)\n", run_number);
    printf("This test is designed to show learning compounding over multiple invocations.\n");
    printf("========================================\n");
    
    MelvinFile file;
    bool is_new_file = false;
    
    // Try to load existing file, or create new
    if (access(file_path, F_OK) == 0) {
        printf("Loading existing file: %s\n", file_path);
        if (melvin_m_map(file_path, &file) < 0) {
            printf("Failed to load, creating new file\n");
            is_new_file = true;
        }
    } else {
        printf("Creating new file: %s\n", file_path);
        is_new_file = true;
    }
    
    if (is_new_file) {
        GraphParams params = {0};
        params.decay_rate = 0.97f;
        params.learning_rate = 0.02f;  // Higher learning rate for tests (2x default)
        params.homeostasis_target = 0.5f;
        
        if (melvin_m_init_new_file(file_path, &params) < 0) {
            printf("Failed to create file\n");
            return;
        }
        
        if (melvin_m_map(file_path, &file) < 0) {
            printf("Failed to map file\n");
            return;
        }
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        printf("Failed to init runtime\n");
        close_file(&file);
        return;
    }
    
    // Node IDs: byte_value + 1000000ULL
    uint64_t node_a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t node_b_id = (uint64_t)'B' + 1000000ULL;
    uint64_t node_c_id = (uint64_t)'C' + 1000000ULL;
    
    // Find and log A->B edge BEFORE training
    float weight_before = 0.0f;
    float trace_before = 0.0f;
    GraphHeaderDisk *gh = file.graph_header;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_a_id && file.edges[i].dst == node_b_id) {
            weight_before = file.edges[i].weight;
            trace_before = file.edges[i].trace;
            break;
        }
    }
    
    log_edge_state("BEFORE_TRAIN", &file, node_a_id, node_b_id);
    diagnose_graph_state(&file, "Before training");
    
    // Train on ABC pattern with EXPLICIT prediction_error
    printf("\nTraining on ABC pattern (500 iterations) with explicit targets...\n");
    
    // Find node indices
    uint64_t node_a_idx = find_node_index_by_id(&file, node_a_id);
    uint64_t node_b_idx = find_node_index_by_id(&file, node_b_id);
    uint64_t node_c_idx = find_node_index_by_id(&file, node_c_id);
    
    // Define explicit targets for supervised learning
    float target_A = 0.0f;  // A should be low after activation
    float target_B = 0.0f;  // B should be low after activation
    float target_C = 1.0f;  // C should be HIGH (we want C to activate when A->B->C chain fires)
    
    for (int i = 0; i < 500; i++) {
        // Forward pass: A -> B -> C
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        // ========================================================================
        // EXPLICIT PREDICTION ERROR SETUP
        // ========================================================================
        // Set prediction_error = target - activation for supervised learning
        // This ensures learning has a clear signal to work with
        // ========================================================================
        if (node_a_idx != UINT64_MAX) {
            NodeDisk *node_a = &file.nodes[node_a_idx];
            node_a->prediction_error = target_A - node_a->state;
        }
        
        if (node_b_idx != UINT64_MAX) {
            NodeDisk *node_b = &file.nodes[node_b_idx];
            node_b->prediction_error = target_B - node_b->state;
        }
        
        if (node_c_idx != UINT64_MAX) {
            NodeDisk *node_c = &file.nodes[node_c_idx];
            node_c->prediction_error = target_C - node_c->state;
        }
        
        // Log epsilon stats every 100 iterations
        if ((i + 1) % 100 == 0 && node_a_idx != UINT64_MAX && node_b_idx != UINT64_MAX && node_c_idx != UINT64_MAX) {
            printf("[EPS_STATS] iter=%d eps_A=%.6f eps_B=%.6f eps_C=%.6f\n",
                   i + 1,
                   file.nodes[node_a_idx].prediction_error,
                   file.nodes[node_b_idx].prediction_error,
                   file.nodes[node_c_idx].prediction_error);
        }
        
        // ========================================================================
        // EXPLICIT LEARNING CALL
        // ========================================================================
        // Call learning kernel directly to ensure it runs with our explicit errors
        // ========================================================================
        strengthen_edges_with_prediction_and_reward(&rt);
        
        // Log A->B edge learning every 50 iterations
        if ((i + 1) % 50 == 0) {
            // Find A->B edge and log learning details
            for (uint64_t j = 0; j < gh->num_edges && j < gh->edge_capacity; j++) {
                if (file.edges[j].src == node_a_id && file.edges[j].dst == node_b_id) {
                    EdgeDisk *e_ab = &file.edges[j];
                    NodeDisk *node_b = (node_b_idx != UINT64_MAX) ? &file.nodes[node_b_idx] : NULL;
                    
                    if (node_b) {
                        printf("[AB_LEARN] iter=%d eps=%.6f elig=%.6f lr=%.6f weight=%.6f trace=%.2f\n",
                               i + 1,
                               node_b->prediction_error,
                               e_ab->eligibility,
                               gh->learning_rate,
                               e_ab->weight,
                               e_ab->trace);
                    }
                    break;
                }
            }
        }
        
        // Also trigger homeostasis to ensure eligibility traces are updated
        if ((i + 1) % 50 == 0) {
            MelvinEvent homeo_ev = { .type = EV_HOMEOSTASIS_SWEEP };
            melvin_event_enqueue(&rt.evq, &homeo_ev);
            melvin_process_n_events(&rt, 10);
        }
    }
    
    // Final epsilon stats
    if (node_a_idx != UINT64_MAX && node_b_idx != UINT64_MAX && node_c_idx != UINT64_MAX) {
        printf("\n[EPS_STATS] Final: eps_A=%.6f eps_B=%.6f eps_C=%.6f\n",
               file.nodes[node_a_idx].prediction_error,
               file.nodes[node_b_idx].prediction_error,
               file.nodes[node_c_idx].prediction_error);
    }
    
    // Find and log A->B edge AFTER training
    float weight_after = 0.0f;
    float trace_after = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_a_id && file.edges[i].dst == node_b_id) {
            weight_after = file.edges[i].weight;
            trace_after = file.edges[i].trace;
            break;
        }
    }
    
    log_edge_state("AFTER_TRAIN", &file, node_a_id, node_b_id);
    diagnose_graph_state(&file, "After training");
    diagnose_learning_progress(&file, "ABC");
    
    // Check if learning compounded
    float max_weight_ab = weight_after;
    float max_weight_bc = 0.0f;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_b_id && file.edges[i].dst == node_c_id) {
            if (file.edges[i].weight > max_weight_bc) {
                max_weight_bc = file.edges[i].weight;
            }
        }
    }
    
    printf("\n=== LEARNING RESULTS ===\n");
    printf("A->B weight: %.6f (was %.6f, change: %.6f)\n", 
           max_weight_ab, weight_before, max_weight_ab - weight_before);
    printf("B->C weight: %.6f\n", max_weight_bc);
    printf("Total pattern strength: %.6f\n", max_weight_ab + max_weight_bc);
    
    // ASSERTION: If trace increased significantly, weight should increase
    float trace_increase = trace_after - trace_before;
    float weight_increase = weight_after - weight_before;
    
    printf("\n=== COMPOUNDING CHECK ===\n");
    printf("Trace increase: %.2f\n", trace_increase);
    printf("Weight increase: %.6f\n", weight_increase);
    
    if (trace_increase > 100.0f) {
        // Heavy usage - weight should have increased
        if (weight_increase < 0.001f) {
            printf("WARNING: Trace increased by %.2f but weight only increased by %.6f\n",
                   trace_increase, weight_increase);
            printf("         Learning may not be compounding properly!\n");
        } else {
            printf("SUCCESS: Weight increased with trace (learning is compounding)\n");
        }
    }
    
    // Save state
    melvin_m_sync(&file);
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    printf("\nFile saved. Next run will build on this learning.\n");
}

static void test_evolution_2_multistep_reasoning(const char *file_path, int run_number) {
    printf("\n========================================\n");
    printf("EVOLUTION TEST 2: Multi-Step Reasoning\n");
    printf("Run #%d (building on previous runs)\n", run_number);
    printf("========================================\n");
    
    MelvinFile file;
    bool is_new_file = false;
    
    if (access(file_path, F_OK) == 0) {
        printf("Loading existing file: %s\n", file_path);
        if (melvin_m_map(file_path, &file) < 0) {
            is_new_file = true;
        }
    } else {
        is_new_file = true;
    }
    
    if (is_new_file) {
        GraphParams params = {0};
        params.decay_rate = 0.97f;
        params.learning_rate = 0.01f;
        
        if (melvin_m_init_new_file(file_path, &params) < 0) return;
        if (melvin_m_map(file_path, &file) < 0) return;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        close_file(&file);
        return;
    }
    
    diagnose_graph_state(&file, "Before training");
    
    // Node IDs: byte_value + 1000000ULL
    uint64_t node_a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t node_b_id = (uint64_t)'B' + 1000000ULL;
    uint64_t node_c_id = (uint64_t)'C' + 1000000ULL;
    uint64_t node_d_id = (uint64_t)'D' + 1000000ULL;
    
    // Find node indices
    uint64_t node_a_idx = find_node_index_by_id(&file, node_a_id);
    uint64_t node_b_idx = find_node_index_by_id(&file, node_b_id);
    uint64_t node_c_idx = find_node_index_by_id(&file, node_c_id);
    uint64_t node_d_idx = find_node_index_by_id(&file, node_d_id);
    
    // Define explicit targets for supervised learning
    float target_A = 0.0f;  // A should be low after activation
    float target_B = 0.0f;  // B should be low after activation
    float target_C = 0.0f;  // C should be low after activation
    float target_D = 1.0f;  // D should be HIGH (we want D to activate when A->B->C->D chain fires)
    
    // Train on A->B->C->D chain with EXPLICIT prediction_error
    printf("\nTraining on A->B->C->D chain (1000 iterations) with explicit targets...\n");
    for (int i = 0; i < 1000; i++) {
        // Forward pass: A -> B -> C -> D
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'D', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        // ========================================================================
        // EXPLICIT PREDICTION ERROR SETUP
        // ========================================================================
        if (node_a_idx != UINT64_MAX) {
            file.nodes[node_a_idx].prediction_error = target_A - file.nodes[node_a_idx].state;
        }
        if (node_b_idx != UINT64_MAX) {
            file.nodes[node_b_idx].prediction_error = target_B - file.nodes[node_b_idx].state;
        }
        if (node_c_idx != UINT64_MAX) {
            file.nodes[node_c_idx].prediction_error = target_C - file.nodes[node_c_idx].state;
        }
        if (node_d_idx != UINT64_MAX) {
            file.nodes[node_d_idx].prediction_error = target_D - file.nodes[node_d_idx].state;
        }
        
        // Log epsilon stats every 200 iterations
        if ((i + 1) % 200 == 0 && node_d_idx != UINT64_MAX) {
            printf("[EPS_STATS] iter=%d eps_D=%.6f (target=%.1f, state=%.6f)\n",
                   i + 1,
                   file.nodes[node_d_idx].prediction_error,
                   target_D,
                   file.nodes[node_d_idx].state);
        }
        
        // ========================================================================
        // EXPLICIT LEARNING CALL
        // ========================================================================
        strengthen_edges_with_prediction_and_reward(&rt);
        
        // Trigger homeostasis periodically
        if ((i + 1) % 100 == 0) {
            MelvinEvent homeo_ev = { .type = EV_HOMEOSTASIS_SWEEP };
            melvin_event_enqueue(&rt.evq, &homeo_ev);
            melvin_process_n_events(&rt, 10);
        }
    }
    
    // Final epsilon stats
    if (node_d_idx != UINT64_MAX) {
        printf("\n[EPS_STATS] Final: eps_D=%.6f (target=%.1f, state=%.6f)\n",
               file.nodes[node_d_idx].prediction_error,
               target_D,
               file.nodes[node_d_idx].state);
    }
    
    diagnose_graph_state(&file, "After training");
    
    GraphHeaderDisk *gh = file.graph_header;
    float weight_ab = 0.0f, weight_bc = 0.0f, weight_cd = 0.0f;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_a_id && file.edges[i].dst == node_b_id) {
            weight_ab = file.edges[i].weight;
        }
        if (file.edges[i].src == node_b_id && file.edges[i].dst == node_c_id) {
            weight_bc = file.edges[i].weight;
        }
        if (file.edges[i].src == node_c_id && file.edges[i].dst == node_d_id) {
            weight_cd = file.edges[i].weight;
        }
    }
    
    printf("\n=== MULTI-STEP CHAIN STRENGTH ===\n");
    printf("A->B: %.6f\n", weight_ab);
    printf("B->C: %.6f\n", weight_bc);
    printf("C->D: %.6f\n", weight_cd);
    printf("Chain complete: %s\n", 
           (weight_ab > 0.3f && weight_bc > 0.3f && weight_cd > 0.3f) ? "YES" : "NO");
    printf("Total chain strength: %.6f\n", weight_ab + weight_bc + weight_cd);
    
    // Test prediction: Activate A, see if D eventually activates
    printf("\n=== TESTING PREDICTION ===\n");
    
    // Reset activations
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (file.nodes[i].id == node_a_id || 
            file.nodes[i].id == node_b_id ||
            file.nodes[i].id == node_c_id ||
            file.nodes[i].id == node_d_id) {
            file.nodes[i].state = 0.0f;
        }
    }
    
    // Activate A (node_a_idx already defined above in training loop)
    // Re-find it here since we're in a different scope
    uint64_t node_a_idx_test = find_node_index_by_id(&file, node_a_id);
    if (node_a_idx_test != UINT64_MAX) {
        inject_pulse(&rt, node_a_id, 2.0f);
        printf("Activated A with strength 2.0\n");
        
        // Re-find node_d_idx for this scope
        uint64_t node_d_idx_test = find_node_index_by_id(&file, node_d_id);
        
        // Process events to propagate
        for (int i = 0; i < 100; i++) {
            melvin_process_n_events(&rt, 10);
            
            if (node_d_idx_test != UINT64_MAX) {
                float d_activation = file.nodes[node_d_idx_test].state;
                if (i % 20 == 0) {
                    printf("  After %d events: D activation = %.6f\n", i * 10, d_activation);
                }
            }
        }
        
        if (node_d_idx_test != UINT64_MAX) {
            float d_final = file.nodes[node_d_idx_test].state;
            printf("Final D activation: %.6f\n", d_final);
            printf("Multi-step prediction: %s\n", d_final > 0.1f ? "SUCCESS" : "FAILED");
        }
    }
    
    melvin_m_sync(&file);
    
    runtime_cleanup(&rt);
    close_file(&file);
}

static void test_evolution_3_learning_compounding(const char *file_path, int run_number) {
    printf("\n========================================\n");
    printf("EVOLUTION TEST 3: Learning Compounding\n");
    printf("Run #%d (checking if learning compounds)\n", run_number);
    printf("========================================\n");
    
    MelvinFile file;
    bool is_new_file = false;
    
    if (access(file_path, F_OK) == 0) {
        if (melvin_m_map(file_path, &file) < 0) {
            is_new_file = true;
        }
    } else {
        is_new_file = true;
    }
    
    if (is_new_file) {
        GraphParams params = {0};
        params.decay_rate = 0.97f;
        params.learning_rate = 0.02f;  // Higher learning rate for tests
        
        if (melvin_m_init_new_file(file_path, &params) < 0) return;
        if (melvin_m_map(file_path, &file) < 0) return;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        close_file(&file);
        return;
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Node IDs: byte_value + 1000000ULL
    uint64_t node_x_id = (uint64_t)'X' + 1000000ULL;
    uint64_t node_y_id = (uint64_t)'Y' + 1000000ULL;
    
    // Check existing learning
    float weight_before = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_x_id && file.edges[i].dst == node_y_id) {
            weight_before = file.edges[i].weight;
            break;
        }
    }
    
    printf("Weight before training: %.6f\n", weight_before);
    
    // Train on X->Y pattern
    // Find node indices
    uint64_t node_x_idx = find_node_index_by_id(&file, node_x_id);
    uint64_t node_y_idx = find_node_index_by_id(&file, node_y_id);
    
    // Define explicit targets
    float target_X = 0.0f;  // X should be low after activation
    float target_Y = 1.0f;  // Y should be HIGH (we want Y to activate when X->Y fires)
    
    printf("Training on X->Y pattern (500 iterations) with explicit targets...\n");
    for (int i = 0; i < 500; i++) {
        // Forward pass: X -> Y
        ingest_byte(&rt, 1ULL, 'X', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'Y', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        // Set explicit prediction_error
        if (node_x_idx != UINT64_MAX) {
            file.nodes[node_x_idx].prediction_error = target_X - file.nodes[node_x_idx].state;
        }
        if (node_y_idx != UINT64_MAX) {
            file.nodes[node_y_idx].prediction_error = target_Y - file.nodes[node_y_idx].state;
        }
        
        // Call learning explicitly
        strengthen_edges_with_prediction_and_reward(&rt);
        
        // Log every 100 iterations
        if ((i + 1) % 100 == 0 && node_y_idx != UINT64_MAX) {
            printf("[EPS_STATS] iter=%d eps_Y=%.6f (target=%.1f, state=%.6f)\n",
                   i + 1,
                   file.nodes[node_y_idx].prediction_error,
                   target_Y,
                   file.nodes[node_y_idx].state);
        }
        
        // Trigger homeostasis periodically
        if ((i + 1) % 50 == 0) {
            MelvinEvent homeo_ev = { .type = EV_HOMEOSTASIS_SWEEP };
            melvin_event_enqueue(&rt.evq, &homeo_ev);
            melvin_process_n_events(&rt, 10);
        }
    }
    
    // Final epsilon stats
    if (node_y_idx != UINT64_MAX) {
        printf("\n[EPS_STATS] Final: eps_Y=%.6f (target=%.1f, state=%.6f)\n",
               file.nodes[node_y_idx].prediction_error,
               target_Y,
               file.nodes[node_y_idx].state);
    }
    
    // Check weight after
    float weight_after = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_x_id && file.edges[i].dst == node_y_id) {
            weight_after = file.edges[i].weight;
            break;
        }
    }
    
    printf("Weight after training: %.6f\n", weight_after);
    printf("Weight change: %.6f\n", weight_after - weight_before);
    printf("Learning compounded: %s\n", 
           (weight_after > weight_before) ? "YES" : "NO");
    
    diagnose_edge(&file, node_x_id, node_y_id, "X->Y after training");
    
    melvin_m_sync(&file);
    
    runtime_cleanup(&rt);
    close_file(&file);
}

// ========================================================================
// MULTI-RUN COMPOUNDING TEST
// ========================================================================

/*
 * This test demonstrates true compounding across separate process runs.
 * 
 * Usage:
 *   Phase 1: ./test_evolution_diagnostic --phase=1
 *   Phase 2: ./test_evolution_diagnostic --phase=2
 * 
 * Expected behavior:
 *   weight_run2_end > weight_run1_end > initial_weight
 * 
 * This proves that learning persists and compounds across process boundaries.
 */
static void test_multi_run_compounding(int phase) {
    const char *file_path = "evolution_compound.m";
    
    printf("\n========================================\n");
    printf("MULTI-RUN COMPOUNDING TEST\n");
    printf("Phase %d\n", phase);
    printf("========================================\n");
    
    MelvinFile file;
    bool is_new_file = false;
    
    if (access(file_path, F_OK) == 0) {
        printf("Loading existing file: %s\n", file_path);
        if (melvin_m_map(file_path, &file) < 0) {
            printf("Failed to load, creating new file\n");
            is_new_file = true;
        }
    } else {
        printf("Creating new file: %s\n", file_path);
        is_new_file = true;
    }
    
    if (is_new_file) {
        GraphParams params = {0};
        params.decay_rate = 0.97f;
        params.learning_rate = 0.02f;  // Higher learning rate for tests
        params.homeostasis_target = 0.5f;
        
        if (melvin_m_init_new_file(file_path, &params) < 0) {
            printf("Failed to create file\n");
            return;
        }
        
        if (melvin_m_map(file_path, &file) < 0) {
            printf("Failed to map file\n");
            return;
        }
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        printf("Failed to init runtime\n");
        close_file(&file);
        return;
    }
    
    // Node IDs: byte_value + 1000000ULL
    uint64_t node_a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t node_b_id = (uint64_t)'B' + 1000000ULL;
    
    // Find A->B edge
    GraphHeaderDisk *gh = file.graph_header;
    float weight_start = 0.0f;
    bool edge_exists = false;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_a_id && file.edges[i].dst == node_b_id) {
            weight_start = file.edges[i].weight;
            edge_exists = true;
            break;
        }
    }
    
    printf("\n=== PHASE %d START ===\n", phase);
    if (edge_exists) {
        printf("A->B edge found: weight = %.6f\n", weight_start);
        log_edge_state("PHASE_START", &file, node_a_id, node_b_id);
    } else {
        printf("A->B edge not found (will be created during training)\n");
    }
    diagnose_graph_state(&file, "Phase start");
    
    // Train on A->B->C pattern
    printf("\nTraining on A->B->C pattern (1000 iterations)...\n");
    for (int i = 0; i < 1000; i++) {
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 10);
    }
    
    // Find A->B edge after training
    float weight_end = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == node_a_id && file.edges[i].dst == node_b_id) {
            weight_end = file.edges[i].weight;
            break;
        }
    }
    
    printf("\n=== PHASE %d END ===\n", phase);
    if (weight_end > 0.0f || edge_exists) {
        printf("A->B edge: weight = %.6f (was %.6f, change: %.6f)\n",
               weight_end, weight_start, weight_end - weight_start);
        log_edge_state("PHASE_END", &file, node_a_id, node_b_id);
    } else {
        printf("A->B edge still not found after training\n");
    }
    diagnose_graph_state(&file, "Phase end");
    
    // Save to file for next phase
    melvin_m_sync(&file);
    
    printf("\n=== PHASE %d SUMMARY ===\n", phase);
    printf("Starting weight: %.6f\n", weight_start);
    printf("Ending weight: %.6f\n", weight_end);
    printf("Weight change: %.6f\n", weight_end - weight_start);
    
    if (phase == 2) {
        // In phase 2, we can compare to phase 1
        // Expected: weight_end_phase2 > weight_end_phase1
        printf("\n=== MULTI-RUN COMPOUNDING CHECK ===\n");
        printf("Phase 2 ending weight: %.6f\n", weight_end);
        printf("Expected: weight_phase2_end > weight_phase1_end\n");
        printf("(Check phase 1 output to compare)\n");
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    printf("\nFile saved. Run phase %d next.\n", phase + 1);
}

// ========================================================================
// MAIN
// ========================================================================

int main(int argc, char **argv) {
    // Check for multi-run test phase
    int phase = 0;
    if (argc > 1) {
        if (strcmp(argv[1], "--phase=1") == 0) {
            phase = 1;
        } else if (strcmp(argv[1], "--phase=2") == 0) {
            phase = 2;
        }
    }
    
    if (phase > 0) {
        // Run multi-run compounding test
        test_multi_run_compounding(phase);
        return 0;
    }
    
    // Normal evolution diagnostic tests
    printf("========================================\n");
    printf("EVOLUTION DIAGNOSTIC TEST SUITE\n");
    printf("========================================\n\n");
    printf("These tests use PERSISTENT .m files that build over time.\n");
    printf("Learning should COMPOUND across runs.\n\n");
    printf("For multi-run test across process boundaries:\n");
    printf("  Phase 1: %s --phase=1\n", argv[0]);
    printf("  Phase 2: %s --phase=2\n\n", argv[0]);
    
    const char *file1 = "evolution_compounding.m";
    const char *file2 = "evolution_multistep.m";
    const char *file3 = "evolution_learning.m";
    
    // Run evolution tests multiple times to see compounding
    printf("=== RUN 1 ===\n");
    test_evolution_1_compounding_learning(file1, 1);
    test_evolution_2_multistep_reasoning(file2, 1);
    test_evolution_3_learning_compounding(file3, 1);
    
    printf("\n\n=== RUN 2 (should build on Run 1) ===\n");
    test_evolution_1_compounding_learning(file1, 2);
    test_evolution_2_multistep_reasoning(file2, 2);
    test_evolution_3_learning_compounding(file3, 2);
    
    printf("\n\n=== RUN 3 (should build on Runs 1-2) ===\n");
    test_evolution_1_compounding_learning(file1, 3);
    test_evolution_2_multistep_reasoning(file2, 3);
    test_evolution_3_learning_compounding(file3, 3);
    
    printf("\n========================================\n");
    printf("Evolution tests complete.\n");
    printf("Files saved: %s, %s, %s\n", file1, file2, file3);
    printf("Run again to see learning compound further.\n");
    printf("========================================\n");
    
    return 0;
}


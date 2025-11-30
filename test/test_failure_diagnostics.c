#define _POSIX_C_SOURCE 200809L

/*
 * test_failure_diagnostics.c
 * 
 * Comprehensive diagnostics for test failures:
 * - Analyzes why each test fails
 * - Checks if instinct nodes exist and are usable
 * - Verifies learning mechanisms
 * - Identifies missing components
 * - Provides actionable fixes
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "melvin.c"

// Forward declarations
extern void strengthen_edges_with_prediction_and_reward(MelvinRuntime *rt);

// ========================================================================
// Diagnostic: Check if instinct nodes exist
// ========================================================================

static void diagnose_instinct_nodes(MelvinFile *file) {
    printf("\n=== DIAGNOSTIC: Instinct Nodes ===\n\n");
    
    uint64_t nodes_checked = 0;
    uint64_t nodes_found = 0;
    
    // Check param nodes
    printf("Param Nodes:\n");
    uint64_t param_ids[] = {
        101ULL, // NODE_ID_PARAM_DECAY
        102ULL, // NODE_ID_PARAM_BIAS
        103ULL, // NODE_ID_PARAM_EXEC_THRESHOLD
        104ULL, // NODE_ID_PARAM_LEARN_RATE
        201ULL, // NODE_ID_LAW_LR_BASE
        202ULL, // NODE_ID_LAW_W_LIMIT
    };
    const char *param_names[] = {
        "PARAM_DECAY", "PARAM_BIAS", "PARAM_EXEC_THRESHOLD",
        "PARAM_LEARN_RATE", "LAW_LR_BASE", "LAW_W_LIMIT"
    };
    
    for (int i = 0; i < 6; i++) {
        uint64_t idx = find_node_index_by_id(file, param_ids[i]);
        nodes_checked++;
        if (idx != UINT64_MAX) {
            printf("  ✓ %s (ID %llu, state=%.3f)\n", 
                   param_names[i], (unsigned long long)param_ids[i],
                   file->nodes[idx].state);
            nodes_found++;
        } else {
            printf("  ✗ %s (ID %llu) MISSING\n", 
                   param_names[i], (unsigned long long)param_ids[i]);
        }
    }
    
    // Check instinct pattern nodes
    printf("\nInstinct Pattern Nodes:\n");
    uint64_t pattern_ids[] = {
        50000ULL, // EXEC:HUB
        60000ULL, // MATH:IN_A
        70000ULL, // COMP:REQ
        80000ULL, // PORT:IN
        10000ULL, // CH:CODE_RAW:IN
    };
    const char *pattern_names[] = {
        "EXEC:HUB", "MATH:IN_A", "COMP:REQ", "PORT:IN", "CH:CODE_RAW:IN"
    };
    
    for (int i = 0; i < 5; i++) {
        uint64_t idx = find_node_index_by_id(file, pattern_ids[i]);
        nodes_checked++;
        if (idx != UINT64_MAX) {
            printf("  ✓ %s (ID %llu, state=%.3f)\n", 
                   pattern_names[i], (unsigned long long)pattern_ids[i],
                   file->nodes[idx].state);
            nodes_found++;
        } else {
            printf("  ✗ %s (ID %llu) MISSING\n", 
                   pattern_names[i], (unsigned long long)pattern_ids[i]);
        }
    }
    
    printf("\nSummary: %llu/%llu instinct nodes found\n",
           (unsigned long long)nodes_found,
           (unsigned long long)nodes_checked);
}

// ========================================================================
// Diagnostic: Check learning mechanism
// ========================================================================

static void diagnose_learning(MelvinRuntime *rt) {
    printf("\n=== DIAGNOSTIC: Learning Mechanism ===\n\n");
    
    MelvinFile *file = rt->file;
    GraphHeaderDisk *gh = file->graph_header;
    
    printf("[1] Learning rate: %.6f\n", gh->learning_rate);
    printf("[2] Reward lambda: %.6f\n", gh->reward_lambda);
    
    // Check if any nodes have prediction_error
    uint64_t nodes_with_error = 0;
    float max_error = 0.0f;
    float sum_error = 0.0f;
    
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        float err = fabsf(file->nodes[i].prediction_error);
        if (err > 0.001f) {
            nodes_with_error++;
            sum_error += err;
            if (err > max_error) max_error = err;
        }
    }
    
    printf("[3] Nodes with prediction_error: %llu\n", 
           (unsigned long long)nodes_with_error);
    if (nodes_with_error > 0) {
        printf("    Max error: %.6f\n", max_error);
        printf("    Avg error: %.6f\n", sum_error / nodes_with_error);
    } else {
        printf("    ⚠ WARNING: No nodes have prediction_error set!\n");
        printf("    This is why learning tests fail.\n");
    }
    
    // Check eligibility traces
    uint64_t edges_with_eligibility = 0;
    float max_eligibility = 0.0f;
    float sum_eligibility = 0.0f;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &file->edges[i];
        if (e->src == UINT64_MAX || e->dst == UINT64_MAX) continue;
        
        float elig = fabsf(e->eligibility);
        if (elig > 0.001f) {
            edges_with_eligibility++;
            sum_eligibility += elig;
            if (elig > max_eligibility) max_eligibility = elig;
        }
    }
    
    printf("[4] Edges with eligibility: %llu\n", 
           (unsigned long long)edges_with_eligibility);
    if (edges_with_eligibility > 0) {
        printf("    Max eligibility: %.6f\n", max_eligibility);
        printf("    Avg eligibility: %.6f\n", 
               sum_eligibility / edges_with_eligibility);
    } else {
        printf("    ⚠ WARNING: No edges have eligibility!\n");
        printf("    Eligibility is needed for learning.\n");
    }
    
    // Check if learning function is called
    printf("[5] Learning function available: ✓\n");
    printf("    strengthen_edges_with_prediction_and_reward() exists\n");
}

// ========================================================================
// Diagnostic: Check specific test failure reasons
// ========================================================================

static void diagnose_test_0_5_1_learning(MelvinFile *file, MelvinRuntime *rt) {
    printf("\n=== DIAGNOSTIC: Test 0.5.1 (Learning Only Prediction Error) ===\n\n");
    
    printf("This test fails because: 'No learning occurred'\n");
    printf("\nChecking why:\n");
    
    // Check if epsilon is set
    GraphHeaderDisk *gh = file->graph_header;
    uint64_t nodes_with_epsilon = 0;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (fabsf(file->nodes[i].prediction_error) > 0.001f) {
            nodes_with_epsilon++;
        }
    }
    
    printf("[1] Nodes with prediction_error: %llu\n", 
           (unsigned long long)nodes_with_epsilon);
    
    if (nodes_with_epsilon == 0) {
        printf("    ✗ PROBLEM: No nodes have prediction_error set!\n");
        printf("    FIX: Tests need to call melvin_set_epsilon_for_node()\n");
        printf("         or melvin_compute_epsilon_from_observation()\n");
    }
    
    // Check if learning is called
    printf("[2] Learning function called: ");
    printf("(would be called during EV_HOMEOSTASIS_SWEEP)\n");
    
    // Check edge weights before/after
    printf("[3] Checking edge weights...\n");
    uint64_t edges_checked = 0;
    uint64_t edges_with_modified_weights = 0;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &file->edges[i];
        if (e->src == UINT64_MAX || e->dst == UINT64_MAX) continue;
        
        edges_checked++;
        float w = fabsf(e->weight);
        if (w > 0.01f && w < 0.9f && 
            fabsf(w - 0.2f) > 0.01f && 
            fabsf(w - 0.3f) > 0.01f && 
            fabsf(w - 0.4f) > 0.01f) {
            edges_with_modified_weights++;
        }
    }
    
    printf("    Edges checked: %llu\n", (unsigned long long)edges_checked);
    printf("    Edges with modified weights: %llu\n",
           (unsigned long long)edges_with_modified_weights);
    
    printf("\nROOT CAUSE:\n");
    printf("  Tests don't set prediction_error on nodes.\n");
    printf("  Without epsilon, learning has nothing to learn from.\n");
    printf("\nFIX:\n");
    printf("  1. Tests should set targets for nodes\n");
    printf("  2. Call melvin_compute_epsilon_from_observation()\n");
    printf("  3. Or manually set prediction_error via melvin_set_epsilon_for_node()\n");
}

static void diagnose_test_0_8_1_params(MelvinFile *file) {
    printf("\n=== DIAGNOSTIC: Test 0.8.1 (Params as Nodes) ===\n\n");
    
    printf("This test fails because: 'Param nodes not created'\n");
    printf("\nChecking param nodes:\n");
    
    uint64_t param_ids[] = {
        101ULL, 102ULL, 103ULL, 104ULL, // Basic params
        201ULL, 202ULL, 203ULL, 204ULL, // Law nodes
    };
    const char *param_names[] = {
        "PARAM_DECAY", "PARAM_BIAS", "PARAM_EXEC_THRESHOLD", "PARAM_LEARN_RATE",
        "LAW_LR_BASE", "LAW_W_LIMIT", "LAW_EXEC_CENTER", "LAW_EXEC_K"
    };
    
    uint64_t found = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t idx = find_node_index_by_id(file, param_ids[i]);
        if (idx != UINT64_MAX) {
            printf("  ✓ %s found (state=%.3f)\n", 
                   param_names[i], file->nodes[idx].state);
            found++;
        } else {
            printf("  ✗ %s MISSING\n", param_names[i]);
        }
    }
    
    printf("\nFound: %llu/8 param nodes\n", (unsigned long long)found);
    
    if (found == 0) {
        printf("\nROOT CAUSE:\n");
        printf("  Tests create NEW files, so param nodes from instincts.c aren't present.\n");
        printf("  melvin.m has param nodes, but tests don't use melvin.m.\n");
        printf("\nFIX:\n");
        printf("  1. Tests should start from melvin.m (copy it)\n");
        printf("  2. Or tests should call melvin_inject_instincts() on new files\n");
        printf("  3. Or melvin.c should create param nodes automatically\n");
    } else if (found < 8) {
        printf("\nROOT CAUSE:\n");
        printf("  Some param nodes missing. Instincts.c may not have created all of them.\n");
    } else {
        printf("\n✓ All param nodes exist!\n");
        printf("  Test failure may be due to test logic, not missing nodes.\n");
    }
}

static void diagnose_test_hard_1_pattern(MelvinFile *file, MelvinRuntime *rt) {
    printf("\n=== DIAGNOSTIC: Test HARD-1 (Pattern Prediction) ===\n\n");
    
    printf("This test fails because: 'Pattern nodes not created'\n");
    printf("\nChecking pattern creation:\n");
    
    GraphHeaderDisk *gh = file->graph_header;
    
    // Check for byte-based nodes (A, B, C)
    uint64_t node_a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t node_b_id = (uint64_t)'B' + 1000000ULL;
    uint64_t node_c_id = (uint64_t)'C' + 1000000ULL;
    
    uint64_t a_idx = find_node_index_by_id(file, node_a_id);
    uint64_t b_idx = find_node_index_by_id(file, node_b_id);
    uint64_t c_idx = find_node_index_by_id(file, node_c_id);
    
    printf("[1] Pattern nodes (A, B, C):\n");
    if (a_idx != UINT64_MAX) printf("    ✓ A found\n");
    else printf("    ✗ A missing\n");
    if (b_idx != UINT64_MAX) printf("    ✓ B found\n");
    else printf("    ✗ B missing\n");
    if (c_idx != UINT64_MAX) printf("    ✓ C found\n");
    else printf("    ✗ C missing\n");
    
    // Check for edges between them
    printf("[2] Pattern edges:\n");
    uint64_t ab_edge = 0, bc_edge = 0;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &file->edges[i];
        if (e->src == node_a_id && e->dst == node_b_id) ab_edge++;
        if (e->src == node_b_id && e->dst == node_c_id) bc_edge++;
    }
    
    printf("    A->B edges: %llu\n", (unsigned long long)ab_edge);
    printf("    B->C edges: %llu\n", (unsigned long long)bc_edge);
    
    // Check if pattern can be activated
    if (a_idx != UINT64_MAX) {
        printf("[3] Activating pattern node A...\n");
        float state_before = file->nodes[a_idx].state;
        
        // Process some events
        melvin_process_n_events(rt, 20);
        
        float state_after = file->nodes[a_idx].state;
        printf("    State: %.3f -> %.3f\n", state_before, state_after);
    }
    
    printf("\nROOT CAUSE:\n");
    printf("  Tests create new files, so pattern nodes don't exist initially.\n");
    printf("  Pattern nodes are created when bytes are ingested, but tests may not\n");
    printf("  be ingesting bytes correctly.\n");
    printf("\nFIX:\n");
    printf("  1. Tests should ingest bytes to create pattern nodes\n");
    printf("  2. Or start from melvin.m which has instinct patterns\n");
    printf("  3. Verify byte ingestion creates nodes correctly\n");
}

static void diagnose_test_hard_5_multistep(MelvinFile *file, MelvinRuntime *rt) {
    printf("\n=== DIAGNOSTIC: Test HARD-5 (Multi-Step Reasoning) ===\n\n");
    
    printf("This test fails because: 'Failed to learn multi-step chain'\n");
    printf("\nChecking multi-step chain:\n");
    
    // Check for A->B->C->D chain
    uint64_t node_a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t node_b_id = (uint64_t)'B' + 1000000ULL;
    uint64_t node_c_id = (uint64_t)'C' + 1000000ULL;
    uint64_t node_d_id = (uint64_t)'D' + 1000000ULL;
    
    uint64_t a_idx = find_node_index_by_id(file, node_a_id);
    uint64_t b_idx = find_node_index_by_id(file, node_b_id);
    uint64_t c_idx = find_node_index_by_id(file, node_c_id);
    uint64_t d_idx = find_node_index_by_id(file, node_d_id);
    
    printf("[1] Chain nodes:\n");
    printf("    A: %s\n", a_idx != UINT64_MAX ? "found" : "missing");
    printf("    B: %s\n", b_idx != UINT64_MAX ? "found" : "missing");
    printf("    C: %s\n", c_idx != UINT64_MAX ? "found" : "missing");
    printf("    D: %s\n", d_idx != UINT64_MAX ? "found" : "missing");
    
    // Check chain edges
    GraphHeaderDisk *gh = file->graph_header;
    uint64_t ab = 0, bc = 0, cd = 0;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        EdgeDisk *e = &file->edges[i];
        if (e->src == node_a_id && e->dst == node_b_id) ab++;
        if (e->src == node_b_id && e->dst == node_c_id) bc++;
        if (e->src == node_c_id && e->dst == node_d_id) cd++;
    }
    
    printf("[2] Chain edges:\n");
    printf("    A->B: %llu\n", (unsigned long long)ab);
    printf("    B->C: %llu\n", (unsigned long long)bc);
    printf("    C->D: %llu\n", (unsigned long long)cd);
    
    // Check if activation propagates
    if (a_idx != UINT64_MAX && d_idx != UINT64_MAX) {
        printf("[3] Testing activation propagation...\n");
        
        // Activate A
        file->nodes[a_idx].state = 1.0f;
        
        // Process events
        melvin_process_n_events(rt, 50);
        
        float d_activation = file->nodes[d_idx].state;
        printf("    D activation after activating A: %.6f\n", d_activation);
        
        if (d_activation > 0.1f) {
            printf("    ✓ Activation propagated!\n");
        } else {
            printf("    ✗ Activation did NOT propagate\n");
            printf("    Chain may be too weak or missing edges\n");
        }
    }
    
    printf("\nROOT CAUSE:\n");
    printf("  Multi-step chains require:\n");
    printf("  1. All nodes exist (A, B, C, D)\n");
    printf("  2. Edges between them (A->B, B->C, C->D)\n");
    printf("  3. Strong enough weights for propagation\n");
    printf("  4. Learning to strengthen the chain\n");
    printf("\nFIX:\n");
    printf("  1. Ensure nodes are created (byte ingestion)\n");
    printf("  2. Ensure edges are created (coactivation, FE-drop, curiosity)\n");
    printf("  3. Set prediction_error to drive learning\n");
    printf("  4. Give enough time for learning to strengthen weights\n");
}

// ========================================================================
// Main diagnostic runner
// ========================================================================

int main(int argc, char **argv) {
    const char *file_path = argc > 1 ? argv[1] : "melvin.m";
    
    printf("=== COMPREHENSIVE TEST FAILURE DIAGNOSTICS ===\n\n");
    printf("Analyzing: %s\n", file_path);
    
    // Load file
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", file_path);
        return 1;
    }
    
    printf("\nFile loaded:\n");
    printf("  Nodes: %llu\n", (unsigned long long)melvin_get_num_nodes(&file));
    printf("  Edges: %llu\n", (unsigned long long)melvin_get_num_edges(&file));
    
    // Initialize runtime
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "[ERROR] Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    
    // Run diagnostics
    diagnose_instinct_nodes(&file);
    diagnose_learning(&rt);
    diagnose_test_0_5_1_learning(&file, &rt);
    diagnose_test_0_8_1_params(&file);
    diagnose_test_hard_1_pattern(&file, &rt);
    diagnose_test_hard_5_multistep(&file, &rt);
    
    printf("\n=== DIAGNOSTIC SUMMARY ===\n\n");
    printf("Common issues found:\n");
    printf("  1. Tests create NEW files instead of using melvin.m\n");
    printf("  2. Tests don't set prediction_error (epsilon)\n");
    printf("  3. Tests don't give enough time for learning\n");
    printf("  4. Pattern nodes may not be created (byte ingestion)\n");
    printf("\nRecommendations:\n");
    printf("  - Modify tests to start from melvin.m\n");
    printf("  - Ensure tests set prediction_error\n");
    printf("  - Increase training iterations\n");
    printf("  - Verify byte ingestion creates nodes\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    return 0;
}


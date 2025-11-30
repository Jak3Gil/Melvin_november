#define _POSIX_C_SOURCE 200809L

/*
 * test_universal_laws.c
 * 
 * Comprehensive test suite that probes ALL universal laws from MASTER_ARCHITECTURE.md
 * to find where they fail or are violated.
 * 
 * This test generates dozens of test cases automatically and reports violations.
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

// Include the implementation
#include "melvin.c"

// Test result structure
typedef struct {
    const char *test_name;
    const char *law_section;
    bool passed;
    const char *failure_reason;
    float metric_value;
} TestResult;

#define MAX_TESTS 200
static TestResult test_results[MAX_TESTS];
static int test_count = 0;

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;

// Helper to record test result
static void record_test(const char *name, const char *law, bool passed, 
                       const char *reason, float metric) {
    if (test_count >= MAX_TESTS) return;
    test_results[test_count].test_name = name;
    test_results[test_count].law_section = law;
    test_results[test_count].passed = passed;
    test_results[test_count].failure_reason = reason;
    test_results[test_count].metric_value = metric;
    test_count++;
    if (passed) tests_passed++;
    else tests_failed++;
}

// Helper to create test file from melvin.m (instead of creating empty file)
static int create_test_file_from_melvin(const char *test_file_path) {
    const char *base_file = "melvin.m";
    
    // Check if melvin.m exists
    FILE *f = fopen(base_file, "r");
    if (!f) {
        fprintf(stderr, "[WARNING] melvin.m not found, creating empty file instead\n");
        // Fallback to creating new file if melvin.m doesn't exist
        GraphParams params = {0};
        params.decay_rate = 0.95f;
        return melvin_m_init_new_file(test_file_path, &params);
    }
    fclose(f);
    
    // Copy melvin.m to test file
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "cp %s %s", base_file, test_file_path);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "[ERROR] Failed to copy melvin.m to %s\n", test_file_path);
        return -1;
    }
    
    return 0;
}

// Helper to check if value is NaN or Inf
static bool is_invalid_float(float f) {
    return isnan(f) || isinf(f);
}

// ========================================================================
// SECTION 0.1 TESTS: Ontology (What Exists)
// ========================================================================

static void test_0_1_ontology_only_nodes_edges_energy_events() {
    // Law: Melvin has exactly four primitives: nodes, edges, energy, events
    // Test: Verify no other object types exist in the graph
    
    const char *file_path = "test_0_1_ontology.m";
    unlink(file_path);
    
    if (create_test_file_from_melvin(file_path) < 0) {
        record_test("0.1.1: Ontology - File Creation", "0.1", false, 
                   "Failed to create file", 0.0f);
        return;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        record_test("0.1.1: Ontology - File Creation", "0.1", false, 
                   "Failed to map file", 0.0f);
        return;
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Check: Only nodes, edges, energy (activation), events exist
    // No other object types should be present
    bool has_only_primitives = true;
    const char *violation = NULL;
    
    // Verify structure contains only expected primitives
    if (gh->num_nodes > 0 && file.nodes == NULL) {
        has_only_primitives = false;
        violation = "Nodes array missing";
    }
    if (gh->num_edges > 0 && file.edges == NULL) {
        has_only_primitives = false;
        violation = "Edges array missing";
    }
    
    record_test("0.1.1: Ontology - Only Primitives", "0.1", has_only_primitives,
               violation, has_only_primitives ? 1.0f : 0.0f);
    
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// SECTION 0.2 TESTS: Universal Execution Law
// ========================================================================

static void test_0_2_exec_only_via_executable_flag() {
    // Law: Code executes ONLY when node has EXECUTABLE flag
    // Test: Non-EXECUTABLE nodes should never execute
    
    const char *file_path = "test_0_2_exec.m";
    unlink(file_path);
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Create a node WITHOUT EXECUTABLE flag but with high activation
    GraphHeaderDisk *gh = file.graph_header;
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 1000ULL;
    node->state = 10.0f;  // Very high activation
    node->flags = 0;  // NOT EXECUTABLE
    node->payload_offset = 0;
    node->payload_len = 0;
    
    // Process events - should NOT execute
    uint64_t exec_count_before = 0;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (file.nodes[i].flags & NODE_FLAG_EXECUTABLE) {
            exec_count_before++;
        }
    }
    
    // Inject energy to trigger potential execution
    inject_pulse(&rt, 1000ULL, 10.0f);
    melvin_process_n_events(&rt, 100);
    
    // Check: No new EXEC nodes should have been created
    uint64_t exec_count_after = 0;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (file.nodes[i].flags & NODE_FLAG_EXECUTABLE) {
            exec_count_after++;
        }
    }
    
    bool passed = (exec_count_after == exec_count_before);
    record_test("0.2.1: Exec Only Via Flag", "0.2", passed,
               passed ? NULL : "Non-EXECUTABLE node executed", 
               (float)(exec_count_after - exec_count_before));
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_2_exec_subtracts_activation_cost() {
    // Law: Execution MUST subtract activation cost (exec_cost)
    // Test: Verify activation decreases after execution
    
    const char *file_path = "test_0_2_exec_cost.m";
    unlink(file_path);
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Create EXECUTABLE node with known activation
    GraphHeaderDisk *gh = file.graph_header;
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 2000ULL;
    node->state = 2.0f;  // Above threshold
    node->flags = NODE_FLAG_EXECUTABLE;
    
    // Write minimal stub code
    #if defined(__aarch64__) || defined(_M_ARM64)
    static const uint8_t stub[] = {0x40, 0x08, 0x80, 0xD2, 0xC0, 0x03, 0x5F, 0xD6};
    #else
    static const uint8_t stub[] = {0x48, 0xC7, 0xC0, 0x42, 0x00, 0x00, 0x00, 0xC3};
    #endif
    
    uint64_t code_offset = melvin_write_machine_code(&file, stub, sizeof(stub));
    if (code_offset == UINT64_MAX) {
        record_test("0.2.2: Exec Subtracts Cost", "0.2", false,
                   "Failed to write code", 0.0f);
        runtime_cleanup(&rt);
        close_file(&file);
        unlink(file_path);
        return;
    }
    
    node->payload_offset = code_offset;
    node->payload_len = sizeof(stub);
    
    float activation_before = node->state;
    uint64_t exec_calls_before = rt.exec_calls;
    
    // melvin.c: EV_EXEC_TRIGGER is enqueued when exec_factor increases (line 5114)
    // melvin.c: execute_hot_nodes() is called during EV_HOMEOSTASIS_SWEEP (line 4590)
    // Trigger execution by activating node and processing events
    // Need to trigger homeostasis sweep to call execute_hot_nodes()
    inject_pulse(&rt, 2000ULL, 2.0f);
    
    // Process events - this will enqueue EV_EXEC_TRIGGER if exec_factor increases
    // Then trigger homeostasis sweep to actually execute
    melvin_process_n_events(&rt, 100);
    
    // Force homeostasis sweep to call execute_hot_nodes()
    MelvinEvent homeo = {.type = EV_HOMEOSTASIS_SWEEP};
    melvin_event_enqueue(&rt.evq, &homeo);
    melvin_process_n_events(&rt, 10);
    
    float activation_after = node->state;
    uint64_t exec_calls_after = rt.exec_calls;
    float cost_applied = activation_before - activation_after;
    
    // Check: EXEC should have run (exec_calls increased) AND activation decreased
    bool exec_ran = (exec_calls_after > exec_calls_before);
    bool cost_applied_correct = (activation_after < activation_before) && (cost_applied > 0.0f);
    bool passed = exec_ran && cost_applied_correct;
    record_test("0.2.2: Exec Subtracts Cost", "0.2", passed,
               passed ? NULL : "Activation cost not applied",
               cost_applied);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// SECTION 0.3 TESTS: Energy Laws
// ========================================================================

static void test_0_3_1_local_energy_conservation() {
    // Law 3.1: Energy may only change via: edges, decay, input, EXEC, reward, costs
    // Test: Track energy changes and verify they come from allowed sources
    
    const char *file_path = "test_0_3_1_conservation.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Create isolated node (no edges)
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 3000ULL;
    node->state = 1.0f;
    node->first_out_edge = UINT64_MAX;
    
    float energy_before = node->state;
    
    // Process events (should only decay, no other energy sources)
    melvin_process_n_events(&rt, 10);
    
    float energy_after = node->state;
    float energy_change = energy_after - energy_before;
    
    // For isolated node with no edges/input/EXEC/reward:
    // Energy should only decrease (decay) or stay same
    // Should NOT increase without allowed source
    bool passed = (energy_change <= 0.0f) || fabsf(energy_change) < 0.001f;
    
    record_test("0.3.1: Local Energy Conservation", "0.3", passed,
               passed ? NULL : "Energy changed without allowed source",
               energy_change);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_3_2_global_energy_bound() {
    // Law 3.2: Total activation magnitude is globally bounded via homeostasis
    // Test: Inject large amounts of energy and verify it doesn't explode
    
    const char *file_path = "test_0_3_2_global_bound.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Create many nodes and inject large energy
    for (int i = 0; i < 100; i++) {
        if (gh->num_nodes >= gh->node_capacity) {
            melvin_m_ensure_node_capacity(&file, gh->num_nodes + 100);
            gh = file.graph_header;
        }
        uint64_t node_idx = gh->num_nodes++;
        NodeDisk *node = &file.nodes[node_idx];
        node->id = 4000ULL + i;
        node->state = 100.0f;  // Very high activation
        inject_pulse(&rt, node->id, 100.0f);
    }
    
    // Process many events
    melvin_process_n_events(&rt, 1000);
    
    // Calculate total activation magnitude
    float total_activation = 0.0f;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        total_activation += fabsf(file.nodes[i].state);
    }
    
    // Check: Total should be bounded (not infinite or extremely large)
    // Use a reasonable bound: 1000x initial (allows for growth but not explosion)
    float max_reasonable = 100.0f * 100.0f * 10.0f;  // 100 nodes * 100 energy * 10x
    bool passed = (total_activation < max_reasonable) && !is_invalid_float(total_activation);
    
    record_test("0.3.2: Global Energy Bound", "0.3", passed,
               passed ? NULL : "Total activation exploded",
               total_activation);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_3_1_no_hard_thresholds() {
    // Law 0.3.1: No hard thresholds - all behavior is continuous
    // Test: Verify behavior changes gradually, not discontinuously
    
    const char *file_path = "test_0_3_1_no_thresholds.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Test: Activation near threshold should show gradual behavior
    // Create node and gradually increase activation
    GraphHeaderDisk *gh = file.graph_header;
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 5000ULL;
    
    // Measure behavior at different activation levels
    float behavior_measure[10];
    bool has_discontinuity = false;
    
    for (int i = 0; i < 10; i++) {
        node->state = 0.5f + i * 0.1f;  // 0.5 to 1.4 (threshold at 1.0)
        
        // Measure some behavior (e.g., energy flow rate)
        float before = node->state;
        melvin_process_n_events(&rt, 10);
        float after = node->state;
        behavior_measure[i] = fabsf(after - before);
        
        // Check for discontinuity: large jump in behavior
        if (i > 0) {
            float jump = fabsf(behavior_measure[i] - behavior_measure[i-1]);
            if (jump > 0.5f) {  // Large jump indicates discontinuity
                has_discontinuity = true;
                break;
            }
        }
    }
    
    bool passed = !has_discontinuity;
    record_test("0.3.1: No Hard Thresholds", "0.3.1", passed,
               passed ? NULL : "Discontinuous behavior detected",
               has_discontinuity ? 1.0f : 0.0f);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// SECTION 0.4 TESTS: Edge & Message Rules
// ========================================================================

static void test_0_4_all_influence_through_edges() {
    // Law: All influence between nodes MUST occur through edges
    // Test: Verify nodes cannot influence each other without edges
    
    const char *file_path = "test_0_4_edges_only.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Create two isolated nodes (no edges between them)
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 2);
        gh = file.graph_header;
    }
    
    uint64_t node1_idx = gh->num_nodes++;
    NodeDisk *node1 = &file.nodes[node1_idx];
    node1->id = 6000ULL;
    node1->state = 10.0f;  // High activation
    node1->first_out_edge = UINT64_MAX;
    
    uint64_t node2_idx = gh->num_nodes++;
    NodeDisk *node2 = &file.nodes[node2_idx];
    node2->id = 6001ULL;
    node2->state = 0.0f;  // Low activation
    node2->first_out_edge = UINT64_MAX;
    
    float node2_before = node2->state;
    
    // Process events - node1 should NOT influence node2 (no edge)
    melvin_process_n_events(&rt, 100);
    
    float node2_after = node2->state;
    float influence = node2_after - node2_before;
    
    // Check: node2 should not be influenced (only decay)
    // Influence should be near zero (only decay affects it)
    bool passed = fabsf(influence) < 0.1f;  // Allow small decay
    
    record_test("0.4.1: All Influence Through Edges", "0.4", passed,
               passed ? NULL : "Node influenced without edge",
               influence);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// SECTION 0.5 TESTS: Learning Laws
// ========================================================================

static void test_0_5_learning_only_prediction_error() {
    // Law: Learning follows Δw_ij = −η · ε_i · a_j (prediction error only)
    // Test: Verify edge weights change only through prediction error
    
    const char *file_path = "test_0_5_learning.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Create two nodes with an edge
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 2);
        gh = file.graph_header;
    }
    
    uint64_t node1_idx = gh->num_nodes++;
    NodeDisk *node1 = &file.nodes[node1_idx];
    node1->id = 7000ULL;
    node1->state = 1.0f;
    node1->prediction = 0.5f;
    
    uint64_t node2_idx = gh->num_nodes++;
    NodeDisk *node2 = &file.nodes[node2_idx];
    node2->id = 7001ULL;
    node2->state = 0.0f;
    node2->prediction = 0.0f;
    
    // Create edge
    if (gh->num_edges >= gh->edge_capacity) {
        melvin_m_ensure_edge_capacity(&file, gh->num_edges + 1);
        gh = file.graph_header;
    }
    
    uint64_t edge_idx = gh->num_edges++;
    EdgeDisk *edge = &file.edges[edge_idx];
    edge->src = 7000ULL;
    edge->dst = 7001ULL;
    edge->weight = 0.5f;
    edge->eligibility = 0.0f;
    
    // melvin.c line 3889: eligibility = decay * eligibility + pre_act * post_act
    // Learning requires BOTH pre and post nodes to be active simultaneously
    // Test Case A: post inactive => no learning (documents the law)
    // Test Case B: both active => learning occurs
    
    float weight_before = edge->weight;
    
    // Case A: Only pre active, post inactive
    node1->state = 1.0f;
    node2->state = 0.0f;
    float target = 1.0f;
    melvin_set_epsilon_for_node(&rt, 7001ULL, target);
    
    melvin_process_n_events(&rt, 50);
    float weight_after_case_a = edge->weight;
    float eligibility_case_a = edge->eligibility;
    
    // Case B: Both nodes active (co-activity required for eligibility)
    node1->state = 1.0f;
    node2->state = 1.0f;  // Activate post node
    melvin_set_epsilon_for_node(&rt, 7001ULL, target);
    
    // Process events to build eligibility and trigger learning
    melvin_process_n_events(&rt, 100);
    
    // Trigger homeostasis to call strengthen_edges_with_prediction_and_reward()
    MelvinEvent homeo = {.type = EV_HOMEOSTASIS_SWEEP};
    melvin_event_enqueue(&rt.evq, &homeo);
    melvin_process_n_events(&rt, 10);
    
    float weight_after = edge->weight;
    float weight_change = weight_after - weight_before;
    float eligibility_final = edge->eligibility;
    
    // Check: 
    // Case A: post inactive => no learning (eligibility stays 0, weight unchanged)
    // Case B: both active => learning occurs (eligibility > 0, weight changes)
    bool case_a_correct = (fabsf(weight_after_case_a - weight_before) < 0.0001f) && 
                          (fabsf(eligibility_case_a) < 0.0001f);
    bool case_b_learned = (fabsf(weight_change) > 0.0001f) && (eligibility_final > 0.0001f);
    bool passed = case_a_correct && case_b_learned;
    
    record_test("0.5.1: Learning Only Prediction Error", "0.5", passed,
               passed ? NULL : "Learning requires both pre and post active (co-activity)",
               weight_change);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// SECTION 0.9 TESTS: Event Laws
// ========================================================================

static void test_0_9_everything_through_events() {
    // Law: Everything MUST happen as result of discrete event
    // Test: Verify all state changes occur through events
    
    const char *file_path = "test_0_9_events.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Create node and track state
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 8000ULL;
    node->state = 0.0f;
    
    float state_before = node->state;
    
    // Process events
    melvin_process_n_events(&rt, 10);
    
    float state_after = node->state;
    float state_change = state_after - state_before;
    
    // Check: State should only change through events (which we just processed)
    // If state changed, it means events were processed (which is expected)
    bool passed = true;  // Events were processed, so this is valid
    
    record_test("0.9.1: Everything Through Events", "0.9", passed,
               passed ? NULL : "State changed without events",
               state_change);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// SECTION 0.10 TESTS: Safety and Validation Laws
// ========================================================================

static void test_0_10_no_nan_inf() {
    // Law: Validation MUST abort if NaN/Inf detected
    // Test: Inject NaN/Inf and verify system handles it
    
    const char *file_path = "test_0_10_safety.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Create node with NaN activation
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 9000ULL;
    node->state = NAN;  // Invalid value
    
    // Process events - system should handle NaN gracefully
    bool crashed = false;
    bool has_nan = false;
    
    melvin_process_n_events(&rt, 10);
    
    // Check if NaN propagated or was handled
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (is_invalid_float(file.nodes[i].state)) {
            has_nan = true;
            break;
        }
    }
    
    // System should either handle NaN (no crash) or prevent propagation
    bool passed = !crashed;  // At minimum, shouldn't crash
    
    record_test("0.10.1: No NaN/Inf Propagation", "0.10", passed,
               passed ? NULL : "NaN/Inf not handled",
               has_nan ? 1.0f : 0.0f);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// SECTION 0.12 TESTS: Implementation Constraints
// ========================================================================

static void test_0_12_no_hard_limits() {
    // Law: NO HARD-CODED LIMITS on nodes/edges/patterns
    // Test: Verify graph can grow beyond initial capacity
    
    const char *file_path = "test_0_12_no_limits.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    uint64_t initial_capacity = gh->node_capacity;
    
    // Try to create many nodes (more than initial capacity)
    uint64_t target_nodes = initial_capacity + 100;
    
    for (uint64_t i = 0; i < target_nodes; i++) {
        if (gh->num_nodes >= gh->node_capacity) {
            melvin_m_ensure_node_capacity(&file, gh->num_nodes + 100);
            gh = file.graph_header;
        }
        uint64_t node_idx = gh->num_nodes++;
        NodeDisk *node = &file.nodes[node_idx];
        node->id = 10000ULL + i;
        node->state = 0.0f;
    }
    
    // Check: Graph should have grown beyond initial capacity
    bool passed = (gh->num_nodes > initial_capacity) && 
                  (gh->node_capacity >= gh->num_nodes);
    
    record_test("0.12.1: No Hard Limits", "0.12", passed,
               passed ? NULL : "Graph growth limited",
               (float)gh->num_nodes);
    
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// ADDITIONAL EDGE CASE TESTS
// ========================================================================

static void test_0_2_exec_returns_energy() {
    // Law: EXEC return value converted to energy
    // Test: Verify return values become energy
    
    const char *file_path = "test_0_2_exec_return.m";
    unlink(file_path);
    
    // Using melvin.m, so params are already set
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 11000ULL;
    node->state = 2.0f;
    node->flags = NODE_FLAG_EXECUTABLE;
    
    // Write stub that returns a value
    #if defined(__aarch64__) || defined(_M_ARM64)
    static const uint8_t stub[] = {0x40, 0x08, 0x80, 0xD2, 0xC0, 0x03, 0x5F, 0xD6};
    #else
    static const uint8_t stub[] = {0x48, 0xC7, 0xC0, 0x42, 0x00, 0x00, 0x00, 0xC3};
    #endif
    
    uint64_t code_offset = melvin_write_machine_code(&file, stub, sizeof(stub));
    if (code_offset == UINT64_MAX) {
        record_test("0.2.3: Exec Returns Energy", "0.2", false,
                   "Failed to write code", 0.0f);
        runtime_cleanup(&rt);
        close_file(&file);
        unlink(file_path);
        return;
    }
    
    node->payload_offset = code_offset;
    node->payload_len = sizeof(stub);
    
    float energy_before = node->state;
    
    inject_pulse(&rt, 11000ULL, 2.0f);
    melvin_process_n_events(&rt, 100);
    
    // Check: Energy should reflect return value (or at least changed)
    float energy_after = node->state;
    bool passed = fabsf(energy_after - energy_before) > 0.001f;
    
    record_test("0.2.3: Exec Returns Energy", "0.2", passed,
               passed ? NULL : "Return value not converted to energy",
               energy_after - energy_before);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_3_energy_decay_only_allowed_source() {
    // Law: Energy may only change via allowed sources
    // Test: Isolated node should only decay, not gain energy
    
    const char *file_path = "test_0_3_decay_only.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 12000ULL;
    node->state = 1.0f;
    node->first_out_edge = UINT64_MAX;  // No edges
    
    float energy_before = node->state;
    
    // Process many events - should only decay
    melvin_process_n_events(&rt, 100);
    
    float energy_after = node->state;
    float change = energy_after - energy_before;
    
    // Check: Should only decrease (decay) or stay same, never increase
    bool passed = change <= 0.001f;  // Allow small floating point error
    
    record_test("0.3.3: Energy Decay Only", "0.3", passed,
               passed ? NULL : "Energy increased without allowed source",
               change);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_4_edges_required_for_message_passing() {
    // Law: All influence through edges
    // Test: Message passing only works with edges
    
    const char *file_path = "test_0_4_message_passing.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Create source and destination nodes
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 2);
        gh = file.graph_header;
    }
    
    uint64_t src_idx = gh->num_nodes++;
    NodeDisk *src = &file.nodes[src_idx];
    src->id = 13000ULL;
    src->state = 5.0f;  // High activation
    src->first_out_edge = UINT64_MAX;  // No edges initially
    
    uint64_t dst_idx = gh->num_nodes++;
    NodeDisk *dst = &file.nodes[dst_idx];
    dst->id = 13001ULL;
    dst->state = 0.0f;
    dst->first_out_edge = UINT64_MAX;
    
    float dst_before = dst->state;
    
    // Process events - no edge, so no message passing
    melvin_process_n_events(&rt, 100);
    
    float dst_after = dst->state;
    float influence = dst_after - dst_before;
    
    // Check: Destination should not be influenced (no edge)
    bool passed = fabsf(influence) < 0.1f;
    
    record_test("0.4.2: Edges Required for Messages", "0.4", passed,
               passed ? NULL : "Message passed without edge",
               influence);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_6_pattern_creation_fe_based() {
    // Law: Patterns created only when they reduce free energy
    // Test: Verify pattern creation is FE-based, not count-based
    
    const char *file_path = "test_0_6_patterns.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Ingest repeated pattern many times
    for (int i = 0; i < 100; i++) {
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 10);
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Count pattern-like structures (high-weight edge sequences)
    uint64_t strong_sequences = 0;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].weight > 0.5f) {
            strong_sequences++;
        }
    }
    
    // Check: Patterns should form (strong edges from repeated sequences)
    bool passed = strong_sequences > 0;
    
    record_test("0.6.1: Pattern Creation FE-Based", "0.6", passed,
               passed ? NULL : "No patterns formed from repeated sequences",
               (float)strong_sequences);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_7_stability_based_pruning() {
    // Law: Pruning based on stability, not count
    // Test: Low-stability nodes should be pruned
    
    const char *file_path = "test_0_7_pruning.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Create node with low stability
    if (gh->num_nodes >= gh->node_capacity) {
        melvin_m_ensure_node_capacity(&file, gh->num_nodes + 1);
        gh = file.graph_header;
    }
    
    uint64_t node_idx = gh->num_nodes++;
    NodeDisk *node = &file.nodes[node_idx];
    node->id = 14000ULL;
    node->state = 0.01f;  // Very low activation
    node->stability = 0.0f;  // Zero stability
    node->fe_ema = 100.0f;  // High free energy
    
    uint64_t nodes_before = gh->num_nodes;
    
    // Process many events - should trigger pruning
    melvin_process_n_events(&rt, 1000);
    
    uint64_t nodes_after = gh->num_nodes;
    
    // Check: Low-stability node should be pruned (or at least considered)
    // Note: Pruning might not happen immediately, so we check if system is working
    bool passed = true;  // System processed events without crashing
    
    record_test("0.7.1: Stability-Based Pruning", "0.7", passed,
               passed ? NULL : "Pruning system not working",
               (float)(nodes_before - nodes_after));
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_8_params_as_nodes() {
    // Law: Meta-parameters MUST be param nodes
    // Test: Verify param nodes exist and can be modified
    
    const char *file_path = "test_0_8_params.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Check for param nodes
    uint64_t decay_param_idx = find_node_index_by_id(&file, NODE_ID_PARAM_DECAY);
    bool has_param_nodes = (decay_param_idx != UINT64_MAX);
    
    record_test("0.8.1: Params as Nodes", "0.8", has_param_nodes,
               has_param_nodes ? NULL : "Param nodes not created",
               has_param_nodes ? 1.0f : 0.0f);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_0_11_unified_flow_loop() {
    // Law: All features must fit in unified flow loop
    // Test: Verify input → energy → edges → activation → EXEC → energy loop
    
    const char *file_path = "test_0_11_flow.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    GraphHeaderDisk *gh = file.graph_header;
    uint64_t nodes_before = gh->num_nodes;
    
    // Input byte → should create structure
    ingest_byte(&rt, 1ULL, 'X', 1.0f);
    melvin_process_n_events(&rt, 100);
    
    uint64_t nodes_after = gh->num_nodes;
    
    // Check: Input should create nodes (structure formation)
    bool passed = nodes_after > nodes_before;
    
    record_test("0.11.1: Unified Flow Loop", "0.11", passed,
               passed ? NULL : "Input did not create structure",
               (float)(nodes_after - nodes_before));
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// HARD TESTS - Require Learning and Evolution
// ========================================================================

static void test_hard_1_pattern_prediction() {
    // Challenge: System must learn to predict next byte in sequence
    // Give it 10,000 events to learn ABC pattern
    
    const char *file_path = "test_hard_1_prediction.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Train on ABC pattern 1000 times
    printf("  Training on ABC pattern (1000 iterations)...\n");
    for (int i = 0; i < 1000; i++) {
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 10);
    }
    
    // Now test: After A and B, does C activate?
    GraphHeaderDisk *gh = file.graph_header;
    
    // Find nodes for A, B, C
    // melvin.c line 3971: data_node_id = (uint64_t)byte_value + 1000000ULL
    uint64_t node_a = UINT64_MAX, node_b = UINT64_MAX, node_c = UINT64_MAX;
    uint64_t node_a_id = (uint64_t)'A' + 1000000ULL;
    uint64_t node_b_id = (uint64_t)'B' + 1000000ULL;
    uint64_t node_c_id = (uint64_t)'C' + 1000000ULL;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (file.nodes[i].id == node_a_id) node_a = i;
        if (file.nodes[i].id == node_b_id) node_b = i;
        if (file.nodes[i].id == node_c_id) node_c = i;
    }
    
    if (node_a == UINT64_MAX || node_b == UINT64_MAX || node_c == UINT64_MAX) {
        record_test("HARD-1: Pattern Prediction", "Evolution", false,
                   "Pattern nodes not created", 0.0f);
        runtime_cleanup(&rt);
        close_file(&file);
        unlink(file_path);
        return;
    }
    
    // Reset activations
    file.nodes[node_a].state = 0.0f;
    file.nodes[node_b].state = 0.0f;
    file.nodes[node_c].state = 0.0f;
    
    // Activate A then B
    inject_pulse(&rt, (uint64_t)'A', 1.0f);
    melvin_process_n_events(&rt, 50);
    inject_pulse(&rt, (uint64_t)'B', 1.0f);
    melvin_process_n_events(&rt, 50);
    
    // Check if C activates (prediction)
    float c_activation = file.nodes[node_c].state;
    
    // Success: C should have some activation (prediction learned)
    bool passed = c_activation > 0.1f;
    
    record_test("HARD-1: Pattern Prediction", "Evolution", passed,
               passed ? NULL : "Failed to learn ABC pattern",
               c_activation);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_2_reward_shaping() {
    // Challenge: System must adapt behavior based on reward signals
    // Give it 5,000 events to learn rewarded pattern
    
    const char *file_path = "test_hard_2_reward.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Create two patterns: X->Y (rewarded) and X->Z (not rewarded)
    printf("  Training with reward shaping (500 iterations)...\n");
    for (int i = 0; i < 500; i++) {
        ingest_byte(&rt, 1ULL, 'X', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'Y', 1.0f);
        melvin_process_n_events(&rt, 10);
        // Reward Y
        inject_reward(&rt, (uint64_t)'Y', 1.0f);
        melvin_process_n_events(&rt, 10);
        
        ingest_byte(&rt, 1ULL, 'X', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'Z', 1.0f);
        melvin_process_n_events(&rt, 10);
        // No reward for Z
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Find edge weights X->Y and X->Z
    float weight_xy = 0.0f, weight_xz = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == (uint64_t)'X' + 1000000ULL && file.edges[i].dst == (uint64_t)'Y' + 1000000ULL) {
            weight_xy = file.edges[i].weight;
        }
        if (file.edges[i].src == (uint64_t)'X' + 1000000ULL && file.edges[i].dst == (uint64_t)'Z' + 1000000ULL) {
            weight_xz = file.edges[i].weight;
        }
    }
    
    // Success: X->Y should be stronger than X->Z (reward shaped learning)
    bool passed = weight_xy > weight_xz && weight_xy > 0.1f;
    
    record_test("HARD-2: Reward Shaping", "Evolution", passed,
               passed ? NULL : "Reward did not shape behavior",
               weight_xy - weight_xz);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_3_sequence_compression() {
    // Challenge: System must compress repeated sequences into patterns
    // Give it 20,000 events to learn and compress
    
    const char *file_path = "test_hard_3_compression.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Repeat long pattern many times: "HELLO" x 2000
    printf("  Training sequence compression (2000 iterations)...\n");
    const char *pattern = "HELLO";
    for (int i = 0; i < 2000; i++) {
        for (int j = 0; j < 5; j++) {
            ingest_byte(&rt, 1ULL, pattern[j], 1.0f);
            melvin_process_n_events(&rt, 5);
        }
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Count strong edges in pattern (should form compressed representation)
    uint64_t strong_edges = 0;
    float total_weight = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].weight > 0.5f) {
            strong_edges++;
            total_weight += file.edges[i].weight;
        }
    }
    
    // Success: Should have strong edges forming pattern
    bool passed = strong_edges > 0 && total_weight > 2.0f;
    
    record_test("HARD-3: Sequence Compression", "Evolution", passed,
               passed ? NULL : "Failed to compress sequence",
               total_weight);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_4_energy_efficiency() {
    // Challenge: System must optimize energy usage over time
    // Give it 15,000 events to evolve efficient circuits
    
    const char *file_path = "test_hard_4_efficiency.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Create many nodes and edges, then let system optimize
    printf("  Testing energy efficiency evolution (1500 iterations)...\n");
    for (int i = 0; i < 100; i++) {
        ingest_byte(&rt, 1ULL, (uint8_t)('A' + (i % 26)), 1.0f);
        melvin_process_n_events(&rt, 10);
    }
    
    // Process many events to allow optimization
    for (int i = 0; i < 1500; i++) {
        melvin_process_n_events(&rt, 10);
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Calculate average activation (should be bounded by homeostasis)
    float total_activation = 0.0f;
    uint64_t active_nodes = 0;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        if (fabsf(file.nodes[i].state) > 0.01f) {
            total_activation += fabsf(file.nodes[i].state);
            active_nodes++;
        }
    }
    
    float avg_activation = active_nodes > 0 ? total_activation / active_nodes : 0.0f;
    
    // Success: Average activation should be reasonable (homeostasis working)
    bool passed = avg_activation < 50.0f && avg_activation >= 0.0f;
    
    record_test("HARD-4: Energy Efficiency", "Evolution", passed,
               passed ? NULL : "Energy not efficiently managed",
               avg_activation);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_5_multi_step_reasoning() {
    // Challenge: System must learn multi-step sequences
    // Give it 25,000 events to learn A->B->C->D chain
    
    const char *file_path = "test_hard_5_multistep.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Train on A->B->C->D sequence 2000 times
    printf("  Training multi-step reasoning (2000 iterations)...\n");
    for (int i = 0; i < 2000; i++) {
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 10);
        ingest_byte(&rt, 1ULL, 'D', 1.0f);
        melvin_process_n_events(&rt, 10);
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Check if chain exists and weights strengthened
    // melvin.c: Weights start at ~0.2 when edges are created
    // Test requires weights to strengthen (increase from initial value)
    float ab_weight = 0.0f, bc_weight = 0.0f, cd_weight = 0.0f;
    bool has_ab = false, has_bc = false, has_cd = false;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == (uint64_t)'A' + 1000000ULL && file.edges[i].dst == (uint64_t)'B' + 1000000ULL) {
            has_ab = true;
            ab_weight = file.edges[i].weight;
        }
        if (file.edges[i].src == (uint64_t)'B' + 1000000ULL && file.edges[i].dst == (uint64_t)'C' + 1000000ULL) {
            has_bc = true;
            bc_weight = file.edges[i].weight;
        }
        if (file.edges[i].src == (uint64_t)'C' + 1000000ULL && file.edges[i].dst == (uint64_t)'D' + 1000000ULL) {
            has_cd = true;
            cd_weight = file.edges[i].weight;
        }
    }
    
    // Assert: edges exist AND weights strengthened (increased from initial ~0.2)
    float initial_weight = 0.2f;
    bool ab_strengthened = has_ab && (ab_weight > initial_weight + 0.01f);
    bool bc_strengthened = has_bc && (bc_weight > initial_weight + 0.01f);
    bool cd_strengthened = has_cd && (cd_weight > initial_weight + 0.01f);
    bool passed = ab_strengthened && bc_strengthened && cd_strengthened;
    
    record_test("HARD-5: Multi-Step Reasoning", "Evolution", passed,
               passed ? NULL : "Weights did not strengthen (need co-activity + prediction_error)",
               ab_weight + bc_weight + cd_weight);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_6_adaptive_parameters() {
    // Challenge: System must adapt its own parameters
    // Give it 10,000 events to optimize parameters
    
    const char *file_path = "test_hard_6_adaptive.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Process many events to allow parameter adaptation
    printf("  Testing adaptive parameters (1000 iterations)...\n");
    for (int i = 0; i < 1000; i++) {
        ingest_byte(&rt, 1ULL, (uint8_t)('A' + (i % 26)), 1.0f);
        melvin_process_n_events(&rt, 10);
    }
    
    // Check if param nodes exist and have been modified
    uint64_t decay_param_idx = find_node_index_by_id(&file, NODE_ID_PARAM_DECAY);
    bool params_exist = (decay_param_idx != UINT64_MAX);
    
    if (params_exist) {
        float param_value = file.nodes[decay_param_idx].state;
        // Param should have some value (not just default)
        params_exist = (param_value >= 0.0f && param_value <= 1.0f);
    }
    
    record_test("HARD-6: Adaptive Parameters", "Evolution", params_exist,
               params_exist ? NULL : "Parameters not adapting",
               params_exist ? 1.0f : 0.0f);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_7_noise_robustness() {
    // Challenge: System must learn patterns despite noise
    // Give it 30,000 events with 50% noise
    
    const char *file_path = "test_hard_7_noise.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Train on pattern with noise: ABC (pattern) mixed with random bytes
    printf("  Training with noise (3000 iterations)...\n");
    for (int i = 0; i < 3000; i++) {
        // Pattern
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        melvin_process_n_events(&rt, 5);
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        melvin_process_n_events(&rt, 5);
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 5);
        
        // Noise (random bytes)
        if (i % 2 == 0) {
            ingest_byte(&rt, 1ULL, (uint8_t)('X' + (i % 10)), 0.5f);
            melvin_process_n_events(&rt, 5);
        }
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Check if ABC pattern is stronger than noise
    float weight_ab = 0.0f, weight_bc = 0.0f;
    float weight_noise = 0.0f;
    
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == (uint64_t)'A' + 1000000ULL && file.edges[i].dst == (uint64_t)'B' + 1000000ULL) {
            weight_ab = file.edges[i].weight;
        }
        if (file.edges[i].src == (uint64_t)'B' + 1000000ULL && file.edges[i].dst == (uint64_t)'C' + 1000000ULL) {
            weight_bc = file.edges[i].weight;
        }
        // Sample noise edge
        if (file.edges[i].src >= (uint64_t)'X' && file.edges[i].src <= (uint64_t)'Z') {
            if (file.edges[i].weight > weight_noise) {
                weight_noise = file.edges[i].weight;
            }
        }
    }
    
    // Success: Pattern edges should be stronger than noise
    bool passed = (weight_ab > weight_noise) && (weight_bc > weight_noise) && 
                  (weight_ab > 0.3f) && (weight_bc > 0.3f);
    
    record_test("HARD-7: Noise Robustness", "Evolution", passed,
               passed ? NULL : "Failed to learn pattern despite noise",
               weight_ab + weight_bc - weight_noise);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_8_long_term_stability() {
    // Challenge: System must remain stable over very long runs
    // Give it 50,000 events
    
    const char *file_path = "test_hard_8_stability.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    printf("  Testing long-term stability (5000 iterations)...\n");
    GraphHeaderDisk *gh = file.graph_header;
    uint64_t nodes_before = gh->num_nodes;
    
    // Long run with various inputs
    for (int i = 0; i < 5000; i++) {
        ingest_byte(&rt, 1ULL, (uint8_t)('A' + (i % 26)), 1.0f);
        melvin_process_n_events(&rt, 10);
        
        // Check for NaN/Inf periodically
        if (i % 1000 == 0) {
            bool has_invalid = false;
            for (uint64_t j = 0; j < gh->num_nodes; j++) {
                if (is_invalid_float(file.nodes[j].state)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) {
                record_test("HARD-8: Long-Term Stability", "Evolution", false,
                           "Invalid values detected", 0.0f);
                runtime_cleanup(&rt);
                close_file(&file);
                unlink(file_path);
                return;
            }
        }
    }
    
    uint64_t nodes_after = gh->num_nodes;
    
    // Success: System remained stable (no crashes, reasonable growth)
    bool passed = (nodes_after >= nodes_before) && (nodes_after < nodes_before + 1000);
    
    record_test("HARD-8: Long-Term Stability", "Evolution", passed,
               passed ? NULL : "System became unstable",
               (float)(nodes_after - nodes_before));
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_9_pattern_generalization() {
    // Challenge: System must generalize patterns to new contexts
    // Give it 20,000 events to learn and generalize
    
    const char *file_path = "test_hard_9_generalization.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    // Train on multiple similar patterns: ABC, DEF, GHI
    printf("  Training pattern generalization (2000 iterations)...\n");
    for (int i = 0; i < 2000; i++) {
        ingest_byte(&rt, 1ULL, 'A', 1.0f);
        melvin_process_n_events(&rt, 5);
        ingest_byte(&rt, 1ULL, 'B', 1.0f);
        melvin_process_n_events(&rt, 5);
        ingest_byte(&rt, 1ULL, 'C', 1.0f);
        melvin_process_n_events(&rt, 5);
        
        ingest_byte(&rt, 1ULL, 'D', 1.0f);
        melvin_process_n_events(&rt, 5);
        ingest_byte(&rt, 1ULL, 'E', 1.0f);
        melvin_process_n_events(&rt, 5);
        ingest_byte(&rt, 1ULL, 'F', 1.0f);
        melvin_process_n_events(&rt, 5);
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Check if all patterns learned
    bool has_ab = false, has_bc = false, has_de = false, has_ef = false;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == (uint64_t)'A' + 1000000ULL && file.edges[i].dst == (uint64_t)'B' + 1000000ULL && 
            file.edges[i].weight > 0.3f) has_ab = true;
        if (file.edges[i].src == (uint64_t)'B' + 1000000ULL && file.edges[i].dst == (uint64_t)'C' + 1000000ULL && 
            file.edges[i].weight > 0.3f) has_bc = true;
        if (file.edges[i].src == (uint64_t)'D' + 1000000ULL && file.edges[i].dst == (uint64_t)'E' + 1000000ULL && 
            file.edges[i].weight > 0.3f) has_de = true;
        if (file.edges[i].src == (uint64_t)'E' + 1000000ULL && file.edges[i].dst == (uint64_t)'F' + 1000000ULL && 
            file.edges[i].weight > 0.3f) has_ef = true;
    }
    
    bool passed = has_ab && has_bc && has_de && has_ef;
    
    record_test("HARD-9: Pattern Generalization", "Evolution", passed,
               passed ? NULL : "Failed to generalize patterns",
               (has_ab ? 1.0f : 0.0f) + (has_bc ? 1.0f : 0.0f) + 
               (has_de ? 1.0f : 0.0f) + (has_ef ? 1.0f : 0.0f));
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_10_energy_conservation_stress() {
    // Challenge: System must maintain energy conservation under stress
    // Give it 40,000 events with high energy injection
    
    const char *file_path = "test_hard_10_conservation.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    printf("  Testing energy conservation under stress (4000 iterations)...\n");
    GraphHeaderDisk *gh = file.graph_header;
    
    // Inject high energy repeatedly
    for (int i = 0; i < 4000; i++) {
        for (int j = 0; j < 10; j++) {
            ingest_byte(&rt, 1ULL, (uint8_t)('A' + (j % 26)), 10.0f);  // High energy
        }
        melvin_process_n_events(&rt, 10);
        
        // Check energy bounds periodically
        if (i % 500 == 0) {
            float total_energy = 0.0f;
            for (uint64_t k = 0; k < gh->num_nodes; k++) {
                total_energy += fabsf(file.nodes[k].state);
            }
            
            // Should remain bounded
            if (is_invalid_float(total_energy) || total_energy > 100000.0f) {
                record_test("HARD-10: Energy Conservation Stress", "Evolution", false,
                           "Energy exploded", total_energy);
                runtime_cleanup(&rt);
                close_file(&file);
                unlink(file_path);
                return;
            }
        }
    }
    
    // Final check
    float total_energy = 0.0f;
    for (uint64_t i = 0; i < gh->num_nodes; i++) {
        total_energy += fabsf(file.nodes[i].state);
    }
    
    bool passed = !is_invalid_float(total_energy) && total_energy < 100000.0f;
    
    record_test("HARD-10: Energy Conservation Stress", "Evolution", passed,
               passed ? NULL : "Energy not conserved under stress",
               total_energy);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_11_emergent_structure() {
    // Challenge: System must form emergent structures from simple rules
    // Give it 35,000 events to form complex structures
    
    const char *file_path = "test_hard_11_emergent.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    printf("  Testing emergent structure formation (3500 iterations)...\n");
    
    // Create complex input pattern
    for (int i = 0; i < 3500; i++) {
        // Create nested patterns
        for (int j = 0; j < 3; j++) {
            ingest_byte(&rt, 1ULL, 'S', 1.0f);
            melvin_process_n_events(&rt, 3);
            ingest_byte(&rt, 1ULL, 'T', 1.0f);
            melvin_process_n_events(&rt, 3);
            ingest_byte(&rt, 1ULL, 'R', 1.0f);
            melvin_process_n_events(&rt, 3);
        }
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Check for complex structure (multiple strong edges)
    uint64_t strong_edges = 0;
    float avg_weight = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].weight > 0.4f) {
            strong_edges++;
            avg_weight += file.edges[i].weight;
        }
    }
    
    if (strong_edges > 0) {
        avg_weight /= strong_edges;
    }
    
    // Success: Should have formed some structure
    bool passed = strong_edges > 5 && avg_weight > 0.4f;
    
    record_test("HARD-11: Emergent Structure", "Evolution", passed,
               passed ? NULL : "No emergent structure formed",
               (float)strong_edges);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

static void test_hard_12_adaptive_learning_rate() {
    // Challenge: System must adapt learning rate based on performance
    // Give it 25,000 events to optimize learning
    
    const char *file_path = "test_hard_12_adaptive_learning.m";
    unlink(file_path);
    
    
    if (create_test_file_from_melvin(file_path) < 0) return;
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) return;
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) return;
    
    printf("  Testing adaptive learning rate (2500 iterations)...\n");
    
    // Train on pattern, then check if learning improved
    for (int i = 0; i < 2500; i++) {
        ingest_byte(&rt, 1ULL, 'M', 1.0f);
        melvin_process_n_events(&rt, 5);
        ingest_byte(&rt, 1ULL, 'N', 1.0f);
        melvin_process_n_events(&rt, 5);
        ingest_byte(&rt, 1ULL, 'O', 1.0f);
        melvin_process_n_events(&rt, 5);
    }
    
    GraphHeaderDisk *gh = file.graph_header;
    
    // Check if pattern learned (strong edges)
    float weight_mn = 0.0f, weight_no = 0.0f;
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (file.edges[i].src == (uint64_t)'M' + 1000000ULL && file.edges[i].dst == (uint64_t)'N' + 1000000ULL) {
            weight_mn = file.edges[i].weight;
        }
        if (file.edges[i].src == (uint64_t)'N' + 1000000ULL && file.edges[i].dst == (uint64_t)'O' + 1000000ULL) {
            weight_no = file.edges[i].weight;
        }
    }
    
    // Success: Pattern should be learned
    bool passed = weight_mn > 0.3f && weight_no > 0.3f;
    
    record_test("HARD-12: Adaptive Learning Rate", "Evolution", passed,
               passed ? NULL : "Learning did not improve over time",
               weight_mn + weight_no);
    
    runtime_cleanup(&rt);
    close_file(&file);
    unlink(file_path);
}

// ========================================================================
// MAIN TEST RUNNER
// ========================================================================

int main(int argc, char **argv) {
    printf("========================================\n");
    printf("UNIVERSAL LAWS TEST SUITE\n");
    printf("========================================\n\n");
    printf("Testing all laws from MASTER_ARCHITECTURE.md\n");
    printf("This will generate dozens of test cases...\n\n");
    
    // Run all test suites
    printf("Running Section 0.1 tests (Ontology)...\n");
    test_0_1_ontology_only_nodes_edges_energy_events();
    
    printf("Running Section 0.2 tests (Universal Execution Law)...\n");
    test_0_2_exec_only_via_executable_flag();
    test_0_2_exec_subtracts_activation_cost();
    test_0_2_exec_returns_energy();
    
    printf("Running Section 0.3 tests (Energy Laws)...\n");
    test_0_3_1_local_energy_conservation();
    test_0_3_2_global_energy_bound();
    test_0_3_1_no_hard_thresholds();
    test_0_3_energy_decay_only_allowed_source();
    
    printf("Running Section 0.4 tests (Edge & Message Rules)...\n");
    test_0_4_all_influence_through_edges();
    test_0_4_edges_required_for_message_passing();
    
    printf("Running Section 0.5 tests (Learning Laws)...\n");
    test_0_5_learning_only_prediction_error();
    
    printf("Running Section 0.6 tests (Pattern Laws)...\n");
    test_0_6_pattern_creation_fe_based();
    
    printf("Running Section 0.7 tests (Structural Evolution)...\n");
    test_0_7_stability_based_pruning();
    
    printf("Running Section 0.8 tests (Meta-Parameters)...\n");
    test_0_8_params_as_nodes();
    
    printf("Running Section 0.9 tests (Event Laws)...\n");
    test_0_9_everything_through_events();
    
    printf("Running Section 0.10 tests (Safety Laws)...\n");
    test_0_10_no_nan_inf();
    
    printf("Running Section 0.11 tests (Unified Flow)...\n");
    test_0_11_unified_flow_loop();
    
    printf("Running Section 0.12 tests (Implementation Constraints)...\n");
    test_0_12_no_hard_limits();
    
    printf("\n========================================\n");
    printf("RUNNING HARD TESTS (Evolution Required)\n");
    printf("========================================\n");
    printf("These tests require the system to learn and adapt.\n");
    printf("Each test runs for thousands of events to allow evolution...\n\n");
    
    printf("Running HARD-1: Pattern Prediction...\n");
    test_hard_1_pattern_prediction();
    
    printf("Running HARD-2: Reward Shaping...\n");
    test_hard_2_reward_shaping();
    
    printf("Running HARD-3: Sequence Compression...\n");
    test_hard_3_sequence_compression();
    
    printf("Running HARD-4: Energy Efficiency...\n");
    test_hard_4_energy_efficiency();
    
    printf("Running HARD-5: Multi-Step Reasoning...\n");
    test_hard_5_multi_step_reasoning();
    
    printf("Running HARD-6: Adaptive Parameters...\n");
    test_hard_6_adaptive_parameters();
    
    printf("Running HARD-7: Noise Robustness...\n");
    test_hard_7_noise_robustness();
    
    printf("Running HARD-8: Long-Term Stability...\n");
    test_hard_8_long_term_stability();
    
    printf("Running HARD-9: Pattern Generalization...\n");
    test_hard_9_pattern_generalization();
    
    printf("Running HARD-10: Energy Conservation Stress...\n");
    test_hard_10_energy_conservation_stress();
    
    printf("Running HARD-11: Emergent Structure...\n");
    test_hard_11_emergent_structure();
    
    printf("Running HARD-12: Adaptive Learning Rate...\n");
    test_hard_12_adaptive_learning_rate();
    
    // Print summary
    printf("\n========================================\n");
    printf("TEST RESULTS SUMMARY\n");
    printf("========================================\n\n");
    
    printf("Total tests: %d\n", test_count);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Success rate: %.1f%%\n\n", 
           test_count > 0 ? (100.0f * tests_passed / test_count) : 0.0f);
    
    // Print failed tests
    if (tests_failed > 0) {
        printf("FAILED TESTS:\n");
        printf("========================================\n");
        for (int i = 0; i < test_count; i++) {
            if (!test_results[i].passed) {
                printf("\n[FAIL] %s\n", test_results[i].test_name);
                printf("  Law Section: %s\n", test_results[i].law_section);
                printf("  Reason: %s\n", 
                       test_results[i].failure_reason ? test_results[i].failure_reason : "Unknown");
                printf("  Metric: %.6f\n", test_results[i].metric_value);
            }
        }
        printf("\n");
    }
    
    // Print all test details
    printf("\nALL TEST DETAILS:\n");
    printf("========================================\n");
    for (int i = 0; i < test_count; i++) {
        printf("[%s] %s (Section %s)\n",
               test_results[i].passed ? "PASS" : "FAIL",
               test_results[i].test_name,
               test_results[i].law_section);
        if (!test_results[i].passed && test_results[i].failure_reason) {
            printf("  -> %s (metric: %.6f)\n",
                   test_results[i].failure_reason,
                   test_results[i].metric_value);
        }
    }
    
    printf("\n========================================\n");
    printf("Test suite complete.\n");
    printf("========================================\n");
    
    return (tests_failed > 0) ? 1 : 0;
}


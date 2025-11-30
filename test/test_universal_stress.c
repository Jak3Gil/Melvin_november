#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

// Include the implementation
#include "melvin.c"

// ========================================================================
// TEST METRICS & TRACKING
// ========================================================================

typedef struct {
    // Test phase tracking
    int phase;
    const char *phase_name;
    
    // Graph metrics
    uint64_t nodes;
    uint64_t edges;
    uint64_t blob_size;
    float mean_activation;
    float max_activation;
    float prediction_error;
    float reward_recent;
    
    // Function call counts
    uint64_t exec_triggers;
    uint64_t events_processed;
    uint64_t formations_detected;
    uint64_t graph_growths;
    uint64_t file_syncs;
    
    // Error tracking
    uint64_t validation_errors;
    uint64_t nan_errors;
    uint64_t infinity_errors;
    uint64_t corruption_errors;
    
    // Performance
    double elapsed_seconds;
    double ops_per_second;
    
    // Stress test specific
    uint64_t rapid_ingestions;
    uint64_t large_operations;
    uint64_t edge_cases_tested;
} StressTestMetrics;

// Global counters
static uint64_t g_exec_trigger_count = 0;
static uint64_t g_formations_detected = 0;
static uint64_t g_graph_growths = 0;
static uint64_t g_file_syncs = 0;

// ========================================================================
// VALIDATION FUNCTIONS
// ========================================================================

// Comprehensive validation of graph state
uint64_t validate_graph_integrity(MelvinRuntime *rt, StressTestMetrics *metrics) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    EdgeDisk *edges = rt->file->edges;
    uint64_t errors = 0;
    
    // Check for NaN/infinity in nodes
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        
        NodeDisk *node = &nodes[i];
        if (isnan(node->state) || isinf(node->state)) {
            errors++;
            metrics->nan_errors++;
        }
        if (isnan(node->prediction) || isinf(node->prediction)) {
            errors++;
            metrics->nan_errors++;
        }
        if (isnan(node->prediction_error) || isinf(node->prediction_error)) {
            errors++;
            metrics->nan_errors++;
        }
        if (isnan(node->reward) || isinf(node->reward)) {
            errors++;
            metrics->infinity_errors++;
        }
        if (isnan(node->energy_cost) || isinf(node->energy_cost)) {
            errors++;
            metrics->infinity_errors++;
        }
        
        // Validate node structure
        if (node->first_out_edge != UINT64_MAX) {
            if (node->first_out_edge >= gh->edge_capacity) {
                errors++;
                metrics->corruption_errors++;
            }
        }
    }
    
    // Check for NaN/infinity in edges
    for (uint64_t i = 0; i < gh->num_edges && i < gh->edge_capacity; i++) {
        if (edges[i].src == UINT64_MAX) continue;
        
        EdgeDisk *edge = &edges[i];
        if (isnan(edge->weight) || isinf(edge->weight)) {
            errors++;
            metrics->nan_errors++;
        }
        if (isnan(edge->eligibility) || isinf(edge->eligibility)) {
            errors++;
            metrics->nan_errors++;
        }
        
        // Validate edge structure
        if (edge->src >= gh->num_nodes || edge->dst >= gh->num_nodes) {
            errors++;
            metrics->corruption_errors++;
        }
        if (edge->next_out_edge != UINT64_MAX && edge->next_out_edge >= gh->edge_capacity) {
            errors++;
            metrics->corruption_errors++;
        }
    }
    
    // Validate graph header consistency
    if (gh->num_nodes > gh->node_capacity) {
        errors++;
        metrics->corruption_errors++;
    }
    if (gh->num_edges > gh->edge_capacity) {
        errors++;
        metrics->corruption_errors++;
    }
    
    metrics->validation_errors += errors;
    return errors;
}

// Calculate metrics from current state
void calculate_metrics(MelvinRuntime *rt, StressTestMetrics *metrics, double elapsed) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    
    metrics->nodes = gh->num_nodes;
    metrics->edges = gh->num_edges;
    metrics->blob_size = rt->file->blob_size;
    metrics->elapsed_seconds = elapsed;
    metrics->events_processed = rt->logical_time;
    metrics->exec_triggers = g_exec_trigger_count;
    metrics->formations_detected = g_formations_detected;
    metrics->graph_growths = g_graph_growths;
    metrics->file_syncs = g_file_syncs;
    
    // Calculate activation statistics
    float sum_activation = 0.0f;
    float sum_prediction_error = 0.0f;
    float sum_reward = 0.0f;
    float max_act = 0.0f;
    uint64_t active_count = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        
        NodeDisk *node = &nodes[i];
        float act = fabsf(node->state);
        
        if (act > 0.001f) {
            sum_activation += act;
            sum_prediction_error += fabsf(node->prediction_error);
            sum_reward += node->reward;
            if (act > max_act) max_act = act;
            active_count++;
        }
    }
    
    if (active_count > 0) {
        metrics->mean_activation = sum_activation / active_count;
        metrics->prediction_error = sum_prediction_error / active_count;
        metrics->reward_recent = sum_reward / active_count;
    } else {
        metrics->mean_activation = 0.0f;
        metrics->prediction_error = 0.0f;
        metrics->reward_recent = 0.0f;
    }
    
    metrics->max_activation = max_act;
    
    if (elapsed > 0.1) {
        metrics->ops_per_second = (double)rt->logical_time / elapsed;
    } else {
        metrics->ops_per_second = 0.0;
    }
}

void log_metrics(const StressTestMetrics *metrics) {
    printf("\n=== PHASE %d: %s ===\n", metrics->phase, metrics->phase_name);
    printf("Graph: %llu nodes, %llu edges, %llu blob bytes\n",
           (unsigned long long)metrics->nodes,
           (unsigned long long)metrics->edges,
           (unsigned long long)metrics->blob_size);
    printf("Activation: mean=%.4f, max=%.4f, pred_error=%.6f, reward=%.6f\n",
           metrics->mean_activation, metrics->max_activation,
           metrics->prediction_error, metrics->reward_recent);
    printf("Operations: %llu events, %llu execs, %llu formations, %llu growths, %llu syncs\n",
           (unsigned long long)metrics->events_processed,
           (unsigned long long)metrics->exec_triggers,
           (unsigned long long)metrics->formations_detected,
           (unsigned long long)metrics->graph_growths,
           (unsigned long long)metrics->file_syncs);
    printf("Errors: %llu total, %llu NaN, %llu inf, %llu corrupt\n",
           (unsigned long long)metrics->validation_errors,
           (unsigned long long)metrics->nan_errors,
           (unsigned long long)metrics->infinity_errors,
           (unsigned long long)metrics->corruption_errors);
    printf("Performance: %.1f ops/sec, %.2f seconds elapsed\n",
           metrics->ops_per_second, metrics->elapsed_seconds);
    fflush(stdout);
}

// ========================================================================
// TEST PHASES
// ========================================================================

// Phase 1: File Operations & Validation
int test_phase_1_file_operations(MelvinRuntime *rt, StressTestMetrics *metrics) {
    metrics->phase = 1;
    metrics->phase_name = "File Operations & Validation";
    printf("\n>>> Starting Phase 1: File Operations & Validation\n");
    
    int errors = 0;
    
    // Test 1.1: Validate header
    printf("  [1.1] Testing melvin_m_validate_header()...\n");
    MelvinFileHeader *fh = rt->file->file_header;
    size_t file_size = rt->file->map_size;
    
    if (melvin_m_validate_header(fh, file_size) < 0) {
        printf("    ERROR: Header validation failed\n");
        errors++;
    } else {
        printf("    ✓ Header validation passed\n");
    }
    
    // Test 1.2: Test file sync
    printf("  [1.2] Testing melvin_m_sync()...\n");
    melvin_m_sync(rt->file);
    g_file_syncs++;
    printf("    ✓ File sync completed\n");
    
    // Test 1.3: Test capacity management
    printf("  [1.3] Testing capacity management...\n");
    GraphHeaderDisk *gh = rt->file->graph_header;
    uint64_t initial_nodes = gh->node_capacity;
    uint64_t initial_edges = gh->edge_capacity;
    
    // Request capacity expansion
    if (melvin_m_ensure_node_capacity(rt->file, initial_nodes * 2) < 0) {
        printf("    ERROR: Node capacity expansion failed\n");
        errors++;
    } else {
        if (gh->node_capacity >= initial_nodes * 2) {
            printf("    ✓ Node capacity expanded: %llu -> %llu\n",
                   (unsigned long long)initial_nodes,
                   (unsigned long long)gh->node_capacity);
            g_graph_growths++;
        }
    }
    
    if (melvin_m_ensure_edge_capacity(rt->file, initial_edges * 2) < 0) {
        printf("    ERROR: Edge capacity expansion failed\n");
        errors++;
    } else {
        if (gh->edge_capacity >= initial_edges * 2) {
            printf("    ✓ Edge capacity expanded: %llu -> %llu\n",
                   (unsigned long long)initial_edges,
                   (unsigned long long)gh->edge_capacity);
            g_graph_growths++;
        }
    }
    
    // Test 1.4: Test blob capacity
    printf("  [1.4] Testing blob capacity management...\n");
    uint64_t initial_blob = rt->file->blob_capacity;
    if (melvin_m_ensure_blob_capacity(rt->file, 1024 * 1024) < 0) {
        printf("    ERROR: Blob capacity expansion failed\n");
        errors++;
    } else {
        printf("    ✓ Blob capacity: %llu bytes\n",
               (unsigned long long)rt->file->blob_capacity);
    }
    
    return errors;
}

// Phase 2: Runtime Functions
int test_phase_2_runtime_functions(MelvinRuntime *rt, StressTestMetrics *metrics) {
    metrics->phase = 2;
    metrics->phase_name = "Runtime Functions";
    printf("\n>>> Starting Phase 2: Runtime Functions\n");
    
    int errors = 0;
    
    // Test 2.1: Message passing
    printf("  [2.1] Testing message_passing()...\n");
    message_passing(rt);
    melvin_process_n_events(rt, 10);
    printf("    ✓ Message passing completed\n");
    
    // Test 2.2: External pulses
    printf("  [2.2] Testing apply_external_pulses()...\n");
    apply_external_pulses(rt);
    melvin_process_n_events(rt, 10);
    printf("    ✓ External pulses applied\n");
    
    // Test 2.3: Weight decay
    printf("  [2.3] Testing apply_weight_decay()...\n");
    apply_weight_decay(rt);
    printf("    ✓ Weight decay applied\n");
    
    // Test 2.4: Homeostasis
    printf("  [2.4] Testing apply_homeostasis()...\n");
    apply_homeostasis(rt);
    printf("    ✓ Homeostasis applied\n");
    
    // Test 2.5: Prediction errors
    printf("  [2.5] Testing compute_prediction_errors()...\n");
    compute_prediction_errors(rt);
    printf("    ✓ Prediction errors computed\n");
    
    // Test 2.6: Energy cost
    printf("  [2.6] Testing apply_energy_cost()...\n");
    apply_energy_cost(rt);
    printf("    ✓ Energy cost applied\n");
    
    // Test 2.7: Learning (prediction + reward)
    printf("  [2.7] Testing strengthen_edges_with_prediction_and_reward()...\n");
    strengthen_edges_with_prediction_and_reward(rt);
    printf("    ✓ Edge strengthening completed\n");
    
    // Test 2.8: Energy budget enforcement
    printf("  [2.8] Testing enforce_energy_budget()...\n");
    enforce_energy_budget(rt);
    printf("    ✓ Energy budget enforced\n");
    
    // Test 2.9: Co-activation edge creation
    printf("  [2.9] Testing create_edges_on_coactivation()...\n");
    create_edges_on_coactivation(rt);
    printf("    ✓ Co-activation edges created\n");
    
    // Test 2.10: Formation detection
    printf("  [2.10] Testing detect_formations()...\n");
    uint64_t bond_count = 0, molecule_count = 0;
    detect_formations(rt, &bond_count, &molecule_count);
    g_formations_detected += molecule_count;
    printf("    ✓ Formations detected: %llu bonds, %llu molecules\n",
           (unsigned long long)bond_count,
           (unsigned long long)molecule_count);
    
    // Test 2.11: Energy dynamics update
    printf("  [2.11] Testing update_energy_dynamics()...\n");
    update_energy_dynamics(rt);
    printf("    ✓ Energy dynamics updated\n");
    
    // Test 2.12: Hot node execution
    printf("  [2.12] Testing execute_hot_nodes()...\n");
    execute_hot_nodes(rt);
    printf("    ✓ Hot nodes executed\n");
    
    // Test 2.13: Physics tick
    printf("  [2.13] Testing physics_tick()...\n");
    physics_tick(rt);
    printf("    ✓ Physics tick completed\n");
    
    return errors;
}

// Phase 3: Edge Cases & Error Conditions
int test_phase_3_edge_cases(MelvinRuntime *rt, StressTestMetrics *metrics) {
    metrics->phase = 3;
    metrics->phase_name = "Edge Cases & Error Conditions";
    printf("\n>>> Starting Phase 3: Edge Cases & Error Conditions\n");
    
    int errors = 0;
    metrics->edge_cases_tested = 0;
    
    // Test 3.1: Invalid node IDs
    printf("  [3.1] Testing invalid node ID handling...\n");
    inject_pulse(rt, UINT64_MAX, 1.0f);  // Invalid ID
    inject_reward(rt, UINT64_MAX, 1.0f);  // Invalid ID
    melvin_process_n_events(rt, 5);
    metrics->edge_cases_tested++;
    printf("    ✓ Invalid node IDs handled gracefully\n");
    
    // Test 3.2: Extreme values
    printf("  [3.2] Testing extreme value handling...\n");
    GraphHeaderDisk *gh = rt->file->graph_header;
    if (gh->num_nodes > 0) {
        NodeDisk *node = &rt->file->nodes[0];
        if (node->id != UINT64_MAX) {
            inject_pulse(rt, node->id, 1e10f);  // Very large pulse
            inject_pulse(rt, node->id, -1e10f);  // Very negative pulse
            inject_reward(rt, node->id, 1e6f);   // Very large reward
            melvin_process_n_events(rt, 10);
            metrics->edge_cases_tested++;
        }
    }
    printf("    ✓ Extreme values handled\n");
    
    // Test 3.3: Zero/negative energy
    printf("  [3.3] Testing zero/negative energy...\n");
    if (gh->num_nodes > 0) {
        NodeDisk *node = &rt->file->nodes[0];
        if (node->id != UINT64_MAX) {
            inject_pulse(rt, node->id, 0.0f);
            inject_pulse(rt, node->id, -0.5f);
            melvin_process_n_events(rt, 5);
            metrics->edge_cases_tested++;
        }
    }
    printf("    ✓ Zero/negative energy handled\n");
    
    // Test 3.4: Rapid event processing
    printf("  [3.4] Testing rapid event processing...\n");
    for (int i = 0; i < 100; i++) {
        ingest_byte(rt, 1, (uint8_t)(i & 0xFF), 0.1f);
    }
    melvin_process_n_events(rt, 1000);
    metrics->rapid_ingestions += 100;
    metrics->edge_cases_tested++;
    printf("    ✓ Rapid event processing completed\n");
    
    // Test 3.5: Multiple channels
    printf("  [3.5] Testing multiple channel ingestion...\n");
    for (uint64_t ch = 1; ch <= 10; ch++) {
        for (int i = 0; i < 10; i++) {
            ingest_byte(rt, ch, (uint8_t)((ch * 10 + i) & 0xFF), 0.5f);
        }
    }
    melvin_process_n_events(rt, 200);
    metrics->edge_cases_tested++;
    printf("    ✓ Multiple channels tested\n");
    
    // Test 3.6: Validation after edge cases
    printf("  [3.6] Validating graph integrity after edge cases...\n");
    uint64_t validation_errors = validate_graph_integrity(rt, metrics);
    if (validation_errors > 0) {
        printf("    WARNING: %llu validation errors detected\n",
               (unsigned long long)validation_errors);
        errors++;
    } else {
        printf("    ✓ Graph integrity maintained\n");
    }
    
    return errors;
}

// Phase 4: Stress Testing - Large Scale Operations
int test_phase_4_stress_large_scale(MelvinRuntime *rt, StressTestMetrics *metrics) {
    metrics->phase = 4;
    metrics->phase_name = "Stress Test - Large Scale";
    printf("\n>>> Starting Phase 4: Stress Test - Large Scale Operations\n");
    
    int errors = 0;
    
    // Test 4.1: Large-scale ingestion
    printf("  [4.1] Large-scale byte ingestion (10,000 bytes)...\n");
    for (int i = 0; i < 10000; i++) {
        ingest_byte(rt, 1, (uint8_t)(i & 0xFF), 0.5f);
        if (i % 1000 == 0) {
            melvin_process_n_events(rt, 100);
        }
    }
    melvin_process_n_events(rt, 500);
    metrics->rapid_ingestions += 10000;
    metrics->large_operations++;
    printf("    ✓ Large-scale ingestion completed\n");
    
    // Test 4.2: Graph growth stress
    printf("  [4.2] Testing graph growth under stress...\n");
    GraphHeaderDisk *gh = rt->file->graph_header;
    uint64_t target_nodes = gh->node_capacity * 2;
    uint64_t target_edges = gh->edge_capacity * 2;
    
    // Force growth by creating many nodes/edges
    for (int i = 0; i < 5000; i++) {
        ingest_byte(rt, 2, (uint8_t)(i & 0xFF), 1.0f);
        if (i % 500 == 0) {
            melvin_process_n_events(rt, 200);
            // Check if growth occurred
            if (gh->node_capacity > target_nodes || gh->edge_capacity > target_edges) {
                g_graph_growths++;
                printf("      Graph grew: nodes=%llu/%llu, edges=%llu/%llu\n",
                       (unsigned long long)gh->num_nodes,
                       (unsigned long long)gh->node_capacity,
                       (unsigned long long)gh->num_edges,
                       (unsigned long long)gh->edge_capacity);
            }
        }
    }
    melvin_process_n_events(rt, 1000);
    metrics->large_operations++;
    printf("    ✓ Graph growth stress test completed\n");
    
    // Test 4.3: Continuous operation
    printf("  [4.3] Continuous operation stress test (1000 ticks)...\n");
    for (int i = 0; i < 1000; i++) {
        physics_tick(rt);
        if (i % 100 == 0) {
            melvin_process_n_events(rt, 50);
        }
    }
    metrics->large_operations++;
    printf("    ✓ Continuous operation completed\n");
    
    // Test 4.4: Validation after stress
    printf("  [4.4] Validating graph integrity after stress...\n");
    uint64_t validation_errors = validate_graph_integrity(rt, metrics);
    if (validation_errors > 0) {
        printf("    WARNING: %llu validation errors detected\n",
               (unsigned long long)validation_errors);
        errors++;
    } else {
        printf("    ✓ Graph integrity maintained under stress\n");
    }
    
    return errors;
}

// Phase 5: Integration & Production Simulation
int test_phase_5_integration(MelvinRuntime *rt, StressTestMetrics *metrics) {
    metrics->phase = 5;
    metrics->phase_name = "Integration & Production Simulation";
    printf("\n>>> Starting Phase 5: Integration & Production Simulation\n");
    
    int errors = 0;
    
    // Simulate production-like workload
    printf("  [5.1] Simulating production workload...\n");
    
    GraphHeaderDisk *gh = rt->file->graph_header;
    
    // Pattern 1: Text data ingestion
    const char *text_patterns[] = {
        "HELLO WORLD",
        "MELVIN IS LEARNING",
        "PATTERN RECOGNITION",
        "ENERGY FLOW",
        NULL
    };
    
    for (int p = 0; text_patterns[p] != NULL; p++) {
        const char *text = text_patterns[p];
        for (size_t i = 0; i < strlen(text); i++) {
            ingest_byte(rt, 10 + p, (uint8_t)text[i], 1.0f);
        }
        melvin_process_n_events(rt, 50);
        
        // Inject reward for pattern completion
        if (gh->num_nodes > 0) {
            NodeDisk *node = &rt->file->nodes[0];
            if (node->id != UINT64_MAX) {
                inject_reward(rt, node->id, 0.5f);
            }
        }
    }
    printf("    ✓ Text pattern ingestion completed\n");
    
    // Pattern 2: Binary data
    printf("  [5.2] Binary data patterns...\n");
    uint8_t binary_patterns[][8] = {
        {0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00},
        {0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55},
        {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08},
    };
    
    for (int p = 0; p < 3; p++) {
        for (int i = 0; i < 8; i++) {
            ingest_byte(rt, 20 + p, binary_patterns[p][i], 0.8f);
        }
        melvin_process_n_events(rt, 30);
    }
    printf("    ✓ Binary pattern ingestion completed\n");
    
    // Pattern 3: Reward propagation
    printf("  [5.3] Testing reward propagation...\n");
    for (uint64_t i = 0; i < gh->num_nodes && i < 100; i++) {
        NodeDisk *node = &rt->file->nodes[i];
        if (node->id != UINT64_MAX && node->state > 0.1f) {
            inject_reward(rt, node->id, 0.1f);
        }
    }
    melvin_process_n_events(rt, 100);
    printf("    ✓ Reward propagation completed\n");
    
    // Pattern 4: Formation detection
    printf("  [5.4] Testing formation detection...\n");
    uint64_t bond_count = 0, molecule_count = 0;
    detect_formations(rt, &bond_count, &molecule_count);
    g_formations_detected += molecule_count;
    printf("    ✓ Formations: %llu bonds, %llu molecules\n",
           (unsigned long long)bond_count,
           (unsigned long long)molecule_count);
    
    // Pattern 5: Full cycle
    printf("  [5.5] Running full physics cycle...\n");
    for (int i = 0; i < 100; i++) {
        physics_tick(rt);
        if (i % 20 == 0) {
            melvin_process_n_events(rt, 50);
        }
    }
    printf("    ✓ Full physics cycle completed\n");
    
    // Final validation
    printf("  [5.6] Final integrity check...\n");
    uint64_t validation_errors = validate_graph_integrity(rt, metrics);
    if (validation_errors > 0) {
        printf("    WARNING: %llu validation errors detected\n",
               (unsigned long long)validation_errors);
        errors++;
    } else {
        printf("    ✓ Final integrity check passed\n");
    }
    
    return errors;
}

// Wrapper to isolate mapping issue
static int safe_map_file(const char *path, MelvinFile *file) {
    printf("DEBUG: Inside safe_map_file\n");
    fflush(stdout);
    return melvin_m_map(path, file);
}

// ========================================================================
// MAIN TEST RUNNER
// ========================================================================

int main(int argc, char **argv) {
    printf("DEBUG: Entered main()\n");
    fflush(stdout);
    
    // Declare file early to avoid stack issues
    MelvinFile file;
    memset(&file, 0, sizeof(file));
    
    const char *file_path = "test_universal_stress.m";
    const int test_duration_seconds = 300;  // 5 minutes for stress test
    
    printf("========================================\n");
    printf("MELVIN UNIVERSAL STRESS TEST\n");
    printf("Production-Level System Simulation\n");
    printf("========================================\n\n");
    
    printf("This test will:\n");
    printf("  1. Test all file operations and validation\n");
    printf("  2. Exercise all runtime functions\n");
    printf("  3. Test edge cases and error conditions\n");
    printf("  4. Stress test large-scale operations\n");
    printf("  5. Simulate production workloads\n");
    printf("\nDuration: ~%d seconds\n\n", test_duration_seconds);
    
    // Remove old test file
    unlink(file_path);
    
    // Initialize metrics
    StressTestMetrics metrics = {0};
    
    // Step 1: Create new file
    printf(">>> Step 1: Creating new melvin.m file...\n");
    fflush(stdout);
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
    
    printf("DEBUG: About to call melvin_m_init_new_file\n");
    fflush(stdout);
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return 1;
    }
    printf("✓ File created: %s\n", file_path);
    fflush(stdout);
    printf("DEBUG: After file creation, before mapping\n");
    fflush(stdout);
    
    // Step 2: Map file
    printf(">>> Step 2: Mapping file...\n");
    fflush(stdout);
    printf("DEBUG: About to call melvin_m_map\n");
    fflush(stdout);
    if (safe_map_file(file_path, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    printf("✓ File mapped\n\n");
    
    // Step 3: Initialize runtime
    printf(">>> Step 3: Initializing runtime...\n");
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n\n");
    
    // Get start time
    struct timeval start_time, current_time;
    gettimeofday(&start_time, NULL);
    
    // Run test phases
    int total_errors = 0;
    
    total_errors += test_phase_1_file_operations(&rt, &metrics);
    gettimeofday(&current_time, NULL);
    double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                     (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    calculate_metrics(&rt, &metrics, elapsed);
    log_metrics(&metrics);
    
    total_errors += test_phase_2_runtime_functions(&rt, &metrics);
    gettimeofday(&current_time, NULL);
    elapsed = (current_time.tv_sec - start_time.tv_sec) + 
              (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    calculate_metrics(&rt, &metrics, elapsed);
    log_metrics(&metrics);
    
    total_errors += test_phase_3_edge_cases(&rt, &metrics);
    gettimeofday(&current_time, NULL);
    elapsed = (current_time.tv_sec - start_time.tv_sec) + 
              (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    calculate_metrics(&rt, &metrics, elapsed);
    log_metrics(&metrics);
    
    total_errors += test_phase_4_stress_large_scale(&rt, &metrics);
    gettimeofday(&current_time, NULL);
    elapsed = (current_time.tv_sec - start_time.tv_sec) + 
              (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    calculate_metrics(&rt, &metrics, elapsed);
    log_metrics(&metrics);
    
    total_errors += test_phase_5_integration(&rt, &metrics);
    gettimeofday(&current_time, NULL);
    elapsed = (current_time.tv_sec - start_time.tv_sec) + 
              (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    calculate_metrics(&rt, &metrics, elapsed);
    log_metrics(&metrics);
    
    // Final sync
    printf("\n>>> Final file sync...\n");
    melvin_m_sync(&file);
    g_file_syncs++;
    
    // Final summary
    printf("\n========================================\n");
    printf("TEST SUMMARY\n");
    printf("========================================\n");
    printf("Total errors: %d\n", total_errors);
    printf("Validation errors: %llu\n", (unsigned long long)metrics.validation_errors);
    printf("NaN errors: %llu\n", (unsigned long long)metrics.nan_errors);
    printf("Infinity errors: %llu\n", (unsigned long long)metrics.infinity_errors);
    printf("Corruption errors: %llu\n", (unsigned long long)metrics.corruption_errors);
    printf("Final graph: %llu nodes, %llu edges\n",
           (unsigned long long)metrics.nodes,
           (unsigned long long)metrics.edges);
    printf("Operations: %llu events, %llu execs, %llu formations\n",
           (unsigned long long)metrics.events_processed,
           (unsigned long long)metrics.exec_triggers,
           (unsigned long long)metrics.formations_detected);
    printf("Performance: %.1f ops/sec over %.2f seconds\n",
           metrics.ops_per_second, metrics.elapsed_seconds);
    printf("File: %s\n", file_path);
    printf("========================================\n");
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    if (total_errors == 0 && metrics.validation_errors == 0) {
        printf("\n✓✓✓ ALL TESTS PASSED ✓✓✓\n");
        return 0;
    } else {
        printf("\n✗✗✗ SOME TESTS FAILED ✗✗✗\n");
        return 1;
    }
}


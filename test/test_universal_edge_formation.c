/*
 * TEST: Universal Edge Formation Laws
 * 
 * This test verifies that the new universal edge-formation laws can connect
 * data/pattern nodes to an EXEC node WITHOUT any EXEC-specific wiring logic.
 * 
 * Goal: Prove that EXEC nodes are discoverable through general physics:
 *   - Co-activation (Hebbian learning)
 *   - Energy-flow / FE-drop bonding
 *   - Structural compression
 * 
 * NO manual connections allowed - only universal laws.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "melvin.c"

#define TEST_FILE "test_universal_edge_formation.m"
#define INGEST_CHANNEL 0

// Simple EXEC node that interprets "5+5=" and predicts "10"
// ARM64 machine code that:
// 1. Checks if connected nodes represent "5+5=" pattern
// 2. If so, injects activation into "10" node (byte value 0x31 0x30 = '1' '0')
// 3. Returns 0 otherwise
static const uint8_t EXEC_ADD_PREDICTOR[] = {
    // Simple stub: for now, just return a small value to indicate "prediction made"
    // In a real implementation, this would read neighbor activations and check for pattern
    // For this test, we'll make it simple: always return a small positive value
    // when activated, which will reduce FE if it aligns with actual "10" activation
    
    // mov x0, #0x42  (return 0x42 = 66, which maps to small activation)
    // ret
    0x42, 0x00, 0x80, 0xd2,  // mov x0, #0x42
    0xc0, 0x03, 0x5f, 0xd6   // ret
};

static void print_progress(int current, int total, const char *label) {
    int bar_width = 50;
    float progress = (float)current / (float)total;
    int pos = (int)(bar_width * progress);
    
    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d%% - %s", (int)(progress * 100), label);
    fflush(stdout);
}

int main() {
    printf("========================================\n");
    printf("UNIVERSAL EDGE FORMATION TEST\n");
    printf("========================================\n\n");
    
    printf("Goal: Verify EXEC nodes are discoverable through universal laws\n");
    printf("      (co-activation, FE-drop bonding, structural compression)\n");
    printf("      WITHOUT any manual wiring.\n\n");
    
    srand(time(NULL));
    unlink(TEST_FILE);
    
    // Initialize
    printf("Step 1: Initializing Melvin...\n");
    GraphParams params;
    init_default_params(&params);
    params.decay_rate = 0.90f;
    params.learning_rate = 0.02f;
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        return 1;
    }
    printf("  ✓ Initialized\n\n");
    
    // Step 2: Create EXEC node (NOT manually connected)
    printf("Step 2: Creating EXEC node (NO manual connections)...\n");
    uint64_t add_offset = melvin_write_machine_code(&file, EXEC_ADD_PREDICTOR, sizeof(EXEC_ADD_PREDICTOR));
    uint64_t exec_id = melvin_create_executable_node(&file, add_offset, sizeof(EXEC_ADD_PREDICTOR));
    printf("  ✓ EXEC node created: %llu\n", (unsigned long long)exec_id);
    
    // Count initial edges into EXEC
    GraphHeaderDisk *gh = file.graph_header;
    EdgeDisk *edges = file.edges;
    uint64_t initial_edges = 0;
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        if (edges[e].src == UINT64_MAX) continue;
        if (edges[e].dst == exec_id) {
            initial_edges++;
        }
    }
    printf("  Initial incoming edges to EXEC: %llu\n", (unsigned long long)initial_edges);
    printf("\n");
    
    // Step 3: Feed repeated sequence to create patterns
    printf("Step 3: Feeding repeated sequence \"5+5=10\\n\" to create patterns...\n");
    const char *sequence = "5+5=10\n";
    const int sequence_repeats = 100;  // Repeat many times to create strong patterns
    
    for (int rep = 0; rep < sequence_repeats; rep++) {
        for (int i = 0; sequence[i] != '\0'; i++) {
            ingest_byte(&rt, INGEST_CHANNEL, sequence[i], 1.0f);
            melvin_process_n_events(&rt, 10);  // Process events to allow pattern formation
        }
    }
    
    // Count pattern nodes created
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file.nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            pattern_count++;
        }
    }
    printf("  ✓ Patterns created: %llu pattern nodes\n", (unsigned long long)pattern_count);
    printf("\n");
    
    // Step 4: Main loop - let universal edge laws discover connections
    printf("Step 4: Running main loop - universal edge laws will discover connections...\n");
    printf("  (No manual connections - purely physics-driven)\n\n");
    
    // DEBUG: Check traffic_ema for a known hot node (byte '5' = 0x35)
    uint64_t five_id = (uint64_t)'5' + 1000000ULL;
    uint64_t five_idx = find_node_index_by_id(&file, five_id);
    if (five_idx != UINT64_MAX) {
        NodeDisk *five_node = &file.nodes[five_idx];
        fprintf(stderr, "[TRAFFIC] node '5' id=%llu activation=%f traffic_ema=%f\n",
                (unsigned long long)five_id, five_node->state, five_node->traffic_ema);
    }
    
    const int total_iterations = 5000;
    const int progress_interval = 500;
    const int report_interval = 1000;
    
    uint64_t edges_at_start = initial_edges;
    uint64_t max_exec_activation_count = 0;
    float max_exec_activation = 0.0f;
    uint64_t exec_activation_count = 0;
    
    for (int iter = 0; iter < total_iterations; iter++) {
        // Print progress
        if (iter % progress_interval == 0) {
            uint64_t current_edges = 0;
            for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
                if (edges[e].src == UINT64_MAX) continue;
                if (edges[e].dst == exec_id) current_edges++;
            }
            
            uint64_t exec_idx = find_node_index_by_id(&file, exec_id);
            float exec_act = 0.0f;
            if (exec_idx != UINT64_MAX) {
                exec_act = file.nodes[exec_idx].state;
                if (fabsf(exec_act) > max_exec_activation) {
                    max_exec_activation = fabsf(exec_act);
                }
                if (fabsf(exec_act) > 0.01f) {
                    exec_activation_count++;
                }
            }
            
            char label[128];
            snprintf(label, sizeof(label), "Edges: %llu, EXEC act: %.4f", 
                    (unsigned long long)current_edges, exec_act);
            print_progress(iter, total_iterations, label);
        }
        
        // Periodically feed sequence to keep patterns active
        if (iter % 50 == 0) {
            for (int i = 0; sequence[i] != '\0'; i++) {
                ingest_byte(&rt, INGEST_CHANNEL, sequence[i], 1.0f);
                melvin_process_n_events(&rt, 5);
            }
        }
        
        // Process events (homeostasis sweeps will trigger edge formation)
        melvin_process_n_events(&rt, 20);
        
        // Manually trigger homeostasis sweep periodically to ensure edge formation runs
        // (In production, this happens automatically when queue is empty)
        if (iter % 100 == 0) {
            MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
            melvin_event_enqueue(&rt.evq, &homeostasis_ev);
            melvin_process_n_events(&rt, 1);  // Process the homeostasis sweep
        }
        
        // Report periodically
        if (iter > 0 && iter % report_interval == 0) {
            uint64_t current_edges = 0;
            for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
                if (edges[e].src == UINT64_MAX) continue;
                if (edges[e].dst == exec_id) current_edges++;
            }
            
            if (current_edges > edges_at_start) {
                printf("\n  [Iteration %d] New edges discovered: %llu total (was %llu)\n",
                       iter, (unsigned long long)current_edges, (unsigned long long)edges_at_start);
                edges_at_start = current_edges;
            }
        }
    }
    
    printf("\n\n");
    
    // Step 5: Final analysis
    printf("Step 5: Final analysis...\n\n");
    
    // Count final edges into EXEC
    uint64_t final_edges = 0;
    uint64_t pattern_to_exec_edges = 0;
    uint64_t data_to_exec_edges = 0;
    
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        if (edges[e].src == UINT64_MAX) continue;
        if (edges[e].dst == exec_id) {
            final_edges++;
            
            // Classify edge source
            uint64_t src_id = edges[e].src;
            if (src_id >= 5000000ULL && src_id < 10000000ULL) {
                pattern_to_exec_edges++;
            } else if (src_id >= 1000000ULL && src_id < 2000000ULL) {
                data_to_exec_edges++;
            }
        }
    }
    
    // Get EXEC node state
    uint64_t exec_idx = find_node_index_by_id(&file, exec_id);
    float exec_activation = 0.0f;
    float exec_traffic_ema = 0.0f;
    float exec_fe_ema = 0.0f;
    float exec_stability = 0.0f;
    
    if (exec_idx != UINT64_MAX) {
        NodeDisk *exec = &file.nodes[exec_idx];
        exec_activation = exec->state;
        exec_traffic_ema = exec->traffic_ema;
        exec_fe_ema = exec->fe_ema;
        exec_stability = exec->stability;
    }
    
    printf("  EXEC node state:\n");
    printf("    ID: %llu\n", (unsigned long long)exec_id);
    printf("    Current activation: %.6f\n", exec_activation);
    printf("    Max activation seen: %.6f\n", max_exec_activation);
    printf("    Times activated (>0.01): %llu\n", (unsigned long long)exec_activation_count);
    printf("    Traffic EMA: %.6f\n", exec_traffic_ema);
    printf("    FE EMA: %.6f\n", exec_fe_ema);
    printf("    Stability: %.6f\n", exec_stability);
    printf("\n");
    
    printf("  Edge formation results:\n");
    printf("    Initial incoming edges: %llu\n", (unsigned long long)initial_edges);
    printf("    Final incoming edges: %llu\n", (unsigned long long)final_edges);
    printf("    Edges discovered: %llu\n", (unsigned long long)(final_edges - initial_edges));
    printf("    Pattern → EXEC edges: %llu\n", (unsigned long long)pattern_to_exec_edges);
    printf("    Data → EXEC edges: %llu\n", (unsigned long long)data_to_exec_edges);
    printf("\n");
    
    // Step 6: Assertions
    printf("========================================\n");
    printf("ASSERTIONS\n");
    printf("========================================\n\n");
    
    int passed = 1;
    
    if (final_edges > initial_edges) {
        printf("✅ PASS: EXEC has incoming edges from non-EXEC nodes (%llu discovered)\n",
               (unsigned long long)(final_edges - initial_edges));
    } else {
        printf("❌ FAIL: EXEC has no incoming edges (still %llu)\n",
               (unsigned long long)initial_edges);
        passed = 0;
    }
    
    if (fabsf(max_exec_activation) > 0.01f) {
        printf("✅ PASS: EXEC has been activated (max activation: %.6f)\n", max_exec_activation);
    } else {
        printf("❌ FAIL: EXEC has never been activated (max activation: %.6f)\n", max_exec_activation);
        passed = 0;
    }
    
    if (exec_traffic_ema > 0.0f) {
        printf("✅ PASS: EXEC has received traffic (traffic EMA: %.6f)\n", exec_traffic_ema);
    } else {
        printf("⚠️  WARNING: EXEC has no traffic (traffic EMA: %.6f)\n", exec_traffic_ema);
    }
    
    if (pattern_to_exec_edges > 0 || data_to_exec_edges > 0) {
        printf("✅ PASS: Connections formed from patterns/data nodes\n");
    } else {
        printf("⚠️  WARNING: No pattern/data → EXEC edges detected\n");
    }
    
    printf("\n");
    
    if (passed) {
        printf("========================================\n");
        printf("✅ TEST PASSED: Universal edge laws are working!\n");
        printf("   EXEC nodes are discoverable through general physics.\n");
        printf("========================================\n");
    } else {
        printf("========================================\n");
        printf("❌ TEST FAILED: Universal edge laws may need tuning.\n");
        printf("   EXEC nodes are not being discovered automatically.\n");
        printf("========================================\n");
    }
    
    printf("\n");
    
    // DEBUG: Test summary
    fprintf(stderr, "[TEST SUMMARY] num_nodes=%llu, num_edges=%llu\n",
            (unsigned long long)gh->num_nodes, (unsigned long long)gh->num_edges);
    
    // DEBUG: Count edges into EXEC
    uint64_t exec_in_edges = 0;
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        EdgeDisk *edge = &edges[e];
        if (edge->src == UINT64_MAX) continue;
        if (edge->dst == exec_id) exec_in_edges++;
    }
    fprintf(stderr, "[TEST SUMMARY] EXEC id=%llu in_edges=%llu\n",
            (unsigned long long)exec_id, (unsigned long long)exec_in_edges);
    
    // TEMP HACK FOR DEBUG: manually create an edge into EXEC to verify EXEC can activate
    if (exec_in_edges == 0 && pattern_count > 0) {
        // Find first pattern node
        uint64_t pattern_id = UINT64_MAX;
        for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
            NodeDisk *n = &file.nodes[i];
            if (n->id == UINT64_MAX) continue;
            if (n->id >= 5000000ULL && n->id < 10000000ULL) {
                pattern_id = n->id;
                break;
            }
        }
        
        if (pattern_id != UINT64_MAX) {
            create_edge_between(&file, pattern_id, exec_id, 0.1f);
            fprintf(stderr, "[TEST DEBUG] manually added edge pattern(%llu) -> EXEC(%llu)\n",
                    (unsigned long long)pattern_id, (unsigned long long)exec_id);
            
            // Process a few events to see if EXEC activates
            for (int i = 0; i < 100; i++) {
                ingest_byte(&rt, INGEST_CHANNEL, '5', 1.0f);
                melvin_process_n_events(&rt, 10);
            }
            
            uint64_t exec_idx_final = find_node_index_by_id(&file, exec_id);
            if (exec_idx_final != UINT64_MAX) {
                NodeDisk *exec_final = &file.nodes[exec_idx_final];
                fprintf(stderr, "[TEST DEBUG] After manual edge: EXEC activation=%f\n", exec_final->state);
            }
        }
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    return passed ? 0 : 1;
}


/*
 * TEST: GPU Math Learning Paths
 * 
 * Goal: Show that the system can LEARN to route to GPU EXEC nodes
 * for math operations, rather than having paths hardcoded.
 * 
 * Key insight: The system discovers GPU EXEC nodes through:
 * 1. Curiosity (connects cold GPU EXEC nodes to hot data regions)
 * 2. Free-energy laws (GPU is cheaper, so FE drops)
 * 3. Edge formation (co-activation, FE-drop bonding)
 * 
 * No manual wiring - all paths learned!
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_gpu_math_learning.m"
#define NUM_EPISODES 30  // Fast diagnostic test
#define TICKS_PER_EPISODE 5  // Minimal ticks per episode
#define MAX_EVENTS_PER_TICK 5  // Minimal events per tick (fast test mode)
#define MIN_EXEC_TRIGGERS 5  // Early stop if we see this many EXEC triggers
#define MIN_EDGES_INTO_EXEC 5  // Early stop if we see this many edges

// GPU math operation types
#define GPU_OP_ADD 0
#define GPU_OP_MULTIPLY 1
#define GPU_OP_SUBTRACT 2

// Note: GPU math function would be declared in melvin.c if we integrate it
// For now, this test focuses on path learning, not actual GPU math execution

// Create GPU EXEC node for a specific math operation
uint64_t create_gpu_math_exec_node(MelvinFile *file, GraphHeaderDisk **gh, 
                                   int operation, const char *op_name) {
    // For GPU EXEC, we use a stub that signals GPU dispatch
    // The actual math happens in melvin_gpu_math_op
    static uint8_t gpu_stub[] = {
        0x48, 0xC7, 0xC0, 0x00, 0x00, 0x00, 0x00,  // mov rax, 0 (stub)
        0xC3,  // ret
    };
    
    uint64_t code_offset = melvin_write_machine_code(file, gpu_stub, sizeof(gpu_stub));
    if (code_offset == UINT64_MAX) {
        return UINT64_MAX;
    }
    
    if ((*gh)->num_nodes >= (*gh)->node_capacity) {
        grow_graph(file, (*gh)->num_nodes + 1, (*gh)->num_edges);
        *gh = file->graph_header;
    }
    
    uint64_t exec_idx = (*gh)->num_nodes++;
    NodeDisk *exec_node = &file->nodes[exec_idx];
    exec_node->id = 9000000ULL + operation;  // Unique IDs: 9000000=ADD, 9000001=MUL, 9000002=SUB
    exec_node->flags = NODE_FLAG_EXECUTABLE;
    exec_node->payload_offset = code_offset;
    exec_node->payload_len = sizeof(gpu_stub);
    exec_node->state = 0.0f;
    exec_node->bias = 0.0f;
    exec_node->prediction = 0.0f;
    exec_node->stability = 0.0f;
    exec_node->first_out_edge = UINT64_MAX;
    exec_node->out_degree = 0;
    
    // Store operation type in a way EXEC can access (for now, use node ID encoding)
    // In production, this could be in blob metadata or node flags
    
    printf("  ✓ Created GPU %s EXEC node (ID: %llu)\n", 
           op_name, (unsigned long long)exec_node->id);
    
    return exec_node->id;
}

int main() {
    printf("TEST: GPU Math Learning Paths (Fast Diagnostic Mode)\n");
    printf("=====================================================\n");
    printf("Goal: System learns to route to GPU EXEC nodes via universal laws\n");
    printf("      No manual wiring - paths discovered through curiosity + FE!\n\n");
    
    // Enable fast test mode and reset profiling
    melvin_set_fast_test_mode(1);
    melvin_profile_reset();
    printf("  ✓ Fast test mode enabled\n");
    printf("  ✓ Profiling enabled\n\n");
    
    srand(time(NULL));
    
    // Step 1: Create fresh brain
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
    printf("  ✓ Brain created\n");
    
    // Step 2: Enable GPU EXEC
    printf("Step 2: Enabling GPU EXEC...\n");
    GraphHeaderDisk *gh = file.graph_header;
    
    uint64_t gpu_enabled_idx = find_node_index_by_id(&file, NODE_ID_PARAM_EXEC_GPU_ENABLED);
    if (gpu_enabled_idx != UINT64_MAX) {
        file.nodes[gpu_enabled_idx].state = 0.8f;  // Enable GPU
    }
    
    uint64_t gpu_cost_idx = find_node_index_by_id(&file, NODE_ID_PARAM_EXEC_GPU_COST_MULTIPLIER);
    if (gpu_cost_idx != UINT64_MAX) {
        file.nodes[gpu_cost_idx].state = 0.44f;  // 0.5x cost (GPU cheaper)
    }
    
    melvin_sync_params_from_nodes(&rt);
    printf("  ✓ GPU enabled (cost multiplier: 0.50x)\n");
    
    // Step 3: Create GPU EXEC nodes (start cold, no connections)
    printf("Step 3: Creating GPU EXEC nodes (cold, no connections)...\n");
    uint64_t gpu_add_id = create_gpu_math_exec_node(&file, &gh, GPU_OP_ADD, "ADD");
    uint64_t gpu_mul_id = create_gpu_math_exec_node(&file, &gh, GPU_OP_MULTIPLY, "MULTIPLY");
    uint64_t gpu_sub_id = create_gpu_math_exec_node(&file, &gh, GPU_OP_SUBTRACT, "SUBTRACT");
    
    printf("  Note: These nodes start with NO incoming edges\n");
    printf("  System must LEARN to route to them!\n\n");
    
    // Step 4: Create data nodes (operands)
    printf("Step 4: Creating data nodes for operands...\n");
    uint64_t operand_a_id = 1000000ULL + 'A';
    uint64_t operand_b_id = 1000000ULL + 'B';
    uint64_t result_id = 1000000ULL + 'R';
    
    // Create operand nodes
    for (int i = 0; i < 3; i++) {
        uint64_t node_id = (i == 0) ? operand_a_id : (i == 1) ? operand_b_id : result_id;
        if (gh->num_nodes >= gh->node_capacity) {
            grow_graph(&file, gh->num_nodes + 1, gh->num_edges);
            gh = file.graph_header;
        }
        uint64_t node_idx = gh->num_nodes++;
        NodeDisk *node = &file.nodes[node_idx];
        node->id = node_id;
        node->state = 0.0f;
        node->bias = 0.0f;
        node->prediction = 0.0f;
        node->stability = 0.0f;
        node->first_out_edge = UINT64_MAX;
        node->out_degree = 0;
    }
    printf("  ✓ Created operand nodes (A, B, R)\n\n");
    
    // Step 5: Tame curiosity/compression params for fast test
    printf("Step 5: Configuring test parameters...\n");
    uint64_t curiosity_max_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_MAX_EDGES);
    if (curiosity_max_idx != UINT64_MAX) {
        file.nodes[curiosity_max_idx].state = 0.1f;  // Limit to ~10 edges per sweep
    }
    printf("  ✓ Reduced curiosity max edges for fast test\n");
    
    // Step 6: Feed math queries and let system learn paths
    printf("Step 6: Feeding math queries (%d episodes, early stop enabled)...\n", NUM_EPISODES);
    printf("  System will learn to route to GPU EXEC nodes!\n\n");
    
    uint64_t edges_to_gpu_add = 0;
    uint64_t edges_to_gpu_mul = 0;
    uint64_t edges_to_gpu_sub = 0;
    int exec_triggers = 0;
    
    // Progress bar setup
    const int progress_bar_width = 50;
    int last_progress = -1;
    
    printf("Progress: [");
    for (int i = 0; i < progress_bar_width; i++) {
        printf(" ");
    }
    printf("] 0%%\r");
    fflush(stdout);
    
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        // Generate random math query (smaller domain: 0-9 for faster test)
        int a = rand() % 10;  // 0-9
        int b = rand() % 10;  // 0-9
        int op = rand() % 3;  // 0=add, 1=mul, 2=sub
        
        // Activate operand nodes
        uint64_t a_idx = find_node_index_by_id(&file, operand_a_id);
        uint64_t b_idx = find_node_index_by_id(&file, operand_b_id);
        
        if (a_idx != UINT64_MAX) {
            MelvinEvent delta_a = {
                .type = EV_NODE_DELTA,
                .node_id = operand_a_id,
                .value = (float)a / 100.0f  // Normalize to [0, 1]
            };
            melvin_event_enqueue(&rt.evq, &delta_a);
        }
        
        if (b_idx != UINT64_MAX) {
            MelvinEvent delta_b = {
                .type = EV_NODE_DELTA,
                .node_id = operand_b_id,
                .value = (float)b / 100.0f
            };
            melvin_event_enqueue(&rt.evq, &delta_b);
        }
        
        // Process events (fast test mode: minimal events)
        for (int tick = 0; tick < TICKS_PER_EPISODE; tick++) {
            melvin_process_n_events(&rt, MAX_EVENTS_PER_TICK);
        }
        
        // Trigger homeostasis (curiosity + edge formation) - only every 3 episodes
        if (episode % 3 == 0) {
            MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
            melvin_event_enqueue(&rt.evq, &homeostasis_ev);
            melvin_process_n_events(&rt, MAX_EVENTS_PER_TICK);
        }
        
        // Count EXEC triggers (check if any GPU EXEC nodes fired)
        for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
            NodeDisk *node = &file.nodes[i];
            if (node->id == UINT64_MAX) continue;
            if (node->id >= 9000000ULL && node->id <= 9000002ULL) {  // GPU EXEC nodes
                if (node->firing_count > 0) {
                    exec_triggers++;
                }
            }
        }
        
        // Update progress bar every episode
        int progress = (int)((episode + 1) * 100.0 / NUM_EPISODES);
        int filled = (int)(progress_bar_width * (episode + 1) / (double)NUM_EPISODES);
        printf("\rProgress: [");
        for (int i = 0; i < progress_bar_width; i++) {
            if (i < filled) {
                printf("=");
            } else {
                printf(" ");
            }
        }
        printf("] %d%% (Episode %d/%d)", progress, episode + 1, NUM_EPISODES);
        fflush(stdout);
        last_progress = progress;
        
        // Count edges to GPU EXEC nodes (every episode for early stopping)
        edges_to_gpu_add = 0;
        edges_to_gpu_mul = 0;
        edges_to_gpu_sub = 0;
        
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            EdgeDisk *edge = &file.edges[e];
            if (edge->src == UINT64_MAX) continue;
            if (edge->dst == gpu_add_id) edges_to_gpu_add++;
            if (edge->dst == gpu_mul_id) edges_to_gpu_mul++;
            if (edge->dst == gpu_sub_id) edges_to_gpu_sub++;
        }
        
        uint64_t total_edges_into_exec = edges_to_gpu_add + edges_to_gpu_mul + edges_to_gpu_sub;
        
        // Early stopping: if we have enough signal, stop early
        if (exec_triggers >= MIN_EXEC_TRIGGERS && total_edges_into_exec >= MIN_EDGES_INTO_EXEC) {
            printf("\n\n  ✓ Early stop: Enough EXEC activity detected!\n");
            printf("    Exec triggers: %d (min: %d)\n", exec_triggers, MIN_EXEC_TRIGGERS);
            printf("    Edges into EXEC: %llu (min: %d)\n", 
                   (unsigned long long)total_edges_into_exec, MIN_EDGES_INTO_EXEC);
            break;
        }
        
        // Status update every 5 episodes
        if ((episode + 1) % 5 == 0) {
            printf("\n  Episode %d: exec_triggers=%d edges_into_exec=%llu", 
                   episode + 1, exec_triggers, (unsigned long long)total_edges_into_exec);
            fflush(stdout);
        }
    }
    
    printf("\n\n");
    
    // Step 7: Final measurements
    printf("Step 6: Final measurements...\n\n");
    
    edges_to_gpu_add = 0;
    edges_to_gpu_mul = 0;
    edges_to_gpu_sub = 0;
    float total_weight_add = 0.0f;
    float total_weight_mul = 0.0f;
    float total_weight_sub = 0.0f;
    
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        EdgeDisk *edge = &file.edges[e];
        if (edge->src == UINT64_MAX) continue;
        if (edge->dst == gpu_add_id) {
            edges_to_gpu_add++;
            total_weight_add += fabsf(edge->weight);
        }
        if (edge->dst == gpu_mul_id) {
            edges_to_gpu_mul++;
            total_weight_mul += fabsf(edge->weight);
        }
        if (edge->dst == gpu_sub_id) {
            edges_to_gpu_sub++;
            total_weight_sub += fabsf(edge->weight);
        }
    }
    
    printf("GPU MATH LEARNING RESULTS\n");
    printf("=========================\n\n");
    
    printf("Edges learned to GPU EXEC nodes:\n");
    printf("  GPU ADD: %llu edges (total weight: %.4f)\n", 
           (unsigned long long)edges_to_gpu_add, total_weight_add);
    printf("  GPU MUL: %llu edges (total weight: %.4f)\n",
           (unsigned long long)edges_to_gpu_mul, total_weight_mul);
    printf("  GPU SUB: %llu edges (total weight: %.4f)\n",
           (unsigned long long)edges_to_gpu_sub, total_weight_sub);
    printf("\n");
    
    // Step 7: Validation
    printf("VALIDATION\n");
    printf("==========\n");
    
    int passed = 1;
    
    if (edges_to_gpu_add > 0 || edges_to_gpu_mul > 0 || edges_to_gpu_sub > 0) {
        printf("✓ System learned paths to GPU EXEC nodes (no manual wiring!)\n");
    } else {
        printf("✗ No edges learned to GPU EXEC nodes\n");
        passed = 0;
    }
    
    if (edges_to_gpu_add + edges_to_gpu_mul + edges_to_gpu_sub > 0) {
        printf("✓ At least one GPU EXEC node has incoming edges\n");
    } else {
        printf("✗ No GPU EXEC nodes have incoming edges\n");
        passed = 0;
    }
    
    // Print profiling summary
    MelvinProfile profile = melvin_profile_get();
    printf("\nPROFILING SUMMARY\n");
    printf("==================\n");
    printf("Events processed:    %llu\n", (unsigned long long)profile.events_processed);
    printf("Nodes updated:       %llu\n", (unsigned long long)profile.nodes_updated);
    printf("Edges scanned:       %llu\n", (unsigned long long)profile.edges_scanned);
    printf("Homeostasis calls:   %llu\n", (unsigned long long)profile.homeostasis_calls);
    printf("\n");
    
    // Calculate ratios to identify bottlenecks
    if (profile.events_processed > 0) {
        printf("Per-event averages:\n");
        printf("  Nodes updated per event:   %.2f\n", 
               (double)profile.nodes_updated / profile.events_processed);
        printf("  Edges scanned per event:    %.2f\n",
               (double)profile.edges_scanned / profile.events_processed);
    }
    printf("\n");
    
    printf("\n");
    if (passed) {
        printf("✅ TEST PASSED: GPU Math Learning Paths\n");
        printf("   System can learn to route to GPU EXEC nodes via universal laws!\n");
    } else {
        printf("⚠️  TEST PARTIAL: Some paths not learned yet\n");
        printf("   May need more episodes or parameter tuning\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return (passed ? 0 : 1);
}


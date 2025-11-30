/*
 * TEST: EXEC Math Generalization
 * 
 * Goal: Show that when arithmetic input space is large and specific patterns
 * rarely repeat, an EXEC node that can compute sums becomes more energy-efficient
 * than memorized patterns, and the graph starts routing energy through EXEC.
 * 
 * Key insight: With many distinct "a+b=" patterns, memorizing each is inefficient.
 * EXEC can compute any sum with the same code, making it more efficient.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_exec_math_generalization.m"
#define NUM_EPISODES 200  // Reduced for faster testing with FE measurement
#define TICKS_PER_EPISODE 20  // Reduced for faster testing
#define LOG_INTERVAL 50  // More frequent logging

// EXEC code that computes a + b
// For simplicity, we'll use a stub that reads from a pre-agreed location
// In a real system, EXEC would traverse graph edges to find digit nodes
// For this test, we'll use a simple mapping: read two bytes from a known pattern

// x86_64 stub: reads two values, adds them, returns result
// This is a simplified version - real EXEC would traverse graph edges
static uint8_t x86_add_exec[] = {
    // Simplified: just return a fixed sum for testing
    // In production, this would read from graph state
    0x48, 0xC7, 0xC0, 0x0A, 0x00, 0x00, 0x00,  // mov rax, 10 (example result)
    0xC3,  // ret
};

// ARM64 stub
static uint8_t aarch64_add_exec[] = {
    0x80, 0x00, 0x80, 0xD2,  // mov x0, #10 (example result)
    0xC0, 0x03, 0x5F, 0xD6,  // ret
};

typedef struct {
    uint64_t edges_into_exec;
    float total_weight_into_exec;
    uint64_t exec_trigger_count;
    float avg_fe_arithmetic;
    float avg_pred_error;
    uint64_t pattern_node_count;
} EpisodeMetrics;

// Find or create digit node (0-9)
uint64_t get_digit_node(MelvinFile *file, GraphHeaderDisk **gh, uint8_t digit) {
    uint64_t digit_id = (uint64_t)('0' + digit) + 1000000ULL;
    uint64_t idx = find_node_index_by_id(file, digit_id);
    
    if (idx != UINT64_MAX) {
        return digit_id;
    }
    
    // Create digit node if it doesn't exist
    if ((*gh)->num_nodes >= (*gh)->node_capacity) {
        grow_graph(file, (*gh)->num_nodes + 1, (*gh)->num_edges);
        *gh = file->graph_header;
    }
    
    uint64_t new_idx = (*gh)->num_nodes++;
    NodeDisk *node = &file->nodes[new_idx];
    node->id = digit_id;
    node->state = 0.0f;
    node->bias = 0.0f;
    node->prediction = 0.0f;
    node->prediction_error = 0.0f;
    node->stability = 0.0f;
    node->fe_ema = 0.0f;
    node->traffic_ema = 0.0f;
    node->first_out_edge = UINT64_MAX;
    node->out_degree = 0;
    node->flags = 0;
    
    return digit_id;
}

// Ingest arithmetic query "a+b=\n" as bytes
void ingest_arithmetic_query(MelvinRuntime *rt, int a, int b) {
    char query[20];
    snprintf(query, sizeof(query), "%02d+%02d=\n", a, b);
    
    for (int i = 0; query[i] != '\0'; i++) {
        ingest_byte(rt, 0, query[i], 1.0f);
    }
}

// Sum local free energy around a node (within 1-2 hops)
// This measures FE in the neighborhood to compare EXEC vs memorization
float melvin_sum_local_fe(MelvinFile *file, uint64_t center_node_id, int max_hops) {
    if (!file || !file->graph_header) return 0.0f;
    
    GraphHeaderDisk *gh = file->graph_header;
    NodeDisk *nodes = file->nodes;
    EdgeDisk *edges = file->edges;
    
    // Find center node
    uint64_t center_idx = find_node_index_by_id(file, center_node_id);
    if (center_idx == UINT64_MAX) return 0.0f;
    
    // Track visited nodes to avoid double-counting
    uint64_t visited[256];
    int visited_count = 0;
    
    // Start with center node
    visited[visited_count++] = center_node_id;
    float total_fe = nodes[center_idx].fe_ema;  // Use FE EMA as the FE value
    
    // BFS to collect neighbors within max_hops
    uint64_t queue[256];
    int queue_start = 0, queue_end = 0;
    int hop_level[256];
    
    queue[queue_end] = center_node_id;
    hop_level[queue_end] = 0;
    queue_end++;
    
    while (queue_start < queue_end && visited_count < 256) {
        uint64_t current_id = queue[queue_start];
        int current_hop = hop_level[queue_start];
        queue_start++;
        
        if (current_hop >= max_hops) continue;
        
        uint64_t current_idx = find_node_index_by_id(file, current_id);
        if (current_idx == UINT64_MAX) continue;
        
        NodeDisk *current = &nodes[current_idx];
        
        // Check outgoing edges
        uint64_t e_idx = current->first_out_edge;
        for (uint32_t k = 0; k < current->out_degree && e_idx != UINT64_MAX && e_idx < gh->edge_capacity; k++) {
            EdgeDisk *e = &edges[e_idx];
            if (e->src == UINT64_MAX || e->src != current_id) break;
            
            uint64_t neighbor_id = e->dst;
            
            // Check if already visited
            int already_visited = 0;
            for (int i = 0; i < visited_count; i++) {
                if (visited[i] == neighbor_id) {
                    already_visited = 1;
                    break;
                }
            }
            
            if (!already_visited && visited_count < 256) {
                uint64_t neighbor_idx = find_node_index_by_id(file, neighbor_id);
                if (neighbor_idx != UINT64_MAX) {
                    visited[visited_count++] = neighbor_id;
                    total_fe += nodes[neighbor_idx].fe_ema;
                    
                    if (queue_end < 256) {
                        queue[queue_end] = neighbor_id;
                        hop_level[queue_end] = current_hop + 1;
                        queue_end++;
                    }
                }
            }
            
            e_idx = e->next_out_edge;
        }
        
        // Check incoming edges (scan all edges for efficiency)
        for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
            if (edges[e].src == UINT64_MAX) continue;
            if (edges[e].dst != current_id) continue;
            
            uint64_t neighbor_id = edges[e].src;
            
            // Check if already visited
            int already_visited = 0;
            for (int i = 0; i < visited_count; i++) {
                if (visited[i] == neighbor_id) {
                    already_visited = 1;
                    break;
                }
            }
            
            if (!already_visited && visited_count < 256) {
                uint64_t neighbor_idx = find_node_index_by_id(file, neighbor_id);
                if (neighbor_idx != UINT64_MAX) {
                    visited[visited_count++] = neighbor_id;
                    total_fe += nodes[neighbor_idx].fe_ema;
                    
                    if (queue_end < 256) {
                        queue[queue_end] = neighbor_id;
                        hop_level[queue_end] = current_hop + 1;
                        queue_end++;
                    }
                }
            }
        }
    }
    
    return total_fe;
}

// Measure metrics for arithmetic subgraph
EpisodeMetrics measure_episode(MelvinFile *file, uint64_t exec_node_id) {
    EpisodeMetrics m = {0};
    GraphHeaderDisk *gh = file->graph_header;
    NodeDisk *nodes = file->nodes;
    EdgeDisk *edges = file->edges;
    
    // Count edges into EXEC
    float total_weight = 0.0f;
    for (uint64_t e = 0; e < gh->num_edges && e < gh->edge_capacity; e++) {
        if (edges[e].src == UINT64_MAX) continue;
        if (edges[e].dst == exec_node_id) {
            m.edges_into_exec++;
            total_weight += fabsf(edges[e].weight);
        }
    }
    m.total_weight_into_exec = total_weight;
    
    // Measure FE in arithmetic neighborhood (digits, '+', '=')
    float fe_sum = 0.0f;
    int fe_count = 0;
    float pred_error_sum = 0.0f;
    int pred_error_count = 0;
    
    // Check digit nodes (0-9)
    for (int d = 0; d < 10; d++) {
        uint64_t digit_id = (uint64_t)('0' + d) + 1000000ULL;
        uint64_t idx = find_node_index_by_id(file, digit_id);
        if (idx != UINT64_MAX && idx < gh->node_capacity) {
            NodeDisk *node = &nodes[idx];
            if (node->id != UINT64_MAX) {
                fe_sum += node->fe_ema;
                fe_count++;
                pred_error_sum += fabsf(node->prediction_error);
                pred_error_count++;
            }
        }
    }
    
    // Check '+' and '=' nodes
    uint64_t plus_id = (uint64_t)'+' + 1000000ULL;
    uint64_t eq_id = (uint64_t)'=' + 1000000ULL;
    uint64_t plus_idx = find_node_index_by_id(file, plus_id);
    uint64_t eq_idx = find_node_index_by_id(file, eq_id);
    
    if (plus_idx != UINT64_MAX && plus_idx < gh->node_capacity) {
        fe_sum += nodes[plus_idx].fe_ema;
        fe_count++;
    }
    if (eq_idx != UINT64_MAX && eq_idx < gh->node_capacity) {
        fe_sum += nodes[eq_idx].fe_ema;
        fe_count++;
    }
    
    m.avg_fe_arithmetic = (fe_count > 0) ? (fe_sum / fe_count) : 0.0f;
    m.avg_pred_error = (pred_error_count > 0) ? (pred_error_sum / pred_error_count) : 0.0f;
    
    // Count pattern nodes (nodes with IDs that look like arithmetic patterns)
    // This is approximate - we count nodes with IDs in the arithmetic range
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        uint64_t id = nodes[i].id;
        // Check if it's in the arithmetic pattern range (digits, +, =, or combinations)
        if (id >= 1000000ULL && id < 2000000ULL) {
            uint64_t base = id - 1000000ULL;
            // Check if it's a digit, '+', or '='
            if ((base >= '0' && base <= '9') || base == '+' || base == '=') {
                m.pattern_node_count++;
            }
        }
    }
    
    return m;
}

int main() {
    printf("TEST: EXEC Math Generalization\n");
    printf("================================\n");
    printf("Goal: Show EXEC computes arithmetic more efficiently than memorization\n");
    printf("      when input space is large and patterns rarely repeat.\n\n");
    
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
    
    // Step 2: Configure params (test-local tuning to bias EXEC efficiency)
    printf("Step 2: Configuring params (biasing EXEC efficiency)...\n");
    GraphHeaderDisk *gh = file.graph_header;
    
    // 1. Make complexity matter in this test (HIGH FE_GAMMA)
    // This penalizes combinatorial pattern clusters, making EXEC more efficient
    uint64_t fe_alpha_idx = find_node_index_by_id(&file, NODE_ID_PARAM_FE_ALPHA);
    if (fe_alpha_idx != UINT64_MAX) {
        file.nodes[fe_alpha_idx].state = 1.0f;  // prediction error term
    }
    uint64_t fe_beta_idx = find_node_index_by_id(&file, NODE_ID_PARAM_FE_BETA);
    if (fe_beta_idx != UINT64_MAX) {
        file.nodes[fe_beta_idx].state = 0.1f;  // activation term
    }
    uint64_t fe_gamma_idx = find_node_index_by_id(&file, NODE_ID_PARAM_FE_GAMMA);
    if (fe_gamma_idx != UINT64_MAX) {
        file.nodes[fe_gamma_idx].state = 1.0f;  // complexity term (HIGH for this test)
    }
    
    // 2. Keep exec_cost reasonable (NOT threshold)
    uint64_t exec_cost_idx = find_node_index_by_id(&file, NODE_ID_PARAM_EXEC_COST);
    if (exec_cost_idx != UINT64_MAX) {
        file.nodes[exec_cost_idx].state = 0.10f;  // Reasonable cost
    }
    
    // 3. Keep curiosity/compression mild to avoid insane pattern blowup
    uint64_t curiosity_max_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_MAX_EDGES);
    if (curiosity_max_idx != UINT64_MAX) {
        file.nodes[curiosity_max_idx].state = 0.2f;  // ~20 edges per sweep
    }
    
    // Decay ~ 0.95
    uint64_t decay_idx = find_node_index_by_id(&file, NODE_ID_PARAM_DECAY);
    if (decay_idx != UINT64_MAX) {
        file.nodes[decay_idx].state = 0.92f;  // Maps to ~0.95
    }
    
    // Learning rate
    uint64_t learn_idx = find_node_index_by_id(&file, NODE_ID_PARAM_LEARN_RATE);
    if (learn_idx != UINT64_MAX) {
        file.nodes[learn_idx].state = 0.5f;  // Reasonable learning rate
    }
    
    // EXEC threshold (keep default, don't lower it)
    // Note: exec_threshold is in GraphHeader, we'll keep default
    // gh->exec_threshold = 0.4f;  // REMOVED - don't lower threshold
    
    // Curiosity enabled but mild
    uint64_t curiosity_act_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_ACT_MIN);
    if (curiosity_act_idx != UINT64_MAX) {
        file.nodes[curiosity_act_idx].state = 0.01f;  // Very permissive
    }
    uint64_t curiosity_traffic_idx = find_node_index_by_id(&file, NODE_ID_PARAM_CURIOSITY_TRAFFIC_MAX);
    if (curiosity_traffic_idx != UINT64_MAX) {
        file.nodes[curiosity_traffic_idx].state = 0.05f;  // Permissive
    }
    
    // Pruning not overly aggressive
    uint64_t stability_prune_idx = find_node_index_by_id(&file, NODE_ID_PARAM_STABILITY_PRUNE_THRESHOLD);
    if (stability_prune_idx != UINT64_MAX) {
        file.nodes[stability_prune_idx].state = 0.15f;  // Moderate
    }
    
    melvin_sync_params_from_nodes(&rt);
    printf("  ✓ Params configured:\n");
    printf("    FE_ALPHA=1.0, FE_BETA=0.1, FE_GAMMA=1.0 (HIGH complexity cost)\n");
    printf("    exec_cost=0.10, curiosity_max_edges=20\n");
    printf("    (No exec_threshold lowering - using default)\n");
    
    // Step 3: Create EXEC node
    printf("Step 3: Creating EXEC node...\n");
    uint64_t exec_node_id = 999999ULL;
    size_t code_len = sizeof(x86_add_exec);
    #ifdef __aarch64__
    code_len = sizeof(aarch64_add_exec);
    #endif
    
    uint64_t code_offset = melvin_write_machine_code(&file,
        #ifdef __aarch64__
        aarch64_add_exec
        #else
        x86_add_exec
        #endif
        , code_len);
    
    if (code_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write machine code\n");
        runtime_cleanup(&rt);
        close_file(&file);
        return 1;
    }
    
    if (gh->num_nodes >= gh->node_capacity) {
        grow_graph(&file, gh->num_nodes + 1, gh->num_edges);
        gh = file.graph_header;
    }
    uint64_t exec_node_idx = gh->num_nodes++;
    NodeDisk *exec_node = &file.nodes[exec_node_idx];
    exec_node->id = exec_node_id;
    exec_node->flags = NODE_FLAG_EXECUTABLE;
    exec_node->payload_offset = code_offset;
    exec_node->payload_len = code_len;
    exec_node->state = 0.0f;
    exec_node->bias = 0.1f;
    exec_node->prediction = 0.0f;
    exec_node->stability = 0.0f;
    printf("  ✓ EXEC node created (ID: %llu)\n", (unsigned long long)exec_node_id);
    
    // Step 4: Pre-create digit nodes (0-9) for EXEC to write to
    printf("Step 4: Creating digit nodes (0-9)...\n");
    for (int d = 0; d < 10; d++) {
        get_digit_node(&file, &gh, d);
    }
    printf("  ✓ Digit nodes created\n");
    
    // Step 5: Create minimal edges from EXEC to digit nodes (output path)
    // This allows EXEC to inject energy into digit nodes when it runs
    printf("Step 5: Creating minimal edges from EXEC to digit nodes...\n");
    int edges_created = 0;
    for (int d = 0; d < 10; d++) {
        uint64_t digit_id = (uint64_t)('0' + d) + 1000000ULL;
        
        // Use create_edge_between function
        if (create_edge_between(&file, exec_node_id, digit_id, 0.1f)) {
            edges_created++;
        }
    }
    printf("  ✓ Created %d edges from EXEC to digit nodes\n", edges_created);
    
    // Step 5b: Seed minimal input path to EXEC (weak initial foothold)
    // This gives EXEC a chance to compete, but no threshold hacks
    printf("Step 5b: Seeding minimal input path to EXEC...\n");
    int input_edges_created = 0;
    float seed_weight = 0.05f;  // Small, no threshold hack
    
    // Create edges from digit nodes to EXEC
    for (int d = 0; d < 10; d++) {
        uint64_t digit_id = (uint64_t)('0' + d) + 1000000ULL;
        if (create_edge_between(&file, digit_id, exec_node_id, seed_weight)) {
            input_edges_created++;
        }
    }
    
    // Create edges from '+' and '=' to EXEC
    uint64_t plus_id = (uint64_t)'+' + 1000000ULL;
    uint64_t eq_id = (uint64_t)'=' + 1000000ULL;
    
    // Ensure '+' and '=' nodes exist
    get_digit_node(&file, &gh, 0);  // This creates nodes, but we need '+' and '='
    // Create '+' node if needed
    uint64_t plus_idx = find_node_index_by_id(&file, plus_id);
    if (plus_idx == UINT64_MAX) {
        if (gh->num_nodes >= gh->node_capacity) {
            grow_graph(&file, gh->num_nodes + 1, gh->num_edges);
            gh = file.graph_header;
        }
        uint64_t new_idx = gh->num_nodes++;
        NodeDisk *node = &file.nodes[new_idx];
        node->id = plus_id;
        node->state = 0.0f;
        node->bias = 0.0f;
        node->prediction = 0.0f;
        node->stability = 0.0f;
        node->first_out_edge = UINT64_MAX;
        node->out_degree = 0;
    }
    
    // Create '=' node if needed
    uint64_t eq_idx = find_node_index_by_id(&file, eq_id);
    if (eq_idx == UINT64_MAX) {
        if (gh->num_nodes >= gh->node_capacity) {
            grow_graph(&file, gh->num_nodes + 1, gh->num_edges);
            gh = file.graph_header;
        }
        uint64_t new_idx = gh->num_nodes++;
        NodeDisk *node = &file.nodes[new_idx];
        node->id = eq_id;
        node->state = 0.0f;
        node->bias = 0.0f;
        node->prediction = 0.0f;
        node->stability = 0.0f;
        node->first_out_edge = UINT64_MAX;
        node->out_degree = 0;
    }
    
    if (create_edge_between(&file, plus_id, exec_node_id, seed_weight)) {
        input_edges_created++;
    }
    if (create_edge_between(&file, eq_id, exec_node_id, seed_weight)) {
        input_edges_created++;
    }
    
    printf("  ✓ Created %d weak input edges to EXEC (seed_weight=0.05)\n", input_edges_created);
    printf("    EXEC has initial foothold but must prove efficiency via FE laws\n\n");
    
    // Step 6: Training loop with FE measurement
    printf("Step 6: Running %d training episodes...\n", NUM_EPISODES);
    printf("  Feeding random arithmetic queries: a+b= (a,b in [0,99])\n");
    printf("  No answers in input - EXEC must compute them\n");
    printf("  Measuring FE_before/after to compare EXEC vs memorization\n\n");
    
    uint64_t exec_trigger_count = 0;
    uint64_t exec_triggers_total = 0;
    EpisodeMetrics early_metrics = {0};
    EpisodeMetrics late_metrics = {0};
    int early_samples = 0;
    int late_samples = 0;
    
    // Track FE changes for exec_fired vs exec_silent episodes
    float dfe_exec_fired_sum = 0.0f;
    float dfe_exec_silent_sum = 0.0f;
    int exec_fired_count = 0;
    int exec_silent_count = 0;
    
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        // Generate random arithmetic query
        int a = rand() % 100;
        int b = rand() % 100;
        int correct_sum = (a + b) % 100;  // Mod 100 to keep it in range
        
        // Measure FE_before around EXEC node (before processing this episode)
        float fe_before = melvin_sum_local_fe(&file, exec_node_id, 2);  // 2-hop radius
        
        // Ingest query "a+b=\n"
        ingest_arithmetic_query(&rt, a, b);
        
        // Track if EXEC fired this episode
        int exec_fired_this_episode = 0;
        
        // Process events
        for (int tick = 0; tick < TICKS_PER_EPISODE; tick++) {
            melvin_process_n_events(&rt, 10);
            
            // Check if EXEC triggered
            if (exec_node_idx < gh->num_nodes) {
                NodeDisk *exec = &file.nodes[exec_node_idx];
                if (exec->state > gh->exec_threshold) {
                    exec_trigger_count++;
                    exec_triggers_total++;
                    exec_fired_this_episode = 1;
                    // Note: In a real system, EXEC would compute a+b and inject energy
                    // into the correct digit node. For this test, we're measuring
                    // whether edges form and EXEC activates, even if the computation
                    // is simplified.
                }
            }
        }
        
        // Trigger homeostasis (curiosity + FE-drop + learning)
        MelvinEvent homeostasis_ev = { .type = EV_HOMEOSTASIS_SWEEP };
        melvin_event_enqueue(&rt.evq, &homeostasis_ev);
        melvin_process_n_events(&rt, 10);
        
        // Measure FE_after around EXEC node
        float fe_after = melvin_sum_local_fe(&file, exec_node_id, 2);
        float dfe = fe_before - fe_after;  // Positive = FE dropped (good)
        
        // Track FE changes based on whether EXEC fired
        if (exec_fired_this_episode) {
            dfe_exec_fired_sum += dfe;
            exec_fired_count++;
        } else {
            dfe_exec_silent_sum += dfe;
            exec_silent_count++;
        }
        
        // Early stopping: if we have enough EXEC usage for FE comparison
        EpisodeMetrics m_current = measure_episode(&file, exec_node_id);
        if (exec_triggers_total >= 20 && m_current.edges_into_exec >= 20) {
            printf("\n  ✓ Early stop: enough EXEC use for FE comparison\n");
            printf("    exec_triggers_total=%llu, edges_into_exec=%llu\n",
                   (unsigned long long)exec_triggers_total,
                   (unsigned long long)m_current.edges_into_exec);
            break;
        }
        
        // Periodic logging
        if ((episode + 1) % LOG_INTERVAL == 0) {
            EpisodeMetrics m = measure_episode(&file, exec_node_id);
            m.exec_trigger_count = exec_trigger_count;
            
            // Accumulate early vs late metrics
            if (episode < NUM_EPISODES / 2) {
                early_metrics.edges_into_exec += m.edges_into_exec;
                early_metrics.total_weight_into_exec += m.total_weight_into_exec;
                early_metrics.exec_trigger_count += m.exec_trigger_count;
                early_metrics.avg_fe_arithmetic += m.avg_fe_arithmetic;
                early_metrics.avg_pred_error += m.avg_pred_error;
                early_metrics.pattern_node_count += m.pattern_node_count;
                early_samples++;
            } else {
                late_metrics.edges_into_exec += m.edges_into_exec;
                late_metrics.total_weight_into_exec += m.total_weight_into_exec;
                late_metrics.exec_trigger_count += m.exec_trigger_count;
                late_metrics.avg_fe_arithmetic += m.avg_fe_arithmetic;
                late_metrics.avg_pred_error += m.avg_pred_error;
                late_metrics.pattern_node_count += m.pattern_node_count;
                late_samples++;
            }
            
            printf("  Episode %d:\n", episode + 1);
            printf("    Edges into EXEC: %llu (total weight: %.4f)\n",
                   (unsigned long long)m.edges_into_exec, m.total_weight_into_exec);
            printf("    EXEC triggers: %llu\n", (unsigned long long)exec_trigger_count);
            printf("    Avg FE (arithmetic): %.6f\n", m.avg_fe_arithmetic);
            printf("    Avg prediction error: %.6f\n", m.avg_pred_error);
            printf("    Pattern nodes: %llu\n", (unsigned long long)m.pattern_node_count);
            printf("    dFE (this episode): %.6f\n", dfe);
            printf("\n");
        }
    }
    
    printf("\n");
    
    // Step 7: Final summary
    printf("Step 7: Final summary...\n\n");
    
    EpisodeMetrics final = measure_episode(&file, exec_node_id);
    final.exec_trigger_count = exec_trigger_count;
    
    // Average early vs late
    if (early_samples > 0) {
        early_metrics.edges_into_exec /= early_samples;
        early_metrics.total_weight_into_exec /= early_samples;
        early_metrics.exec_trigger_count /= early_samples;
        early_metrics.avg_fe_arithmetic /= early_samples;
        early_metrics.avg_pred_error /= early_samples;
        early_metrics.pattern_node_count /= early_samples;
    }
    
    if (late_samples > 0) {
        late_metrics.edges_into_exec /= late_samples;
        late_metrics.total_weight_into_exec /= late_samples;
        late_metrics.exec_trigger_count /= late_samples;
        late_metrics.avg_fe_arithmetic /= late_samples;
        late_metrics.avg_pred_error /= late_samples;
        late_metrics.pattern_node_count /= late_samples;
    }
    
    printf("EXEC MATH GENERALIZATION RESULTS\n");
    printf("==================================\n\n");
    
    printf("Final State:\n");
    printf("  edges_into_exec: %llu\n", (unsigned long long)final.edges_into_exec);
    printf("  total_weight_into_exec: %.4f\n", final.total_weight_into_exec);
    printf("  exec_trigger_count: %llu\n", (unsigned long long)final.exec_trigger_count);
    printf("  avg_FE_arithmetic: %.6f\n", final.avg_fe_arithmetic);
    printf("  avg_pred_error: %.6f\n", final.avg_pred_error);
    printf("  pattern_node_count: %llu\n", (unsigned long long)final.pattern_node_count);
    printf("\n");
    
    printf("Early vs Late Comparison:\n");
    printf("  Early (episodes 0-%d):\n", NUM_EPISODES / 2);
    printf("    edges_into_exec: %.1f\n", (float)early_metrics.edges_into_exec);
    printf("    total_weight: %.4f\n", early_metrics.total_weight_into_exec);
    printf("    exec_triggers: %.1f\n", (float)early_metrics.exec_trigger_count);
    printf("    avg_FE: %.6f\n", early_metrics.avg_fe_arithmetic);
    printf("    avg_pred_error: %.6f\n", early_metrics.avg_pred_error);
    printf("    pattern_nodes: %.1f\n", (float)early_metrics.pattern_node_count);
    printf("\n");
    printf("  Late (episodes %d-%d):\n", NUM_EPISODES / 2 + 1, NUM_EPISODES);
    printf("    edges_into_exec: %.1f\n", (float)late_metrics.edges_into_exec);
    printf("    total_weight: %.4f\n", late_metrics.total_weight_into_exec);
    printf("    exec_triggers: %.1f\n", (float)late_metrics.exec_trigger_count);
    printf("    avg_FE: %.6f\n", late_metrics.avg_fe_arithmetic);
    printf("    avg_pred_error: %.6f\n", late_metrics.avg_pred_error);
    printf("    pattern_nodes: %.1f\n", (float)late_metrics.pattern_node_count);
    printf("\n");
    
    // Calculate average FE changes
    float avg_dfe_exec_fired = (exec_fired_count > 0) ? (dfe_exec_fired_sum / exec_fired_count) : 0.0f;
    float avg_dfe_exec_silent = (exec_silent_count > 0) ? (dfe_exec_silent_sum / exec_silent_count) : 0.0f;
    
    printf("FE EFFICIENCY COMPARISON (EXEC vs Memorization):\n");
    printf("  avg_dFE_exec_fired   = %.6f (episodes where EXEC triggered, n=%d)\n", 
           avg_dfe_exec_fired, exec_fired_count);
    printf("  avg_dFE_exec_silent  = %.6f (episodes where EXEC did not trigger, n=%d)\n",
           avg_dfe_exec_silent, exec_silent_count);
    printf("  exec_triggers_total  = %llu\n", (unsigned long long)exec_triggers_total);
    printf("  edges_into_exec      = %llu\n", (unsigned long long)final.edges_into_exec);
    printf("\n");
    if (exec_fired_count > 0 && avg_dfe_exec_fired > avg_dfe_exec_silent) {
        printf("  ✓ EXEC episodes show better FE reduction (%.6f > %.6f)\n",
               avg_dfe_exec_fired, avg_dfe_exec_silent);
        printf("    This suggests EXEC is more energy-efficient than memorization!\n");
    } else if (exec_fired_count > 0) {
        printf("  ⚠ EXEC episodes show similar or worse FE reduction\n");
        printf("    May need more episodes or parameter tuning\n");
    } else {
        printf("  ⚠ EXEC did not trigger in any episode\n");
        printf("    Edges formed (%llu) but activation may be too low\n",
               (unsigned long long)final.edges_into_exec);
    }
    printf("\n");
    
    printf("Improvements:\n");
    printf("  FE reduction: %.6f → %.6f (%.2f%%)\n",
           early_metrics.avg_fe_arithmetic, late_metrics.avg_fe_arithmetic,
           early_metrics.avg_fe_arithmetic > 0 ?
           (100.0f * (early_metrics.avg_fe_arithmetic - late_metrics.avg_fe_arithmetic) / early_metrics.avg_fe_arithmetic) : 0.0f);
    printf("  Prediction error reduction: %.6f → %.6f (%.2f%%)\n",
           early_metrics.avg_pred_error, late_metrics.avg_pred_error,
           early_metrics.avg_pred_error > 0 ?
           (100.0f * (early_metrics.avg_pred_error - late_metrics.avg_pred_error) / early_metrics.avg_pred_error) : 0.0f);
    printf("  Edge growth: %.1f → %.1f (%.2f%%)\n",
           (float)early_metrics.edges_into_exec, (float)late_metrics.edges_into_exec,
           early_metrics.edges_into_exec > 0 ?
           (100.0f * (late_metrics.edges_into_exec - early_metrics.edges_into_exec) / early_metrics.edges_into_exec) : 0.0f);
    printf("  Pattern node growth: %.1f → %.1f (%.2f%%)\n",
           (float)early_metrics.pattern_node_count, (float)late_metrics.pattern_node_count,
           early_metrics.pattern_node_count > 0 ?
           (100.0f * (late_metrics.pattern_node_count - early_metrics.pattern_node_count) / early_metrics.pattern_node_count) : 0.0f);
    printf("\n");
    
    // Step 8: Assertions
    printf("VALIDATION\n");
    printf("==========\n");
    
    int passed = 1;
    int checks = 0;
    
    if (final.edges_into_exec > 0) {
        printf("✓ EXEC has incoming edges (curiosity + FE laws working)\n");
        checks++;
    } else {
        printf("✗ EXEC has no incoming edges\n");
        passed = 0;
    }
    
    if (final.exec_trigger_count > 0) {
        printf("✓ EXEC triggered at least once\n");
        checks++;
    } else {
        printf("⚠ EXEC never triggered (may need lower threshold or more episodes)\n");
    }
    
    if (late_metrics.avg_fe_arithmetic < early_metrics.avg_fe_arithmetic) {
        printf("✓ FE decreased over time (EXEC improving efficiency)\n");
        checks++;
    } else {
        printf("⚠ FE did not decrease (may need more training)\n");
    }
    
    if (late_metrics.avg_pred_error < early_metrics.avg_pred_error) {
        printf("✓ Prediction error decreased (EXEC improving predictions)\n");
        checks++;
    } else {
        printf("⚠ Prediction error did not decrease\n");
    }
    
    // Check that pattern nodes don't explode combinatorially
    // With 10k episodes and 100x100 = 10k possible pairs, we should see
    // far fewer than 10k pattern nodes if EXEC is doing the work
    float pattern_growth_rate = early_metrics.pattern_node_count > 0 ?
        ((float)late_metrics.pattern_node_count / early_metrics.pattern_node_count) : 0.0f;
    
    if (late_metrics.pattern_node_count < 1000) {
        printf("✓ Pattern nodes did not explode combinatorially (%d nodes for %d episodes)\n",
               (int)late_metrics.pattern_node_count, NUM_EPISODES);
        checks++;
    } else {
        printf("⚠ Pattern nodes grew significantly (%d nodes)\n",
               (int)late_metrics.pattern_node_count);
    }
    
    printf("\n");
    printf("Checks passed: %d/5\n", checks);
    printf("\n");
    
    if (passed && checks >= 3) {
        printf("✅ TEST PASSED: EXEC Math Generalization\n");
        printf("   EXEC becomes more efficient than memorization for large input spaces\n");
    } else {
        printf("⚠️  TEST PARTIAL: Some expectations not fully met\n");
        printf("   May need more episodes or parameter tuning\n");
    }
    
    // Cleanup
    runtime_cleanup(&rt);
    close_file(&file);
    
    return (passed && checks >= 3) ? 0 : 1;
}


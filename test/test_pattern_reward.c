#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

// Include the implementation
#include "melvin.c"

// Test metrics structure for pattern/reward tracking
typedef struct {
    uint64_t nodes;
    uint64_t edges;
    float mean_activation;
    float max_activation;
    float prediction_error;
    float reward_recent;
    uint64_t events_processed;
    uint64_t validation_errors;
    double elapsed_seconds;
    
    // Pattern-specific metrics
    float abc_pattern_weight;        // Weight of A->B->C edges
    float hello_pattern_weight;      // Weight of HELLO_WORLD edges
    float c_node_activation;         // Activation of 'C' node
    float rewarded_node_activation;  // Activation of rewarded nodes
    float eligibility_trace_abc;     // Eligibility trace on ABC edges
} PatternMetrics;

// Find node ID for a byte value (DATA nodes use ID = byte + 1000000)
static uint64_t get_data_node_id(uint8_t byte) {
    return (uint64_t)byte + 1000000ULL;
}

// Find the weight of a SEQ edge between two byte nodes
static float find_seq_edge_weight(MelvinRuntime *rt, uint8_t src_byte, uint8_t dst_byte) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    EdgeDisk *edges = rt->file->edges;
    NodeDisk *nodes = rt->file->nodes;
    
    uint64_t src_id = get_data_node_id(src_byte);
    uint64_t dst_id = get_data_node_id(dst_byte);
    
    // Find source node
    uint64_t src_idx = UINT64_MAX;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == src_id) {
            src_idx = i;
            break;
        }
    }
    
    if (src_idx == UINT64_MAX) return 0.0f;
    
    // Traverse outgoing edges from source node
    NodeDisk *src_node = &nodes[src_idx];
    uint64_t edge_idx = src_node->first_out_edge;
    
    for (uint32_t i = 0; i < src_node->out_degree && edge_idx != UINT64_MAX; i++) {
        if (edge_idx >= gh->edge_capacity) break;
        EdgeDisk *e = &edges[edge_idx];
        
        if ((e->flags & EDGE_FLAG_SEQ) && e->src == src_id && e->dst == dst_id) {
            return e->weight;
        }
        
        edge_idx = e->next_out_edge;
        if (edge_idx == UINT64_MAX) break;
    }
    
    return 0.0f;
}

// Find average eligibility trace for ABC pattern edges
static float find_abc_eligibility(MelvinRuntime *rt) {
    float sum = 0.0f;
    int count = 0;
    
    // A->B edge
    float w1 = find_seq_edge_weight(rt, 'A', 'B');
    if (w1 > 0.0f) {
        GraphHeaderDisk *gh = rt->file->graph_header;
        EdgeDisk *edges = rt->file->edges;
        NodeDisk *nodes = rt->file->nodes;
        
        uint64_t a_id = get_data_node_id('A');
        uint64_t a_idx = UINT64_MAX;
        for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
            if (nodes[i].id == a_id) {
                a_idx = i;
                break;
            }
        }
        
        if (a_idx != UINT64_MAX) {
            NodeDisk *a_node = &nodes[a_idx];
            uint64_t edge_idx = a_node->first_out_edge;
            for (uint32_t i = 0; i < a_node->out_degree && edge_idx != UINT64_MAX; i++) {
                if (edge_idx >= gh->edge_capacity) break;
                EdgeDisk *e = &edges[edge_idx];
                if (e->dst == get_data_node_id('B')) {
                    sum += e->eligibility;
                    count++;
                    break;
                }
                edge_idx = e->next_out_edge;
                if (edge_idx == UINT64_MAX) break;
            }
        }
    }
    
    // B->C edge
    float w2 = find_seq_edge_weight(rt, 'B', 'C');
    if (w2 > 0.0f) {
        GraphHeaderDisk *gh = rt->file->graph_header;
        EdgeDisk *edges = rt->file->edges;
        NodeDisk *nodes = rt->file->nodes;
        
        uint64_t b_id = get_data_node_id('B');
        uint64_t b_idx = UINT64_MAX;
        for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
            if (nodes[i].id == b_id) {
                b_idx = i;
                break;
            }
        }
        
        if (b_idx != UINT64_MAX) {
            NodeDisk *b_node = &nodes[b_idx];
            uint64_t edge_idx = b_node->first_out_edge;
            for (uint32_t i = 0; i < b_node->out_degree && edge_idx != UINT64_MAX; i++) {
                if (edge_idx >= gh->edge_capacity) break;
                EdgeDisk *e = &edges[edge_idx];
                if (e->dst == get_data_node_id('C')) {
                    sum += e->eligibility;
                    count++;
                    break;
                }
                edge_idx = e->next_out_edge;
                if (edge_idx == UINT64_MAX) break;
            }
        }
    }
    
    return count > 0 ? sum / count : 0.0f;
}

// Find node activation by byte value
static float find_node_activation(MelvinRuntime *rt, uint8_t byte) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    uint64_t target_id = get_data_node_id(byte);
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == target_id) {
            return nodes[i].state;
        }
    }
    
    return 0.0f;
}

// Calculate pattern-specific metrics
void calculate_pattern_metrics(MelvinRuntime *rt, PatternMetrics *metrics, double elapsed_seconds) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    
    // Basic metrics
    metrics->nodes = gh->num_nodes;
    metrics->edges = gh->num_edges;
    metrics->elapsed_seconds = elapsed_seconds;
    metrics->events_processed = rt->logical_time;
    
    // Calculate mean and max activation, prediction error
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
            sum_prediction_error += node->prediction_error;
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
    
    // Pattern-specific metrics
    metrics->abc_pattern_weight = (find_seq_edge_weight(rt, 'A', 'B') + 
                                   find_seq_edge_weight(rt, 'B', 'C')) / 2.0f;
    metrics->c_node_activation = find_node_activation(rt, 'C');
    metrics->eligibility_trace_abc = find_abc_eligibility(rt);
    metrics->rewarded_node_activation = metrics->c_node_activation; // C is our rewarded node
    
    // Validation errors
    metrics->validation_errors = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        NodeDisk *node = &nodes[i];
        if (isnan(node->state) || isinf(node->state) ||
            isnan(node->prediction) || isinf(node->prediction)) {
            metrics->validation_errors++;
        }
    }
}

// Log pattern metrics
void log_pattern_metrics(const PatternMetrics *metrics, int minute) {
    printf("\n--- [%02d:00] PATTERN METRICS ---\n", minute);
    printf("nodes: %llu\n", (unsigned long long)metrics->nodes);
    printf("edges: %llu\n", (unsigned long long)metrics->edges);
    printf("mean_activation: %.4f\n", metrics->mean_activation);
    printf("max_activation: %.4f\n", metrics->max_activation);
    printf("prediction_error: %.6f\n", metrics->prediction_error);
    printf("reward_recent: %.6f\n", metrics->reward_recent);
    printf("events_processed: %llu\n", (unsigned long long)metrics->events_processed);
    printf("validation_errors: %llu\n", (unsigned long long)metrics->validation_errors);
    printf("elapsed: %.1f seconds (%.2f minutes)\n", metrics->elapsed_seconds, metrics->elapsed_seconds / 60.0);
    
    printf("\n--- PATTERN LEARNING ---\n");
    printf("ABC_pattern_weight: %.6f\n", metrics->abc_pattern_weight);
    printf("C_node_activation: %.6f\n", metrics->c_node_activation);
    printf("eligibility_trace_ABC: %.6f\n", metrics->eligibility_trace_abc);
    printf("rewarded_node_activation: %.6f\n", metrics->rewarded_node_activation);
    
    if (metrics->elapsed_seconds > 0.1) {
        printf("events_processed_per_sec: %.1f\n", 
               (double)metrics->events_processed / metrics->elapsed_seconds);
    }
    
    fflush(stdout);
}

// Ingest repetitive pattern
void ingest_repetitive_pattern(MelvinRuntime *rt, const char *pattern, size_t repetitions, uint64_t channel_id) {
    size_t pattern_len = strlen(pattern);
    
    printf("Ingesting pattern \"%s\" repeated %zu times (channel %llu)...\n", 
           pattern, repetitions, (unsigned long long)channel_id);
    
    for (size_t rep = 0; rep < repetitions; rep++) {
        for (size_t i = 0; i < pattern_len; i++) {
            ingest_byte(rt, channel_id, (uint8_t)pattern[i], 1.0f);
        }
        
        // Process events periodically
        if (rep % 10 == 0) {
            melvin_process_n_events(rt, 100);
        }
    }
    
    // Process remaining events
    melvin_process_n_events(rt, 500);
    
    printf("  Ingested %zu bytes (%zu * %zu)\n", pattern_len * repetitions, pattern_len, repetitions);
}

int main(int argc, char **argv) {
    const char *file_path = "test_20min.m";  // Reuse stable baseline
    const int test_duration_seconds = 10 * 60;  // 10 minutes
    const int metrics_interval_seconds = 30;    // Log every 30 seconds
    
    printf("========================================\n");
    printf("MELVIN PATTERN STRESS-TEST + REWARD LOOP\n");
    printf("========================================\n\n");
    printf("Goal: Test pattern learning with minimal reward\n");
    printf("Duration: %d seconds (%d minutes)\n", test_duration_seconds, test_duration_seconds / 60);
    printf("Metrics interval: Every %d seconds\n\n", metrics_interval_seconds);
    
    // Step 1: Try to load existing stable baseline, or create new
    printf("Step 1: Loading baseline file...\n");
    MelvinFile file;
    
    if (access(file_path, R_OK) == 0) {
        printf("Found existing baseline: %s\n", file_path);
        if (melvin_m_map(file_path, &file) < 0) {
            fprintf(stderr, "ERROR: Failed to map existing file, creating new...\n");
            goto create_new;
        }
        printf("✓ Loaded existing baseline\n\n");
    } else {
create_new:
        printf("Creating new baseline file...\n");
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
            fprintf(stderr, "ERROR: Failed to create file\n");
            return 1;
        }
        
        if (melvin_m_map(file_path, &file) < 0) {
            fprintf(stderr, "ERROR: Failed to map file\n");
            return 1;
        }
        printf("✓ Created new baseline\n\n");
    }
    
    // Step 2: Initialize runtime
    printf("Step 2: Initializing runtime...\n");
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        return 1;
    }
    printf("✓ Runtime initialized\n\n");
    
    // Step 3: Ingest repetitive patterns
    printf("Step 3: Ingesting repetitive patterns...\n");
    
    // Pattern A: ABCABCABC... (100 repetitions)
    ingest_repetitive_pattern(&rt, "ABC", 100, 1);
    
    // Pattern B: HELLO_WORLDHELLO_WORLD... (50 repetitions)
    ingest_repetitive_pattern(&rt, "HELLO_WORLD", 50, 2);
    
    // Pattern C: Small C snippet (20 repetitions)
    const char *c_snippet = "int main() { return 0; }";
    ingest_repetitive_pattern(&rt, c_snippet, 20, 10);
    
    printf("\n✓ Pattern ingestion complete\n\n");
    
    // Step 4: Start test with reward injection
    printf("Step 4: Starting pattern stress-test with reward...\n");
    printf("Reward strategy: Inject +1.0 reward on 'C' node after ABC patterns\n");
    printf("========================================\n\n");
    
    // Start timer
    struct timeval start_time, current_time, last_metrics_time, last_reward_time;
    gettimeofday(&start_time, NULL);
    last_metrics_time = start_time;
    last_reward_time = start_time;
    
    PatternMetrics metrics;
    uint64_t reward_count = 0;
    uint64_t pattern_count = 0;
    float baseline_abc_weight = 0.0f;
    float baseline_pred_error = 0.0f;
    int baseline_set = 0;
    
    // Get baseline metrics
    gettimeofday(&current_time, NULL);
    double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                     (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    calculate_pattern_metrics(&rt, &metrics, elapsed);
    baseline_abc_weight = metrics.abc_pattern_weight;
    baseline_pred_error = metrics.prediction_error;
    baseline_set = 1;
    
    printf("--- [BASELINE] AFTER PATTERN INGESTION ---\n");
    log_pattern_metrics(&metrics, 0);
    
    // Main test loop
    uint64_t abc_pattern_injections = 0;
    double last_progress_log = 0.0;
    
    while (1) {
        gettimeofday(&current_time, NULL);
        elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                  (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        // Check if we've reached the time limit
        if (elapsed >= test_duration_seconds) {
            printf("\n=== Test duration reached ===\n");
            break;
        }
        
        // Process events continuously
        melvin_process_n_events(&rt, 1000);
        
        // Inject more ABC patterns periodically to keep the pattern active
        if ((int)elapsed % 30 == 0 && (int)elapsed != (int)last_progress_log) {
            // Inject ABC pattern every 30 seconds
            ingest_byte(&rt, 1, 'A', 1.0f);
            melvin_process_n_events(&rt, 50);
            ingest_byte(&rt, 1, 'B', 1.0f);
            melvin_process_n_events(&rt, 50);
            ingest_byte(&rt, 1, 'C', 1.0f);
            melvin_process_n_events(&rt, 100);
            
            abc_pattern_injections++;
            pattern_count++;
            
            // Inject reward on 'C' node after ABC pattern
            uint64_t c_node_id = get_data_node_id('C');
            inject_reward(&rt, c_node_id, 1.0f);
            reward_count++;
            
            // Process reward events
            melvin_process_n_events(&rt, 50);
        }
        
        // Progress messages every 30 seconds
        if (elapsed - last_progress_log >= 30.0) {
            printf("[%.0f seconds - ABC patterns: %llu, Rewards: %llu]\n", 
                   elapsed, (unsigned long long)abc_pattern_injections, (unsigned long long)reward_count);
            fflush(stdout);
            last_progress_log = elapsed;
        }
        
        // Log metrics at intervals
        double time_since_last_metrics = (current_time.tv_sec - last_metrics_time.tv_sec) + 
                                        (current_time.tv_usec - last_metrics_time.tv_usec) / 1000000.0;
        
        if (time_since_last_metrics >= metrics_interval_seconds) {
            calculate_pattern_metrics(&rt, &metrics, elapsed);
            
            // Check for failures
            if (metrics.validation_errors > 0) {
                printf("\n❌ FAILURE: Validation errors detected! (%llu)\n", 
                       (unsigned long long)metrics.validation_errors);
                break;
            }
            
            int current_minute = (int)(elapsed / 60);
            log_pattern_metrics(&metrics, current_minute);
            
            // Show learning progress
            if (baseline_set) {
                float weight_delta = metrics.abc_pattern_weight - baseline_abc_weight;
                float error_delta = baseline_pred_error - metrics.prediction_error;
                
                printf("\n--- LEARNING PROGRESS ---\n");
                printf("ABC weight change: %+.6f (baseline: %.6f, current: %.6f)\n",
                       weight_delta, baseline_abc_weight, metrics.abc_pattern_weight);
                printf("Prediction error change: %+.6f (baseline: %.6f, current: %.6f)\n",
                       error_delta, baseline_pred_error, metrics.prediction_error);
                
                // Signs of learning
                if (weight_delta > 0.001f) {
                    printf("✓ ABC pattern edges strengthening!\n");
                }
                if (error_delta > 0.001f) {
                    printf("✓ Prediction error decreasing!\n");
                }
                if (metrics.eligibility_trace_abc > 0.01f) {
                    printf("✓ Eligibility traces active!\n");
                }
            }
            
            last_metrics_time = current_time;
        }
    }
    
    // Final metrics
    printf("\n========================================\n");
    printf("FINAL METRICS\n");
    printf("========================================\n");
    gettimeofday(&current_time, NULL);
    elapsed = (current_time.tv_sec - start_time.tv_sec) + 
              (current_time.tv_usec - start_time.tv_usec) / 1000000.0;
    calculate_pattern_metrics(&rt, &metrics, elapsed);
    log_pattern_metrics(&metrics, (int)(elapsed / 60));
    
    // Final summary
    printf("\n========================================\n");
    printf("TEST SUMMARY\n");
    printf("========================================\n");
    printf("Duration: %.1f seconds (%.1f minutes)\n", elapsed, elapsed / 60.0);
    printf("ABC patterns injected: %llu\n", (unsigned long long)abc_pattern_injections);
    printf("Rewards injected: %llu\n", (unsigned long long)reward_count);
    printf("Final nodes: %llu\n", (unsigned long long)metrics.nodes);
    printf("Final edges: %llu\n", (unsigned long long)metrics.edges);
    printf("Validation errors: %llu\n", (unsigned long long)metrics.validation_errors);
    
    if (baseline_set) {
        float weight_delta = metrics.abc_pattern_weight - baseline_abc_weight;
        float error_delta = baseline_pred_error - metrics.prediction_error;
        
        printf("\n--- LEARNING RESULTS ---\n");
        printf("ABC pattern weight: %.6f -> %.6f (change: %+.6f)\n",
               baseline_abc_weight, metrics.abc_pattern_weight, weight_delta);
        printf("Prediction error: %.6f -> %.6f (change: %+.6f)\n",
               baseline_pred_error, metrics.prediction_error, error_delta);
        
        // Evaluate success
        int success = 1;
        if (metrics.validation_errors > 0) {
            printf("\n❌ FAILED: Validation errors\n");
            success = 0;
        }
        if (weight_delta <= 0.0f) {
            printf("\n⚠ WARNING: ABC edges did not strengthen (may need more time/patterns)\n");
        }
        if (error_delta <= 0.0f) {
            printf("\n⚠ WARNING: Prediction error did not decrease (may need more time/patterns)\n");
        }
        
        if (success && weight_delta > 0.0f && error_delta > 0.0f) {
            printf("\n✅ TEST PASSED: Pattern learning detected!\n");
            printf("   - ABC pattern edges strengthened\n");
            printf("   - Prediction error decreased\n");
            printf("   - System showed learning behavior\n");
        } else if (success) {
            printf("\n⚠ TEST PARTIAL: System stable but learning unclear\n");
            printf("   - May need longer duration or more pattern repetitions\n");
        }
    }
    
    // Sync and close
    printf("\nSyncing file to disk...\n");
    melvin_m_sync(&file);
    printf("✓ File synced\n\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    printf("========================================\n");
    printf("TEST COMPLETE\n");
    printf("========================================\n");
    printf("File: %s\n", file_path);
    
    return 0;
}


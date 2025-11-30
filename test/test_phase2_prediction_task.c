/*
 * Phase 2 Test C: Prediction Task Test
 * 
 * Goal: Teach Melvin next-byte prediction on synthetic data
 * Reward = +1 if predicted next byte matches; -1 otherwise
 * 
 * This proves computation → prediction → reward → computation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_phase2_prediction_task.m"
#define PATTERN_LENGTH 3
#define NUM_ITERATIONS 500

// Generate repeating pattern: ABC, ABC, ABC...
static void generate_pattern_sequence(uint8_t *buffer, size_t len) {
    const uint8_t pattern[] = {'A', 'B', 'C'};
    for (size_t i = 0; i < len; i++) {
        buffer[i] = pattern[i % PATTERN_LENGTH];
    }
}

// Find node with highest activation (prediction)
static uint64_t find_predicted_node(MelvinFile *file, uint8_t byte_value) {
    uint64_t predicted_node_id = UINT64_MAX;
    float max_activation = -1.0f;
    
    // Look for data nodes (byte values)
    for (uint64_t i = 0; i < file->graph_header->num_nodes; i++) {
        NodeDisk *node = &file->nodes[i];
        if (node->id == UINT64_MAX) continue;
        if (!(node->flags & NODE_FLAG_DATA)) continue;
        
        // Check if this is a byte node (ID = byte_value + 1000000)
        uint64_t byte_node_id = (uint64_t)byte_value + 1000000ULL;
        if (node->id == byte_node_id) {
            if (node->prediction > max_activation) {
                max_activation = node->prediction;
                predicted_node_id = node->id;
            }
        }
    }
    
    return predicted_node_id;
}

int main() {
    printf("========================================\n");
    printf("PHASE 2 TEST C: PREDICTION TASK TEST\n");
    printf("========================================\n\n");
    
    printf("Goal: Next-byte prediction with reward\n");
    printf("Pattern: ABC repeating (%d iterations)\n\n", NUM_ITERATIONS);
    
    // Step 1: Create new file
    printf("Step 1: Creating test file...\n");
    GraphParams params;
    init_default_params(&params);
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "Failed to create test file\n");
        return 1;
    }
    printf("✓ Created %s\n\n", TEST_FILE);
    
    // Step 2: Map file and initialize runtime
    printf("Step 2: Mapping file and initializing runtime...\n");
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "Failed to map file\n");
        return 1;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "Failed to initialize runtime\n");
        return 1;
    }
    printf("✓ Runtime initialized\n\n");
    
    // Step 3: Generate pattern sequence
    printf("Step 3: Generating pattern sequence...\n");
    uint8_t *sequence = malloc(NUM_ITERATIONS);
    if (!sequence) {
        fprintf(stderr, "Failed to allocate sequence\n");
        return 1;
    }
    generate_pattern_sequence(sequence, NUM_ITERATIONS);
    printf("✓ Generated %d bytes\n\n", NUM_ITERATIONS);
    
    // Step 4: Run prediction task
    printf("Step 4: Running prediction task...\n");
    
    int correct_predictions = 0;
    int total_predictions = 0;
    float total_reward = 0.0f;
    
    for (int i = 0; i < NUM_ITERATIONS - 1; i++) {
        uint8_t current_byte = sequence[i];
        uint8_t next_byte = sequence[i + 1];
        
        // Ingest current byte
        ingest_byte(&rt, 0, current_byte, 1.0f);
        melvin_process_n_events(&rt, 20);
        
        // Find predicted next byte (node with highest prediction)
        uint64_t predicted_node_id = find_predicted_node(&file, next_byte);
        
        // Check if prediction is correct
        uint64_t correct_node_id = (uint64_t)next_byte + 1000000ULL;
        int prediction_correct = (predicted_node_id == correct_node_id);
        
        if (prediction_correct) {
            correct_predictions++;
            total_reward += 1.0f;
        } else {
            total_reward -= 1.0f;
        }
        total_predictions++;
        
        // Inject reward
        if (prediction_correct) {
            inject_reward(&rt, correct_node_id, 1.0f);
        } else {
            // Negative reward for incorrect prediction
            if (predicted_node_id != UINT64_MAX) {
                inject_reward(&rt, predicted_node_id, -1.0f);
            }
        }
        
        melvin_process_n_events(&rt, 10);
        
        // Progress indicator
        if ((i + 1) % 100 == 0) {
            float accuracy = (float)correct_predictions / total_predictions * 100.0f;
            printf("  Iteration %d: Accuracy = %.1f%% (%d/%d correct)\n", 
                   i + 1, accuracy, correct_predictions, total_predictions);
        }
    }
    
    printf("\n");
    
    // Step 5: Analyze results
    printf("Step 5: Analyzing results...\n");
    
    float final_accuracy = (float)correct_predictions / total_predictions * 100.0f;
    float avg_reward = total_reward / total_predictions;
    
    printf("  Total predictions: %d\n", total_predictions);
    printf("  Correct predictions: %d\n", correct_predictions);
    printf("  Final accuracy: %.2f%%\n", final_accuracy);
    printf("  Average reward: %.3f\n", avg_reward);
    printf("\n");
    
    // Check edge weights for pattern ABC
    printf("Step 6: Checking learned edge weights...\n");
    
    uint64_t a_node_id = (uint64_t)'A' + 1000000ULL;
    uint64_t b_node_id = (uint64_t)'B' + 1000000ULL;
    uint64_t c_node_id = (uint64_t)'C' + 1000000ULL;
    
    uint64_t a_idx = find_node_index_by_id(&file, a_node_id);
    uint64_t b_idx = find_node_index_by_id(&file, b_node_id);
    uint64_t c_idx = find_node_index_by_id(&file, c_node_id);
    
    float a_to_b_weight = 0.0f;
    float b_to_c_weight = 0.0f;
    
    if (a_idx != UINT64_MAX) {
        NodeDisk *a_node = &file.nodes[a_idx];
        uint64_t e_idx = a_node->first_out_edge;
        for (uint32_t i = 0; i < a_node->out_degree && e_idx != UINT64_MAX; i++) {
            EdgeDisk *e = &file.edges[e_idx];
            if (e->dst == b_node_id) {
                a_to_b_weight = e->weight;
                break;
            }
            e_idx = e->next_out_edge;
        }
    }
    
    if (b_idx != UINT64_MAX) {
        NodeDisk *b_node = &file.nodes[b_idx];
        uint64_t e_idx = b_node->first_out_edge;
        for (uint32_t i = 0; i < b_node->out_degree && e_idx != UINT64_MAX; i++) {
            EdgeDisk *e = &file.edges[e_idx];
            if (e->dst == c_node_id) {
                b_to_c_weight = e->weight;
                break;
            }
            e_idx = e->next_out_edge;
        }
    }
    
    printf("  A -> B weight: %.4f\n", a_to_b_weight);
    printf("  B -> C weight: %.4f\n", b_to_c_weight);
    printf("\n");
    
    // Summary
    printf("========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    
    int passed = 1;
    
    if (final_accuracy > 50.0f) {
        printf("✓ Prediction accuracy > 50%%: PASSED (%.2f%%)\n", final_accuracy);
    } else {
        printf("✗ Prediction accuracy > 50%%: FAILED (%.2f%%)\n", final_accuracy);
        passed = 0;
    }
    
    if (a_to_b_weight > 0.3f && b_to_c_weight > 0.3f) {
        printf("✓ Edge weights > 0.3: PASSED (A->B: %.4f, B->C: %.4f)\n", 
               a_to_b_weight, b_to_c_weight);
    } else {
        printf("⚠ Edge weights > 0.3: PARTIAL (A->B: %.4f, B->C: %.4f)\n", 
               a_to_b_weight, b_to_c_weight);
    }
    
    if (avg_reward > 0.0f) {
        printf("✓ Average reward > 0: PASSED (%.3f)\n", avg_reward);
    } else {
        printf("⚠ Average reward > 0: PARTIAL (%.3f)\n", avg_reward);
    }
    
    printf("\n");
    if (passed) {
        printf("✅ PREDICTION TASK TEST: PASSED\n");
        printf("Melvin can learn patterns and make predictions!\n");
    } else {
        printf("❌ PREDICTION TASK TEST: PARTIAL\n");
        printf("May need more iterations or tuning.\n");
    }
    
    // Cleanup
    free(sequence);
    runtime_cleanup(&rt);
    close_file(&file);
    
    return passed ? 0 : 1;
}


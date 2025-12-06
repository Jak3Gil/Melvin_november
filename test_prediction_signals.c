/*
 * Test Prediction Signals - Sanity Checks
 * 
 * Tests the three prediction channels:
 * A. Internal next-state prediction
 * B. Sensory next-input prediction  
 * C. Value-delta prediction
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "src/melvin.h"

#define TEST_A_STEPS 100
#define TEST_B_BYTES 200
#define TEST_C_STEPS 150

/* Test A: Internal prediction sanity */
void test_internal_prediction(Graph *g) {
    printf("\n=== TEST A: Internal Next-State Prediction ===\n");
    printf("Running %d physics steps...\n", TEST_A_STEPS);
    
    /* Track a few sample nodes */
    uint32_t sample_nodes[] = {0, 10, 50, 100, 200};
    uint32_t sample_count = 5;
    
    printf("\nNode ID | Energy | Predicted | Error | Notes\n");
    printf("--------|--------|----------|-------|------\n");
    
    for (int step = 0; step < TEST_A_STEPS; step++) {
        /* Feed a small amount of input to create activity */
        if (step % 10 == 0) {
            melvin_feed_byte(g, 0, (uint8_t)(step % 256), 0.1f);
        }
        
        melvin_run_physics(g);
        
        /* Print sample every 20 steps */
        if (step % 20 == 0 || step == TEST_A_STEPS - 1) {
            printf("\nStep %d:\n", step);
            for (uint32_t i = 0; i < sample_count; i++) {
                uint32_t node_id = sample_nodes[i];
                if (node_id >= g->node_count) continue;
                
                Node *n = &g->nodes[node_id];
                const char *note = "";
                
                if (fabsf(n->prediction_error) > 1.0f) {
                    note = "LARGE_ERROR";
                } else if (fabsf(n->prediction_error) < 0.01f && n->energy > 0.01f) {
                    note = "GOOD_PRED";
                } else if (n->predicted_activation < 0.0f || n->predicted_activation > 10.0f) {
                    note = "OUT_OF_RANGE";
                }
                
                printf("  %6u | %6.3f | %8.3f | %5.3f | %s\n",
                       node_id, n->energy, n->predicted_activation, n->prediction_error, note);
            }
        }
    }
    
    printf("\n✓ Test A Complete\n");
    printf("  Check: predicted_activation should be in similar range as energy\n");
    printf("  Check: prediction_error should be small when stable, larger around surprises\n");
}

/* Test B: Sensory prediction sanity */
void test_sensory_prediction(Graph *g) {
    printf("\n=== TEST B: Sensory Next-Input Prediction ===\n");
    printf("Feeding repeating pattern 'ABABAB...' (%d bytes)\n", TEST_B_BYTES);
    
    uint8_t pattern[] = {'A', 'B'};
    uint32_t pattern_len = 2;
    
    printf("\nStep | Last Byte | Predicted | Actual | Error | Confidence\n");
    printf("-----|-----------|----------|-------|-------|-----------\n");
    
    float total_error = 0.0f;
    uint32_t correct_predictions = 0;
    
    for (int i = 0; i < TEST_B_BYTES; i++) {
        uint8_t byte = pattern[i % pattern_len];
        
        /* Predict before feeding */
        melvin_predict_sensory(g);
        uint8_t predicted = g->predicted_next_byte;
        
        /* Feed the actual byte */
        melvin_feed_byte(g, 0, byte, 0.2f);
        melvin_run_physics(g);
        
        /* Compute error */
        float byte_error = fabsf((float)predicted - (float)byte) / 255.0f;
        total_error += byte_error;
        
        if (predicted == byte) {
            correct_predictions++;
        }
        
        /* Print every 20 bytes */
        if (i % 20 == 0 || i == TEST_B_BYTES - 1) {
            printf("%4d | %9u | %8u | %5u | %5.3f | %10u\n",
                   i, g->last_input_byte, predicted, byte, byte_error, g->predicted_byte_confidence);
        }
        
        /* Check sensory_pred_error for DATA nodes */
        if (i == TEST_B_BYTES - 1) {
            printf("\nFinal sensory_pred_error for DATA nodes (0-255):\n");
            uint32_t sample_data_nodes = (g->node_count < 256) ? g->node_count : 256;
            uint32_t active_data_count = 0;
            float avg_sensory_error = 0.0f;
            
            for (uint32_t j = 0; j < sample_data_nodes; j++) {
                Node *n = &g->nodes[j];
                if (n->type == NODE_TYPE_DATA && n->sensory_pred_error > 0.0f) {
                    avg_sensory_error += n->sensory_pred_error;
                    active_data_count++;
                }
            }
            
            if (active_data_count > 0) {
                avg_sensory_error /= (float)active_data_count;
                printf("  Active DATA nodes with sensory_pred_error: %u\n", active_data_count);
                printf("  Average sensory_pred_error: %.4f\n", avg_sensory_error);
            }
        }
    }
    
    float avg_error = total_error / (float)TEST_B_BYTES;
    float accuracy = (float)correct_predictions / (float)TEST_B_BYTES * 100.0f;
    
    printf("\n✓ Test B Complete\n");
    printf("  Total bytes: %d\n", TEST_B_BYTES);
    printf("  Correct predictions: %u (%.1f%%)\n", correct_predictions, accuracy);
    printf("  Average error: %.4f\n", avg_error);
    printf("  Check: error should be high at first, then drop as pattern repeats\n");
}

/* Test C: Value-delta prediction sanity */
void test_value_delta_prediction(Graph *g) {
    printf("\n=== TEST C: Value-Delta Prediction ===\n");
    printf("Running %d steps with alternating stable/noisy regimes\n", TEST_C_STEPS);
    
    printf("\nStep | Global Value | Predicted Delta | Actual Delta | Value Error\n");
    printf("-----|--------------|-----------------|-------------|------------\n");
    
    float prev_global_value = 0.0f;
    
    for (int step = 0; step < TEST_C_STEPS; step++) {
        /* Alternate between stable (low energy) and noisy (high energy) regimes */
        bool noisy_regime = (step / 30) % 2 == 1;
        
        if (noisy_regime) {
            /* Inject random high-energy inputs */
            for (int i = 0; i < 5; i++) {
                uint8_t byte = (uint8_t)(rand() % 256);
                melvin_feed_byte(g, 0, byte, 0.5f);
            }
        } else {
            /* Stable regime: feed same byte repeatedly */
            melvin_feed_byte(g, 0, (uint8_t)('A' + (step % 26)), 0.1f);
        }
        
        melvin_run_physics(g);
        
        /* Get current global value */
        float current_value = g->global_value_estimate;
        float actual_delta = current_value - prev_global_value;
        float value_error = g->predicted_global_value_delta - actual_delta;
        
        /* Print every 15 steps */
        if (step % 15 == 0 || step == TEST_C_STEPS - 1) {
            const char *regime = noisy_regime ? "NOISY" : "STABLE";
            printf("%4d | %12.4f | %15.4f | %11.4f | %10.4f [%s]\n",
                   step, current_value, g->predicted_global_value_delta, actual_delta, value_error, regime);
        }
        
        prev_global_value = current_value;
    }
    
    /* Check value_pred_error for EXEC nodes */
    printf("\nValue prediction error for EXEC nodes:\n");
    uint32_t exec_count = 0;
    float avg_value_error = 0.0f;
    
    for (uint64_t i = 0; i < g->node_count && exec_count < 20; i++) {
        Node *n = &g->nodes[i];
        if (n->type == NODE_TYPE_EXEC && fabsf(n->value_pred_error) > 0.001f) {
            printf("  EXEC node %lu: value_pred_error = %.4f, value = %.4f\n",
                   (unsigned long)i, n->value_pred_error, n->value);
            avg_value_error += fabsf(n->value_pred_error);
            exec_count++;
        }
    }
    
    if (exec_count > 0) {
        avg_value_error /= (float)exec_count;
        printf("  Average |value_pred_error| for EXEC nodes: %.4f\n", avg_value_error);
    }
    
    printf("\n✓ Test C Complete\n");
    printf("  Check: predicted_global_value_delta should track trends\n");
    printf("  Check: value_pred_error should shrink as system sees same regime again\n");
}

int main(int argc, char **argv) {
    const char *brain_file = (argc > 1) ? argv[1] : "test_prediction_brain.m";
    
    printf("=== Prediction Signals Sanity Check ===\n");
    printf("Brain file: %s\n", brain_file);
    
    /* Open or create brain */
    Graph *g = melvin_open(brain_file, 10000, 50000, 1024*1024);
    if (!g) {
        fprintf(stderr, "Failed to open brain file: %s\n", brain_file);
        return 1;
    }
    
    printf("Graph opened: %llu nodes, %llu edges\n",
           (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    
    /* Run all three tests */
    test_internal_prediction(g);
    test_sensory_prediction(g);
    test_value_delta_prediction(g);
    
    /* Summary */
    printf("\n=== SUMMARY ===\n");
    printf("All three prediction channels tested.\n");
    printf("Review output above to verify:\n");
    printf("  - Internal: predicted_activation ~ energy, prediction_error reasonable\n");
    printf("  - Sensory: error drops as pattern repeats\n");
    printf("  - Value-delta: predicted delta tracks actual trends\n");
    
    melvin_close(g);
    return 0;
}


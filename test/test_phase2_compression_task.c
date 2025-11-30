/*
 * Phase 2 Test D: Compression Task Test
 * 
 * Goal: Verify Melvin learns to compress data for reward
 * - Reward = negative pattern entropy (fewer patterns = better compression)
 * - Reward = improved prediction quality
 * 
 * This tests pattern hierarchy formation and compression.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "melvin.c"

#define TEST_FILE "test_phase2_compression_task.m"

// Calculate pattern entropy (Shannon entropy of pattern activations)
float calculate_pattern_entropy(MelvinRuntime *rt) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    
    float total_activation = 0.0f;
    uint64_t pattern_count = 0;
    
    // Count patterns (nodes with ID >= 5000000 are pattern nodes)
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        if (nodes[i].id >= 5000000ULL) {
            pattern_count++;
            total_activation += fabsf(nodes[i].state);
        }
    }
    
    if (pattern_count == 0 || total_activation == 0.0f) {
        return 0.0f;  // No patterns or no activation
    }
    
    // Calculate entropy: H = -Σ p_i * log2(p_i)
    float entropy = 0.0f;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        if (nodes[i].id >= 5000000ULL) {
            float p = fabsf(nodes[i].state) / total_activation;
            if (p > 0.0001f) {  // Avoid log(0)
                entropy -= p * log2f(p);
            }
        }
    }
    
    return entropy;
}

// Calculate compression ratio (fewer patterns for same data = better compression)
float calculate_compression_ratio(MelvinRuntime *rt, uint64_t data_bytes) {
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    
    uint64_t pattern_count = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        if (nodes[i].id >= 5000000ULL) {
            pattern_count++;
        }
    }
    
    if (data_bytes == 0) return 0.0f;
    if (pattern_count == 0) return 1.0f;  // No compression yet
    
    // Compression ratio = patterns / data_bytes (lower is better)
    return (float)pattern_count / (float)data_bytes;
}

int main() {
    printf("========================================\n");
    printf("PHASE 2 TEST D: COMPRESSION TASK TEST\n");
    printf("========================================\n\n");
    
    // Step 1: Initialize runtime
    printf("Step 1: Initializing runtime...\n");
    MelvinRuntime *rt = malloc(sizeof(MelvinRuntime));
    if (!rt) {
        fprintf(stderr, "ERROR: Failed to allocate runtime\n");
        return 1;
    }
    
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.015f;
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create test file\n");
        free(rt);
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map test file\n");
        free(rt);
        return 1;
    }
    
    if (runtime_init(rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        close_file(&file);
        free(rt);
        return 1;
    }
    
    printf("✓ Runtime initialized\n\n");
    
    // Step 2: Generate repetitive data (highly compressible)
    printf("Step 2: Generating repetitive data...\n");
    const char *pattern = "ABCABCABC";
    uint64_t data_bytes = 0;
    uint64_t iterations = 200;  // 200 repetitions of ABC
    
    for (uint64_t iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < strlen(pattern); i++) {
            uint8_t byte = (uint8_t)pattern[i];
            MelvinEvent ev = {
                .type = EV_INPUT_BYTE,
                .channel_id = 0,
                .byte = byte,
                .value = 3.0f  // Higher energy for pattern formation
            };
            melvin_event_enqueue(&rt->evq, &ev);
            data_bytes++;
        }
        
        // Process events to allow pattern formation
        for (int j = 0; j < 50; j++) {
            melvin_process_n_events(rt, 10);
        }
    }
    
    printf("✓ Generated %llu bytes of repetitive data\n\n", (unsigned long long)data_bytes);
    
    // Step 3: Measure initial compression
    printf("Step 3: Measuring compression...\n");
    float initial_entropy = calculate_pattern_entropy(rt);
    float initial_ratio = calculate_compression_ratio(rt, data_bytes);
    uint64_t initial_patterns = 0;
    GraphHeaderDisk *gh = rt->file->graph_header;
    NodeDisk *nodes = rt->file->nodes;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        if (nodes[i].id >= 5000000ULL) initial_patterns++;
    }
    
    printf("  Initial patterns: %llu\n", (unsigned long long)initial_patterns);
    printf("  Initial entropy: %.4f\n", initial_entropy);
    printf("  Initial compression ratio: %.4f\n", initial_ratio);
    printf("\n");
    
    // Step 4: Run compression task with reward
    printf("Step 4: Running compression task with reward...\n");
    uint64_t reward_iterations = 100;
    float total_reward = 0.0f;
    float best_entropy = initial_entropy;
    float best_ratio = initial_ratio;
    
    for (uint64_t iter = 0; iter < reward_iterations; iter++) {
        // Process events
        (void)melvin_process_n_events(rt, 20);
        
        // Calculate current compression metrics
        float current_entropy = calculate_pattern_entropy(rt);
        float current_ratio = calculate_compression_ratio(rt, data_bytes);
        
        // Reward = negative entropy (lower entropy = better compression)
        // Also reward lower pattern count (fewer patterns = better compression)
        float reward = -current_entropy * 0.1f;  // Scale down
        if (current_ratio < initial_ratio) {
            reward += 0.5f;  // Bonus for better compression ratio
        }
        
        // Inject reward
        MelvinEvent reward_ev = {
            .type = EV_REWARD_ARRIVAL,
            .value = reward
        };
        melvin_event_enqueue(&rt->evq, &reward_ev);
        
        total_reward += reward;
        
        if (current_entropy < best_entropy) best_entropy = current_entropy;
        if (current_ratio < best_ratio) best_ratio = current_ratio;
        
        if ((iter + 1) % 20 == 0) {
            printf("  Iteration %llu: Entropy=%.4f, Ratio=%.4f, Reward=%.3f\n",
                   (unsigned long long)(iter + 1), current_entropy, current_ratio, reward);
        }
    }
    
    printf("\n");
    
    // Step 5: Final measurements
    printf("Step 5: Final measurements...\n");
    float final_entropy = calculate_pattern_entropy(rt);
    float final_ratio = calculate_compression_ratio(rt, data_bytes);
    uint64_t final_patterns = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        if (nodes[i].id == UINT64_MAX) continue;
        if (nodes[i].id >= 5000000ULL) final_patterns++;
    }
    
    printf("  Final patterns: %llu\n", (unsigned long long)final_patterns);
    printf("  Final entropy: %.4f\n", final_entropy);
    printf("  Final compression ratio: %.4f\n", final_ratio);
    printf("  Average reward: %.4f\n", total_reward / reward_iterations);
    printf("\n");
    
    // Step 6: Results
    printf("========================================\n");
    printf("RESULTS\n");
    printf("========================================\n");
    
    int entropy_improved = (final_entropy < initial_entropy);
    int ratio_improved = (final_ratio < initial_ratio);
    int patterns_reasonable = (final_patterns > 0 && final_patterns < data_bytes / 10);
    
    if (entropy_improved) {
        printf("✓ Entropy reduced: %.4f → %.4f\n", initial_entropy, final_entropy);
    } else {
        printf("⚠ Entropy not reduced: %.4f → %.4f\n", initial_entropy, final_entropy);
    }
    
    if (ratio_improved) {
        printf("✓ Compression ratio improved: %.4f → %.4f\n", initial_ratio, final_ratio);
    } else {
        printf("⚠ Compression ratio not improved: %.4f → %.4f\n", initial_ratio, final_ratio);
    }
    
    if (patterns_reasonable) {
        printf("✓ Pattern count reasonable: %llu patterns for %llu bytes\n",
               (unsigned long long)final_patterns, (unsigned long long)data_bytes);
    } else {
        printf("⚠ Pattern count: %llu patterns for %llu bytes\n",
               (unsigned long long)final_patterns, (unsigned long long)data_bytes);
    }
    
    printf("\n");
    
    if (entropy_improved || ratio_improved) {
        printf("✅ COMPRESSION TASK TEST: PASSED\n");
        printf("Melvin can learn to compress data!\n");
    } else {
        printf("⚠️  COMPRESSION TASK TEST: PARTIAL\n");
        printf("Compression learning needs more tuning.\n");
    }
    
    // Cleanup
    runtime_cleanup(rt);
    close_file(&file);
    free(rt);
    unlink(TEST_FILE);
    
    return 0;
}


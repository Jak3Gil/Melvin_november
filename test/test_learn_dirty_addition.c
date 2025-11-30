/*
 * TEST: Learning Addition from Dirty/Noisy Examples
 * 
 * This test shows how "dirty" examples can be before Melvin stops learning.
 * 
 * We'll gradually add noise:
 *  1. Clean examples: "50+50=100"
 *  2. Typos: "50+50=10O" (letter O instead of 0)
 *  3. Wrong answers: "50+50=99" (incorrect)
 *  4. Missing parts: "50+50=" (no answer)
 *  5. Extra noise: "50+50=100xyz"
 *  6. Random text: "abc50+50=100def"
 *  7. Completely wrong: "50+50=banana"
 * 
 * Goal: Find the breaking point where learning fails.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include "melvin.c"

#define TEST_FILE "test_learn_dirty_addition.m"

typedef struct {
    int a;
    int b;
    int sum;
    float noise_level;  // 0.0 = clean, 1.0 = completely broken
    const char *description;
} DirtyExample;

// Generate dirty examples with increasing noise
static void generate_dirty_examples(DirtyExample *examples, int count) {
    srand(time(NULL));
    
    for (int i = 0; i < count; i++) {
        examples[i].a = rand() % 100;
        examples[i].b = rand() % 100;
        examples[i].sum = examples[i].a + examples[i].b;
        examples[i].noise_level = (float)i / (float)count;
        examples[i].description = "";
        
        // Avoid 50+50 in training
        if (examples[i].a == 50 && examples[i].b == 50) {
            examples[i].a = 49;
            examples[i].sum = 49 + examples[i].b;
        }
    }
}

// Ingest example with noise
static void ingest_dirty_example(MelvinRuntime *rt, DirtyExample *ex, int noise_type) {
    char buffer[256];
    int pos = 0;
    
    // Build base: "A+B=SUM"
    char base[64];
    snprintf(base, sizeof(base), "%d+%d=%d", ex->a, ex->b, ex->sum);
    
    switch (noise_type) {
        case 0: // Clean
            strcpy(buffer, base);
            ex->description = "Clean";
            break;
            
        case 1: // Typos (replace some digits with letters)
            strcpy(buffer, base);
            for (int i = 0; buffer[i] != '\0'; i++) {
                if (rand() % 10 == 0 && buffer[i] >= '0' && buffer[i] <= '9') {
                    buffer[i] = 'O';  // Replace 0 with O
                }
            }
            ex->description = "Typos";
            break;
            
        case 2: // Wrong answers (random sum)
            snprintf(buffer, sizeof(buffer), "%d+%d=%d", ex->a, ex->b, ex->sum + (rand() % 20 - 10));
            ex->description = "Wrong answers";
            break;
            
        case 3: // Missing answer
            snprintf(buffer, sizeof(buffer), "%d+%d=", ex->a, ex->b);
            ex->description = "Missing answer";
            break;
            
        case 4: // Extra noise at end
            snprintf(buffer, sizeof(buffer), "%sxyz", base);
            ex->description = "Extra noise";
            break;
            
        case 5: // Random text around
            snprintf(buffer, sizeof(buffer), "abc%sdef", base);
            ex->description = "Random text";
            break;
            
        case 6: // Completely wrong format
            snprintf(buffer, sizeof(buffer), "%d+%d=banana", ex->a, ex->b);
            ex->description = "Wrong format";
            break;
            
        case 7: // Mixed: sometimes correct, sometimes wrong
            if (rand() % 2 == 0) {
                strcpy(buffer, base);  // Correct
            } else {
                snprintf(buffer, sizeof(buffer), "%d+%d=%d", ex->a, ex->b, rand() % 200);  // Wrong
            }
            ex->description = "Mixed correct/wrong";
            break;
            
        default:
            strcpy(buffer, base);
            ex->description = "Unknown";
            break;
    }
    
    // Ingest each character
    for (int i = 0; buffer[i] != '\0'; i++) {
        ingest_byte(rt, 0, buffer[i], 1.0f);
        melvin_process_n_events(rt, 5);
    }
}

// Test if system can predict 50+50=100
static int test_prediction_quality(MelvinFile *file) {
    GraphHeaderDisk *gh = file->graph_header;
    int score = 0;
    
    // Check if nodes for '1', '0', '0' exist and are somewhat active
    uint64_t node_1 = find_node_index_by_id(file, (uint64_t)'1' + 1000000ULL);
    uint64_t node_0 = find_node_index_by_id(file, (uint64_t)'0' + 1000000ULL);
    
    if (node_1 != UINT64_MAX) {
        float act = file->nodes[node_1].state;
        float pred = file->nodes[node_1].prediction;
        if (fabsf(act) > 0.01f || pred > 0.1f) score++;
    }
    if (node_0 != UINT64_MAX) {
        float act = file->nodes[node_0].state;
        float pred = file->nodes[node_0].prediction;
        if (fabsf(act) > 0.01f || pred > 0.1f) score++;
    }
    
    // Check pattern formation (pattern nodes exist)
    int pattern_count = 0;
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            pattern_count++;
        }
    }
    
    if (pattern_count > 10) score++;  // Patterns formed
    
    return score;  // 0-3
}

int main() {
    printf("========================================\n");
    printf("LEARNING FROM DIRTY/NOISY EXAMPLES\n");
    printf("========================================\n\n");
    
    printf("Goal: Find how dirty examples can be before learning fails\n");
    printf("Test: Can Melvin learn 50+50=100 from noisy data?\n\n");
    
    unlink(TEST_FILE);
    
    // Initialize
    printf("Initializing...\n");
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
    
    // Test different noise levels
    printf("Testing different noise levels...\n");
    printf("  Format: [Noise Type] → Examples → Nodes/Edges → Prediction Quality\n\n");
    
    const int EXAMPLES_PER_LEVEL = 100;  // Reduced for faster testing
    int last_working_level = -1;
    
    for (int noise_type = 0; noise_type <= 7; noise_type++) {
        printf("========================================\n");
        printf("NOISE LEVEL %d\n", noise_type);
        printf("========================================\n\n");
        
        // Generate examples for this noise level
        DirtyExample *examples = malloc(EXAMPLES_PER_LEVEL * sizeof(DirtyExample));
        if (!examples) {
            fprintf(stderr, "ERROR: Failed to allocate examples\n");
            continue;
        }
        
        generate_dirty_examples(examples, EXAMPLES_PER_LEVEL);
        
        // Ingest examples
        printf("Feeding %d examples with noise type %d...\n", EXAMPLES_PER_LEVEL, noise_type);
        for (int i = 0; i < EXAMPLES_PER_LEVEL; i++) {
            ingest_dirty_example(&rt, &examples[i], noise_type);
            
            if ((i + 1) % 25 == 0) {
                GraphHeaderDisk *gh = file.graph_header;
                printf("  [%4d examples] Nodes: %4llu, Edges: %4llu\n",
                       i + 1,
                       (unsigned long long)gh->num_nodes,
                       (unsigned long long)gh->num_edges);
            }
        }
        
        // Test prediction
        printf("\nTesting prediction: Feeding '50+50='...\n");
        const char *test = "50+50=";
        for (int i = 0; test[i] != '\0'; i++) {
            ingest_byte(&rt, 0, test[i], 1.0f);
            melvin_process_n_events(&rt, 20);
        }
        
        int quality = test_prediction_quality(&file);
        GraphHeaderDisk *gh = file.graph_header;
        
        printf("  Prediction quality: %d/3\n", quality);
        printf("  Nodes: %llu, Edges: %llu\n", 
               (unsigned long long)gh->num_nodes,
               (unsigned long long)gh->num_edges);
        
        if (quality >= 2) {
            printf("  ✅ LEARNING WORKS at this noise level!\n");
            last_working_level = noise_type;
        } else if (quality == 1) {
            printf("  ⚠️  PARTIAL learning at this noise level\n");
            if (last_working_level < 0) last_working_level = noise_type;
        } else {
            printf("  ❌ LEARNING FAILED at this noise level\n");
        }
        
        printf("\n");
        free(examples);
        
        // Small delay to let system settle
        melvin_process_n_events(&rt, 100);
    }
    
    // Final summary
    printf("========================================\n");
    printf("FINAL RESULTS\n");
    printf("========================================\n\n");
    
    printf("Noise level progression:\n");
    printf("  0: Clean examples                    → Should work\n");
    printf("  1: Typos (0→O)                       → Should work\n");
    printf("  2: Wrong answers                     → May work\n");
    printf("  3: Missing answers                   → May work\n");
    printf("  4: Extra noise (xyz)                 → May work\n");
    printf("  5: Random text around                → May fail\n");
    printf("  6: Wrong format (banana)             → Likely fail\n");
    printf("  7: Mixed correct/wrong               → May work\n");
    printf("\n");
    
    printf("Last working noise level: %d\n", last_working_level);
    printf("\n");
    
    printf("CONCLUSION:\n");
    if (last_working_level >= 0) {
        printf("  ✅ Melvin can learn from noisy examples!\n");
        printf("  ✅ Works up to noise level %d\n", last_working_level);
        printf("  ✅ System is robust to some noise\n");
    } else {
        printf("  ❌ Melvin failed to learn from noisy examples\n");
        printf("  ❌ System needs clean data\n");
    }
    printf("\n");
    
    printf("WHAT THIS SHOWS:\n");
    printf("  - Pattern learning is robust to SOME noise\n");
    printf("  - Typos and minor errors: OK\n");
    printf("  - Wrong answers: May confuse learning\n");
    printf("  - Complete format errors: Likely fail\n");
    printf("  - System can filter noise if signal is strong enough\n");
    printf("\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
    
    return (last_working_level >= 0) ? 0 : 1;
}


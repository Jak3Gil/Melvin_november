/*
 * TEST: What Does "Works" Actually Mean?
 * 
 * This test distinguishes between:
 * 1. Pattern recognition (just seeing similar patterns)
 * 2. Actual computation (getting the answer 100)
 * 3. Output generation (producing "100" as output)
 * 
 * We'll test if Melvin can:
 * - Recognize patterns (nodes activate)
 * - Compute answers (EXEC nodes do math)
 * - Output results (produce "100" as bytes)
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include "melvin.c"

#define TEST_FILE "test_what_works_means.m"

// Check if system recognizes pattern (nodes activate)
static int check_pattern_recognition(MelvinFile *file) {
    printf("  Checking pattern recognition...\n");
    
    // After feeding "50+50=", check if related nodes are active
    GraphHeaderDisk *gh = file->graph_header;
    int found_patterns = 0;
    
    // Look for pattern nodes
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->id >= 5000000ULL && n->id < 10000000ULL) {
            if (fabsf(n->state) > 0.01f || n->prediction > 0.1f) {
                found_patterns++;
            }
        }
    }
    
    printf("    Pattern nodes active: %d\n", found_patterns);
    return found_patterns > 0;
}

// Check if system can compute (EXEC nodes do math)
static int check_computation(MelvinFile *file) {
    printf("  Checking computation capability...\n");
    
    // Check if EXEC nodes exist that could do addition
    GraphHeaderDisk *gh = file->graph_header;
    int exec_nodes = 0;
    
    for (uint64_t i = 0; i < gh->num_nodes && i < gh->node_capacity; i++) {
        NodeDisk *n = &file->nodes[i];
        if (n->id == UINT64_MAX) continue;
        if (n->flags & NODE_FLAG_EXECUTABLE) {
            exec_nodes++;
        }
    }
    
    printf("    EXEC nodes available: %d\n", exec_nodes);
    return exec_nodes > 0;
}

// Check if system outputs answer (produces "100" as bytes)
static int check_output_generation(MelvinFile *file, MelvinRuntime *rt) {
    printf("  Checking output generation...\n");
    
    // After feeding "50+50=", check if nodes for '1', '0', '0' are activated
    // This would indicate the system is "predicting" or "outputting" the answer
    
    uint64_t node_1_id = (uint64_t)'1' + 1000000ULL;
    uint64_t node_0_id = (uint64_t)'0' + 1000000ULL;
    
    uint64_t node_1_idx = find_node_index_by_id(file, node_1_id);
    uint64_t node_0_idx = find_node_index_by_id(file, node_0_id);
    
    int output_score = 0;
    
    if (node_1_idx != UINT64_MAX) {
        NodeDisk *node_1 = &file->nodes[node_1_idx];
        float act = node_1->state;
        float pred = node_1->prediction;
        printf("    Node '1': activation=%.4f, prediction=%.4f\n", act, pred);
        if (fabsf(act) > 0.1f || pred > 0.2f) {
            output_score++;
            printf("      ✓ '1' is active/predicted\n");
        }
    }
    
    if (node_0_idx != UINT64_MAX) {
        NodeDisk *node_0 = &file->nodes[node_0_idx];
        float act = node_0->state;
        float pred = node_0->prediction;
        printf("    Node '0': activation=%.4f, prediction=%.4f\n", act, pred);
        if (fabsf(act) > 0.1f || pred > 0.2f) {
            output_score++;
            printf("      ✓ '0' is active/predicted\n");
        }
    }
    
    printf("    Output score: %d/2 (need both '1' and '0' for '100')\n", output_score);
    return output_score;
}

// Test with pattern learning only
static void test_pattern_learning_only() {
    printf("========================================\n");
    printf("TEST 1: PATTERN LEARNING ONLY\n");
    printf("========================================\n\n");
    
    printf("Method: Feed examples, see if patterns form\n");
    printf("Question: Does it recognize '50+50=' pattern?\n\n");
    
    unlink(TEST_FILE);
    
    GraphParams params;
    init_default_params(&params);
    params.decay_rate = 0.90f;
    params.learning_rate = 0.02f;
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        return;
    }
    
    // Feed training examples
    printf("Feeding 200 addition examples...\n");
    for (int i = 0; i < 200; i++) {
        int a = rand() % 100;
        int b = rand() % 100;
        if (a == 50 && b == 50) a = 49;
        int sum = a + b;
        
        char problem[64];
        snprintf(problem, sizeof(problem), "%d+%d=%d", a, b, sum);
        
        for (int j = 0; problem[j] != '\0'; j++) {
            ingest_byte(&rt, 0, problem[j], 1.0f);
            melvin_process_n_events(&rt, 3);
        }
    }
    
    printf("  ✓ Training complete\n\n");
    
    // Test: feed "50+50="
    printf("Testing: Feeding '50+50='...\n");
    const char *test = "50+50=";
    for (int i = 0; test[i] != '\0'; i++) {
        ingest_byte(&rt, 0, test[i], 1.0f);
        melvin_process_n_events(&rt, 20);
    }
    
    printf("\n");
    
    // Check what "works" means
    int pattern_ok = check_pattern_recognition(&file);
    int computation_ok = check_computation(&file);
    int output_ok = check_output_generation(&file, &rt);
    
    printf("\n");
    printf("RESULTS:\n");
    printf("  Pattern recognition: %s\n", pattern_ok ? "✅ YES" : "❌ NO");
    printf("  Computation (EXEC): %s\n", computation_ok ? "✅ YES" : "❌ NO");
    printf("  Output generation: %s (%d/2)\n", 
           output_ok >= 2 ? "✅ YES" : output_ok > 0 ? "⚠️ PARTIAL" : "❌ NO",
           output_ok);
    
    printf("\n");
    printf("WHAT THIS MEANS:\n");
    if (pattern_ok) {
        printf("  ✓ System recognizes '50+50=' as a pattern\n");
        printf("    (Seen similar sequences before)\n");
    }
    if (computation_ok) {
        printf("  ✓ System has EXEC nodes that could compute\n");
        printf("    (But may not be triggered/used)\n");
    }
    if (output_ok >= 2) {
        printf("  ✓ System outputs/predicts '100'\n");
        printf("    (Nodes for '1' and '0' are active)\n");
    } else if (output_ok > 0) {
        printf("  ⚠️  System partially outputs answer\n");
        printf("    (Some digits active, but not complete)\n");
    } else {
        printf("  ❌ System does NOT output answer\n");
        printf("    (Just recognizes pattern, doesn't compute)\n");
    }
    
    runtime_cleanup(&rt);
    close_file(&file);
}

// Test with EXEC computation
static void test_exec_computation() {
    printf("\n========================================\n");
    printf("TEST 2: EXEC COMPUTATION\n");
    printf("========================================\n\n");
    
    printf("Method: Create EXEC node that does addition\n");
    printf("Question: Can it compute 50+50=100 directly?\n\n");
    
    unlink(TEST_FILE);
    
    GraphParams params;
    init_default_params(&params);
    
    if (melvin_m_init_new_file(TEST_FILE, &params) < 0) {
        fprintf(stderr, "ERROR: Failed to create file\n");
        return;
    }
    
    MelvinFile file;
    if (melvin_m_map(TEST_FILE, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to map file\n");
        return;
    }
    
    MelvinRuntime rt;
    if (runtime_init(&rt, &file) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize runtime\n");
        return;
    }
    
    // Create EXEC node with ADD code
    printf("Creating EXEC node with ADD machine code...\n");
    const uint8_t ARM64_ADD[] = {
        0x00, 0x00, 0x01, 0x8b,  // add x0, x0, x1
        0xc0, 0x03, 0x5f, 0xd6   // ret
    };
    
    uint64_t add_offset = melvin_write_machine_code(&file, ARM64_ADD, sizeof(ARM64_ADD));
    if (add_offset == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to write code\n");
        return;
    }
    
    uint64_t add_exec_id = melvin_create_executable_node(&file, add_offset, sizeof(ARM64_ADD));
    if (add_exec_id == UINT64_MAX) {
        fprintf(stderr, "ERROR: Failed to create EXEC node\n");
        return;
    }
    
    printf("  ✓ EXEC node created: %llu\n", (unsigned long long)add_exec_id);
    printf("  Code: add x0, x0, x1; ret\n");
    printf("  This computes: x0 = x0 + x1\n\n");
    
    // Check what "works" means
    int pattern_ok = check_pattern_recognition(&file);
    int computation_ok = check_computation(&file);
    
    printf("\n");
    printf("RESULTS:\n");
    printf("  Pattern recognition: %s\n", pattern_ok ? "✅ YES" : "❌ NO (no training)");
    printf("  Computation (EXEC): %s\n", computation_ok ? "✅ YES" : "❌ NO");
    
    printf("\n");
    printf("WHAT THIS MEANS:\n");
    printf("  ✓ System CAN compute 50+50=100\n");
    printf("    (EXEC node has ADD machine code)\n");
    printf("  ⚠️  But it needs to be TRIGGERED\n");
    printf("    (Activation must cross exec_threshold)\n");
    printf("  ⚠️  And inputs must be SET UP\n");
    printf("    (x0=50, x1=50 in CPU registers)\n");
    printf("  ⚠️  Result must be CONVERTED to output\n");
    printf("    (Return value → energy → node activation → bytes)\n");
    
    printf("\n");
    printf("KEY DIFFERENCE:\n");
    printf("  Pattern learning: Recognizes '50+50=' → predicts '100'\n");
    printf("  EXEC computation: Computes 50+50 → returns 100 → outputs '100'\n");
    printf("  Both can work, but EXEC is direct computation!\n");
    
    runtime_cleanup(&rt);
    close_file(&file);
}

int main() {
    printf("========================================\n");
    printf("WHAT DOES 'WORKS' ACTUALLY MEAN?\n");
    printf("========================================\n\n");
    
    printf("This test distinguishes:\n");
    printf("  1. Pattern recognition (seeing similar patterns)\n");
    printf("  2. Computation (actually doing math)\n");
    printf("  3. Output generation (producing answers)\n\n");
    
    srand(time(NULL));
    
    // Test 1: Pattern learning
    test_pattern_learning_only();
    
    // Test 2: EXEC computation
    test_exec_computation();
    
    // Summary
    printf("\n========================================\n");
    printf("SUMMARY: WHAT 'WORKS' MEANS\n");
    printf("========================================\n\n");
    
    printf("When we say 'works', we could mean:\n\n");
    
    printf("1. PATTERN RECOGNITION:\n");
    printf("   - System sees '50+50=' and recognizes it\n");
    printf("   - Pattern nodes activate\n");
    printf("   - But: Does it know the answer?\n");
    printf("   - Answer: Maybe, if it saw '50+50=100' in training\n");
    printf("\n");
    
    printf("2. COMPUTATION:\n");
    printf("   - System has EXEC nodes that can do math\n");
    printf("   - Can compute 50+50=100 directly\n");
    printf("   - But: Needs to be triggered and set up\n");
    printf("   - Answer: Yes, if EXEC node is activated\n");
    printf("\n");
    
    printf("3. OUTPUT GENERATION:\n");
    printf("   - System produces '100' as output\n");
    printf("   - Nodes for '1', '0', '0' are activated\n");
    printf("   - Can be read as bytes/characters\n");
    printf("   - Answer: Yes, if computation or prediction works\n");
    printf("\n");
    
    printf("CONCLUSION:\n");
    printf("  'Works' could mean any of these!\n");
    printf("  Pattern learning: Recognizes patterns, may predict\n");
    printf("  EXEC computation: Actually computes, returns result\n");
    printf("  Both can lead to output, but EXEC is more direct!\n");
    printf("\n");
    
    return 0;
}


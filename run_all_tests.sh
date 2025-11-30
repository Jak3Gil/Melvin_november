#!/bin/bash
# Run all tests on melvin.m with instincts embedded

set -e

MELVIN_FILE="melvin_with_instincts.m"
TEST_RESULTS="test_results_$(date +%Y%m%d_%H%M%S).txt"

echo "=== RUNNING ALL TESTS ON MELVIN.M WITH INSTINCTS ===" > "$TEST_RESULTS"
echo "Started: $(date)" >> "$TEST_RESULTS"
echo "" >> "$TEST_RESULTS"

# Step 1: Create fresh melvin.m with instincts
echo "=== STEP 1: Creating fresh melvin.m with instincts ===" | tee -a "$TEST_RESULTS"
echo ""

# We'll need to create this using a simple C program since test_with_instincts has compilation issues
cat > create_melvin_with_instincts.c << 'EOFCREATE'
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include "melvin.c"
#include "instincts.c"

int main(int argc, char **argv) {
    const char *file_path = argc > 1 ? argv[1] : "melvin_with_instincts.m";
    
    unlink(file_path);
    
    GraphParams params = {0};
    params.decay_rate = 0.95f;
    params.exec_threshold = 0.75f;
    params.learning_rate = 0.015f;
    params.reward_lambda = 0.1f;
    params.energy_cost_mu = 0.01f;
    params.homeostasis_target = 0.5f;
    params.homeostasis_strength = 0.01f;
    params.weight_decay = 0.01f;
    params.global_energy_budget = 10000.0f;
    
    if (melvin_m_init_new_file(file_path, &params) < 0) {
        fprintf(stderr, "Failed to create file\n");
        return 1;
    }
    
    MelvinFile file;
    if (melvin_m_map(file_path, &file) < 0) {
        fprintf(stderr, "Failed to map file\n");
        return 1;
    }
    
    fprintf(stderr, "Injecting instincts...\n");
    melvin_inject_instincts(&file);
    
    uint64_t nodes = melvin_get_num_nodes(&file);
    uint64_t edges = melvin_get_num_edges(&file);
    fprintf(stderr, "Created melvin.m with instincts:\n");
    fprintf(stderr, "  Nodes: %llu\n", (unsigned long long)nodes);
    fprintf(stderr, "  Edges: %llu\n", (unsigned long long)edges);
    
    melvin_m_sync(&file);
    close_file(&file);
    
    fprintf(stderr, "✓ melvin.m created successfully\n");
    return 0;
}
EOFCREATE

# Try to compile and run (may fail due to type issues, but we'll try)
if gcc -std=c11 -O0 -DMELVIN_DIAGNOSTIC_MODE create_melvin_with_instincts.c -lm -o create_melvin_with_instincts 2>/dev/null; then
    ./create_melvin_with_instincts "$MELVIN_FILE" 2>&1 | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
else
    echo "⚠ Could not compile create_melvin_with_instincts (type conflicts)" | tee -a "$TEST_RESULTS"
    echo "  Using existing melvin.m files from previous tests" | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
fi

# Step 2: Run learning kernel test
echo "=== STEP 2: Learning Kernel Test ===" | tee -a "$TEST_RESULTS"
echo "" | tee -a "$TEST_RESULTS"
if [ -f test_learning_kernel ]; then
    ./test_learning_kernel 2>&1 | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
else
    echo "⚠ test_learning_kernel not found" | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
fi

# Step 3: Run evolution diagnostic test
echo "=== STEP 3: Evolution Diagnostic Test ===" | tee -a "$TEST_RESULTS"
echo "" | tee -a "$TEST_RESULTS"
if [ -f test_evolution_diagnostic ]; then
    # Clean up old test files
    rm -f evolution_compounding.m evolution_multistep.m evolution_learning.m
    
    echo "Running phase 1..." | tee -a "$TEST_RESULTS"
    ./test_evolution_diagnostic --phase=1 2>&1 | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
    
    echo "Running phase 2..." | tee -a "$TEST_RESULTS"
    ./test_evolution_diagnostic --phase=2 2>&1 | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
else
    echo "⚠ test_evolution_diagnostic not found" | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
fi

# Step 4: Run universal laws test
echo "=== STEP 4: Universal Laws Test ===" | tee -a "$TEST_RESULTS"
echo "" | tee -a "$TEST_RESULTS"
if [ -f test_universal_laws ]; then
    # Clean up old test files
    rm -f test_*.m
    
    ./test_universal_laws 2>&1 | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
else
    echo "⚠ test_universal_laws not found" | tee -a "$TEST_RESULTS"
    echo "" | tee -a "$TEST_RESULTS"
fi

# Step 5: Summary
echo "=== TEST SUMMARY ===" | tee -a "$TEST_RESULTS"
echo "Completed: $(date)" | tee -a "$TEST_RESULTS"
echo "" | tee -a "$TEST_RESULTS"
echo "Results saved to: $TEST_RESULTS" | tee -a "$TEST_RESULTS"

echo ""
echo "✓ All tests completed!"
echo "  Results: $TEST_RESULTS"


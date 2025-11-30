# Test Failure Diagnostics

## Executive Summary

**Root Cause:** Tests create NEW empty files instead of using `melvin.m` with instinct patterns.

**Impact:** 11/30 tests fail because:
- No instinct patterns available
- No param nodes
- No scaffolding for learning/reasoning

## Detailed Findings

### 1. Test 0.5.1: Learning Only Prediction Error
**Status:** FAILED  
**Reason:** "No learning occurred"

**Root Cause:**
- Tests don't set `prediction_error` on nodes
- Without epsilon, learning has no signal
- Learning function exists but has nothing to learn from

**Evidence:**
- `melvin.m` has 1 node with prediction_error
- But fresh test files have 0 nodes with prediction_error
- Eligibility traces are 0 (no learning activity)

**Fix:**
```c
// Tests must set prediction_error:
melvin_set_epsilon_for_node(&rt, node_id, target_value);
// OR
melvin_compute_epsilon_from_observation(&rt);
```

### 2. Test 0.8.1: Params as Nodes
**Status:** FAILED  
**Reason:** "Param nodes not created"

**Root Cause:**
- Tests create NEW files (0 nodes, 0 edges)
- `melvin.m` HAS param nodes (8/8 found in diagnostics)
- But tests don't use `melvin.m`

**Evidence:**
- `melvin.m`: Has PARAM_DECAY, PARAM_BIAS, LAW_LR_BASE, etc.
- Fresh files: 0 nodes, 0 edges

**Fix:**
```c
// Option 1: Start from melvin.m
system("cp melvin.m test_file.m");

// Option 2: Inject instincts
melvin_inject_instincts(&file);
```

### 3. Test HARD-1: Pattern Prediction
**Status:** FAILED  
**Reason:** "Pattern nodes not created"

**Root Cause:**
- Tests create fresh files
- Pattern nodes (A, B, C) created by byte ingestion
- But byte ingestion may not happen or nodes not created

**Evidence:**
- `melvin.m`: Has instinct patterns (EXEC:HUB, MATH:IN_A, etc.)
- Fresh files: No pattern nodes
- Pattern nodes A, B, C missing in fresh files

**Fix:**
- Start from `melvin.m` (has instinct patterns)
- OR ensure byte ingestion creates nodes
- OR use instinct patterns as starting point

### 4. Test HARD-5: Multi-Step Reasoning
**Status:** FAILED  
**Reason:** "Failed to learn multi-step chain"

**Root Cause:**
- Chain nodes (A, B, C, D) don't exist
- Chain edges (A->B, B->C, C->D) don't exist
- Even if they exist, weights may be too weak
- Learning may not have enough time

**Evidence:**
- `melvin.m`: Has instinct patterns that could help
- Fresh files: No chain nodes, no chain edges

**Fix:**
- Start from `melvin.m` with instinct scaffolding
- Ensure byte ingestion creates A, B, C, D nodes
- Set prediction_error to drive learning
- Increase training iterations

### 5. Test HARD-6: Adaptive Parameters
**Status:** FAILED  
**Reason:** "Parameters not adapting"

**Root Cause:**
- Law nodes exist in `melvin.m` but tests create fresh files
- `melvin_sync_params_from_nodes()` may not be called
- Law nodes may not be connected/wired

**Evidence:**
- `melvin.m`: Has LAW_LR_BASE, LAW_W_LIMIT, etc.
- Fresh files: No law nodes

**Fix:**
- Start from `melvin.m` (has law nodes)
- Verify `melvin_sync_params_from_nodes()` is called
- Ensure law nodes are wired to update `g_params`

## Critical Discovery

**The tests are NOT using `melvin.m`!**

Every test does:
```c
melvin_m_init_new_file(file_path, &params);  // Creates EMPTY file
```

Instead of:
```c
system("cp melvin.m test_file.m");  // Start with instincts
```

## Statistics

**melvin.m (with instincts):**
- Nodes: 117
- Edges: 122
- Param nodes: 8/8 found
- Instinct patterns: 11/11 found

**Fresh test files:**
- Nodes: 0
- Edges: 0
- Param nodes: 0/8
- Instinct patterns: 0/11

## Recommended Fixes

### Immediate (Quick Wins):
1. **Modify tests to start from melvin.m:**
   ```c
   // Before each test:
   system("cp melvin.m test_file.m");
   // Then map test_file.m instead of creating new
   ```

2. **Set prediction_error in tests:**
   ```c
   // After forward pass:
   melvin_set_epsilon_for_node(&rt, target_node_id, target_value);
   ```

### Long-term (Proper Fix):
1. **Create test helper function:**
   ```c
   MelvinFile* create_test_file_from_melvin(const char *test_name) {
       char cmd[256];
       snprintf(cmd, sizeof(cmd), "cp melvin.m %s.m", test_name);
       system(cmd);
       // Map and return
   }
   ```

2. **Ensure all tests use this helper**

3. **Add epsilon computation to test framework**

## Conclusion

The test failures are NOT due to broken physics. They're due to:
1. Tests creating empty files instead of using `melvin.m`
2. Tests not setting `prediction_error`
3. Tests not giving enough time for learning

**With `melvin.m` and proper epsilon setting, most tests should pass.**

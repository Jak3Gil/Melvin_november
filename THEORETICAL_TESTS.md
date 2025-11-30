# Theoretical Capability Tests

This document describes the tests created to validate theoretical capabilities described in `MASTER_ARCHITECTURE.md`.

## Overview

These tests probe the boundaries of what's theoretically possible vs. what actually works in the current implementation. Each test targets a specific theoretical capability from Section 18 of the architecture document.

## Test Suite

### 1. `test_self_modify.c` - Self-Modifying Code

**Theoretical Capability:** Section 18.1 - Self-Modifying Code Evolution

**What it tests:**
- Can EXEC nodes write new machine code to the blob?
- Does `melvin_write_machine_code()` work when called from within EXEC nodes?
- Can newly written code be executed?

**How it works:**
1. Creates an EXEC node with a "writer" function
2. Writer function calls `melvin_write_machine_code()` to write a simple return function
3. Checks if blob size increased (new code written)
4. Attempts to execute the newly written code

**Expected Result:** EXEC nodes should be able to write code. This is foundational for code evolution.

**Status:** ⚠️ Needs testing

---

### 2. `test_code_evolution.c` - Code Evolution

**Theoretical Capability:** Section 18.1 - Mutation and Recombination

**What it tests:**
- Can EXEC nodes modify other EXEC nodes' code?
- Does code mutation work (changing bytes in existing code)?
- Can one EXEC node mutate another EXEC node's machine code?

**How it works:**
1. Creates a "target" EXEC node with simple code (returns 0x10)
2. Creates a "mutator" EXEC node that modifies the target's code
3. Activates mutator to change target's code
4. Compares original vs modified code

**Expected Result:** EXEC nodes should be able to modify other nodes' code. This enables mutation in evolution.

**Status:** ⚠️ Needs testing

---

### 3. `test_auto_exec.c` - Automatic EXEC Creation

**Theoretical Capability:** Section 18.7 - Automatic EXEC Creation from Learned Patterns

**What it tests:**
- Can high-energy patterns trigger automatic EXEC node creation?
- Does the system detect patterns and create EXEC nodes for them?
- Can patterns become "actions" automatically?

**How it works:**
1. Ingests repeated pattern "ABC" to create high-energy nodes
2. Creates an "auto-creator" EXEC node that checks for high-energy patterns
3. Auto-creator creates new EXEC nodes when patterns detected
4. Checks if new EXEC node was created

**Expected Result:** High-energy patterns should trigger EXEC creation. This enables patterns to become actions.

**Status:** ⚠️ Needs testing

---

### 4. `test_meta_learning.c` - Meta-Learning

**Theoretical Capability:** Section 18.8 - Meta-Learning and Self-Optimization

**What it tests:**
- Can EXEC nodes modify physics parameters (decay_rate, learning_rate, exec_threshold)?
- Does the system optimize its own parameters?
- Can parameters be changed at runtime?

**How it works:**
1. Creates a "meta-optimizer" EXEC node
2. Optimizer reads current performance (prediction error)
3. Optimizer modifies physics parameters in GraphHeaderDisk
4. Checks if parameters actually changed

**Expected Result:** EXEC nodes should be able to modify physics parameters. This enables self-optimization.

**Status:** ⚠️ Needs testing

---

### 5. `test_emergent_algo.c` - Emergent Algorithm Formation

**Theoretical Capability:** Section 18.2 - Emergent Algorithm Formation

**What it tests:**
- Do repeated patterns form strong edge sequences?
- Are patterns "reusable" (activating start predicts end)?
- Can patterns become functional abstractions?

**How it works:**
1. Ingests repeated patterns "ABC" and "XYZ" many times
2. Analyzes edge weights to see if patterns formed
3. Tests reusability by activating A and checking if C activates
4. Checks if energy flows through pattern chains

**Expected Result:** Repeated patterns should form strong edges. Patterns should be reusable (energy flows through them).

**Status:** ⚠️ Needs testing

---

## Running the Tests

### Individual Tests

```bash
# Compile and run each test
gcc -o test_self_modify test_self_modify.c -lm -std=c11
./test_self_modify

gcc -o test_code_evolution test_code_evolution.c -lm -std=c11
./test_code_evolution

gcc -o test_auto_exec test_auto_exec.c -lm -std=c11
./test_auto_exec

gcc -o test_meta_learning test_meta_learning.c -lm -std=c11
./test_meta_learning

gcc -o test_emergent_algo test_emergent_algo.c -lm -std=c11
./test_emergent_algo
```

### Run All Tests

```bash
./run_theoretical_tests.sh
```

This script compiles and runs all tests, providing a summary of what works and what doesn't.

---

## What We're Learning

These tests help us understand:

1. **What's Actually Implemented**
   - Which theoretical capabilities already work
   - What needs to be implemented
   - What limitations exist

2. **Architecture Validation**
   - Does the architecture support these capabilities?
   - Are there missing mechanisms?
   - Do we need new functions/APIs?

3. **Gap Analysis**
   - What's theoretical vs. what's proven
   - What needs more work
   - What might not be possible

---

## Expected Outcomes

### Best Case
All tests pass - all theoretical capabilities work!

### Realistic Case
Some tests pass, some fail - we learn what works and what needs implementation.

### Worst Case
Most tests fail - architecture may need adjustments, or capabilities need more implementation.

---

## Next Steps

After running tests:

1. **Document Results**
   - Update MASTER_ARCHITECTURE.md with test results
   - Mark what's proven vs. theoretical
   - Note any limitations discovered

2. **Fix Issues**
   - Implement missing capabilities
   - Fix bugs discovered
   - Add missing APIs

3. **Expand Tests**
   - Add more edge cases
   - Test failure modes
   - Test long-term behavior

---

## Notes

- These tests are **exploratory** - they probe what's possible
- Some tests may fail due to missing implementation, not architecture flaws
- Results help guide future development priorities
- Tests should be run on both x86_64 and aarch64 architectures


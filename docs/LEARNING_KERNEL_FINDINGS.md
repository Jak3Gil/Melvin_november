# Learning Kernel Test Findings

## Test Status: ✓ RUNNING ON JETSON

The learning kernel test is now successfully running on the Jetson via USB connection.

## Critical Finding: Learning Sign is Backwards!

### Test Results
- **Initial weight**: 0.200000
- **Final weight**: 0.057708  
- **Change**: -0.142292 (NEGATIVE - weight DECREASED!)

### Root Cause

The learning formula in `strengthen_edges_with_prediction_and_reward()` is:

```c
float fitness_signal = -epsilon + lambda * reward;
float delta_w = learning_rate * fitness_signal * e->eligibility * use_factor;
```

With:
- `prediction_error (epsilon) = 1.0` (predicted too low)
- `reward = 0.0`
- `fitness_signal = -1.0 + 0.0 = -1.0`
- `delta_w = 0.1 * (-1.0) * 0.95 * 0.999 = -0.095` (NEGATIVE!)

### The Problem

When `prediction_error > 0` (meaning we predicted too LOW, actual > predicted), we want to **INCREASE** the weight to make stronger predictions. But the current formula makes `fitness_signal` negative, which **DECREASES** the weight.

### The Fix

The sign convention is backwards. Options:

1. **Remove the negative sign**:
   ```c
   float fitness_signal = epsilon + lambda * reward;  // Remove the minus
   ```

2. **OR flip the interpretation of prediction_error**:
   - If `prediction_error = actual - predicted`, then positive means predicted too low
   - But the learning rule treats it as "predicted too high"
   - Need to verify what `prediction_error` actually represents in the codebase

### Evidence

The kernel test shows:
- Learning updates ARE happening (delta_w is nonzero)
- Eligibility traces ARE working (decaying from 1.0 → 0.95 → 0.90...)
- The math is correct, just the SIGN is wrong
- Weight saturation (tanh) is working (weights stay in bounds)

### Next Steps

1. Verify what `prediction_error` actually represents in the codebase
2. Fix the sign in the learning formula
3. Re-run the kernel test to verify weights now increase
4. Re-run evolution tests to see if learning compounds correctly

## Test Infrastructure

- ✓ Test compiles and runs on Jetson
- ✓ Debug instrumentation working
- ✓ Can isolate learning kernel from rest of system
- ✓ Can track weight changes over iterations

## Files

- `test_learning_kernel.c` - Minimal test that proves learning kernel works
- `melvin.c` - Contains `strengthen_edges_with_prediction_and_reward()` with instrumentation


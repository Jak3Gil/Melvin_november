# Learning Compounding Patch - Summary

## Problem Identified

Edge weights were stuck at 0.2 and not compounding over time, even though:
- Edges were being created correctly
- Traces were increasing (showing usage)
- Multi-step reasoning was working
- Files were persisting correctly

## Root Causes Found

1. **Main learning function never called**: `strengthen_edges_with_prediction_and_reward()` was defined but **never invoked**
2. **Weight clamps too low**: Learning in `message_passing()` was clamped to [-1.0, 1.0]
3. **Learning rate too low**: Default was ~0.005, too slow to see compounding
4. **No diagnostics**: Couldn't track weight changes over time

## Changes Made

### 1. Documented Weight Learning Mechanism

**Location**: `melvin.c` lines ~3286-3300

Added comprehensive comments explaining:
- **WHEN**: Called during `message_passing()`, which runs on every event
- **WHAT**: Free-Energy learning rule: `Δw_ij = −η · ε_i · a_j`
- **TYPE**: Error-driven Hebbian learning
- **INPUTS**: prediction_error, presynaptic activation, learning_rate
- **UPDATE**: Applies to existing edges, not just new ones

### 2. Removed Hidden Caps

**Location**: `melvin.c` lines ~3295-3300 and ~3153-3158

**Changed**:
- Weight clamp from `[-1.0, 1.0]` → `[-10.0, 10.0]`
- This allows weights to compound naturally through learning
- Safety ceiling only, NOT a learning target

**Also fixed**: Reward-based learning clamp (line ~3153) from 1.0 to 10.0

### 3. Fixed Missing Learning Function Call

**Location**: `melvin.c` line ~5041 (EV_HOMEOSTASIS_SWEEP case)

**Added**:
```c
strengthen_edges_with_prediction_and_reward(rt);
```

**Why critical**: This function:
- Applies learning to ALL edges with eligibility > 0.001
- Uses fitness signal: `-epsilon + lambda * reward`
- Allows weights up to 10.0 (not capped at 1.0)
- Also applies trace-based strengthening

**Without this**: Learning was limited to weak updates in message_passing() only

### 4. Increased Learning Rate for Tests

**Location**: 
- `melvin.c` line ~4458: Increased max learning rate from 0.01 to 0.02
- `test_evolution_diagnostic.c`: Set test learning rate to 0.02 (2x default)

**Why**: Allows weights to compound visibly on human timescales

### 5. Added Diagnostic Logging

**Location**: `melvin.c` line ~4467

**Added**: Learning rate logging when it changes (under `#ifdef MELVIN_DIAGNOSTIC_MODE`)

### 6. Enhanced Evolution Diagnostic Test

**Location**: `test_evolution_diagnostic.c`

**Added**:
- `log_edge_state()`: Helper to log edge state (weight, trace, eligibility, usage)
- Before/after tracking: Logs edge state before and after training
- Compounding check: Warns if trace increases but weight doesn't
- Higher learning rate: 0.02 for tests (was 0.01)

### 7. Added Multi-Run Compounding Test

**Location**: `test_evolution_diagnostic.c` function `test_multi_run_compounding()`

**Usage**:
```bash
# Phase 1
./test_evolution_diagnostic --phase=1

# Phase 2 (in separate process run)
./test_evolution_diagnostic --phase=2
```

**Purpose**: Proves learning compounds across process boundaries:
- Phase 1: Trains and saves weight
- Phase 2: Loads file, trains more, checks weight > phase 1

## Expected Results

After these changes:

1. **Weights should compound**: Edge weights should increase over multiple training runs
2. **No 0.2 cap**: Weights can grow beyond 0.2 (up to 10.0 safety limit)
3. **Learning on existing edges**: All edges get learning updates during homeostasis sweeps
4. **Visible progress**: With 0.02 learning rate, changes should be visible in tests
5. **Diagnostics show progress**: Logs clearly show weight increases

## Testing

Run the diagnostic test:
```bash
# Compile with diagnostics
gcc -std=c11 -Wall -Wextra -O0 -lm -DMELVIN_DIAGNOSTIC_MODE \
    -o test_evolution_diagnostic test_evolution_diagnostic.c

# Run evolution tests (multiple runs in same process)
./test_evolution_diagnostic

# Run multi-run test (across process boundaries)
./test_evolution_diagnostic --phase=1
./test_evolution_diagnostic --phase=2
```

**Expected output**:
- `[EDGE_EVOLUTION]` logs showing weight increases
- `SUCCESS: Weight increased with trace` messages
- Phase 2 weight > Phase 1 weight

## Files Modified

1. `melvin.c`:
   - Added learning mechanism documentation
   - Removed 1.0 weight clamps (changed to 10.0)
   - Added call to `strengthen_edges_with_prediction_and_reward()`
   - Increased max learning rate to 0.02
   - Added diagnostic logging

2. `test_evolution_diagnostic.c`:
   - Added `log_edge_state()` helper
   - Enhanced `test_evolution_1_compounding_learning()` with before/after tracking
   - Added `test_multi_run_compounding()` for cross-process testing
   - Increased test learning rate to 0.02

## Verification Checklist

- [x] Learning mechanism documented
- [x] 0.2 cap removed (no hard limit at 0.2)
- [x] 1.0 clamp increased to 10.0
- [x] `strengthen_edges_with_prediction_and_reward()` now called
- [x] Learning rate increased for tests
- [x] Diagnostics added
- [x] Multi-run test added
- [x] Code compiles without errors
- [ ] Tests run and show compounding (needs verification)

## Next Steps

1. Run `./test_evolution_diagnostic` and verify weights increase
2. Run multi-run test and verify cross-process compounding
3. If weights still don't compound, investigate:
   - Eligibility trace decay rate
   - Prediction error calculation
   - Homeostasis sweep frequency


# Explicit Epsilon Patch - Summary

## Goal

Make learning actually move weights in the evolution test by:
1. Setting explicit `prediction_error` values (target - activation)
2. Calling `strengthen_edges_with_prediction_and_reward()` explicitly
3. Adding detailed logging to verify epsilon and weight updates

## Problem

The kernel test proved learning works when `epsilon` is manually set, but the evolution test showed weights stuck at 0.2. The issue was that `prediction_error` was effectively zero in the evolution path - there was no explicit target signal.

## Changes Made

### 1. Modified `test_evolution_diagnostic.c`

**Test 1 (ABC pattern):**
- Set explicit targets: `target_A = 0.0f`, `target_B = 0.0f`, `target_C = 1.0f`
- After each forward pass, compute: `prediction_error = target - activation`
- Call `strengthen_edges_with_prediction_and_reward(&rt)` explicitly after setting errors
- Add `[EPS_STATS]` logging every 100 iterations
- Add `[AB_LEARN]` logging every 50 iterations showing epsilon, eligibility, learning rate, weight, trace

**Test 2 (A->B->C->D chain):**
- Set explicit targets: `target_A = 0.0f`, `target_B = 0.0f`, `target_C = 0.0f`, `target_D = 1.0f`
- Same pattern: compute errors, call learning explicitly, log stats

**Test 3 (X->Y pattern):**
- Set explicit targets: `target_X = 0.0f`, `target_Y = 1.0f`
- Same pattern

### 2. Made `find_node_index_by_id()` non-static

Changed from `static` to public function so test code can access it to find node indices.

### 3. Added Forward Declarations

Added `extern` declarations for `strengthen_edges_with_prediction_and_reward()` and `find_node_index_by_id()` in test file.

## Expected Behavior

When running the test, you should now see:

1. **Nonzero epsilon values** in `[EPS_STATS]` output:
   ```
   [EPS_STATS] iter=100 eps_A=-0.123456 eps_B=-0.234567 eps_C=0.654321
   ```
   - For A and B: negative epsilon (they're active but target is 0)
   - For C/D: positive epsilon (they should be active, target is 1)

2. **Weight updates** in `[AB_LEARN]` output:
   ```
   [AB_LEARN] iter=50 eps=0.123456 elig=0.456789 lr=0.010448 weight=0.200123 trace=50.00
   ```
   - `eps` should be nonzero
   - `weight` should increase from 0.2 over iterations

3. **Final weight > 0.2** after training, especially if trace increased significantly

## Next Steps

1. **Run the test** and verify epsilon values are nonzero
2. **Check if weights move** - if they still don't, check:
   - Is `eligibility` nonzero? (should be ~0.4-0.5 for active edges)
   - Is `learning_rate` reasonable? (should be ~0.01-0.02)
   - Are there other writers to `edge->weight` that might be resetting it?

3. **If weights still don't move**, temporarily disable non-learning writers to `edge->weight`:
   - Normalization passes
   - Homeostasis adjustments
   - Pruning adjustments
   - Any other maintenance code

4. **Once learning works**, generalize the error signal beyond explicit targets:
   - Use prediction-based error (prediction vs actual)
   - Use reward signals
   - Let the graph learn its own targets

## Files Modified

- `test_evolution_diagnostic.c`: Added explicit epsilon setup and learning calls
- `melvin.c`: Made `find_node_index_by_id()` non-static

## Verification

Code compiles successfully. Ready to test on Jetson to verify:
- Epsilon values are nonzero
- Weights increase from 0.2
- Learning compounds over iterations


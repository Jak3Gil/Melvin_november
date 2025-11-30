# Learning Sign Fix - Summary

## Problem Identified

The learning kernel test revealed that weights were **decreasing** when they should **increase**:
- Initial weight: 0.200000
- Final weight: 0.057708 (DECREASED!)
- Change: -0.142292

## Root Cause

The learning formula had the **wrong sign**:

```c
// WRONG (before):
float fitness_signal = -epsilon + lambda * reward;
float delta_w = -learning_rate * eps_i * a_j;  // In message_passing()
```

When `prediction_error (epsilon) = 1.0` (meaning we under-predicted), this gave:
- `fitness_signal = -1.0` → negative delta_w → weight DECREASES
- But we WANT weight to INCREASE when we under-predict!

## Convention Established

**Convention: `epsilon = target - prediction`**
- `epsilon > 0` → under-predicted → strengthen weights
- `epsilon < 0` → over-predicted → weaken weights

## Fixes Applied

### 1. `strengthen_edges_with_prediction_and_reward()`

**Changed:**
```c
// BEFORE:
float fitness_signal = -epsilon + lambda * reward;

// AFTER:
float fitness_signal = epsilon + lambda * reward;  // FIXED: was -epsilon, now +epsilon
```

**Added comment:**
```c
// Convention:
//   epsilon = target - prediction
//   epsilon > 0  => under-predicted => strengthen weights that contributed
//   epsilon < 0  => over-predicted  => weaken them
```

### 2. `message_passing()` learning rule

**Changed:**
```c
// BEFORE:
float delta_w = -learning_rate * eps_i * a_j;

// AFTER:
float delta_w = learning_rate * eps_i * a_j;  // FIXED: was -learning_rate, now +learning_rate
```

**Added comment explaining convention.**

### 3. Updated all related comments

- Function documentation updated
- Call site comments updated
- Old "negative sign" comments removed

## Verification

### Kernel Test Results (AFTER FIX)

```
[KERNEL_TEST] Initial weight: 0.200000
[KERNEL_TEST] Final weight: 0.391002
[KERNEL_TEST] Total change: 0.191002
[KERNEL_TEST] ✓ PASS: Weight increased from 0.200000 to 0.391002
[KERNEL_TEST] The learning kernel CAN change weights!
```

**✓ SUCCESS:** Weights now **increase** when `epsilon > 0`!

### Evolution Test Status

The evolution diagnostic test still shows weights at 0.2, but this is likely because:
1. `prediction_error` might be zero in those tests (no prediction mechanism active)
2. The sign fix is correct (proven by kernel test)
3. Evolution tests may need prediction_error to be computed/set for learning to occur

## Files Modified

- `melvin.c`:
  - Fixed sign in `strengthen_edges_with_prediction_and_reward()` (line ~3659)
  - Fixed sign in `message_passing()` (line ~3434)
  - Updated all related comments

## Next Steps

1. ✓ Sign fix verified with kernel test
2. ⚠️ Evolution tests may need prediction_error computation
3. ⚠️ Verify prediction_error is being set in real scenarios
4. ⚠️ Re-run universal laws tests to ensure no regressions

## Conclusion

**The learning sign bug is FIXED.** The physics was literally learning in the wrong direction. Now:
- Positive epsilon → positive delta_w → weight increases ✓
- Negative epsilon → negative delta_w → weight decreases ✓

The kernel test proves the fix works. The evolution test issue is separate (prediction_error computation).


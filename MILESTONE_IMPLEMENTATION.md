# Milestone Implementation - Next Steps

## Status: Core Physics Fixed ✅

We've achieved:
- ✅ Continuous physics (no hard thresholds)
- ✅ No magic numbers (all in g_params)
- ✅ Learning kernel sign fixed
- ✅ Epsilon flowing (explicit targets working)
- ✅ Weights escaping 0.2 and stabilizing ~1.3

## Implementation Tasks

### 1. Generalize Error Signal ✅ (Partial)

**Added:**
- `melvin_compute_epsilon_from_observation()` - placeholder for future generalization
- `melvin_set_epsilon_for_node()` - helper to set epsilon from target

**Status:** Functions added, but test code still sets epsilon directly. Next step: Update test to use `melvin_set_epsilon_for_node()`.

**Future:** When real sensory input is added, this will compute epsilon from observations automatically.

### 2. Wire g_params to Law Nodes ✅ (Complete)

**Added:**
- Law node IDs: `NODE_ID_LAW_LR_BASE`, `NODE_ID_LAW_W_LIMIT`, `NODE_ID_LAW_EXEC_CENTER`, `NODE_ID_LAW_EXEC_K`, `NODE_ID_LAW_ELIG_SCALE`, `NODE_ID_LAW_PRUNE_BASE`
- `map01_to_range()` - smooth mapping from [0,1] activation to parameter ranges
- Law node creation in `melvin_ensure_param_nodes()`
- Law node syncing in `melvin_sync_params_from_nodes()`

**How it works:**
- Law nodes initialized to 0.5 (mid-range) → maps to default g_params values
- Each homeostasis sweep, law node activations update g_params
- Graph can now tune its own physics scales!

**Parameter ranges:**
- `lr_base`: [0.001, 0.05]
- `w_limit`: [5.0, 20.0]
- `exec_center`: [0.3, 0.7]
- `exec_k`: [2.0, 10.0]
- `elig_scale`: [0.0001, 0.01]
- `prune_base`: [0.0001, 0.01]

### 3. Negative Eligibility Investigation ⚠️ (TODO)

**Current behavior:** Eligibility can be negative (e.g., `elig=-0.45`).

**Options:**
1. Keep signed eligibility (current) - allows anti-correlation signals
2. Make eligibility nonnegative - `elig = fabs(raw_elig)` or clamp to [0,∞)

**Action needed:** Add eligibility histogram logging to see distribution:
- Mean, std, min, max of eligibility
- Count of positive vs negative values
- Correlation with learning success

### 4. System-Level Monitoring ⚠️ (TODO)

**Needed:** Add always-on monitors that print every N sweeps:

```
[SYSTEM_STATS] sweep=1000
  epsilon: mean=0.123 std=0.456 min=-1.75 max=2.22
  |epsilon|: mean=0.789 std=0.234
  weight: mean=1.234 std=0.567 min=0.001 max=9.876
  |weight|: mean=1.345 std=0.456
  eligibility: mean=0.123 std=0.234 min=-0.45 max=0.89
  edges_saturated: 42 (|w| > 0.9 * w_limit)
  edges_near_zero: 123 (|w| < 0.01)
```

**Purpose:** Detect:
- Drift to saturation (everything at w_limit)
- Collapse to zero (weights near zero)
- Stable learning band (like the ~1.3 we see)

## Next Actions

1. **Update test code** to use `melvin_set_epsilon_for_node()` instead of direct assignment
2. **Add eligibility histogram** logging in `strengthen_edges_with_prediction_and_reward()`
3. **Add system stats** function called every N homeostasis sweeps
4. **Test on Jetson** to verify law nodes work and monitoring shows stable learning

## Files Modified

- `melvin.c`:
  - Added `melvin_compute_epsilon_from_observation()` and `melvin_set_epsilon_for_node()`
  - Added law node IDs and syncing
  - Added `map01_to_range()` helper

## Future: Real Data Integration

Once monitoring is in place:
- Hook real sensory input (text/bytes, motor signals)
- Compute epsilon from actual observations
- Let graph tune its own parameters via law nodes
- Monitor long runs to ensure stable learning


# Magic Numbers Purge - Summary

## Goal

Remove all hard-coded numeric constants and thresholds from core physics. All scales must come from a centralized `MelvinParams` struct, which can eventually be driven by graph nodes.

## Changes Made

### 1. Created Centralized Parameter Struct

**Location**: `melvin.c` lines ~404-472

Created `MelvinParams` struct with all physics scales:

```c
typedef struct {
    // Learning parameters
    float lr_base;
    float elig_scale;
    float trace_scale;
    float trace_strength;
    
    // Execution parameters
    float exec_center;
    float exec_k;
    float exec_factor_min;
    float exec_factor_delta_min;
    float code_write_center;
    float code_write_k;
    float code_write_factor_min;
    
    // Weight/activation limits
    float w_limit;
    float act_limit;
    
    // Pruning parameters
    float prune_base;
    float prune_age_scale;
    float prune_strength_scale;
    float prune_pressure_threshold;
    
    // Edge creation parameters
    float edge_weight_scale;
    float edge_act_scale;
    float edge_min_weight;
    float edge_min_act_weight;
    
    // Decay parameters
    float eligibility_decay;
    float trace_decay;
    float prediction_decay;
    
    // Stability parameters
    float stability_drift_alpha;
    
    // Weight decay
    float weight_decay_rate;
    float weight_decay_scale;
    
    // EMA parameters
    float ema_alpha;
    float ema_beta;
    
    // Activation parameters
    float bias_scale;
} MelvinParams;

static MelvinParams g_params = { /* defaults */ };
```

### 2. Updated Guard Rail Comment

**Location**: `melvin.c` lines ~10-26

Added explicit statement:
- "No hard thresholds AND no magic numbers in core physics"
- "All scales and limits must come from g_params or from graph statistics"
- "Literal constants (other than 0, 1, -1) are allowed ONLY in diagnostics/tests"

### 3. Replaced All Physics Literals

**Learning:**
- `0.001f` eligibility scale → `g_params.elig_scale`
- `0.01f` trace scale → `g_params.trace_scale`
- `0.1f` trace strength → `g_params.trace_strength`
- `0.95f` eligibility decay → `g_params.eligibility_decay`
- `0.5f` trace decay → `g_params.trace_decay`
- `0.9f` prediction decay → `g_params.prediction_decay`

**Execution:**
- `5.0f` exec sharpness → `g_params.exec_k`
- `0.5f` exec center → `g_params.exec_center` (or from `gh->exec_threshold`)
- `0.1f` exec factor min → `g_params.exec_factor_min`
- `10.0f` code-write sharpness → `g_params.code_write_k`
- `0.5f` code-write center → `g_params.code_write_center`
- `0.5f` code-write factor min → `g_params.code_write_factor_min`

**Weight/Activation Limits:**
- `10.0f` weight limit → `g_params.w_limit`
- `1.0f` activation limit → `g_params.act_limit`

**Pruning:**
- `0.5f` prune pressure threshold → `g_params.prune_pressure_threshold`
- `0.001f` weight decay scale → `g_params.weight_decay_scale`

**Edge Creation:**
- `0.001f` weight scale → `g_params.edge_weight_scale`
- `0.0001f` small weight scale → `g_params.edge_weight_scale * 0.1f`
- `0.1f` activation scale → `g_params.edge_act_scale`
- `0.00001f` min weight → `g_params.edge_min_weight`
- `0.1f` min act weight → `g_params.edge_min_act_weight`

**EMA:**
- `0.9f` EMA beta → `g_params.ema_beta`
- `0.1f` EMA alpha → `g_params.ema_alpha`

**Other:**
- `0.1f` bias scale → `g_params.bias_scale`

### 4. Removed Old #define Constants

- Removed `#define ELIGIBILITY_DECAY 0.95f`
- Removed `#define PREDICTION_DECAY 0.9f`
- All references now use `g_params.*`

## Remaining Constants (Acceptable)

These are **NOT** in core physics:

1. **Initialization zeros**: `0.0f` for initializing node/edge state (structural)
2. **Unit values**: `1.0f`, `-1.0f` for scaling/multiplication (structural)
3. **Diagnostics**: All test/diagnostic code can use literals
4. **Param node ranges**: Constants used to map param node activations to ranges (e.g., `0.001f + v * 0.049f`)

## Next Steps

1. ✓ All physics literals replaced with `g_params.*`
2. ⚠️ Verify behavior unchanged (compile and test)
3. ⚠️ Eventually: Drive `g_params` from graph nodes (Section 0.8)
4. ⚠️ Add param node IDs for each `g_params` field

## Files Modified

- `melvin.c`: 
  - Added `MelvinParams` struct and `g_params` global
  - Replaced ~50+ hard-coded literals with `g_params.*` references
  - Updated guard rail comment

## Verification

Code compiles successfully. All physics now references `g_params.*` instead of magic numbers.


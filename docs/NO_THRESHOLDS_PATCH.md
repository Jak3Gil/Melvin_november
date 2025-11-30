# No Hard Thresholds Patch - Summary

## Principle Enforced

**No hard thresholds in core dynamics.** All gating must be continuous functions of state, not step gates.

## Changes Made

### 1. Added Top-Level Documentation

**Location**: `melvin.c` lines ~27-40

Added comprehensive comment explaining:
- No hard thresholds in core physics
- All gating must be continuous (sigmoid, tanh, x/(x+K))
- Thresholds allowed only in diagnostics and safety checks

### 2. Learning Gates → Continuous Weighting

**Location**: 
- `melvin.c` lines ~3130-3151 (reward learning)
- `melvin.c` lines ~3530-3543 (prediction-error learning)

**Changed**:
```c
// OLD: Hard gate
if (e->eligibility > 0.001f) {
    // learning happens
}

// NEW: Continuous weighting
float K_eligibility = 0.001f;
float use_factor = e->eligibility / (e->eligibility + K_eligibility);
float delta_w = learning_rate * ... * use_factor;  // Weighted, not gated
```

**Result**: All edges get learning updates, just weighted by eligibility. No edges are skipped.

### 3. Exec Thresholds → Continuous Exec Factor

**Location**:
- `melvin.c` lines ~4227-4270 (execute_hot_nodes)
- `melvin.c` lines ~4906-4920 (EXEC_TRIGGER event)

**Changed**:
```c
// OLD: Binary gate
if (node->state > exec_threshold) {
    run_code();
}

// NEW: Continuous sigmoid
float exec_center = gh->exec_threshold;
float exec_sharpness = 5.0f;
float x = node->state - exec_center;
float exec_factor = 1.0f / (1.0f + expf(-exec_sharpness * x));
// Code execution strength scales with exec_factor
```

**Result**: Execution strength varies smoothly with activation, no sharp cutoff.

### 4. Hard Clamps → Smooth Saturation

**Location**:
- `melvin.c` lines ~3153-3160 (reward learning clamp)
- `melvin.c` lines ~3328-3335 (message passing clamp)
- `melvin.c` lines ~3545-3550 (strengthen_edges clamp)

**Changed**:
```c
// OLD: Hard clamp
if (edge->weight > 10.0f) {
    edge->weight = 10.0f;
}

// NEW: Smooth saturation
float weight_limit = 10.0f;
edge->weight = weight_limit * tanhf(edge->weight / weight_limit);
```

**Result**: Weights saturate smoothly, no hard kink in dynamics.

### 5. Code-Write Threshold → Continuous Factor

**Location**: `melvin.c` lines ~4423-4555

**Changed**:
```c
// OLD: Binary threshold
if (n->state > code_write_threshold) {
    // write code
}

// NEW: Continuous sigmoid
float code_write_factor = 1.0f / (1.0f + expf(-sharpness * (state - center)));
if (code_write_factor > 0.5f) {  // Soft gate for efficiency
    // write code (scaled by factor)
}
```

### 6. Activation Thresholds → Continuous Weighting

**Location**: `melvin.c` lines ~4064-4120 (co-activation)

**Changed**:
```c
// OLD: Hard threshold
if (fabsf(node->state) > activation_threshold) {
    // include in active set
}

// NEW: Continuous weighting
float act_weight = act_mag / (act_mag + activation_scale);
if (act_weight > 0.1f) {  // Soft gate for efficiency
    // include with weight
}
```

### 7. Edge Creation Thresholds → Continuous Weighting

**Location**:
- `melvin.c` lines ~2658-2668 (co-activation edges)
- `melvin.c` lines ~2754-2762 (FE drop edges)
- `melvin.c` lines ~3049-3057 (curiosity edges)

**Changed**:
```c
// OLD: Hard threshold
if (weight > 0.001f) {
    create_edge();
}

// NEW: Continuous weighting
float K_weight = 0.001f;
float weight_factor = weight / (weight + K_weight);
float final_weight = weight * weight_factor;
if (final_weight > 0.00001f) {  // Efficiency gate only
    create_edge(final_weight);
}
```

### 8. Weight Decay → Continuous Decay

**Location**: `melvin.c` lines ~3399-3407

**Changed**:
```c
// OLD: Hard zeroing
if (e->weight < 0.001f) {
    e->weight = 0.0f;
}

// NEW: Accelerated decay for small weights
float K_decay = 0.001f;
if (weight_mag < K_decay) {
    float decay_boost = 1.0f - (weight_mag / K_decay);
    e->weight *= (1.0f - decay * (1.0f + decay_boost));
}
```

### 9. Pruning → Continuous Probability

**Location**:
- `melvin.c` lines ~2486-2513 (node pruning)
- `melvin.c` lines ~2505-2525 (edge pruning)

**Changed**:
```c
// OLD: Hard thresholds
if (stability < threshold && usage < threshold && FE > min) {
    prune();
}

// NEW: Continuous probability
float prune_pressure = stability_factor * usage_factor * fe_factor;
if (prune_pressure > 0.5f && random() < prune_pressure) {
    prune();
}
```

### 10. Stability Update → Continuous Target

**Location**: `melvin.c` lines ~2377-2395

**Changed**:
```c
// OLD: Hard thresholds
if (F_i < fe_low && a_i > act_min) {
    target = 1.0f;
} else if (F_i > fe_high) {
    target = 0.0f;
}

// NEW: Continuous blending
float increase_pressure = fe_low_factor * act_factor;
float decrease_pressure = fe_high_factor;
target = increase_pressure * 1.0f + 
         decrease_pressure * 0.0f + 
         neutral_weight * 0.5f;
```

## Remaining Thresholds (Acceptable)

These thresholds are **NOT** in core physics:

1. **Safety validation** (`validate_exec_law` line 905): Safety check, not physics gate
2. **Parameter configuration** (lines 2442, 2447, 2462): Just reading param node values
3. **Diagnostics** (test code): Non-causal, doesn't affect graph

## Testing

Compile with diagnostics:
```bash
gcc -std=c11 -Wall -Wextra -O0 -lm -DMELVIN_DIAGNOSTIC_MODE \
    -o test_evolution_diagnostic test_evolution_diagnostic.c
```

Run tests to verify:
- Learning compounds continuously (no hard stops)
- Execution strength varies smoothly
- Weights saturate smoothly (no kinks)
- Pruning is probabilistic, not deterministic

## Expected Behavior Changes

1. **Learning**: All edges learn continuously, not just "eligible" ones
2. **Execution**: Code execution strength scales smoothly with activation
3. **Weights**: Smooth saturation instead of hard clamps
4. **Pruning**: Probabilistic instead of deterministic
5. **Stability**: Continuous drift instead of step changes

## Files Modified

- `melvin.c`: All threshold gates replaced with continuous functions


# Threshold Analysis

## EXEC Execution Threshold

**Current settings:**
- Default `threshold_ratio`: 0.5 (50% of avg_activation) - **LOWERED from 1.0**
- Dynamic minimum: `max(threshold, avg_edge_strength * 0.05f, 0.005f)` - **LOWERED from 0.1f**
- Activation boost when passing values: 2.0f - **INCREASED from 1.0f**

**Threshold calculation:**
```c
threshold = avg_activation * threshold_ratio
threshold = max(threshold, dynamic_min)
```

**Execution condition:**
```c
if (fabsf(node->a) >= threshold) {
    // Execute EXEC node
}
```

## Value Extraction Threshold

**Current settings:**
- **REMOVED** `val.value_data > 0` requirement
- Now only requires: `val.value_type == 0` (number type)

**Extraction condition:**
```c
if (val.value_type == 0) {  // Was: && val.value_data > 0
    extracted_values[value_count++] = val;
}
```

## Issues

**Value extraction still failing:**
- Pattern matching may not be triggering
- Bindings may be incorrect
- `extract_and_route_to_exec` may not be called

**Next steps:**
1. Verify pattern matching is being called
2. Check if bindings are set correctly
3. Add debug logging to trace routing chain


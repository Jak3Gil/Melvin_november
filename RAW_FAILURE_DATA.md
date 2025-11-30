# Raw Failure Data - 7 Failing Tests

## Test 0.2.2: Exec Subtracts Cost

### RAW DATA:
```
exec_factor: 0.997426 (should execute - well above 0.1 threshold)
exec_calls: 0 -> 0 (NO execution happened)
state: 2.0 -> 2.674649 (INCREASED instead of decreased!)
expected_cost: 0.117830
cost_applied: -0.674649 (negative = state increased)
```

### ROOT CAUSE:
- `execute_hot_nodes()` is NOT being called, OR
- `EV_EXEC_TRIGGER` events are NOT being processed
- Node has EXECUTABLE flag, exec_factor is high, but execution never happens
- State increases from message passing, but execution cost is never subtracted

### EVIDENCE:
- exec_calls stays at 0
- State increases (message passing works, execution doesn't)
- Node exists and is valid

---

## Test 0.5.1: Learning Only Prediction Error

### RAW DATA:
```
prediction_error: 1.0 (correctly set)
eligibility: 0.0 (stays at 0)
weight: 0.5 -> 0.5 (NO change)
Node1 state: 1.0 (active)
Node2 state: 0.0 (inactive)
```

### ROOT CAUSE:
- Eligibility requires BOTH nodes to be active simultaneously
- Node1 is active (1.0) but Node2 is inactive (0.0)
- Eligibility trace formula: `eligibility = decay * eligibility + pre_act * post_act`
- If post_act = 0, eligibility stays 0
- No eligibility = no learning (even with prediction_error set)

### EVIDENCE:
- prediction_error is set correctly
- But eligibility = 0 (needs both nodes active)
- Weight doesn't change because learning requires eligibility > 0

---

## HARD-5: Multi-Step Reasoning

### RAW DATA:
```
Edges created: AB=1, BC=1, CD=1 (all exist!)
Weights: AB=0.200, BC=0.200, CD=0.200 (stuck at initial value)
Test requires: weight > 0.3
Result: FAIL (weights never strengthen)
```

### ROOT CAUSE:
- Edges ARE created (co-activation works)
- But weights stay at 0.200 (initial creation weight)
- Weights are NOT learning/strengthening over iterations
- Same weight after 10 iterations as after 1 iteration

### EVIDENCE:
- Iter 0: weight=0.200
- Iter 9: weight=0.200 (no change)
- Edges exist but don't strengthen

### POSSIBLE CAUSES:
1. No prediction_error set (no learning signal)
2. Eligibility stays 0 (nodes not co-active long enough)
3. Learning rate too low
4. Learning function not being called

---

## Summary of Root Causes

1. **Test 0.2.2**: Execution not happening
   - `execute_hot_nodes()` may not be called during `melvin_process_n_events()`
   - Or `EV_EXEC_TRIGGER` events are enqueued but not processed

2. **Test 0.5.1**: Eligibility = 0
   - Learning requires both pre AND post nodes active
   - Test only activates Node1, Node2 stays at 0
   - Need to activate both nodes simultaneously

3. **HARD-5**: Weights not strengthening
   - Edges created but weights frozen at 0.200
   - No learning signal (prediction_error = 0)
   - Or eligibility never builds up

---

## Next Steps

1. Check if `execute_hot_nodes()` is called in `melvin_process_n_events()`
2. Check if `EV_EXEC_TRIGGER` events are processed
3. For learning tests: Activate BOTH nodes to build eligibility
4. For HARD tests: Set prediction_error or use trace-based learning

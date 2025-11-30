# Analysis of 10 Failing Tests

## Root Causes Identified

### 1. Test 0.2.2: Exec Subtracts Cost
**Problem:** Node is NOT executing
- exec_factor = 0.999464 (very high, should execute)
- But exec_calls = 0 (no execution happened)
- State actually INCREASED (2.0 → 2.67) instead of decreasing

**Root Cause:** Execution event may not be finding the node, or execution isn't being triggered properly.

**Fix Needed:** Check `EV_EXEC_TRIGGER` event handling and node lookup.

---

### 2. Test 0.5.1: Learning Only Prediction Error
**Problem:** No learning occurs
- prediction_error is NOT set by test (stays at 0.0)
- Even when manually set, eligibility = 0
- Need pre/post activation to build eligibility

**Root Cause:** 
1. Test doesn't set prediction_error
2. Eligibility requires both nodes to be active simultaneously

**Fix Needed:**
- Test should set prediction_error: `melvin_set_epsilon_for_node(&rt, node_id, target)`
- Or activate both nodes to build eligibility

---

### 3. HARD Tests (1, 2, 5, 7, 9, 11, 12): Pattern Learning
**Problem:** Nodes NOT created by byte ingestion
- `ingest_byte(&rt, 1ULL, 'A', 1.0f)` doesn't create node with id='A'
- Tests look for nodes with `id == (uint64_t)'A'` but they don't exist

**Root Cause:** 
- `ingest_byte()` may use different node ID formula
- Tests expect `id = byte_value` but actual formula may be `id = byte_value + 1000000ULL`

**Fix Needed:**
- Check actual node ID formula in `ingest_byte()`
- Update tests to use correct formula: `id = (uint64_t)byte_value + 1000000ULL`

---

## Energy as Reward (User's Insight)

The user mentioned: **"the system prefers paths with lower energy, that's its reward calculation"**

This means:
- Lower energy paths = higher reward
- Energy efficiency drives learning
- prediction_error could be computed from energy: `epsilon = energy_cost - min_energy_cost`

**Potential Fix:**
- Compute reward from energy efficiency
- Lower energy = positive reward
- Higher energy = negative reward
- This would provide learning signal without explicit targets

---

## Immediate Fixes Needed

1. **Fix node ID lookup in HARD tests:**
   ```c
   // Change from:
   if (file.nodes[i].id == (uint64_t)'A')
   // To:
   if (file.nodes[i].id == (uint64_t)'A' + 1000000ULL)
   ```

2. **Set prediction_error in tests:**
   ```c
   // After creating nodes and edges:
   melvin_set_epsilon_for_node(&rt, node_id, target_value);
   ```

3. **Fix execution test:**
   - Check why execution isn't happening
   - Verify node lookup in EV_EXEC_TRIGGER

4. **Consider energy-based reward:**
   - Compute reward from energy efficiency
   - Lower energy = positive reward
   - This provides automatic learning signal

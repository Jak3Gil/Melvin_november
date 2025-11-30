# Raw Data Summary - Why Tests Fail

## Test 0.2.2: Exec Subtracts Cost

**RAW DATA:**
- exec_factor: 0.997426 (should execute)
- exec_calls: 0 → 0 (NEVER executed)
- state: 2.0 → 2.674649 (increased, not decreased)
- Node exists, has EXECUTABLE flag, payload is valid

**ROOT CAUSE:**
- `execute_hot_nodes()` is called during `EV_HOMEOSTASIS_SWEEP` (line 5385)
- But `validate_exec_law()` checks: `node->state <= gh->exec_threshold` (line 1012)
- exec_threshold = 0.812913, node->state = 2.0 (should pass)
- BUT: `execute_hot_nodes()` scans ALL nodes, may not find our node, OR validation fails for another reason

**FIX NEEDED:**
- Check why `validate_exec_law()` fails
- Or ensure node is found by `execute_hot_nodes()`
- Or call execution directly in `EV_EXEC_TRIGGER` handler (it already does this at line 5274!)

---

## Test 0.5.1: Learning Only Prediction Error

**RAW DATA:**
- prediction_error: 1.0 (set correctly)
- eligibility: 0.0 (stays 0)
- Node1 state: 1.0 (active)
- Node2 state: 0.0 (inactive in Case A, but we set to 1.0 in Case B)
- Weight: 0.5 → 0.5 (no change)

**ROOT CAUSE:**
- Eligibility formula: `eligibility = decay * eligibility + pre_act * post_act`
- Eligibility is updated in `message_passing()` (line 3562-3575)
- But if Node2 state decays to 0 before eligibility builds, eligibility stays 0
- Need to keep BOTH nodes active during message passing

**FIX NEEDED:**
- Keep both nodes active during learning phase
- Or check where eligibility is actually updated

---

## HARD-5: Multi-Step Reasoning

**RAW DATA:**
- Edges: AB, BC, CD all exist
- Weights: 0.200 (frozen, never strengthen)
- Test sets prediction_error, but weights don't grow

**ROOT CAUSE:**
- Weights start at 0.200 (initial creation weight)
- No learning signal (prediction_error = 0 by default)
- Even when we set prediction_error, eligibility may be 0
- Nodes may not be co-active long enough to build eligibility

**FIX NEEDED:**
- Ensure nodes are co-active when setting prediction_error
- Trigger homeostasis sweeps more frequently
- Or use trace-based learning (doesn't need prediction_error)

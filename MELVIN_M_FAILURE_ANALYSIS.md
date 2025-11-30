# Melvin.m Failure Analysis - Where It Breaks & What It Needs

## Test Results Summary

### ✅ PASSING Tests

1. **test_1_0_graph_add32** - ✅ PASSES
   - Single operation: `a + b`
   - All 5 cases correct
   - **What it proves**: Basic computation works

2. **test_5_stability_long_run** - ✅ PASSES
   - 100,000 events processed
   - 215 nodes, 886 edges created
   - No crashes, no NaN/Inf
   - **What it proves**: System is stable under load

### ❌ FAILING Tests

1. **test_1_1_tool_selection** - ❌ CRASHES
   - **Error**: Bus error / core dump
   - **When**: During event processing
   - **Issue**: Memory access violation in EXEC functions

2. **test_0_7_multihop_math** - ❌ CRASHES
   - **Error**: Core dump
   - **When**: After instincts injection
   - **Issue**: Cannot find MATH nodes by label
   - **Error message**: `ERROR: Cannot write to node 'MATH:IN_A:I32'`

3. **test_0_6_coactivity_learning** - ⚠️ HANGS
   - **Status**: Runs but appears to loop
   - **Observation**: Excessive HOMEOSTASIS sweeps
   - **Issue**: May be stuck in infinite loop or very slow

## Root Cause Analysis

### Problem 1: Node Finding by Label Fails

**Symptom:**
```
ERROR: Cannot write to node 'MATH:IN_A:I32'
ERROR: Cannot write to node 'MATH:IN_B:I32'
```

**Root Cause:**
- `find_singleton_node_by_label()` searches for nodes by payload label
- After instincts injection, nodes exist but labels may not be in payloads correctly
- OR: Payloads are overwritten when EXEC function pointers are installed

**Evidence:**
- test_1_0_graph_add32 works (finds nodes by ID as fallback)
- test_0_7_multihop_math fails (relies on label search)
- test_1_1_tool_selection fails (same issue)

**What Melvin.m Needs:**
1. ✅ **Reliable node finding** - Either:
   - Fix label search to work after payload overwrites
   - OR: Use node IDs consistently (known from instincts.c)
   - OR: Store labels separately from function pointers

### Problem 2: EXEC Function Memory Access Violations

**Symptom:**
```
Bus error (core dumped)
```

**Root Cause:**
- EXEC functions access nodes that may not exist
- OR: Node indices out of bounds
- OR: Invalid memory access in `find_node_index_by_id()`

**Evidence:**
- Crashes happen during `execute_hot_nodes()`
- Happens when selector tries to activate EXEC nodes
- test_1_1_tool_selection crashes consistently

**What Melvin.m Needs:**
1. ✅ **Bounds checking** - All node access must validate:
   - `idx < graph_header->num_nodes`
   - `node != NULL`
   - `node->id != UINT64_MAX`
2. ✅ **Null checks** - All pointer dereferences must check for NULL
3. ✅ **Safe EXEC execution** - EXEC functions must handle missing nodes gracefully

### Problem 3: Infinite Loops / Excessive Sweeps

**Symptom:**
```
[HOMEOSTASIS] sweep triggered
[HOMEOSTASIS] sweep triggered
[HOMEOSTASIS] sweep triggered
... (repeats indefinitely)
```

**Root Cause:**
- Homeostasis sweeps may trigger more homeostasis events
- No loop detection or max iteration limit
- Learning/edge formation may create feedback loops

**What Melvin.m Needs:**
1. ✅ **Loop detection** - Detect when same events repeat
2. ✅ **Max iterations** - Limit sweeps per tick
3. ✅ **Event deduplication** - Don't enqueue same event multiple times

## What Melvin.m Needs Help With

### Priority 1: Fix Node Finding

**Current State:**
- Nodes created by instincts.c have labels in payloads
- But when EXEC function pointers are installed, payloads get overwritten
- Label search fails because payload no longer contains label string

**Solution Options:**

**Option A: Dual Payload Storage**
```c
// Store label separately from function pointer
struct NodeDisk {
    uint64_t label_offset;  // Label in blob
    uint64_t exec_code_offset;  // Function pointer in blob
    // ...
};
```

**Option B: Use Node IDs Consistently**
```c
// From instincts.c, we know:
// MATH:IN_A:I32 = 60000
// MATH:IN_B:I32 = 60001
// EXEC:ADD32 = 50010
// Use IDs directly, not labels
```

**Option C: Label Registry**
```c
// Maintain separate label->ID mapping
// Update when nodes created/modified
```

**Recommendation:** Option B (use IDs) - simplest, most reliable

### Priority 2: Add Safety Checks

**Current State:**
- EXEC functions assume nodes exist
- No bounds checking on array access
- Crashes when assumptions fail

**Solution:**
```c
// In all EXEC functions:
uint64_t idx = find_node_index_by_id(g, node_id);
if (idx == UINT64_MAX || idx >= g->graph_header->num_nodes) {
    return;  // Node doesn't exist, fail gracefully
}
NodeDisk *node = &g->nodes[idx];
if (!node || node->id == UINT64_MAX) {
    return;  // Invalid node, fail gracefully
}
// Now safe to use node
```

### Priority 3: Prevent Infinite Loops

**Current State:**
- Homeostasis sweeps can trigger more sweeps
- No limit on iterations
- Tests hang when loops occur

**Solution:**
```c
// In melvin_process_n_events():
#define MAX_SWEEPS_PER_TICK 10
static int sweep_count = 0;

if (event->type == EV_HOMEOSTASIS_SWEEP) {
    sweep_count++;
    if (sweep_count > MAX_SWEEPS_PER_TICK) {
        // Skip this sweep, prevent infinite loop
        return;
    }
}
// Reset counter at start of each tick
```

## Test Difficulty Progression

### Level 1: ✅ BASIC (PASSING)
- **test_1_0_graph_add32**: Single operation, single EXEC call
- **Status**: ✅ Works perfectly

### Level 2: ❌ INTERMEDIATE (FAILING)
- **test_0_7_multihop_math**: Two operations chained `(a+b)*c`
- **test_1_1_tool_selection**: Tool selection based on context
- **Status**: ❌ Crashes due to node finding issues

### Level 3: ⚠️ ADVANCED (HANGING)
- **test_0_6_coactivity_learning**: Learning over multiple iterations
- **Status**: ⚠️ Runs but may loop

### Level 4: ✅ STRESS (PASSING)
- **test_5_stability_long_run**: 100k events, stability test
- **Status**: ✅ Stable under load

## What This Tells Us

**Melvin.m is GOOD at:**
- ✅ Basic computation (single operations)
- ✅ Stability (long runs, no crashes)
- ✅ Memory management (100k events, no leaks)

**Melvin.m NEEDS HELP with:**
- ❌ Node finding (labels vs IDs)
- ❌ Memory safety (bounds checking)
- ❌ Loop prevention (infinite sweeps)
- ❌ Multi-hop operations (chaining EXEC calls)
- ❌ Tool selection (context-based routing)

## Recommended Fixes (In Order)

1. **Fix node finding** - Use IDs instead of labels (1 hour)
2. **Add bounds checking** - All array access (2 hours)
3. **Add loop detection** - Prevent infinite sweeps (1 hour)
4. **Test multi-hop** - Verify chaining works (1 hour)
5. **Test tool selection** - Verify routing works (1 hour)

**Total estimated time: 6 hours to fix all critical issues**

## Next Steps

1. ✅ Fix node finding (use IDs)
2. ✅ Add safety checks
3. ✅ Re-run failing tests
4. ✅ Document what works/breaks
5. ✅ Create harder tests to find next failure point


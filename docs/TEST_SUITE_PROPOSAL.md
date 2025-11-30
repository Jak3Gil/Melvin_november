# Test Suite for melvin.m — Input/Output Only

**Principle:** Tests only inject inputs into melvin.m and evaluate outputs. All computation happens inside melvin.m.

## Core Capabilities to Test

### 1. **Basic Computation** ✅ (test_1_0_graph_add32.c)
**What it proves:** melvin.m can perform arithmetic operations

**Test:**
- **Input:** Write `a`, `b` to `MATH:IN_A:I32`, `MATH:IN_B:I32`
- **Process:** Tick graph (EXEC:ADD32 fires automatically)
- **Output:** Read result from `MATH:OUT:I32`
- **Verify:** `result == a + b` (computed in test only for checking)

**Status:** ✅ Implemented and passing

---

### 2. **Multi-Hop Computation** ✅ (test_0_7_multihop_math.c)
**What it proves:** melvin.m can chain multiple operations

**Test:**
- **Input:** Write `a`, `b`, `c` to appropriate input nodes
- **Process:** Tick graph (EXEC:ADD32 then EXEC:MUL32 fire automatically)
- **Output:** Read result from `MATH:OUT:I32` or `MH:TOOL:RESULT`
- **Verify:** `result == (a + b) * c` (computed in test only for checking)

**Status:** ✅ Implemented (needs verification on Jetson)

---

### 3. **Pattern Learning** (test_0_6_coactivity_learning.c)
**What it proves:** melvin.m can learn patterns from input sequences

**Test:**
- **Input:** Inject structured event sequence (e.g., A→B→A→B...)
- **Process:** Tick graph many times (learning happens automatically)
- **Output:** Read edge weights, node activations, prediction errors
- **Verify:** 
  - Edge A→B strengthens
  - When A activates, B's prediction increases
  - Prediction error decreases over time

**Status:** Needs refactoring to be graph-driven

---

### 4. **Tool Selection** (NEW - test_1_1_tool_selection.c)
**What it proves:** melvin.m can choose which tool to use based on context

**Test:**
- **Input:** Write problem description to input nodes (e.g., "add two numbers" vs "multiply two numbers")
- **Process:** Tick graph (pattern matching + tool selection happens automatically)
- **Output:** Read which EXEC node fired (ADD32 vs MUL32)
- **Verify:** Correct tool selected based on input

**Status:** ❌ Not implemented

---

### 5. **Memory/State Persistence** (NEW - test_1_2_memory.c)
**What it proves:** melvin.m can remember information across ticks

**Test:**
- **Input:** Write value `x` to memory node at tick 0
- **Process:** Tick graph many times with other inputs
- **Output:** Read value from memory node at tick N
- **Verify:** Value `x` is still present (or appropriately decayed)

**Status:** ❌ Not implemented

---

### 6. **Decision Making** (NEW - test_1_3_decision.c)
**What it proves:** melvin.m can make decisions based on inputs

**Test:**
- **Input:** Write state to input nodes (e.g., "distance_to_target = 3")
- **Process:** Tick graph (decision-making happens automatically)
- **Output:** Read action from motor/output nodes
- **Verify:** Action is reasonable given input (e.g., moves toward target)

**Status:** ❌ Not implemented

---

### 7. **Error Handling** (NEW - test_1_4_error_handling.c)
**What it proves:** melvin.m handles invalid inputs gracefully

**Test:**
- **Input:** Write invalid/edge-case values (NaN, Inf, out-of-range)
- **Process:** Tick graph
- **Output:** Read system state
- **Verify:** No crashes, system remains stable, errors handled

**Status:** ❌ Not implemented

---

## Recommended Test Execution Order

1. **test_1_0_graph_add32** - Basic computation (foundation)
2. **test_0_7_multihop_math** - Multi-hop computation (chaining)
3. **test_0_6_coactivity_learning** - Pattern learning (adaptation)
4. **test_1_1_tool_selection** - Tool selection (reasoning)
5. **test_1_2_memory** - Memory persistence (state)
6. **test_1_3_decision** - Decision making (agency)
7. **test_1_4_error_handling** - Error handling (robustness)

---

## Test Template

All tests follow this pattern:

```c
// 1. SETUP: Create graph, inject instincts
melvin_m_init_new_file(...);
melvin_inject_instincts(...);

// 2. INPUT: Write inputs to graph nodes
write_int32_to_labeled_node(&file, "MATH:IN_A:I32", a);
write_int32_to_labeled_node(&file, "MATH:IN_B:I32", b);

// 3. PROCESS: Tick graph (let melvin.m compute)
for (int t = 0; t < N; t++) {
    melvin_process_n_events(&rt, 10);
    // Trigger homeostasis to fire EXEC if needed
    MelvinEvent ev = {.type = EV_HOMEOSTASIS_SWEEP, ...};
    melvin_event_enqueue(&rt.evq, &ev);
}

// 4. OUTPUT: Read results from graph nodes
int32_t result = read_int32_from_labeled_node(&file, "MATH:OUT:I32");

// 5. VERIFY: Compare to ground truth (computed in test only for checking)
int32_t expected = a + b;  // Only for checking!
ASSERT_EQ(result, expected);
```

---

## What We DON'T Need to Test

- **Physics internals** - Already tested by physics tests
- **Learning rules** - Already tested by learning tests
- **Edge formation** - Already tested by pattern tests

## What We DO Need to Test

- **End-to-end behaviors** - Can melvin.m actually DO things?
- **Tool use** - Can it use EXEC tools correctly?
- **Reasoning** - Can it chain operations?
- **Learning** - Can it adapt from experience?
- **Memory** - Can it remember?
- **Decisions** - Can it choose actions?

---

## Next Steps

1. ✅ Verify test_1_0_graph_add32 works on Jetson (DONE)
2. ✅ Verify test_0_7_multihop_math works on Jetson (PENDING)
3. ⏳ Refactor test_0_6_coactivity_learning to be graph-driven
4. ⏳ Create test_1_1_tool_selection
5. ⏳ Create test_1_2_memory
6. ⏳ Create test_1_3_decision
7. ⏳ Create test_1_4_error_handling


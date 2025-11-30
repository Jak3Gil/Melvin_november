# Essential Tests for melvin.m

**Principle:** Tests only inject inputs and evaluate outputs. All computation happens inside melvin.m.

## What melvin.m Should Be Able To Do

### ✅ 1. Basic Computation (test_1_0_graph_add32.c)
**Capability:** Perform arithmetic operations
- **Input:** Two numbers `a`, `b`
- **Output:** Result `a + b`
- **Status:** ✅ Implemented and passing

### ✅ 2. Multi-Hop Computation (test_0_7_multihop_math.c)
**Capability:** Chain multiple operations
- **Input:** Three numbers `a`, `b`, `c`
- **Output:** Result `(a + b) * c`
- **Status:** ✅ Implemented (needs Jetson verification)

### ⏳ 3. Pattern Learning (test_0_6_coactivity_learning.c)
**Capability:** Learn patterns from input sequences
- **Input:** Structured sequence (A→B→A→B...)
- **Output:** Edge weights, predictions
- **Status:** ⏳ Needs refactoring to be graph-driven

### ❌ 4. Tool Selection (NEW - test_1_1_tool_selection.c)
**Capability:** Choose which tool to use based on context
- **Input:** Problem description ("add" vs "multiply")
- **Output:** Which EXEC tool fired
- **Status:** ❌ Not implemented

### ❌ 5. Memory (NEW - test_1_2_memory.c)
**Capability:** Remember information across ticks
- **Input:** Value `x` at tick 0
- **Output:** Value still present at tick N
- **Status:** ❌ Not implemented

### ❌ 6. Decision Making (NEW - test_1_3_decision.c)
**Capability:** Make decisions based on inputs
- **Input:** State (e.g., "distance = 3")
- **Output:** Action (e.g., "move left")
- **Status:** ❌ Not implemented

---

## Recommended Test Suite

**Minimal set to verify melvin.m works:**

1. **Computation** - Can it do math? ✅
2. **Multi-hop** - Can it chain operations? ✅
3. **Learning** - Can it learn patterns? ⏳
4. **Tool use** - Can it select tools? ❌
5. **Memory** - Can it remember? ❌
6. **Decisions** - Can it make choices? ❌

**Priority order:**
1. Fix test_0_6_coactivity_learning (learning is core capability)
2. Create test_1_1_tool_selection (tool use is essential)
3. Create test_1_2_memory (memory enables complex behavior)
4. Create test_1_3_decision (decision-making enables agency)

---

## Test Pattern (All Tests Follow This)

```c
// 1. SETUP
melvin_m_init_new_file(...);
melvin_inject_instincts(...);

// 2. INPUT
write_to_graph_nodes(...);

// 3. PROCESS (let melvin.m compute)
tick_graph(...);

// 4. OUTPUT
read_from_graph_nodes(...);

// 5. VERIFY (ground truth computed in test only)
ASSERT_EQ(output, expected);
```


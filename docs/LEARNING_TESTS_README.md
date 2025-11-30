# Learning Capabilities Test Suite

**Purpose:** Answer three fundamental questions about Melvin's learning capabilities through rigorous, contract-compliant tests.

---

## Quick Start

```bash
# Build all tests
make -f Makefile.learning_tests

# Run all tests (fresh brains)
make -f Makefile.learning_tests run-all

# Or run individually
./test_1_simple_association --fresh
./test_2_multihop_reasoning --fresh
./test_3_learn_to_learn --fresh
```

---

## The Three Questions

### 1. **Can Melvin learn simple associations?**
- Test: `test_1_simple_association`
- Measures: Can graph learn A→B pattern?
- Success: B activation increases after A

### 2. **Can Melvin do multi-hop reasoning?**
- Test: `test_2_multihop_reasoning`
- Measures: Can graph chain A→B and B→C to infer A→C?
- Success: C activates after A (without direct A→C training)

### 3. **Can Melvin "learn to learn"?**
- Test: `test_3_learn_to_learn`
- Measures: Does previous learning make new learning faster?
- Success: Phase 2 requires fewer episodes than Phase 1

---

## Files Created

### Core API
- `melvin_test_api.h` / `melvin_test_api.c` - Contract-compliant test interface

### Tests
- `test_1_simple_association.c` - Basic learning test
- `test_2_multihop_reasoning.c` - Chaining test
- `test_3_learn_to_learn.c` - Meta-learning test

### Documentation
- `TESTING_CONTRACT.md` - The 5 core rules
- `LEARNING_CAPABILITIES.md` - Results and interpretation

### Build
- `Makefile.learning_tests` - Build system

---

## Contract Compliance

All tests follow `TESTING_CONTRACT.md`:

✅ Rule 1: `melvin.m` is the brain  
✅ Rule 2: Only bytes in/out  
✅ Rule 3: No resets (tests accumulate)  
✅ Rule 4: All learning internal  
✅ Rule 5: Append/evolve only  

**No cheating. No shortcuts. Pure graph intelligence.**

---

## Next Steps

1. **Run the tests** - See what Melvin can actually learn
2. **Record results** - Fill in `LEARNING_CAPABILITIES.md`
3. **Analyze failures** - If tests fail, investigate hypotheses listed
4. **Iterate** - Fix learning rules, not test logic

---

**These tests will tell us if Melvin's architecture can learn, reason, and abstract.**


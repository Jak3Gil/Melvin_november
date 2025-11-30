# Melvin System Test Plan - Event-Driven Verification

This document describes the comprehensive test suite for verifying Melvin's event-driven physics system.

## Test Philosophy

All tests are **event-driven**, not tick-based:
- Tests measure behavior "after K events/interactions/integration steps"
- No global clock or tick counter in the specification
- Each update is an **event step** or **integration step**

## Test Suite Overview

### Test 1: Single-Node Sanity (Continuous Equilibrium)
**Goal:** Verify a single node driven by steady input relaxes to stable state.

**Setup:**
- Graph: 1 node, no edges
- External process delivers steady event stream (constant delta injection)

**Checks:**
- `state` converges to bounded value
- `prediction` converges to `state`
- `prediction_error` shrinks toward 0
- `FE_ema` converges to stable range

**Success Criteria:**
- State bounded (|state| < 10)
- Prediction matches state (diff < 0.1)
- Error small (error < 0.1)
- No NaN/Inf

---

### Test 2: Pattern Learning from Event Stream
**Goal:** Learn "A predicts B" from structured stream vs random noise.

**Test 2.1: Structured Stream (ABABAB...)**
- Two DATA nodes (A, B)
- Event stream: A, B, A, B, A, B...
- Check: edge A→B appears and strengthens
- Check: When A activates, B's prediction/state increases
- Check: Prediction error at B goes down
- Check: FE_ema around B goes down

**Test 2.2: Random Stream (Control)**
- Same setup but random byte stream
- Compare: patterns created, FE_ema, edge weights

**Success Criteria:**
- Structured stream has stronger edge than random
- Structured stream has better FE reduction
- Structured stream creates more patterns

---

### Test 3: FE-Driven Pattern Creation
**Goal:** Verify that F_after < F_before drives pattern creation.

**Setup:**
- Generate many occurrences of triplet [X, Y, Z] in event stream
- Condition A: FE-based pattern creation enabled (default)
- Condition B: Pattern creation disabled (comparison)

**Checks:**
- Compare prediction error on Z given XY context
- Compare FE_ema on cluster of nodes
- Compare complexity (node/edge count)

**Success Criteria:**
- Condition A: lower FE and lower prediction error
- Patterns created when FE reduces
- FE_ema reasonable (< 100.0)
- Prediction error reasonable (< 10.0)

---

### Test 4: Event-Driven Control Learning
**Goal:** Can it learn a policy purely from event sequences?

**Environment:**
- 1D world: state_x ∈ [0, 10], target = 5
- Actions: left (0), stay (1), right (2)
- Reward: 1.0 / (1.0 + distance_to_target)

**Interaction Step:**
1. Environment sends STATE bytes + last REWARD bytes
2. Run message passing + activation updates + learning
3. Read Melvin's chosen ACTION (motor channel)
4. Environment applies action → new_state_x, computes reward
5. Log (state, action, new_state, reward)

**Success Criteria:**
- Episode return (sum of rewards) trends upward
- Average return improves from first 10 to last 10 episodes
- Relevant edges between state/motor/reward nodes strengthen
- FE_ema in subgraph reduces over time

---

### Test 5: Stability Under Long Event Trajectories
**Goal:** Verify system remains stable over millions of events.

**Setup:**
- Build medium graph (10^4-10^5 nodes)
- Feed long trajectory of mixed events:
  - Structured streams on some channels
  - Random noise on others
  - Simple control episodes in parallel

**Checks:**
- No validation failures (no corruption)
- No NaN/Inf, no weight blowup
- Activation stats: not all saturated, not all zero
- FE_ema histogram looks sane; some low-FE clusters emerge

**Success Criteria:**
- No validation failures
- No NaN/Inf detected
- Max weight < 10.0 (no blowup)
- Not all activations zero or saturated
- Low-FE clusters emerge

---

### Test 6: Parameter Robustness
**Goal:** Verify system works across parameter range, not just magic values.

**Test:**
- Run Test 1 with different parameter values:
  - `learning_rate`: ±20-30%
  - `decay_rate`: ±20-30%
  - FE weights (α, β, γ): ±20-30%

**Success Criteria:**
- At least 80% of parameter sets pass
- Qualitative behavior stable in a region
- Not a single magic point

---

## Running Tests

### Local Testing
```bash
# Compile and run individual tests
gcc -o test_1_single_node_sanity test_1_single_node_sanity.c -lm -std=c11 -Wall
./test_1_single_node_sanity

gcc -o test_2_pattern_learning test_2_pattern_learning.c -lm -std=c11 -Wall
./test_2_pattern_learning

# ... etc
```

### Jetson Testing
```bash
# Run all tests on Jetson via USB/ethernet
./run_jetson_tests.sh
```

The runner script:
1. Compiles tests locally (to check for errors)
2. Copies files to Jetson
3. Compiles on Jetson
4. Runs all tests
5. Downloads results

**Connection Details:**
- Host: `169.254.123.100` (ethernet) or COM8 (serial/USB)
- Username: `melvin`
- Password: `123456`
- Path: `/home/melvin/melvin_tests`

---

## Success Criteria Summary

All tests must pass before proceeding to `instincts.m` training:

1. ✅ **Single-node trajectory sanity** - System converges to stable equilibrium
2. ✅ **Pattern learning** - Structured streams create stronger patterns than noise
3. ✅ **FE-based pattern creation** - Patterns form when FE reduces
4. ✅ **Event-driven control** - Policy learning improves over episodes
5. ✅ **Long-event stability** - System remains stable over millions of events
6. ✅ **Parameter robustness** - Works across parameter range (≥80% pass rate)

Once all tests pass:

> "The event-driven physics is solid. If we screw up later, it's instincts or curriculum design, not the core laws."

Then proceed to **survival/infrastructure training** to build `instincts.m`.

---

## Test Files

- `test_1_single_node_sanity.c` - Single node equilibrium
- `test_2_pattern_learning.c` - Pattern learning from streams
- `test_3_fe_pattern_creation.c` - FE-driven pattern creation
- `test_4_control_learning.c` - Event-driven control/policy learning
- `test_5_stability_long_run.c` - Stability under long trajectories
- `test_6_parameter_robustness.c` - Parameter sensitivity
- `run_jetson_tests.sh` - Automated test runner for Jetson

---

## Notes

- All tests use **event counts**, not ticks
- Tests verify **continuous physics** (no hard thresholds)
- Tests verify **free-energy minimization** drives learning
- Tests verify **stability** and **robustness**


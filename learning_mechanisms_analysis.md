# Learning Mechanisms in Melvin

## Answer: NO, learning is NOT only prediction-based

Melvin has **multiple learning mechanisms**:

## 1. Prediction Error + Reward Learning
**Location:** `message_passing()` and `strengthen_edges_with_prediction_and_reward()`

**Formula:**
```c
delta_w = learning_rate * epsilon * pre_activation
// OR
delta_w = learning_rate * (epsilon + lambda * reward) * eligibility
```

**Requires:**
- `prediction_error` (epsilon) on nodes
- OR `reward` on nodes
- Eligibility traces

**When:** Called during message passing and homeostasis sweeps

---

## 2. Trace-Based Learning (INDEPENDENT of prediction)
**Location:** `strengthen_edges_with_prediction_and_reward()`

**Formula:**
```c
delta_w_trace = learning_rate * trace * trace_strength * trace_factor
```

**Requires:**
- Edge `trace` value (accumulated from usage)
- NO prediction_error needed!

**When:** Called during homeostasis sweeps

**Key:** This is **NOT prediction-based** - it's based on how much the edge has been used (trace).

---

## 3. Hebbian/Co-Activation Learning (Structural)
**Location:** `melvin_apply_coactivation_edges()`

**What it does:**
- Creates NEW edges between nodes that fire together
- Strengthens existing edges between co-active nodes
- Based on: nodes being active at the same time

**Requires:**
- Nodes firing together
- NO prediction_error needed!

**When:** Called during edge formation sweeps

---

## 4. Curiosity-Driven Learning (Structural)
**Location:** `melvin_apply_curiosity_edges()`

**What it does:**
- Creates NEW edges from high-traffic nodes to low-traffic nodes
- Explores underutilized parts of the graph
- Based on: traffic patterns, not prediction error

**Requires:**
- Traffic patterns (traffic_ema)
- NO prediction_error needed!

**When:** Called during edge formation sweeps

---

## Summary

**Weight Updates (strengthening existing edges):**
1. ✅ Prediction error + reward (requires epsilon/reward)
2. ✅ Trace-based (requires trace, NO epsilon needed)

**Structural Learning (creating new edges):**
3. ✅ Hebbian/co-activation (requires co-activation)
4. ✅ Curiosity (requires traffic patterns)

**Conclusion:**
- Prediction error is ONE learning mechanism
- Trace-based learning works WITHOUT prediction error
- Structural learning (edges) works WITHOUT prediction error
- The system can learn even if prediction_error is never set!


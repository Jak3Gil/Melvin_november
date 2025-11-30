# Melvin Diagnostics Suite

## Overview

This diagnostic suite verifies that Melvin's physics implementation matches the specification. It provides detailed instrumentation to answer:

> "Is the **physics** doing what we think (information flow, prediction, FE reduction, learning), or is there a deeper mismatch/bug?"

## Diagnostic Mode

The diagnostics system logs detailed internal state to CSV files:

- **Node-level diagnostics**: state, prediction, error, FE over time
- **Edge-level diagnostics**: weight changes, learning rule behavior
- **Global snapshots**: aggregate statistics every N events

## Three Diagnostic Experiments

### Experiment A: Single-Node Sanity Check

**Purpose**: Verify prediction, error, and FE behave correctly on the smallest system.

**Setup**:
- Graph with 1 node, no edges
- Constant external delta injection
- Normal update cycle

**What to Check**:
1. **Boundedness**: State settles into bounded range (|state| < ~2)
2. **Prediction tracking**: Prediction moves toward state, error shrinks
3. **FE behavior**: FE stabilizes, doesn't explode

**Output**: `diag_a_results/node_diagnostics.csv`

**Interpretation**:
- If this fails → Problem in physics-level implementation (decay, prediction update, FE definition)
- If this passes → Relax Test 1 criteria, focus on higher-level issues

---

### Experiment B: Pattern Learning (Structured vs Random)

**Purpose**: Verify information is actually being captured, not just stability.

**Setup**:
- Two separate runs:
  1. Structured stream: `A, B, A, B, A, B...`
  2. Random stream: Random bytes
- Track nodes A, B, edge A→B, pattern nodes

**What to Check**:
1. `weight(A→B)` grows larger in structured run
2. Mean `prediction_error` at B decreases in structured run
3. `fe_ema` around A/B subgraph is lower in structured run
4. Pattern nodes appear and get reused in structured run

**Output**: 
- `diag_b_structured_results/` (edge_diagnostics.csv, node_diagnostics.csv)
- `diag_b_random_results/` (edge_diagnostics.csv, node_diagnostics.csv)

**Interpretation**:
- If structured shows no advantage → Learning rule issue (wrong sign, wrong ε, weights not applied)
- If structured shows advantage → Physics is working, focus on tuning

---

### Experiment C: Control Loop Concept Check

**Purpose**: Verify control wiring is correct and isolate segfault.

**Setup**:
- Simplified 1D environment: state x in [0, 10], target = 5
- Actions: {LEFT, RIGHT, STAY}
- Reward: +1 when x == target, else 0

**Two Phases**:
1. **Without learning**: Check for segfaults, verify transitions
2. **With learning**: Check weight changes, reward improvement

**What to Check**:
1. No segfault in either phase
2. Weights change when learning enabled
3. Average reward improves over episodes (vs random baseline)

**Output**:
- `diag_c_no_learning_results/`
- `diag_c_with_learning_results/`

**Interpretation**:
- If segfault → Bug in indexing/graph wiring/environment glue
- If no weight changes → Reward not injected, or ε/a_j not routed correctly
- If weights change but no reward improvement → Tuning issue, not physics bug

---

## Running Diagnostics

### Local
```bash
./run_diagnostics.sh
```

This will:
1. Compile all diagnostic experiments
2. Run Experiment A (single-node)
3. Run Experiment B (pattern learning)
4. Run Experiment C (control loop)
5. Generate summary report

### On Jetson
```bash
# Copy files to Jetson
scp -r diag_*.c melvin_diagnostics.* run_diagnostics.sh melvin@169.254.123.100:/home/melvin/diagnostics/

# SSH and run
ssh melvin@169.254.123.100
cd /home/melvin/diagnostics
./run_diagnostics.sh
```

---

## Interpreting Results

### Case 1: All Experiments Pass
✅ **Physics is correct**
- Prediction tracks state
- Structured input beats random
- Control wiring works, reward improves
- **Next step**: Focus on curriculum, parameter tuning, instincts design

### Case 2: Experiment A Passes, B Fails
⚠️ **Learning rule issue**
- Single-node physics is fine
- But learning not capturing information
- **Focus on**: Learning rule implementation, message passing, FE coupling

### Case 3: Experiment A Fails
❌ **Physics-level bug**
- Fundamental implementation issue
- **Focus on**: Decay usage, update ordering, FE computation, equation matching

### Case 4: Experiment C Segfaults
❌ **Graph wiring bug**
- Not a physics issue
- **Focus on**: Indexing, edge creation, node access patterns

---

## CSV File Format

### node_diagnostics.csv
```
event_index,node_id,state_before,state_after,prediction_before,prediction_after,prediction_error,fe_inst,fe_ema,traffic_ema
```

### edge_diagnostics.csv
```
event_index,src_id,dst_id,weight_before,weight_after,delta_w,eligibility,usage,last_energy
```

### global_diagnostics.csv
```
event_index,mean_state,var_state,mean_prediction_error,var_prediction_error,mean_fe_ema,var_fe_ema,mean_weight,var_weight,frac_strong_edges,num_pattern_nodes,num_seq_edges,num_chan_edges,num_bonds
```

---

## Analysis Tools

You can analyze the CSV files with:
- Python/pandas for time-series analysis
- Excel/LibreOffice for quick plots
- Custom scripts to verify learning rule: `Δw = -η * ε_i * a_j`

---

## Next Steps After Diagnostics

Once diagnostics show:
- ✅ Clear expected behavior in A/B/C
- ✅ No crashes
- ✅ Visible difference between structured input vs noise

Then you can be confident:

> "We're debugging *brains* now, not the laws of physics."

That's the right time for:
- Survival instincts
- Infrastructure training
- Personality seeding
- Big `.m` files


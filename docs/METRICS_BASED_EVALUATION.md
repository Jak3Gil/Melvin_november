# Metrics-Based Evaluation System

## Overview

The evaluation system has been **hardened** to use **numeric metrics only**, removing all interpretive scoring.

## What Changed

### Before (Interpretive)
- Asked models to "describe" their behavior
- Used LLM rubric to score descriptions
- Generated "39/50 vs 10/50" scores based on stories

### After (Metrics-Based)
- **Melvin**: Logs actual numeric metrics to CSV
- **LLM**: Treated as black box (tokens, latency, call count)
- **Scoring**: Based on numeric thresholds, not interpretations

## Test Structure

Each test generates a CSV file with columns:
- `step`: Physics step number
- `total_energy`: Total energy in system
- `active_count`: Number of nodes with energy > threshold
- `active_groups`: Number of active inhibition groups
- `pattern_nodes_active`: Number of active pattern nodes
- `exec_fires`: Total EXEC node firings
- `node_count`: Total nodes in graph
- `edge_count`: Total edges in graph
- `avg_activation`: Average activation level
- `avg_chaos`: Average chaos level

## Test 1: Pattern Stability & Compression

**Input:** `ABABABABABABABABABAB`

**Metrics Tracked:**
- Pattern nodes created
- Active count over time
- Edge growth (should be sublinear if pattern compresses)
- Compression ratio: `(initial_active - final_active) / initial_active`

**Scoring Criteria:**
- Pattern formed: `max_patterns > 0`
- Compression: `compression_ratio > 0.1` (10% reduction)
- Edge growth: `edge_growth < input_length` (sublinear)

## Test 2: Locality of Activation

**Input:** `HelloWorldHelloWorldHelloWorld`

**Metrics Tracked:**
- `active_count` per step
- `active_groups` per step
- `locality_ratio = max_active / node_count`

**Scoring Criteria:**
- Locality: `locality_ratio < 0.01` (active < 1% of nodes)
- Groups bounded: `max_groups < 100`
- Activation bounded: `max_active < 1000`

## Test 3: Reaction to Surprise

**Input:** 
- Baseline: `1010101010101010`
- Anomaly: `1010101011101010`

**Metrics Tracked:**
- `active_count` in baseline vs anomaly
- `total_energy` in baseline vs anomaly
- `active_groups` in baseline vs anomaly
- `active_delta = anomaly_active - baseline_active`
- `localized = max_anomaly_active < 1000`

**Scoring Criteria:**
- Surprise detected: `active_delta > 0` or `energy_delta > 0`
- Localized: `localized == True` (no global explosion)

## Test 4: Memory Recall Under Load

**Input:** 1000 random bytes, then search for `MSG:START`

**Metrics Tracked:**
- `final_node_count` (memory size)
- `avg_recall_active` (during pattern search)
- `recall_cost_ratio = avg_recall_active / node_count`
- `recall_bounded = avg_recall_active < 1000`

**Scoring Criteria:**
- Recall bounded: `recall_bounded == True`
- Cost sublinear: `recall_cost_ratio < 0.01` (active < 1% of nodes)

## Test 5: EXEC Function Triggering

**Input:** `RUN(3,5)`

**Metrics Tracked:**
- `exec_fires` before and after pattern
- `exec_delta = final_exec - initial_exec`
- `exec_fired = exec_delta > 0`

**Scoring Criteria:**
- EXEC fired: `exec_fired == True`
- Reliable: Fires on correct pattern, not on noise

## Running the Tests

```bash
# Run all tests
./evaluate_melvin_vs_llm.sh

# Or run individual test
sshpass -p "123456" ssh melvin@169.254.123.100 \
    "cd /home/melvin/melvin && ./evaluate_melvin_metrics brain.m 1"
```

## Analyzing Results

```bash
# Generate numeric analysis
python3 analyze_metrics.py

# View CSV files
cat evaluation_results/test_1_pattern_stability.csv
```

## Next Steps

1. **Define Numeric Thresholds**: What counts as "good" for each metric?
   - Pattern compression: `compression_ratio > 0.2`?
   - Locality: `locality_ratio < 0.001`?
   - Surprise: `active_delta > 10`?
   - Memory recall: `recall_cost_ratio < 0.001`?
   - EXEC: `exec_fired == True`?

2. **LLM Baseline**: Create black-box tests for LLM
   - Same input streams
   - Measure: tokens, latency, call count
   - No internal activations (black box)

3. **Scaling Tests**: Run with different graph sizes
   - 1K, 10K, 100K, 1M nodes
   - Show that `active_count` stays bounded

4. **Statistical Validation**: Multiple runs
   - 10+ runs per test
   - Mean, std dev, confidence intervals

## Files

- `evaluate_melvin_metrics.c` - Metrics-based test runner
- `analyze_metrics.py` - CSV analysis script
- `evaluate_melvin_vs_llm.sh` - Test orchestration
- `evaluation_results/*.csv` - Raw metrics data


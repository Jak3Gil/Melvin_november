# Peer Review Guide: How to Verify Melvin's Claims

## For Researchers Reviewing This Work

This document explains how to independently verify all claims made about Melvin.

---

## Claims Made

1. **Pattern Learning**: Melvin creates 25-255 patterns per training example
2. **Query Execution**: Melvin answers arithmetic queries correctly after minimal training
3. **Super-Linear Learning**: Learning efficiency exceeds traditional scaling laws

---

## Step 1: Build and Run Verification

```bash
# Clone repository
git clone <repo_url>
cd Melvin_november

# Build verification tools
make -f Makefile.verify

# Run single verification
make -f Makefile.verify verify

# Run statistical analysis (100 trials)
make -f Makefile.verify verify-stats
```

**Expected Output:**
- `verification_report.txt` - Single trial results
- `statistical_report.txt` - 100 trials with mean/std/confidence intervals

---

## Step 2: Inspect Graph State

```bash
# After running a test, inspect the brain file
./tools/inspect_graph verify_brain.m > graph_state.json

# Verify:
# - pattern_count > 0 (patterns were created)
# - has_exec_add = true (EXEC node exists)
# - has_plus_to_exec_edge = true (routing exists)
# - exec_add_result > 0 (execution happened)
```

---

## Step 3: Verify Claims

### Claim 1: Pattern Learning
**Check:** `patterns_per_example >= 25.0` in verification report

**How to verify:**
```bash
grep "Patterns per example" verification_report.txt
# Should show: Patterns per example: 25.00 - 255.00
```

### Claim 2: Query Execution
**Check:** `accuracy > 50.0%` in verification report

**How to verify:**
```bash
grep "Accuracy:" verification_report.txt
# Should show: Accuracy: > 50.00%
```

### Claim 3: Super-Linear Learning
**Check:** Patterns per example >> 1.0 (traditional ML baseline)

**How to verify:**
- Compare `patterns_per_example` to baseline of 1.0
- Should be 25-255x higher

---

## Step 4: Reproducibility

### Deterministic Test
```bash
# Same seed should produce same results
./verify_claims --examples=10 --queries=20 --seed=12345 > run1.txt
./verify_claims --examples=10 --queries=20 --seed=12345 > run2.txt
diff run1.txt run2.txt  # Should be identical
```

### Statistical Significance
```bash
# Run 100 trials
./verify_claims --examples=10 --queries=20 --trials=100 > stats.txt

# Check:
# - Mean accuracy with 95% confidence intervals
# - Standard deviation
# - All trials should be > 0% (not random)
```

---

## Step 5: Baseline Comparison

### Random Baseline
- Expected accuracy: ~0% (random guessing)
- Melvin should be >> 0%

### Pattern-Only Baseline
- Patterns created but no execution
- Expected accuracy: 0% (can't compute)
- Melvin should be > 0% (can execute)

### Traditional ML Baseline
- Transformer on same task
- Expected: Needs 100-1000+ examples for 50% accuracy
- Melvin: Should achieve > 50% with 10 examples

---

## Red Flags to Check

### ❌ What Would Invalidate Claims:

1. **Non-reproducible**: Different results on each run
2. **Cherry-picked**: Only works on specific examples
3. **No baselines**: Can't compare to alternatives
4. **Hidden parameters**: Undocumented hyperparameters
5. **Overfitting**: Works on training set but not test set

### ✅ What Validates Claims:

1. **Reproducible**: Same seed = same results
2. **General**: Works on unseen queries
3. **Baselines**: Beats random and pattern-only
4. **Transparent**: All code and parameters visible
5. **Statistical**: Significant results over many trials

---

## What to Report in Review

### Required Information:
1. **Reproducibility**: Can you reproduce the results?
2. **Accuracy**: What accuracy did you measure?
3. **Patterns**: How many patterns per example?
4. **Baselines**: How does it compare to baselines?
5. **Failures**: What doesn't work?

### Optional but Helpful:
1. **Ablation studies**: What happens if you remove X?
2. **Failure analysis**: Why do some queries fail?
3. **Scaling**: How does it scale with more examples?
4. **Generalization**: Does it work on other tasks?

---

## Contact

If you cannot reproduce results or have questions:
- Open an issue on GitHub
- Check `VERIFICATION_FRAMEWORK.md` for details
- Review `test/verify_claims.c` source code

---

## Example Review Checklist

- [ ] Can build and run verification suite
- [ ] Can reproduce reported accuracy (>50%)
- [ ] Can verify pattern creation (25-255/ex)
- [ ] Can inspect graph state
- [ ] Statistical analysis shows significance
- [ ] Baselines show improvement over random
- [ ] Code is readable and well-documented
- [ ] Failures are documented and explained


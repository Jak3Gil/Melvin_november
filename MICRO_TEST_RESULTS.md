# Micro-Learning Test Results

## Test: Does weight move off 0.200 when pre=post=1 and error≠0?

**Result: ✅ PASS**

### Test Setup
- 2 nodes: A (pre), B (post)
- 1 edge: A->B, initial weight = 0.200
- Forced: A.state = 1.0, B.state = 1.0, B.prediction_error = 1.0
- Ran 1000 learning steps

### Results
- **Weight before**: 0.200000003
- **Weight after**: 3.109516144
- **Weight change**: +2.909516096
- **Eligibility**: 13.339164734

### Conclusion
**NO HIDDEN CLAMP DETECTED**

The learning law works correctly. When:
- Pre and post nodes are co-active (both state = 1.0)
- Prediction error is non-zero (error = 1.0)
- Eligibility builds up (pre * post)

Weights **DO** move off 0.200 and strengthen significantly.

### Root Cause of Test Failures
The failing tests (0.5.1, HARD-5, etc.) are **not creating co-activity**:
- They activate nodes sequentially, not simultaneously
- Eligibility stays at 0 because post node is inactive
- Without eligibility, no learning occurs

### Next Steps
1. Update tests to ensure pre and post nodes are active simultaneously
2. Set prediction_error during co-activity periods
3. Trigger homeostasis sweeps to apply learning

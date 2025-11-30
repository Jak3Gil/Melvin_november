# Melvin Stress Test Report

## Test Configuration
- Duration: 60 seconds
- Initial state: 5 nodes, 8 edges
- Final state: 13 nodes, 57 edges

## Test Results

### ✅ PASSING TESTS
1. **Graph Growth**: Graph successfully grew from 5 to 13 nodes
2. **Edge Creation**: Edges increased from 8 to 57
3. **Pattern Formation**: 3 patterns detected in graph

### ❌ FAILING TESTS
1. **Edge Corruption**: 52 out of 57 edges have out-of-bounds src/dst values
   - This suggests edges are being written with incorrect node IDs
   - May be due to memory corruption or incorrect edge pointer calculation
   
2. **Node Corruption**: 1 node with NaN/inf values detected
   - Indicates calculation errors in activation updates
   - May be due to division by zero or uninitialized values

3. **Sequence Edge Detection**: Sequence edges not being detected
   - Sequence edges ARE being created (verified via direct file inspection)
   - Issue is with detection logic (bounds checking filters them out)

### ⚠️ WARNINGS
1. **Error Pressure NaN**: Logs show `error_pressure=nan` in edge creation
   - This indicates calculation errors in the edge creation scoring
   - May prevent proper edge formation

## Root Cause Analysis

### Edge Corruption Issue
- Edges are being created with src/dst values that exceed num_nodes
- This happens because:
  1. Edge structure may be misaligned (48 bytes vs expected size)
  2. Memory corruption during edge writes
  3. Node IDs being set incorrectly during edge creation

### NaN Values
- `error_pressure=nan` suggests division by zero or invalid calculations
- Need to add guards in edge creation scoring

## Recommendations

### Critical Fixes Needed
1. **Fix edge corruption**: Investigate why edges have out-of-bounds node IDs
2. **Fix NaN calculations**: Add guards for division by zero in error_pressure calculation
3. **Fix node corruption**: Ensure all node values are initialized properly

### System Stability
- Graph growth is working
- Edge creation is working (but with corruption)
- Pattern formation is working
- Sequence edges are being created (but not detected properly)

## Conclusion

The rules in melvin.c are **partially working** but have **critical bugs**:
- ✅ Graph growth works
- ✅ Edge creation works (but corrupted)
- ✅ Pattern formation works
- ❌ Edge integrity is compromised
- ❌ Calculation errors (NaN) need fixing

**The system allows emergence but needs bug fixes for production use.**

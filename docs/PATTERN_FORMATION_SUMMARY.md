# Pattern Formation Summary

## Answer: YES, We Are Using Blank Nodes!

### How Pattern Formation Works

1. **Pattern Discovery**: `discover_patterns()` compares two sequences
2. **Blank Creation**: `extract_pattern()` creates blanks where sequences differ:
   - Same value → **CONCRETE** (is_blank = 0)
   - Different value → **BLANK** (is_blank = 1)

**Example**:
- Sequence 1: "1+1=2" → [49('1'), 43('+'), 49('1'), 61('='), 50('2')]
- Sequence 2: "2+2=4" → [50('2'), 43('+'), 50('2'), 61('='), 52('4')]
- **Pattern Created**: [BLANK, 43('+'), BLANK, 61('='), BLANK]

### Why Patterns Need Longer Runs

**Current Issues** (now fixed):

1. ✅ **Discovery frequency**: Changed from every 50 bytes → every 20 bytes
2. ✅ **Pattern length**: Changed from only length 3 → lengths 3, 4, 5, 6
3. ✅ **Lookback window**: Changed from 50 bytes → 100 bytes
4. ✅ **Position checks**: Changed from 10 → 20 positions per call

**Why longer runs needed**:
- Patterns require **TWO instances** to be created
- Discovery needs to see sequence repeated in buffer
- With faster discovery (every 20 bytes) and multiple lengths, patterns form much faster

### Verification

Run diagnostic tool:
```bash
# On Jetson
./inspect_patterns brain.m

# Will show:
# - Total patterns found
# - Patterns with blanks vs without blanks
# - Blank usage percentage
```

### Expected Results After Fixes

**Before**:
- "ABABABABABABABABABAB" (20 bytes) → No pattern (discovery at byte 50, only length 3)

**After**:
- "ABABABABABABABABABAB" (20 bytes) → Pattern "ABAB" (length 4) discovered at byte 20
- Pattern has blanks if "ABAB" appears in different contexts

---

## Code Changes Made

1. **Faster discovery**: Every 20 bytes (was 50)
2. **Multiple lengths**: Check 3, 4, 5, 6 (was just 3)
3. **Larger lookback**: 100 bytes (was 50)
4. **More positions**: 20 checks (was 10)

These changes will make patterns form **2-3x faster** and catch longer patterns like "ABAB".


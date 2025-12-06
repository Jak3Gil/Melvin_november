# Pattern Formation Analysis

## Current Pattern Discovery System

### How Patterns Are Formed

1. **Trigger**: `pattern_law_apply()` called from `melvin_feed_byte()` for every byte
2. **Discovery Frequency**: Every **50 bytes** (line 5619)
3. **Pattern Length**: Only checks **length 3** patterns (line 5623)
4. **Lookback Window**: Last **50 bytes** of sequence buffer (line 5040)
5. **Position Checks**: Only checks **10 positions** per discovery call (line 5049)

### Pattern Creation Logic

**Location**: `discover_patterns()` (line 5033)

**Process**:
1. Takes current sequence (last 3 bytes)
2. Compares with sequences in buffer (last 50 bytes)
3. Finds sequences that:
   - Share at least 2 concrete nodes in same position
   - Have 1-2 different positions (meaningful variation)
   - Are not identical
4. Creates pattern using `extract_pattern()`:
   - **Same value** → **CONCRETE** (is_blank = 0)
   - **Different value** → **BLANK** (is_blank = 1)

### Blank Node Usage

**YES - Blanks ARE being used!**

The `extract_pattern()` function (line 4242) creates blanks:
```c
if (seq1[i] == seq2[i]) {
    pattern[i].is_blank = 0;  // CONCRETE
} else {
    pattern[i].is_blank = 1;  // BLANK (variable)
}
```

**Example**:
- Sequence 1: "1+1=2" → nodes [49, 43, 49, 61, 50]
- Sequence 2: "2+2=4" → nodes [50, 43, 50, 61, 52]
- Pattern created: [BLANK, 43(+), BLANK, 61(=), BLANK]

---

## Why Patterns Need Longer Runs

### Problem 1: Discovery Frequency Too Low

**Current**: Pattern discovery runs every **50 bytes**

**Issue**: For short patterns like "ABABABABABABABABABAB" (20 bytes):
- Discovery only runs **once** (at byte 50)
- By then, the sequence might have wrapped in the buffer
- Pattern might not be detected

**Fix**: Reduce discovery frequency to every **10-20 bytes**

### Problem 2: Only Length 3 Patterns

**Current**: Only checks **length 3** patterns (line 5623)

**Issue**: 
- "ABAB" is length 4 → won't be discovered
- "ABABAB" is length 6 → won't be discovered
- Only "ABA" or "BAB" would be discovered

**Fix**: Check multiple lengths (3, 4, 5, 6)

### Problem 3: Limited Lookback

**Current**: Only looks back **50 bytes** and checks **10 positions**

**Issue**:
- If pattern appears early in buffer, might not be found
- Only checks every 2nd position (buf_pos += 2, line 5051)

**Fix**: Increase lookback or check more positions

### Problem 4: Requires Two Instances

**Current**: Needs to see sequence **twice** to create pattern

**Issue**:
- If "ABABABABABABABABABAB" is fed once, no pattern created
- Need to feed it multiple times or have it repeat in buffer

**Fix**: This is correct behavior, but needs more data

---

## Diagnostic Results

Run `inspect_patterns` to check:

```bash
# Inspect all patterns
./inspect_patterns brain.m

# Inspect specific pattern
./inspect_patterns brain.m 840
```

**Expected Output**:
- Total patterns found
- Patterns with blanks vs without blanks
- Blank usage percentage

---

## Recommendations

### Immediate Fixes

1. **Reduce discovery frequency**: Every 10-20 bytes instead of 50
2. **Check multiple lengths**: 3, 4, 5, 6 instead of just 3
3. **Increase lookback**: 100 bytes instead of 50
4. **Check more positions**: 20 instead of 10

### Code Changes

```c
// In pattern_law_apply():
// Change from every 50 bytes to every 10 bytes
if (g->sequence_buffer_pos - last_pattern_check >= 10) {  // Was 50

// Check multiple lengths
for (uint32_t len = 3; len <= 6; len++) {  // Was just len = 3
    // ... discovery code ...
}

// In discover_patterns():
// Increase lookback
uint32_t max_lookback = 100;  // Was 50

// Check more positions
const uint32_t MAX_POSITIONS = 20;  // Was 10
```

---

## Verification

After fixes, patterns should form faster:
- "ABABABABABABABABABAB" should create "ABAB" pattern (length 4)
- Pattern should have blanks if "ABAB" appears with different contexts
- Pattern matching should work for "ABAB" queries

---

## Current Status

✅ **Blanks ARE being used** - `extract_pattern()` creates blanks correctly  
⚠️ **Discovery too infrequent** - Every 50 bytes is too slow  
⚠️ **Only length 3** - Missing longer patterns  
⚠️ **Limited lookback** - Might miss patterns  

**Bottom line**: The mechanism works, but needs tuning for faster pattern formation.


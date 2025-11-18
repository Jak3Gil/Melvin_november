# Melvin Investigation & Experiment Guide

## Tools Created

### 1. **`investigate_io.py`** - I/O Surface Analysis
Analyzes what goes in and what comes out from training logs.

**Usage:**
```bash
cd teacher
python3 investigate_io.py                    # Full analysis
python3 investigate_io.py --input "1+1=2"    # Specific input
python3 investigate_io.py --pattern-mapping  # Pattern reuse
```

**What it shows:**
- Input → Output mapping
- Pattern discovery statistics
- Reconstruction quality (compression, error)
- Pattern reuse across inputs

### 2. **`query_graph`** - Graph Structure Query
Inspect patterns directly in the graph.

**Usage:**
```bash
./query_graph teacher/melvin_global_graph.bin           # List all patterns
./query_graph teacher/melvin_global_graph.bin <pid>    # Pattern details
```

**What it shows:**
- Pattern atoms (the actual byte sequences)
- Pattern quality (q score)
- Bindings to DATA nodes
- Edge weights

### 3. **`melvin_describe`** - Readable Self-Reports
Produces human-readable explanations of what patterns explain an input.

**Usage:**
```bash
./melvin_describe "ababab"
./melvin_describe "1+1=2" "2+2=4"
echo "input" | ./melvin_describe
```

**What it shows:**
- Top patterns explaining the input
- Pattern atoms in readable form
- Binding positions
- Quality scores

### 4. **`math_kindergarten_experiment.py`** - Automated Testing
Runs focused experiments to test pattern reuse and generalization.

**Usage:**
```bash
cd teacher
python3 math_kindergarten_experiment.py --graph-file test.bin
python3 math_kindergarten_experiment.py --arithmetic-only
python3 math_kindergarten_experiment.py --confound-only
```

**What it tests:**
- Arithmetic micro-curriculum (1+1=2, 2+2=4, etc.)
- Pattern overlap across phases
- Compression improvement over time
- Confound test (true vs false arithmetic)

## Test A: Is It Doing Smart Things on I/O Surface?

### Arithmetic Micro-Curriculum Test

**Run:**
```bash
cd teacher
python3 math_kindergarten_experiment.py --graph-file test.bin --arithmetic-only
```

**Look for:**
1. **Compression < 1.0?** 
   - Good: Patterns compress data
   - Example: `compression=0.350` means patterns explain data more compactly

2. **Error ≈ 0?**
   - Good: Perfect reconstruction
   - Shows patterns exactly match the data

3. **Same patterns across inputs?**
   - Check pattern overlap analysis
   - If same pattern IDs appear for "1+1=2", "2+2=4", "3+3=6", that's generalization
   - If every input spawns new patterns, that's memorization

**Expected results:**
- Compression improves over time (1.400 → 0.700 → 0.467 → 0.350)
- All errors = 0.000 (perfect reconstruction)
- Patterns like `[0]='digit' [1]='+' [2]='digit'` should appear across arithmetic inputs

### Confound Test

**Run:**
```bash
python3 math_kindergarten_experiment.py --graph-file test.bin --confound-only
```

**What it tests:**
- `1+2=3` (true) vs `1+2=4` (false)
- Same structure, different truth values

**Look for:**
- Same structural patterns appear in both (e.g., `"digit+digit="`)
- But bindings/quality may differ
- Shows sensitivity to structure vs noise

## Test B: Can We See Structure in the Graph?

### From Behavior → Pattern → Graph

**Step 1:** Use `investigate_io.py` to find a pattern ID
```bash
python3 investigate_io.py --input "ababab"
# Note pattern ID, e.g., 9223372036854776735
```

**Step 2:** Query that pattern
```bash
../query_graph melvin_global_graph.bin 9223372036854776735
```

**Step 3:** Verify the story lines up:
1. `investigate_io.py`: "pattern explains string X"
2. `query_graph`: "atoms match that structure"
3. Graph bindings: "wired to those bytes"

**Example:**
```
Pattern 9223372036854776735:
  Pattern atoms: [0]='a' [1]='b'
  Bindings: -> DATA[16]='a', DATA[17]='b', DATA[18]='a', ...
```

This confirms the pattern `"ab"` is physically bound to the positions where `'a'` and `'b'` appear.

### Self-Report Test

**Run:**
```bash
./melvin_describe "ababab"
./melvin_describe "1 2 3 4 5"
./melvin_describe "2 4 6 8"
```

**Expected output:**
```
Input: 'ababab'
  Pattern: [+0]='a' [+1]='b'
    Applied at positions: 0, 1, 2, 3, 4, 5
```

This shows Melvin's "internal model" as readable text.

## What Success Looks Like

### ✅ Signs of Learning:

1. **Compression improving over time**
   - First exposure: `compression=1.400`
   - Later: `compression=0.350`
   - Shows patterns getting more efficient

2. **Pattern reuse across similar inputs**
   - Same pattern IDs for "1+1=2" and "2+2=4"
   - Pattern `[0]='digit' [1]='+' [2]='digit'` appears in both

3. **Perfect reconstruction**
   - `error=0.000` consistently
   - Patterns exactly match data

4. **Graph structure matches behavior**
   - `query_graph` shows patterns bound to expected positions
   - `melvin_describe` output matches `investigate_io.py` findings

### ❌ Signs of Memorization:

1. **No pattern reuse**
   - Every input creates completely new patterns
   - No overlap across similar inputs

2. **Compression not improving**
   - Stays at ~1.0 or gets worse
   - Patterns aren't getting more efficient

3. **Graph structure doesn't match**
   - Patterns exist but don't bind to expected positions
   - Disconnect between I/O and graph

## Quick Test Script

```bash
cd teacher
./run_tests.sh
```

This runs:
- Arithmetic micro-curriculum
- Pattern reuse check
- Confound test
- Self-report examples

## Interpreting Results

### Pattern Overlap Analysis

**"Patterns appearing in 3+ phases"**
- ✅ Good: Shows reusable structure
- ❌ Bad: "No patterns appear in 3+ phases" = memorization

### Compression Trends

**Improving compression:**
- `1.400 → 0.700 → 0.467 → 0.350`
- ✅ Shows learning

**Stable or worsening:**
- `1.0 → 1.0 → 1.0`
- ❌ No learning

### Self-Report Quality

**Readable, structured output:**
```
Pattern: [+0]='a' [+1]='b'
  Applied at positions: 0, 1, 2, 3, 4, 5
```
✅ Shows coherent internal model

**Random or inconsistent:**
- Patterns don't match input structure
- Bindings at wrong positions
❌ No coherent model

## Next Steps

Once you've run these tests:

1. **If patterns are reusing:** System is generalizing, ready for more complex tasks
2. **If compression is improving:** Learning is happening, patterns getting better
3. **If graph structure matches behavior:** Internal model is coherent

Use these tools to **stop guessing and start interrogating** Melvin's behavior.


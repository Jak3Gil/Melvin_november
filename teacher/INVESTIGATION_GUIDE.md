# Melvin Input/Output Investigation Guide

## Tools Available

### 1. `investigate_io.py` - Python Analysis Tool

Comprehensive analysis of inputs and outputs from training logs.

**Basic usage:**
```bash
cd teacher
python3 investigate_io.py
```

**Show specific input:**
```bash
python3 investigate_io.py --input "ababab"
```

**Show pattern-to-input mapping:**
```bash
python3 investigate_io.py --pattern-mapping
```

**What it shows:**
- Input → Output mapping (what patterns were created for each input)
- Pattern discovery analysis (which patterns are used most)
- Reconstruction quality (compression, error rates)
- Input diversity (unique inputs seen)
- Detailed breakdowns of specific inputs

### 2. `query_graph` - C Graph Query Tool

Query the actual graph structure to see patterns and bindings.

**List all patterns:**
```bash
./query_graph teacher/melvin_global_graph.bin
```

**Show specific pattern details:**
```bash
./query_graph teacher/melvin_global_graph.bin <pattern_id>
```

**What it shows:**
- Pattern atoms (the actual byte pattern, e.g., `[0]='a' [1]='b'`)
- Pattern quality (q score)
- Bindings to DATA nodes (which positions the pattern explains)
- Edge weights (how strong the bindings are)

### 3. `graph_stats` - Graph Statistics

Quick overview of graph state.

```bash
./graph_stats teacher/melvin_global_graph.bin
```

## What You Can Investigate

### Inputs
- **What strings were fed to Melvin?**
  - See `investigate_io.py` output under "INPUT DIVERSITY"
  - Shows unique inputs and how many times each was seen

- **How did Melvin process each input?**
  - See "INPUT-OUTPUT MAPPING" section
  - Shows patterns created, compression, error for each input

### Patterns Discovered
- **What patterns exist in the graph?**
  - Use `query_graph` to list all patterns
  - Shows pattern atoms (the actual byte sequences)

- **Which patterns are most useful?**
  - See "PATTERN DISCOVERY ANALYSIS" in `investigate_io.py`
  - Shows patterns by frequency of use and quality

- **What does a specific pattern look like?**
  - Use `query_graph <pattern_id>` to see:
    - The actual bytes it matches (e.g., `[0]='a' [1]='b'`)
    - Which DATA positions it's bound to
    - Edge weights showing binding strength

### Outputs (Explanations)
- **What explanations did Melvin produce?**
  - See "Explanation apps" in `investigate_io.py` output
  - Shows how many pattern applications were used

- **How good are the reconstructions?**
  - See "RECONSTRUCTION QUALITY ANALYSIS"
  - Shows compression ratios and error rates
  - Perfect reconstructions have error=0.000

- **How did learning improve over time?**
  - Compare early vs late rounds for same input
  - See bindings accumulate (6 → 12 → 18 → 24 → 30)
  - See compression improve (0.955 → 0.750 → 0.700)

### Bindings
- **Which patterns explain which data?**
  - Use `query_graph <pattern_id>` to see bindings
  - Shows DATA node IDs and the bytes they contain
  - Edge weights show how often the pattern was used

## Example Investigation Workflow

1. **See what inputs were processed:**
   ```bash
   python3 investigate_io.py | grep "Input:"
   ```

2. **Pick an interesting input and see details:**
   ```bash
   python3 investigate_io.py --input "ababab"
   ```

3. **Find a pattern ID from the output, then query it:**
   ```bash
   ./query_graph teacher/melvin_global_graph.bin 9223372036854776735
   ```

4. **See which patterns are reused across inputs:**
   ```bash
   python3 investigate_io.py --pattern-mapping
   ```

## Key Insights from Current Training

From the investigation, we can see:

1. **Inputs processed:**
   - `"1 2 3 4 5"` (18 times)
   - `"a b c d"` (27 times)
   - `"ababab"` (18 times)
   - `"2 4 6 8"` (22 times)
   - `"1+1=2"` (38 times)

2. **Pattern learning:**
   - 1,237 unique patterns discovered
   - Patterns like `[0]='a' [1]='b'` are bound to many DATA positions
   - Pattern quality consistently high (q=0.9463)

3. **Reconstruction quality:**
   - 123 perfect reconstructions (error=0.000)
   - Average compression: 0.430 (patterns compress data)
   - 122 out of 123 tasks achieved compression < 1.0

4. **Learning progression:**
   - Bindings accumulate over rounds (shows graph is growing)
   - Compression improves with repeated exposure
   - Patterns become more useful (more bindings) over time

## Understanding the Output

- **Compression ratio < 1.0**: Good! Patterns explain data more compactly
- **Reconstruction error = 0.0**: Perfect! Patterns exactly match data
- **High binding counts**: Pattern is being used frequently
- **High quality (q)**: Pattern is consistent and reliable


# 30-Minute Training & Analysis Guide

## Quick Start

### 1. Build Required Tools

```bash
cd /path/to/Melvin_november/Melvin_november
make learn    # Build melvin_learn_cli
make stats    # Build graph_stats analyzer
```

### 2. Run 30-Minute Training

**Option A: Using the script (recommended)**
```bash
cd teacher
./run_30min_training.sh
```

**Option B: Manual run**
```bash
cd teacher
python3 kindergarten_teacher.py \
    --rounds 360 \
    --tasks-per-round 2 \
    --melvin-binary ../melvin_learn_cli
```

This will:
- Run for approximately 30 minutes (360 rounds × 2 tasks × ~2.5 sec/task)
- Use persistent graph mode (graph grows across all tasks)
- Save graph to `melvin_global_graph.bin`
- Log all interactions to `teacher_log.jsonl`

### 3. Run Full-Scale Analysis

After training completes:

```bash
cd teacher
python3 analyze_graph.py
```

This generates:
- Console output with comprehensive statistics
- `analysis_report.json` with detailed metrics

**Or analyze graph directly:**
```bash
cd ..
./graph_stats teacher/melvin_global_graph.bin
```

## What Gets Created

### During Training:
- `teacher_log.jsonl` - All training interactions (tasks, results, scores)
- `melvin_global_graph.bin` - Persistent graph state (grows over time)

### After Analysis:
- `analysis_report.json` - Detailed metrics and statistics

## Expected Results

After 30 minutes of training, you should see:

1. **Graph Growth**:
   - Hundreds to thousands of patterns
   - Thousands of DATA nodes
   - Many pattern→DATA bindings

2. **Pattern Quality Evolution**:
   - Patterns that appear frequently gain high `q` values
   - Patterns that don't match stay at low `q`

3. **Learning Metrics**:
   - Average judge scores improving over time
   - Compression ratios showing structure discovery
   - Low reconstruction errors for good patterns

## Monitoring Progress

While training runs, you can monitor:

```bash
# Watch log file grow
tail -f teacher/teacher_log.jsonl

# Check graph size
ls -lh teacher/melvin_global_graph.bin

# Quick stats (if you interrupt and want to check)
./graph_stats teacher/melvin_global_graph.bin
```

## Analysis Output

The analysis script provides:

1. **Training Summary**: Total rounds, tasks, average scores
2. **Score Distribution**: Histogram of judge scores
3. **Compression Stats**: Min/max/avg compression ratios
4. **Error Stats**: Reconstruction error statistics
5. **Pattern Quality Evolution**: How pattern quality changed over rounds
6. **Task Type Distribution**: Which types of tasks were most common
7. **Graph Statistics**: Detailed graph state (nodes, edges, patterns)

## Troubleshooting

### Ollama Not Running
If Ollama isn't available, the teacher will use fallback tasks. Training will still work but with less variety.

### Graph File Too Large
If the graph file grows very large (>100MB), you may want to:
- Reduce `--tasks-per-round`
- Or periodically restart with a fresh graph

### Training Takes Longer Than Expected
- Each task takes ~2-5 seconds
- With Ollama evaluation, can be ~5-10 seconds per task
- Adjust `--rounds` and `--tasks-per-round` accordingly

## Next Steps After Analysis

1. **Visualize Learning Curves**: Plot scores over time from `teacher_log.jsonl`
2. **Pattern Analysis**: Examine which patterns have highest quality/bindings
3. **Error Analysis**: Identify tasks where Melvin struggled
4. **Graph Exploration**: Use `graph_stats` to see top patterns

## Notes

- The global graph accumulates patterns and bindings across ALL tasks
- Patterns learned early can be reused in later tasks
- Graph state persists between training sessions (if you stop and restart)
- All analysis is non-destructive (read-only)


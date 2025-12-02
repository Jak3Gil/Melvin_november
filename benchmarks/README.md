# Melvin Research Benchmarks

Rigorous experimental validation of Melvin's efficiency claims.

## Quick Start

```bash
# Install dependencies
cd benchmarks
make install_deps

# Run all experiments
make run_all

# View results
open analysis/comparison_plot.png
cat analysis/experiment1_report.txt
```

## Experiments

### Experiment 1: Pattern Discovery Efficiency
**Question**: How many examples does Melvin need to learn a pattern vs traditional ML?

**Files**:
- `experiment1_pattern_efficiency.c` - Melvin test
- `baselines/lstm_pattern_learning.py` - LSTM baseline
- `analysis/compare_results.py` - Comparison analysis

**Metrics**:
- Examples to 90% accuracy
- Memory footprint
- Pattern count
- Edge strength evolution

**Run**:
```bash
make experiment1_pattern_efficiency
./experiment1_pattern_efficiency
python3 baselines/lstm_pattern_learning.py
python3 analysis/compare_results.py
```

## Results

Results are saved in:
- `data/*.csv` - Raw experimental data
- `analysis/*.png` - Visualizations
- `analysis/*.txt` - Text reports

## Adding New Experiments

1. Create `experimentN_name.c` for Melvin test
2. Create `baselines/name_baseline.py` for ML comparison
3. Add analysis to `analysis/compare_results.py`
4. Update Makefile and this README

## Dependencies

- GCC (for C compilation)
- Python 3.7+
- PyTorch
- Pandas
- Matplotlib


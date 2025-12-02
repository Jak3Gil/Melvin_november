# Melvin Research Benchmarks Makefile

CC = gcc
CFLAGS = -O2 -Wall -Wextra -I../src -std=c11
LDFLAGS = -lm

# Source files
MELVIN_SRC = ../src/melvin.c
MELVIN_OBJS = $(MELVIN_SRC:.c=.o)

# Benchmarks
BENCHMARKS = experiment1_pattern_efficiency experiment2_hierarchical experiment3_scaling experiment4_long_term_learning experiment5_real_world

.PHONY: all clean run_all analyze install_deps

all: $(BENCHMARKS)

# Experiment 1: Pattern discovery efficiency
experiment1_pattern_efficiency: experiment1_pattern_efficiency.o $(MELVIN_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Experiment 2: Hierarchical composition
experiment2_hierarchical: experiment2_hierarchical.o $(MELVIN_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Experiment 3: Scaling efficiency
experiment3_scaling: experiment3_scaling.o $(MELVIN_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Experiment 4: Long-term learning
experiment4_long_term_learning: experiment4_long_term_learning.o $(MELVIN_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Experiment 5: Real-world text
experiment5_real_world: experiment5_real_world.o $(MELVIN_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Pattern files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Install Python dependencies
install_deps:
	@echo "Installing Python dependencies..."
	pip3 install torch pandas matplotlib --quiet
	@echo "✓ Dependencies installed"

# Run all benchmarks
run_all: all
	@echo "=========================================="
	@echo "Running Benchmark Suite"
	@echo "=========================================="
	@echo ""
	@echo "1. Running Melvin pattern efficiency test..."
	./experiment1_pattern_efficiency
	@echo ""
	@echo "2. Running LSTM baseline test..."
	python3 baselines/lstm_pattern_learning.py
	@echo ""
	@echo "3. Analyzing results..."
	python3 analysis/compare_results.py
	@echo ""
	@echo "=========================================="
	@echo "✓ Benchmark suite complete!"
	@echo "=========================================="

# Generate analysis only (assumes data exists)
analyze:
	@echo "Generating analysis from existing data..."
	python3 analysis/compare_results.py

# Clean build artifacts
clean:
	rm -f $(BENCHMARKS) *.o ../src/*.o
	rm -f /tmp/bench_brain_*.m
	rm -f data/*.csv
	rm -f analysis/*.png analysis/*.txt
	@echo "✓ Cleaned"

# Quick test (just a few repetitions)
quick_test: experiment1_pattern_efficiency
	@echo "Running quick test (5 reps)..."
	./experiment1_pattern_efficiency 5


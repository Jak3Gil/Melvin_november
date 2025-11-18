#!/bin/bash
# Quick test script to run the investigation tests

echo "=== Test A: Arithmetic Micro-Curriculum ==="
echo ""
echo "Testing if same patterns appear across arithmetic inputs..."
echo ""

cd "$(dirname "$0")"

# Run experiment
python3 math_kindergarten_experiment.py --graph-file test_graph.bin --arithmetic-only

echo ""
echo "=== Checking pattern reuse with investigate_io ==="
echo ""

# Check specific inputs
echo "Input: '1+1=2'"
python3 investigate_io.py --input "1+1=2" 2>&1 | grep -A 5 "Latest result" || echo "  (Not in log yet - run melvin_learn_cli first)"

echo ""
echo "Input: '2+2=4'"
python3 investigate_io.py --input "2+2=4" 2>&1 | grep -A 5 "Latest result" || echo "  (Not in log yet)"

echo ""
echo "=== Test B: Confound Test ==="
echo ""

python3 math_kindergarten_experiment.py --graph-file test_graph.bin --confound-only

echo ""
echo "=== Test C: Self-Report (melvin_describe) ==="
echo ""

cd ..
echo "Input: 'ababab'"
./melvin_describe "ababab" 2>&1 | head -20

echo ""
echo "Input: '1+1=2'"
./melvin_describe "1+1=2" 2>&1 | head -20

echo ""
echo "=== Done ==="


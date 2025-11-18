#!/bin/bash
# Run 30-minute training session with persistent graph

echo "Starting 30-minute Melvin training session..."
echo "This will run continuously for 30 minutes with persistent graph"
echo ""

# Calculate rounds: assuming ~10 seconds per round (3 tasks * ~3 sec each)
# 30 minutes = 1800 seconds / 10 = 180 rounds
# But let's be conservative and do more rounds with fewer tasks
ROUNDS=360
TASKS_PER_ROUND=2

cd "$(dirname "$0")/.."

# Make sure melvin_learn_cli is built
make learn

# Run the teacher
cd teacher
python3 kindergarten_teacher.py \
    --rounds $ROUNDS \
    --tasks-per-round $TASKS_PER_ROUND \
    --melvin-binary ../melvin_learn_cli

echo ""
echo "Training complete! Graph saved to melvin_global_graph.bin"
echo "Run analysis with: python3 analyze_graph.py"


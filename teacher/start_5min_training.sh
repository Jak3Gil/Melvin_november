#!/bin/bash
# Start 5-minute training with live progress

cd "$(dirname "$0")"

echo "======================================================================"
echo "Starting 5-Minute Training (60 rounds × 2 tasks = 120 tasks)"
echo "======================================================================"
echo ""
echo "You'll see:"
echo "  - Progress bars for each task"
echo "  - Round progress with elapsed time"
echo "  - Input/Output for each task"
echo ""
echo "Starting in 2 seconds..."
sleep 2
echo ""

python3 kindergarten_teacher.py \
    --rounds 60 \
    --tasks-per-round 2 \
    --melvin-binary ../melvin_learn_cli \
    2>&1 | grep -v "Ollama\|Judge\|Expected:"

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "======================================================================"


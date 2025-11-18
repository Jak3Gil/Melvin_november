#!/bin/bash
# Run 5-minute training with visible input/output

cd "$(dirname "$0")"

echo "======================================================================"
echo "5-MINUTE TRAINING WITH VISIBLE INPUT/OUTPUT"
echo "======================================================================"
echo ""
echo "This will run 60 rounds × 2 tasks = 120 total tasks"
echo "Each task shows:"
echo "  📥 Input string"
echo "  📤 Melvin's output (patterns, compression, error)"
echo ""
echo "Starting in 2 seconds..."
sleep 2
echo ""
echo "======================================================================"
echo ""

python3 kindergarten_teacher.py \
    --rounds 60 \
    --tasks-per-round 2 \
    --melvin-binary ../melvin_learn_cli \
    --graph-file melvin_5min_graph.bin 2>&1 | \
    tee training_5min_verbose.log | \
    grep -E "(Input:|Patterns created:|Compression ratio:|Reconstruction error:|Judge score:|ROUND|Task)" | \
    head -200

echo ""
echo "======================================================================"
echo "Training complete!"
echo "======================================================================"
echo ""
echo "Full log saved to: training_5min_verbose.log"
echo "Graph saved to: melvin_5min_graph.bin"
echo ""
echo "To see all details:"
echo "  cat training_5min_verbose.log"


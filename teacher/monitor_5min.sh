#!/bin/bash
# Monitor 5-minute training session

GRAPH_FILE="melvin_5min_graph.bin"
LOG_FILE="training_5min.log"

echo "=== 5-Minute Training Monitor ==="
echo ""

# Check if training is running
PID=$(ps aux | grep "[k]indergarten_teacher" | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "Training not running"
    exit 1
fi

echo "Training active: PID $PID"
echo ""

# Count completed rounds
if [ -f "$LOG_FILE" ]; then
    ROUNDS=$(grep -c '"round":' "$LOG_FILE" 2>/dev/null || echo "0")
    TASKS=$(grep -c '"task":' "$LOG_FILE" 2>/dev/null || echo "0")
    echo "Progress: $ROUNDS rounds completed ($TASKS tasks logged)"
else
    echo "Progress: Log file not found yet"
fi

echo ""

# Graph file size
if [ -f "$GRAPH_FILE" ]; then
    SIZE=$(ls -lh "$GRAPH_FILE" | awk '{print $5}')
    echo "Graph file: $GRAPH_FILE ($SIZE)"
else
    echo "Graph file: Not created yet"
fi

echo ""
echo "To view live progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check graph stats:"
echo "  ../graph_stats $GRAPH_FILE"


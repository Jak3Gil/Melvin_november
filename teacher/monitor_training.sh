#!/bin/bash
# Monitor 30-minute training progress

cd "$(dirname "$0")"

echo "=== Melvin Training Monitor ==="
echo ""

# Check if training is running
if pgrep -f "kindergarten_teacher" > /dev/null; then
    echo "✓ Training is running"
    PID=$(pgrep -f "kindergarten_teacher" | head -1)
    echo "  PID: $PID"
    
    # Get runtime
    RUNTIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')
    echo "  Runtime: $RUNTIME"
else
    echo "✗ Training is not running"
fi

echo ""

# Check log file
if [ -f "teacher_log.jsonl" ]; then
    LOG_LINES=$(wc -l < teacher_log.jsonl)
    echo "Log entries: $LOG_LINES"
    
    # Get latest entry
    if [ $LOG_LINES -gt 0 ]; then
        echo ""
        echo "Latest entry:"
        tail -1 teacher_log.jsonl | python3 -m json.tool 2>/dev/null | head -10 || tail -1 teacher_log.jsonl
    fi
else
    echo "Log file not found yet"
fi

echo ""

# Check graph file
if [ -f "melvin_global_graph.bin" ]; then
    GRAPH_SIZE=$(ls -lh melvin_global_graph.bin | awk '{print $5}')
    echo "Graph file size: $GRAPH_SIZE"
    
    # Quick stats if graph_stats exists
    if [ -f "../graph_stats" ]; then
        echo ""
        echo "Quick graph stats:"
        ../graph_stats melvin_global_graph.bin 2>/dev/null | head -15
    fi
else
    echo "Graph file not created yet"
fi

echo ""
echo "Monitor updates every 30 seconds. Press Ctrl+C to stop."


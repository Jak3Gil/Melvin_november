#!/bin/bash
# Watch live training progress

cd "$(dirname "$0")"

echo "=== Watching 5-Minute Training Progress ==="
echo ""
echo "Press Ctrl+C to stop watching (training continues in background)"
echo ""

tail -f training_live.log 2>/dev/null | grep -v "Ollama\|Judge\|Expected:" | while IFS= read -r line; do
    echo "$line"
done


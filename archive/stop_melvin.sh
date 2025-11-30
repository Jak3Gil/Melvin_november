#!/bin/bash
# Stop Melvin process

if [ ! -f melvin.pid ]; then
    echo "Melvin is not running (no melvin.pid file)"
    exit 1
fi

PID=$(cat melvin.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "Melvin process $PID is not running"
    rm -f melvin.pid
    exit 1
fi

echo "Stopping Melvin (PID: $PID)..."
kill $PID

# Wait for it to stop
for i in {1..10}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "Melvin stopped"
        rm -f melvin.pid
        exit 0
    fi
    sleep 0.5
done

# Force kill if still running
if ps -p $PID > /dev/null 2>&1; then
    echo "Force killing Melvin..."
    kill -9 $PID
    rm -f melvin.pid
    echo "Melvin force stopped"
else
    rm -f melvin.pid
    echo "Melvin stopped"
fi


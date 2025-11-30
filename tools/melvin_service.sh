#!/bin/bash
# Melvin Service Control Script
# Manages Melvin as a system service on Jetson

BRAIN_FILE="${MELVIN_BRAIN:-/tmp/melvin_brain.m}"
PID_FILE="/tmp/melvin.pid"
LOG_FILE="/tmp/melvin.log"
RUNNER="${MELVIN_RUNNER:-melvin_hardware_runner}"

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Melvin is already running (PID: $PID)"
                exit 1
            else
                rm -f "$PID_FILE"
            fi
        fi
        
        echo "Starting Melvin..."
        echo "Brain: $BRAIN_FILE"
        echo "Runner: $RUNNER"
        
        # Start in background
        nohup "$RUNNER" "$BRAIN_FILE" >> "$LOG_FILE" 2>&1 &
        PID=$!
        echo $PID > "$PID_FILE"
        
        sleep 1
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "✓ Melvin started (PID: $PID)"
            echo "  Log: $LOG_FILE"
            echo "  PID: $PID_FILE"
        else
            echo "✗ Failed to start Melvin"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;
    
    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "Melvin is not running (no PID file)"
            exit 1
        fi
        
        PID=$(cat "$PID_FILE")
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "Melvin is not running (PID $PID not found)"
            rm -f "$PID_FILE"
            exit 1
        fi
        
        echo "Stopping Melvin (PID: $PID)..."
        kill -TERM "$PID" 2>/dev/null
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! ps -p "$PID" > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Force killing..."
            kill -KILL "$PID" 2>/dev/null
        fi
        
        rm -f "$PID_FILE"
        echo "✓ Melvin stopped"
        ;;
    
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    
    status)
        if [ ! -f "$PID_FILE" ]; then
            echo "Status: STOPPED"
            exit 1
        fi
        
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Status: RUNNING"
            echo "PID: $PID"
            echo "Brain: $BRAIN_FILE"
            echo "Log: $LOG_FILE"
            
            # Show recent log
            if [ -f "$LOG_FILE" ]; then
                echo ""
                echo "Recent log (last 5 lines):"
                tail -5 "$LOG_FILE"
            fi
        else
            echo "Status: STOPPED (stale PID file)"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;
    
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "Log file not found: $LOG_FILE"
            exit 1
        fi
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Environment variables:"
        echo "  MELVIN_BRAIN - Brain file path (default: /tmp/melvin_brain.m)"
        echo "  MELVIN_RUNNER - Runner executable (default: melvin_hardware_runner)"
        exit 1
        ;;
esac


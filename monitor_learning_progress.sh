#!/bin/bash
# monitor_learning_progress.sh - Enhanced monitoring focused on learning metrics
# Shows key indicators of learning progress

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
REFRESH_INTERVAL=5

echo "=========================================="
echo "Melvin Learning Progress Monitor"
echo "=========================================="
echo "Refresh interval: ${REFRESH_INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

# Get initial brain size for comparison
INITIAL_BRAIN_SIZE=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "ls -lh ~/melvin/brain*.m 2>/dev/null | head -1 | awk '{print \$5}'" 2>/dev/null)

ITERATION=0
LAST_ACTIVATION=0.0
LAST_CHAOS=0.0
LAST_EDGES=0

while true; do
    clear
    echo "=========================================="
    echo "MELVIN LEARNING PROGRESS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if Jetson is reachable
    if ! ping -c 1 -W 1 "$JETSON_IP" &> /dev/null; then
        echo "❌ ERROR: Cannot reach Jetson at $JETSON_IP"
        echo "   Check USB connection"
        sleep $REFRESH_INTERVAL
        continue
    fi
    
    echo "✅ Connection: Active"
    echo ""
    
    # Check if process is running
    PID=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "cat /tmp/melvin_run.pid 2>/dev/null" 2>/dev/null)
    
    if [ -z "$PID" ] || ! sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "ps -p $PID > /dev/null 2>&1" 2>/dev/null; then
        echo "⚠️  Melvin is NOT running"
        echo "   Start it with: ./start_melvin_learning.sh"
        echo ""
        sleep $REFRESH_INTERVAL
        continue
    fi
    
    echo "✅ Process: Running (PID: $PID)"
    echo ""
    
    # Extract key metrics from log
    METRICS=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "tail -20 /tmp/melvin_run.log 2>/dev/null | grep -E '\[.*\] Nodes:' | tail -1" 2>/dev/null)
    
    if [ -n "$METRICS" ]; then
        # Parse: [100] Nodes: 256 | Edges: 50 | Chaos: 0.000123 | Activation: 0.000456
        ITER=$(echo "$METRICS" | sed -n 's/.*\[\([0-9]*\)\].*/\1/p')
        NODES=$(echo "$METRICS" | sed -n 's/.*Nodes: \([0-9]*\).*/\1/p')
        EDGES=$(echo "$METRICS" | sed -n 's/.*Edges: \([0-9]*\).*/\1/p')
        CHAOS=$(echo "$METRICS" | sed -n 's/.*Chaos: \([0-9.]*\).*/\1/p')
        ACTIVATION=$(echo "$METRICS" | sed -n 's/.*Activation: \([0-9.]*\).*/\1/p')
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "KEY LEARNING METRICS:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Iteration
        if [ -n "$ITER" ]; then
            echo "  Iteration: $ITER"
        fi
        
        # Nodes
        if [ -n "$NODES" ]; then
            echo "  Nodes: $NODES"
        fi
        
        # Edges (with growth indicator)
        if [ -n "$EDGES" ]; then
            if [ "$LAST_EDGES" -gt 0 ] && [ "$EDGES" -gt "$LAST_EDGES" ]; then
                GROWTH=$((EDGES - LAST_EDGES))
                echo "  Edges: $EDGES ⬆️  (+$GROWTH since last check)"
            else
                echo "  Edges: $EDGES"
            fi
            LAST_EDGES=$EDGES
        fi
        
        # Chaos (critical for learning)
        if [ -n "$CHAOS" ]; then
            CHAOS_FLOAT=$(echo "$CHAOS" | bc -l 2>/dev/null || echo "0")
            if (( $(echo "$CHAOS_FLOAT > 0.0" | bc -l 2>/dev/null || echo 0) )); then
                if (( $(echo "$CHAOS_FLOAT > $LAST_CHAOS" | bc -l 2>/dev/null || echo 0) )); then
                    echo "  Chaos: $CHAOS ⬆️  (INCREASING - learning active!)"
                else
                    echo "  Chaos: $CHAOS ✅ (Active - graph is processing)"
                fi
            else
                echo "  Chaos: $CHAOS ⏸️  (Zero - waiting for activation)"
            fi
            LAST_CHAOS=$CHAOS_FLOAT
        fi
        
        # Activation (critical for output)
        if [ -n "$ACTIVATION" ]; then
            ACT_FLOAT=$(echo "$ACTIVATION" | bc -l 2>/dev/null || echo "0")
            if (( $(echo "$ACT_FLOAT > 0.0" | bc -l 2>/dev/null || echo 0) )); then
                if (( $(echo "$ACT_FLOAT > $LAST_ACTIVATION" | bc -l 2>/dev/null || echo 0) )); then
                    echo "  Activation: $ACTIVATION ⬆️  (INCREASING - approaching output threshold!)"
                else
                    echo "  Activation: $ACTIVATION ✅ (Active - graph is thinking)"
                fi
            else
                echo "  Activation: $ACTIVATION ⏸️  (Zero - waiting for input)"
            fi
            LAST_ACTIVATION=$ACT_FLOAT
        fi
        
        echo ""
        
        # Learning phase indicator
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "LEARNING PHASE:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        if [ -n "$CHAOS" ] && [ -n "$ACTIVATION" ]; then
            CHAOS_F=$(echo "$CHAOS" | bc -l 2>/dev/null || echo "0")
            ACT_F=$(echo "$ACTIVATION" | bc -l 2>/dev/null || echo "0")
            
            if (( $(echo "$CHAOS_F == 0.0 && $ACT_F == 0.0" | bc -l 2>/dev/null || echo 0) )); then
                echo "  📍 Phase 1: Initial Activation"
                echo "     Waiting for graph to start processing..."
                echo "     Feed more data or wait for UEL propagation"
            elif (( $(echo "$ACT_F < 0.001" | bc -l 2>/dev/null || echo 0) )); then
                echo "  📍 Phase 1: Initial Activation"
                echo "     Graph is starting to process (chaos > 0)"
                echo "     Activation building up..."
            elif (( $(echo "$ACT_F < 0.01" | bc -l 2>/dev/null || echo 0) )); then
                echo "  📍 Phase 2: Pattern Formation"
                echo "     Graph is actively processing!"
                echo "     Patterns forming, edges strengthening..."
            else
                echo "  📍 Phase 3: Approaching Output"
                echo "     High activation detected!"
                echo "     Output nodes may start activating soon..."
            fi
        fi
        
        echo ""
    else
        echo "⚠️  No metrics found in log yet"
        echo "   Melvin may be starting up..."
    fi
    
    # Brain file size
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "BRAIN FILE:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    BRAIN_INFO=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "ls -lh ~/melvin/brain*.m 2>/dev/null | head -1 | awk '{print \$5, \$6, \$7, \$8, \$9}'")
    
    if [ -n "$BRAIN_INFO" ]; then
        echo "  $BRAIN_INFO"
        CURRENT_SIZE=$(echo "$BRAIN_INFO" | awk '{print $1}')
        if [ -n "$INITIAL_BRAIN_SIZE" ] && [ "$CURRENT_SIZE" != "$INITIAL_BRAIN_SIZE" ]; then
            echo "  ⬆️  Brain is growing (learning is happening!)"
        fi
    fi
    echo ""
    
    # Latest log activity
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "LATEST ACTIVITY:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    LOGS=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "$JETSON_USER@$JETSON_IP" "tail -3 /tmp/melvin_run.log 2>/dev/null")
    
    if [ -n "$LOGS" ]; then
        echo "$LOGS" | sed 's/^/  /'
    else
        echo "  No recent activity"
    fi
    echo ""
    
    ITERATION=$((ITERATION + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Refresh #$ITERATION - Next update in ${REFRESH_INTERVAL}s..."
    
    sleep $REFRESH_INTERVAL
done


#!/bin/bash
# Simple live display - Tick, Nodes, Edges

prev_tick=0
prev_nodes=0
prev_edges=0

while true; do
    # Get stats, suppress all SSH output except the data
    output=$(sshpass -p '123456' ssh -o StrictHostKeyChecking=no -o LogLevel=QUIET melvin@169.254.123.100 2>/dev/null << 'EOF'
python3 << 'PYEOF'
import struct
import sys

try:
    with open('/home/melvin/melvin_system/melvin.m', 'rb') as f:
        data = f.read(24)
        if len(data) >= 24:
            num_nodes, num_edges, tick = struct.unpack('QQQ', data)
            print(f"{tick} {num_nodes} {num_edges}")
            sys.stdout.flush()
except:
    print("0 0 0")
    sys.stdout.flush()
PYEOF
EOF
)
    
    # Parse the numbers (first line that has 3 numbers)
    stats=$(echo "$output" | grep -E "^[0-9]+ [0-9]+ [0-9]+$" | head -1)
    
    if [ -n "$stats" ]; then
        read -r tick nodes edges <<< "$stats"
        
        # Calculate deltas
        tick_delta=$((tick - prev_tick))
        node_delta=$((nodes - prev_nodes))
        edge_delta=$((edges - prev_edges))
        
        # Clear screen and show
        clear
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║              MELVIN GRAPH - LIVE                         ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""
        echo "  Tick:  $tick"
        if [ $tick_delta -gt 0 ]; then
            echo "         (+$tick_delta since last update)"
        fi
        echo ""
        echo "  Nodes: $nodes"
        if [ $node_delta -gt 0 ]; then
            echo "         (+$node_delta since last update)"
        fi
        echo ""
        echo "  Edges: $edges"
        if [ $edge_delta -gt 0 ]; then
            echo "         (+$edge_delta since last update)"
        fi
        echo ""
        echo "  Updated: $(date +%H:%M:%S)"
        echo ""
        echo "  Press Ctrl+C to exit"
        
        prev_tick=$tick
        prev_nodes=$nodes
        prev_edges=$edges
    else
        clear
        echo "Connecting to Jetson..."
    fi
    
    sleep 1
done


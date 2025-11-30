#!/bin/bash
# Live display of Melvin's graph growth

# Function to get graph stats (suppress SSH welcome messages)
get_stats() {
    sshpass -p '123456' ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR melvin@169.254.123.100 2>/dev/null << 'EOF'
python3 << 'PYEOF'
import struct
import os
import sys

try:
    with open('/home/melvin/melvin_system/melvin.m', 'rb') as f:
        data = f.read(24)
        if len(data) >= 24:
            num_nodes, num_edges, tick = struct.unpack('QQQ', data)
            file_size = os.path.getsize('/home/melvin/melvin_system/melvin.m')
            
            # Calculate ratio
            ratio = num_edges / num_nodes if num_nodes > 0 else 0
            
            # Calculate disk usage
            max_size_gb = 4000  # 4TB
            usage_pct = min(100, (file_size / (max_size_gb * 1024 * 1024 * 1024)) * 100)
            
            print(f"{tick}|{num_nodes}|{num_edges}|{file_size}|{ratio:.2f}|{usage_pct:.6f}")
            sys.stdout.flush()
        else:
            print("0|0|0|0|0|0")
            sys.stdout.flush()
except Exception as e:
    print(f"ERROR|{e}")
    sys.stdout.flush()
PYEOF
EOF
}

# Main display loop
clear
prev_nodes=0
prev_edges=0
prev_tick=0

while true; do
    # Get stats and filter out any SSH messages
    stats=$(get_stats 2>&1 | grep -E "^[0-9]+\|[0-9]+\|[0-9]+\|" | head -1)
    
    if [[ -z "$stats" ]] || [[ "$stats" == *"ERROR"* ]]; then
        clear
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║         MELVIN GRAPH - LIVE DISPLAY                      ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""
        echo "  Connecting to Jetson..."
        sleep 2
        continue
    fi
    
    IFS='|' read -r tick nodes edges file_size ratio usage <<< "$stats"
    
    # Calculate growth rate
    node_growth=$((nodes - prev_nodes))
    edge_growth=$((edges - prev_edges))
    tick_growth=$((tick - prev_tick))
    
    # Clear and redraw
    clear
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║         MELVIN GRAPH - LIVE DISPLAY                      ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    printf "  Tick:        %'d" $tick
    if [ $tick_growth -gt 0 ]; then
        echo " (+$tick_growth)"
    else
        echo ""
    fi
    printf "  Nodes:       %'d" $nodes
    if [ $node_growth -gt 0 ]; then
        echo " (+$node_growth)"
    else
        echo ""
    fi
    printf "  Edges:       %'d" $edges
    if [ $edge_growth -gt 0 ]; then
        echo " (+$edge_growth)"
    else
        echo ""
    fi
    echo ""
    
    if [ -n "$file_size" ] && [ "$file_size" != "0" ]; then
        file_size_mb=$(echo "scale=2; $file_size / 1024 / 1024" | bc 2>/dev/null || echo "0")
        echo "  File Size:   ${file_size_mb} MB"
    fi
    
    if [ -n "$ratio" ] && [ "$ratio" != "0" ]; then
        echo "  Ratio:       ${ratio} edges/node"
    fi
    echo ""
    echo "  Status:      Growing organically (no limits)"
    echo "  Limit:       4TB disk space only"
    echo ""
    echo "  Updated:     $(date +%H:%M:%S)"
    echo ""
    echo "  Press Ctrl+C to exit"
    
    prev_nodes=$nodes
    prev_edges=$edges
    prev_tick=$tick
    
    sleep 1
done

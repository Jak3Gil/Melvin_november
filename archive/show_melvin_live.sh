#!/bin/bash
# Simple live display
while true; do
  clear
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║              MELVIN GRAPH - LIVE                         ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""
  
  stats=$(sshpass -p '123456' ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR melvin@169.254.123.100 'python3 -c "import struct; f=open(\"/home/melvin/melvin_system/melvin.m\",\"rb\"); d=f.read(24); n,e,t=struct.unpack(\"QQQ\",d); print(f\"{t} {n} {e}\")" 2>/dev/null | tail -1')
  
  if [ -n "$stats" ]; then
    read tick nodes edges <<< "$stats"
    echo "  Tick:  $(printf "%'d" $tick)"
    echo "  Nodes: $(printf "%'d" $nodes)"
    echo "  Edges: $(printf "%'d" $edges)"
  else
    echo "  Connecting..."
  fi
  
  echo ""
  echo "  Updated: $(date +%H:%M:%S)"
  echo ""
  echo "  Press Ctrl+C to exit"
  sleep 1
done

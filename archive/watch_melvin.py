#!/usr/bin/env python3
"""Live display of Melvin's graph - Tick, Nodes, Edges"""

import subprocess
import struct
import time
import sys
import os

def get_stats():
    """Get graph stats from Jetson"""
    cmd = [
        'sshpass', '-p', '123456',
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'LogLevel=ERROR',
        'melvin@169.254.123.100',
        'python3', '-c', '''
import struct
import sys
try:
    with open("/home/melvin/melvin_system/melvin.m", "rb") as f:
        data = f.read(24)
        if len(data) >= 24:
            num_nodes, num_edges, tick = struct.unpack("QQQ", data)
            print(f"{tick}|{num_nodes}|{num_edges}")
            sys.stdout.flush()
except Exception as e:
    print(f"ERROR|{e}")
    sys.stdout.flush()
'''
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        output = result.stdout.strip()
        if '|' in output and not output.startswith('ERROR'):
            parts = output.split('|')
            if len(parts) == 3:
                return int(parts[0]), int(parts[1]), int(parts[2])
    except:
        pass
    return None, None, None

def main():
    prev_tick = 0
    prev_nodes = 0
    prev_edges = 0
    
    while True:
        tick, nodes, edges = get_stats()
        
        if tick is not None:
            # Calculate deltas
            tick_delta = tick - prev_tick
            node_delta = nodes - prev_nodes
            edge_delta = edges - prev_edges
            
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("╔══════════════════════════════════════════════════════════╗")
            print("║              MELVIN GRAPH - LIVE                         ║")
            print("╚══════════════════════════════════════════════════════════╝")
            print()
            print(f"  Tick:  {tick:,}")
            if tick_delta > 0:
                print(f"         (+{tick_delta:,} since last update)")
            print()
            print(f"  Nodes: {nodes:,}")
            if node_delta > 0:
                print(f"         (+{node_delta:,} since last update)")
            print()
            print(f"  Edges: {edges:,}")
            if edge_delta > 0:
                print(f"         (+{edge_delta:,} since last update)")
            print()
            print(f"  Updated: {time.strftime('%H:%M:%S')}")
            print()
            print("  Press Ctrl+C to exit")
            
            prev_tick = tick
            prev_nodes = nodes
            prev_edges = edges
        else:
            os.system('clear' if os.name != 'nt' else 'cls')
            print("Connecting to Jetson...")
        
        time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


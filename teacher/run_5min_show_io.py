#!/usr/bin/env python3
"""
Run 5-minute training and show I/O interactions clearly
"""

import subprocess
import sys
import os

def main():
    print("=" * 70)
    print("5-MINUTE TRAINING WITH VISIBLE I/O")
    print("=" * 70)
    print()
    print("Running 60 rounds × 2 tasks = 120 total tasks")
    print("Each task shows: Input → Melvin Output")
    print()
    print("Starting...")
    print()
    
    melvin_binary = "../melvin_learn_cli"
    graph_file = "melvin_5min_graph.bin"
    
    cmd = [
        "python3", "kindergarten_teacher.py",
        "--rounds", "60",
        "--tasks-per-round", "2",
        "--melvin-binary", melvin_binary,
        "--graph-file", graph_file
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    round_num = 0
    task_num = 0
    
    try:
        for line in proc.stdout:
            line = line.rstrip()
            
            # Show rounds
            if "ROUND" in line and "/" in line:
                round_num += 1
                print()
                print("=" * 70)
                print(f"ROUND {round_num}/60")
                print("=" * 70)
                sys.stdout.flush()
            
            # Show tasks
            elif "--- Task" in line:
                task_num += 1
                print()
                print(line)
                sys.stdout.flush()
            
            # Show input
            elif "Input:" in line:
                print(f"  📥 {line}")
                sys.stdout.flush()
            
            # Show Melvin output
            elif "📤 Melvin Output:" in line:
                print(f"  {line}")
                sys.stdout.flush()
            elif "Patterns created:" in line or "Explanation apps:" in line or "Compression ratio:" in line or "Reconstruction error:" in line or "Top pattern:" in line:
                if line.startswith("   "):
                    print(line)
                else:
                    print(f"     {line}")
                sys.stdout.flush()
            
            # Skip Ollama/Judge errors
            elif "Ollama" in line or "Judge" in line:
                pass
            
            # Progress updates
            elif task_num > 0 and task_num % 20 == 0 and "ROUND" not in line:
                print(f"\n[Progress: {task_num}/120 tasks]")
                sys.stdout.flush()
        
        proc.wait()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        proc.terminate()
        proc.wait()
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Graph saved to: {graph_file}")
    if os.path.exists(graph_file):
        size = os.path.getsize(graph_file)
        print(f"Graph size: {size:,} bytes ({size/1024:.1f} KB)")

if __name__ == "__main__":
    main()


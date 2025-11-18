#!/usr/bin/env python3
"""
Run 5-minute training with live progress updates
"""

import subprocess
import sys
import time
import os
from datetime import datetime

def format_time(seconds):
    """Format seconds as MM:SS"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def print_progress_bar(current, total, width=50):
    """Print a progress bar"""
    if total == 0:
        return
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {current}/{total} ({percent*100:.1f}%)"

def main():
    print("=" * 70)
    print("5-MINUTE TRAINING WITH LIVE PROGRESS")
    print("=" * 70)
    print()
    print("Using universal graph: melvin_global_graph.bin")
    print("60 rounds × 2 tasks = 120 total tasks")
    print()
    print("Starting training...")
    print()
    
    melvin_binary = "../melvin_learn_cli"
    graph_file = "melvin_global_graph.bin"
    
    cmd = [
        "python3", "kindergarten_teacher.py",
        "--rounds", "60",
        "--tasks-per-round", "2",
        "--melvin-binary", melvin_binary,
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    task_count = 0
    round_count = 0
    start_time = time.time()
    last_update = start_time
    
    print("=" * 70)
    print("LIVE PROGRESS")
    print("=" * 70)
    print()
    
    try:
        for line in proc.stdout:
            line = line.rstrip()
            
            # Track rounds
            if "ROUND" in line and "/" in line:
                round_count += 1
                elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"ROUND {round_count}/60 | Elapsed: {format_time(elapsed)}")
                print(f"{'='*70}")
                sys.stdout.flush()
            
            # Track tasks
            elif "--- Task" in line:
                task_count += 1
                elapsed = time.time() - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                remaining = (120 - task_count) / rate if rate > 0 else 0
                
                print(f"\n{line}")
                print(f"  Progress: {print_progress_bar(task_count, 120)}")
                print(f"  Elapsed: {format_time(elapsed)} | Est. remaining: {format_time(remaining)}")
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
            
            # Periodic status updates (every 5 seconds)
            now = time.time()
            if now - last_update >= 5.0:
                elapsed = now - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                remaining = (120 - task_count) / rate if rate > 0 else 0
                
                print(f"\n[Status: {task_count}/120 tasks | {round_count}/60 rounds | {format_time(elapsed)} elapsed | ~{format_time(remaining)} remaining]")
                sys.stdout.flush()
                last_update = now
            
            # Skip noise
            if "Ollama" in line or "Judge" in line or "Expected:" in line:
                continue
        
        proc.wait()
        
        total_time = time.time() - start_time
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {format_time(total_time)}")
        print(f"Tasks completed: {task_count}/120")
        print(f"Rounds completed: {round_count}/60")
        print(f"Graph saved to: {graph_file}")
        if os.path.exists(graph_file):
            size = os.path.getsize(graph_file)
            print(f"Graph size: {size:,} bytes ({size/1024/1024:.2f} MB)")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        proc.terminate()
        proc.wait()
        print(f"Completed {task_count}/120 tasks before interruption")

if __name__ == "__main__":
    main()


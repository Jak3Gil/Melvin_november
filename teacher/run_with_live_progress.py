#!/usr/bin/env python3
"""
Run training with live progress updates
"""

import subprocess
import sys
import time

def main():
    print("=" * 70)
    print("5-MINUTE TRAINING - LIVE PROGRESS")
    print("=" * 70)
    print()
    
    cmd = [
        "python3", "kindergarten_teacher.py",
        "--rounds", "60",
        "--tasks-per-round", "2",
        "--melvin-binary", "../melvin_learn_cli",
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
    last_progress = time.time()
    
    try:
        for line in proc.stdout:
            line = line.rstrip()
            
            # Skip noise
            if any(x in line for x in ["Ollama", "Judge", "Expected:", "Melvin Kindergarten Teacher", "Running", "Using persistent"]):
                continue
            
            # Show rounds
            if "ROUND" in line and "/" in line:
                round_count += 1
                elapsed = time.time() - start_time
                print()
                print("=" * 70)
                print(f"{line} | Elapsed: {int(elapsed//60):02d}:{int(elapsed%60):02d}")
                print("=" * 70)
                sys.stdout.flush()
            
            # Show tasks
            elif "--- Task" in line:
                task_count += 1
                elapsed = time.time() - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                remaining = (120 - task_count) / rate if rate > 0 else 0
                pct = (task_count / 120 * 100) if 120 > 0 else 0
                bar = '█' * int(pct/2.5) + '░' * (40 - int(pct/2.5))
                
                print()
                print(line)
                print(f"  [{bar}] {task_count}/120 ({pct:.1f}%) | ⏱️  {int(elapsed//60):02d}:{int(elapsed%60):02d} | ~{int(remaining//60):02d}:{int(remaining%60):02d} left")
                sys.stdout.flush()
                last_progress = time.time()
            
            # Show I/O
            elif "Input:" in line:
                print(f"  📥 {line}")
                sys.stdout.flush()
            elif "📤 Melvin Output:" in line:
                print(f"  {line}")
                sys.stdout.flush()
            elif any(x in line for x in ["Patterns created:", "Explanation apps:", "Compression ratio:", "Reconstruction error:", "Top pattern:"]):
                if line.startswith("   "):
                    print(line)
                else:
                    print(f"     {line}")
                sys.stdout.flush()
            
            # Periodic status (every 5 seconds)
            now = time.time()
            if now - last_progress >= 5.0 and task_count > 0:
                elapsed = now - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                remaining = (120 - task_count) / rate if rate > 0 else 0
                pct = (task_count / 120 * 100) if 120 > 0 else 0
                bar = '█' * int(pct/2.5) + '░' * (40 - int(pct/2.5))
                print(f"\n[Status: {bar}] {task_count}/120 tasks | {int(elapsed//60):02d}:{int(elapsed%60):02d} elapsed | ~{int(remaining//60):02d}:{int(remaining%60):02d} remaining")
                sys.stdout.flush()
                last_progress = now
        
        proc.wait()
        
        total_time = time.time() - start_time
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {int(total_time//60):02d}:{int(total_time%60):02d}")
        print(f"Tasks: {task_count}/120")
        print(f"Rounds: {round_count}/60")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()


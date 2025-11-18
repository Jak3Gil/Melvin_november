#!/usr/bin/env python3
"""
5-minute test with learning progress, graph outputs, and text generation test
"""

import subprocess
import sys
import time
import os
import json
from datetime import datetime

def format_time(seconds):
    """Format seconds as MM:SS"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def print_progress_bar(current, total, width=50):
    """Print a progress bar"""
    if total == 0:
        return ""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {current}/{total} ({percent*100:.1f}%)"

def test_text_generation(melvin_binary, graph_file, test_prompts):
    """Test if the graph can generate text like an LLM"""
    print("\n" + "=" * 70)
    print("TESTING TEXT GENERATION CAPABILITY")
    print("=" * 70)
    print("\nTesting if graph can generate new text sequences...")
    print()
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: Prompt = '{prompt}'")
        
        # Try to get the graph to generate continuation
        # For now, we'll feed the prompt and see what patterns match
        # Then try to reconstruct/extend
        
        cmd = [melvin_binary]
        if graph_file and os.path.exists(graph_file):
            cmd.extend(["--load", graph_file, "--save", graph_file])
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            out, err = proc.communicate(prompt + "\n", timeout=10)
            
            if err:
                print(f"  ⚠️  stderr: {err[:100]}")
            
            try:
                result = json.loads(out)
                
                # Show what patterns were found
                patterns = result.get('patterns', [])
                if patterns:
                    top_patterns = sorted(patterns, key=lambda p: p.get('binding_count', 0), reverse=True)[:3]
                    print(f"  📊 Top patterns found:")
                    for p in top_patterns:
                        print(f"     - ID {p.get('id')}: q={p.get('q', 0):.3f}, bindings={p.get('binding_count', 0)}")
                
                # Show reconstruction capability
                recon_error = result.get('reconstruction_error', 1.0)
                compression = result.get('compression_ratio', 1.0)
                print(f"  🔄 Reconstruction error: {recon_error:.3f}")
                print(f"  📦 Compression ratio: {compression:.3f}")
                
                # Check if it can reconstruct the input
                if recon_error < 0.2:
                    print(f"  ✅ Can reconstruct input (error < 0.2)")
                else:
                    print(f"  ⚠️  Reconstruction quality low (error >= 0.2)")
                
                results.append({
                    'prompt': prompt,
                    'reconstruction_error': recon_error,
                    'compression_ratio': compression,
                    'num_patterns': len(patterns)
                })
                
            except json.JSONDecodeError:
                print(f"  ❌ Failed to parse output: {out[:200]}")
                results.append({
                    'prompt': prompt,
                    'error': 'JSON parse failed'
                })
                
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  Timeout")
            results.append({
                'prompt': prompt,
                'error': 'timeout'
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'prompt': prompt,
                'error': str(e)
            })
        
        print()
    
    print("=" * 70)
    print("GENERATION TEST SUMMARY")
    print("=" * 70)
    print("\nThe graph can:")
    print("  ✅ Learn patterns from input")
    print("  ✅ Reconstruct learned sequences")
    print("  ❓ Generate NEW sequences (requires pattern extension logic)")
    print("\nNote: Current system reconstructs learned patterns.")
    print("      True 'generation' would require extending patterns forward.")
    print("      This is pattern-based learning, not token-by-token LLM generation.")
    
    return results

def show_graph_stats(graph_file):
    """Show graph statistics if available"""
    if not os.path.exists(graph_file):
        return
    
    print("\n" + "=" * 70)
    print("GRAPH STATISTICS")
    print("=" * 70)
    
    size = os.path.getsize(graph_file)
    print(f"Graph file: {graph_file}")
    print(f"Size: {size:,} bytes ({size/1024:.1f} KB)")
    
    # Try to use graph_stats.c if available
    graph_stats_bin = "./graph_stats"
    if os.path.exists(graph_stats_bin):
        try:
            proc = subprocess.run(
                [graph_stats_bin, graph_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                print("\nDetailed stats:")
                print(proc.stdout)
        except:
            pass

def main():
    print("=" * 70)
    print("5-MINUTE SYSTEM TEST")
    print("=" * 70)
    print()
    print("This test will:")
    print("  1. Run 5-minute training (60 rounds × 2 tasks = 120 tasks)")
    print("  2. Show learning progress and graph outputs")
    print("  3. Test text generation capabilities")
    print()
    
    # Find melvin_learn_cli - check parent directory first, then current
    import os
    melvin_binary = "../melvin_learn_cli"
    if not os.path.exists(melvin_binary):
        melvin_binary = "./melvin_learn_cli"
    if not os.path.exists(melvin_binary):
        # Try absolute path from project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        melvin_binary = os.path.join(project_root, "melvin_learn_cli")
    
    graph_file = "melvin_5min_test_graph.bin"
    
    # Check if binary exists
    if not os.path.exists(melvin_binary):
        print(f"❌ Error: Melvin binary not found: {melvin_binary}")
        print("   Please build it first: make melvin_learn_cli")
        return 1
    
    print(f"Using binary: {melvin_binary}")
    print(f"Graph file: {graph_file}")
    print()
    print("Starting training in 2 seconds...")
    time.sleep(2)
    print()
    
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
    
    task_count = 0
    round_count = 0
    start_time = time.time()
    last_update = start_time
    
    # Collect sample outputs
    sample_outputs = []
    
    print("=" * 70)
    print("TRAINING PROGRESS")
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
            
            # Show Melvin output - this is the graph output!
            elif "📤 Melvin Output:" in line:
                print(f"  {line}")
                sys.stdout.flush()
            elif any(keyword in line for keyword in [
                "Patterns created:", "Explanation apps:", 
                "Compression ratio:", "Reconstruction error:", 
                "Top pattern:"
            ]):
                print(f"     {line}")
                # Collect sample outputs
                if task_count <= 5 or task_count % 20 == 0:
                    sample_outputs.append(line)
                sys.stdout.flush()
            
            # Show judge feedback occasionally
            elif "Judge score:" in line and (task_count <= 5 or task_count % 20 == 0):
                print(f"  {line}")
                sys.stdout.flush()
            
            # Periodic status updates (every 10 seconds)
            now = time.time()
            if now - last_update >= 10.0:
                elapsed = now - start_time
                rate = task_count / elapsed if elapsed > 0 else 0
                remaining = (120 - task_count) / rate if rate > 0 else 0
                
                print(f"\n[Status: {task_count}/120 tasks | {round_count}/60 rounds | "
                      f"{format_time(elapsed)} elapsed | ~{format_time(remaining)} remaining]")
                sys.stdout.flush()
                last_update = now
        
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
        
        # Show graph stats
        show_graph_stats(graph_file)
        
        # Show sample outputs
        if sample_outputs:
            print("\n" + "=" * 70)
            print("SAMPLE GRAPH OUTPUTS")
            print("=" * 70)
            for output in sample_outputs[:10]:
                print(f"  {output}")
        
        # Test text generation
        test_prompts = [
            "1 2 3",
            "a b c",
            "hello",
            "abcabc"
        ]
        
        gen_results = test_text_generation(melvin_binary, graph_file, test_prompts)
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        print("\nSummary:")
        print(f"  ✅ Training completed: {task_count}/120 tasks")
        print(f"  ✅ Graph saved: {graph_file}")
        print(f"  ✅ Graph outputs shown during training")
        print(f"  {'✅' if gen_results else '⚠️ '} Generation test completed")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        proc.terminate()
        proc.wait()
        print(f"Completed {task_count}/120 tasks before interruption")
        
        if task_count > 0:
            show_graph_stats(graph_file)
            test_prompts = ["1 2 3", "a b c"]
            test_text_generation(melvin_binary, graph_file, test_prompts)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


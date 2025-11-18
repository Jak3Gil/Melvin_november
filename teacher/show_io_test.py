#!/usr/bin/env python3
"""
Show inputs and outputs from Melvin learning - demonstrates the back and forth
"""

import json
import subprocess
import sys
import os

def run_melvin_show_io(input_str, graph_file=None, round_num=1):
    """Run Melvin on an input and show the full input/output interaction."""
    print("=" * 70)
    print(f"ROUND {round_num}")
    print("=" * 70)
    print()
    
    print(f"📥 INPUT TO MELVIN:")
    print(f"   '{input_str}'")
    print(f"   Length: {len(input_str)} bytes")
    print()
    
    melvin_binary = "../melvin_learn_cli"
    if not os.path.exists(melvin_binary):
        melvin_binary = "./melvin_learn_cli"
    
    cmd = [melvin_binary]
    if graph_file:
        cmd.extend(["--load", graph_file, "--save", graph_file])
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = proc.communicate(input=input_str)
        
        if proc.returncode != 0:
            print(f"❌ ERROR: melvin_learn_cli failed")
            print(f"   stderr: {err}")
            return None
        
        if not out.strip():
            print(f"❌ ERROR: No output from melvin_learn_cli")
            return None
        
        try:
            result = json.loads(out)
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: JSON decode failed")
            print(f"   Raw output: {out[:200]}")
            return None
        
        print(f"📤 OUTPUT FROM MELVIN:")
        print()
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            return None
        
        # Show key metrics
        print(f"   Metrics:")
        print(f"     • Patterns created: {result.get('num_patterns', 0)}")
        print(f"     • Explanation applications: {result.get('explanation_apps', 0)}")
        print(f"     • Compression ratio: {result.get('compression_ratio', 0):.3f}")
        print(f"     • Reconstruction error: {result.get('reconstruction_error', 0):.3f}")
        print()
        
        # Show top patterns
        patterns = result.get('patterns', [])
        if patterns:
            print(f"   Top Patterns Discovered:")
            sorted_pats = sorted(patterns, key=lambda p: p.get('binding_count', 0), reverse=True)
            for i, p in enumerate(sorted_pats[:5], 1):
                pid = p.get('id', '?')
                q = p.get('q', 0)
                bindings = p.get('binding_count', 0)
                print(f"     {i}. Pattern {pid}:")
                print(f"        Quality (q): {q:.3f}")
                print(f"        Bindings: {bindings}")
        
        print()
        print(f"   ✅ Success: Patterns explain input with {result.get('compression_ratio', 0):.3f} compression")
        if result.get('reconstruction_error', 1.0) == 0.0:
            print(f"   ✅ Perfect reconstruction (error = 0.000)")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None


def main():
    # Test inputs
    test_inputs = [
        ("1+1=2", "arithmetic pattern"),
        ("2+2=4", "arithmetic pattern"),
        ("ababab", "repeating pattern"),
        ("1 2 3 4 5", "sequential numbers"),
    ]
    
    graph_file = "test_io_graph.bin"
    
    print("=" * 70)
    print("MELVIN INPUT/OUTPUT TEST")
    print("=" * 70)
    print()
    print("This test shows the back-and-forth between:")
    print("  1. Input string fed to Melvin")
    print("  2. Melvin's learning process (internal)")
    print("  3. Output: patterns discovered, compression, error")
    print()
    
    results = []
    
    for i, (input_str, description) in enumerate(test_inputs, 1):
        result = run_melvin_show_io(input_str, graph_file, i)
        if result:
            results.append((input_str, result))
        print()
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    for input_str, result in results:
        print(f"Input: '{input_str}'")
        print(f"  → {result.get('num_patterns', 0)} patterns, "
              f"compression={result.get('compression_ratio', 0):.3f}, "
              f"error={result.get('reconstruction_error', 0):.3f}")
    
    print()
    print(f"Graph saved to: {graph_file}")
    if os.path.exists(graph_file):
        size = os.path.getsize(graph_file)
        print(f"Graph size: {size:,} bytes ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()


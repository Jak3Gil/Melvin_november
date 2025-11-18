#!/usr/bin/env python3
"""
Math Kindergarten Experiment
Tests if Melvin learns reusable patterns across arithmetic tasks.
"""

import json
import subprocess
import sys
import os
from collections import defaultdict
from typing import Dict, List, Set

def run_melvin_on_string(input_str: str, graph_file: str = None) -> Dict:
    """Run melvin_learn_cli on a string and return JSON result."""
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
            return {"error": f"melvin_learn_cli failed: {err}"}
        
        if not out.strip():
            return {"error": "No output from melvin_learn_cli"}
        
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def extract_pattern_ids(result: Dict) -> Set[int]:
    """Extract pattern IDs from a melvin result."""
    patterns = result.get('patterns', [])
    return {p.get('id') for p in patterns if 'id' in p}


def analyze_pattern_overlap(phase_results: List[Dict], phase_names: List[str]):
    """Analyze which patterns appear across phases."""
    print("\n" + "=" * 70)
    print("PATTERN OVERLAP ANALYSIS")
    print("=" * 70)
    
    phase_patterns = {}
    for name, result in zip(phase_names, phase_results):
        if 'error' not in result:
            phase_patterns[name] = extract_pattern_ids(result)
    
    # Find patterns that appear in multiple phases
    all_patterns = set()
    for patterns in phase_patterns.values():
        all_patterns.update(patterns)
    
    print(f"\nTotal unique patterns across all phases: {len(all_patterns)}")
    
    # Count per-phase
    print("\nPatterns per phase:")
    for name, patterns in phase_patterns.items():
        print(f"  {name}: {len(patterns)} patterns")
    
    # Find overlaps
    print("\nPattern reuse across phases:")
    phase_list = list(phase_patterns.items())
    for i in range(len(phase_list)):
        for j in range(i + 1, len(phase_list)):
            name1, pats1 = phase_list[i]
            name2, pats2 = phase_list[j]
            overlap = pats1 & pats2
            if overlap:
                print(f"  {name1} <-> {name2}: {len(overlap)} shared patterns")
    
    # Find patterns that appear in 3+ phases
    pattern_counts = defaultdict(list)
    for name, patterns in phase_patterns.items():
        for pid in patterns:
            pattern_counts[pid].append(name)
    
    multi_phase_patterns = {pid: phases for pid, phases in pattern_counts.items() 
                           if len(phases) >= 3}
    
    if multi_phase_patterns:
        print(f"\nPatterns appearing in 3+ phases: {len(multi_phase_patterns)}")
        print("  (These are strong candidates for reusable structure)")
    else:
        print("\nNo patterns appear in 3+ phases")
        print("  (Patterns are phase-specific, may indicate memorization)")
    
    return phase_patterns, multi_phase_patterns


def run_arithmetic_curriculum(graph_file: str = None):
    """Run arithmetic micro-curriculum."""
    print("=" * 70)
    print("ARITHMETIC MICRO-CURRICULUM")
    print("=" * 70)
    
    # Phase 1: Core arithmetic
    arithmetic_inputs = [
        "1+1=2",
        "2+2=4",
        "3+3=6",
        "4+4=8",
    ]
    
    # Phase 2: Distractors (different structure)
    distractor_inputs = [
        "1 2 3 4 5",
        "2 4 6 8",
        "3 6 9 12",
    ]
    
    # Phase 3: Variants (similar structure, different values)
    variant_inputs = [
        "1+2=3",
        "2+3=5",
        "3+4=7",
    ]
    
    all_inputs = arithmetic_inputs + distractor_inputs + variant_inputs
    phase_names = (
        ["arithmetic"] * len(arithmetic_inputs) +
        ["distractor"] * len(distractor_inputs) +
        ["variant"] * len(variant_inputs)
    )
    
    results = []
    
    print("\nRunning curriculum...")
    for i, (input_str, phase) in enumerate(zip(all_inputs, phase_names)):
        print(f"  [{i+1}/{len(all_inputs)}] {phase}: '{input_str}'")
        result = run_melvin_on_string(input_str, graph_file)
        results.append(result)
        
        if 'error' not in result:
            print(f"    -> {result.get('num_patterns', 0)} patterns, "
                  f"compression={result.get('compression_ratio', 0):.3f}, "
                  f"error={result.get('reconstruction_error', 0):.3f}")
        else:
            print(f"    -> ERROR: {result['error']}")
    
    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    arithmetic_results = [r for r, p in zip(results, phase_names) if p == "arithmetic"]
    distractor_results = [r for r, p in zip(results, phase_names) if p == "distractor"]
    variant_results = [r for r, p in zip(results, phase_names) if p == "variant"]
    
    def print_phase_stats(name: str, phase_results: List[Dict]):
        successful = [r for r in phase_results if 'error' not in r]
        if not successful:
            print(f"\n{name}: No successful runs")
            return
        
        compressions = [r.get('compression_ratio', 1.0) for r in successful]
        errors = [r.get('reconstruction_error', 1.0) for r in successful]
        pattern_counts = [r.get('num_patterns', 0) for r in successful]
        
        print(f"\n{name}:")
        print(f"  Successful: {len(successful)}/{len(phase_results)}")
        print(f"  Avg compression: {sum(compressions)/len(compressions):.3f}")
        print(f"  Avg error: {sum(errors)/len(errors):.3f}")
        print(f"  Avg patterns: {sum(pattern_counts)/len(pattern_counts):.1f}")
        print(f"  Compression < 1.0: {sum(1 for c in compressions if c < 1.0)}/{len(compressions)}")
        print(f"  Error ≈ 0: {sum(1 for e in errors if e < 0.001)}/{len(errors)}")
    
    print_phase_stats("Arithmetic", arithmetic_results)
    print_phase_stats("Distractor", distractor_results)
    print_phase_stats("Variant", variant_results)
    
    # Pattern overlap analysis
    phase_results = [
        arithmetic_results[0] if arithmetic_results else {},
        distractor_results[0] if distractor_results else {},
        variant_results[0] if variant_results else {},
    ]
    phase_names_short = ["arithmetic", "distractor", "variant"]
    
    analyze_pattern_overlap(phase_results, phase_names_short)
    
    return results


def run_confound_test(graph_file: str = None):
    """Run confound test: same structure, different truth values."""
    print("\n" + "=" * 70)
    print("CONFOUND TEST: Structure vs Noise")
    print("=" * 70)
    
    # True arithmetic
    true_inputs = [
        "1+2=3",
        "3+5=8",
    ]
    
    # False/inconsistent arithmetic
    false_inputs = [
        "1+2=4",
        "3+5=9",
    ]
    
    print("\nTrue arithmetic:")
    true_results = []
    for inp in true_inputs:
        print(f"  '{inp}'")
        result = run_melvin_on_string(inp, graph_file)
        true_results.append(result)
        if 'error' not in result:
            print(f"    -> compression={result.get('compression_ratio', 0):.3f}, "
                  f"error={result.get('reconstruction_error', 0):.3f}, "
                  f"patterns={result.get('num_patterns', 0)}")
    
    print("\nFalse/inconsistent arithmetic:")
    false_results = []
    for inp in false_inputs:
        print(f"  '{inp}'")
        result = run_melvin_on_string(inp, graph_file)
        false_results.append(result)
        if 'error' not in result:
            print(f"    -> compression={result.get('compression_ratio', 0):.3f}, "
                  f"error={result.get('reconstruction_error', 0):.3f}, "
                  f"patterns={result.get('num_patterns', 0)}")
    
    # Compare pattern overlap
    print("\n" + "=" * 70)
    print("CONFOUND ANALYSIS")
    print("=" * 70)
    
    true_patterns = set()
    for r in true_results:
        if 'error' not in r:
            true_patterns.update(extract_pattern_ids(r))
    
    false_patterns = set()
    for r in false_results:
        if 'error' not in r:
            false_patterns.update(extract_pattern_ids(r))
    
    overlap = true_patterns & false_patterns
    
    print(f"\nTrue inputs: {len(true_patterns)} unique patterns")
    print(f"False inputs: {len(false_patterns)} unique patterns")
    print(f"Overlap: {len(overlap)} shared patterns")
    
    if overlap:
        print("\n  -> Same structural patterns appear in both (expected)")
        print("  -> Check if bindings/quality differ (structure vs noise sensitivity)")
    else:
        print("\n  -> No pattern overlap (may indicate structure sensitivity)")
    
    # Compare metrics
    true_successful = [r for r in true_results if 'error' not in r]
    false_successful = [r for r in false_results if 'error' not in r]
    
    if true_successful and false_successful:
        true_comp = [r.get('compression_ratio', 1.0) for r in true_successful]
        false_comp = [r.get('compression_ratio', 1.0) for r in false_successful]
        
        print(f"\nCompression comparison:")
        print(f"  True: avg={sum(true_comp)/len(true_comp):.3f}")
        print(f"  False: avg={sum(false_comp)/len(false_comp):.3f}")
    
    return true_results, false_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Math kindergarten experiment for Melvin"
    )
    parser.add_argument("--graph-file", type=str, default=None,
                       help="Persistent graph file (for cross-task learning)")
    parser.add_argument("--arithmetic-only", action="store_true",
                       help="Run only arithmetic curriculum")
    parser.add_argument("--confound-only", action="store_true",
                       help="Run only confound test")
    
    args = parser.parse_args()
    
    graph_file = args.graph_file or "experiment_graph.bin"
    
    if args.confound_only:
        run_confound_test(graph_file)
    elif args.arithmetic_only:
        run_arithmetic_curriculum(graph_file)
    else:
        run_arithmetic_curriculum(graph_file)
        run_confound_test(graph_file)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nGraph saved to: {graph_file}")
    print("\nNext steps:")
    print("  1. Use investigate_io.py to see detailed pattern analysis")
    print("  2. Use query_graph to inspect specific patterns")
    print("  3. Check if same patterns appear across phases (generalization)")


if __name__ == "__main__":
    main()


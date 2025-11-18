#!/usr/bin/env python3
"""
Investigate inputs and outputs in the Melvin system.
Shows what goes in, what patterns are discovered, and what comes out.
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Any

def load_log_entries(log_file: str = "teacher_log.jsonl") -> List[Dict]:
    """Load all entries from teacher log."""
    entries = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except FileNotFoundError:
        print(f"Log file {log_file} not found")
        return []
    return entries


def show_input_output_analysis(entries: List[Dict]):
    """Show detailed input/output analysis."""
    print("=" * 70)
    print("MELVIN INPUT/OUTPUT INVESTIGATION")
    print("=" * 70)
    print()
    
    successful = [e for e in entries if 'error' not in e.get('melvin_result', {})]
    
    print(f"Total entries: {len(entries)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(entries) - len(successful)}")
    print()
    
    # Group by input string
    input_groups = defaultdict(list)
    for e in successful:
        input_str = e['task']['input_str']
        input_groups[input_str].append(e)
    
    print("=" * 70)
    print("INPUT-OUTPUT MAPPING")
    print("=" * 70)
    print()
    
    # Show examples of inputs and their outputs
    print("Sample Input → Output Analysis:\n")
    
    for input_str, group in list(input_groups.items())[:10]:
        print(f"Input: '{input_str}'")
        print(f"  Seen {len(group)} times across rounds")
        
        # Get latest result
        latest = group[-1]
        mr = latest['melvin_result']
        
        print(f"  Latest result (Round {latest['round']}):")
        print(f"    Patterns created: {mr.get('num_patterns', 0)}")
        print(f"    Explanation apps: {mr.get('explanation_apps', 0)}")
        print(f"    Compression ratio: {mr.get('compression_ratio', 0):.3f}")
        print(f"    Reconstruction error: {mr.get('reconstruction_error', 0):.3f}")
        
        # Show pattern qualities
        patterns = mr.get('patterns', [])
        if patterns:
            qs = [p.get('q', 0) for p in patterns]
            bindings = [p.get('binding_count', 0) for p in patterns]
            print(f"    Pattern quality range: {min(qs):.3f} - {max(qs):.3f}")
            print(f"    Total bindings: {sum(bindings)}")
            print(f"    Top 3 patterns by quality:")
            sorted_pats = sorted(patterns, key=lambda p: p.get('q', 0), reverse=True)
            for i, p in enumerate(sorted_pats[:3], 1):
                print(f"      {i}. Pattern {p.get('id', '?')}: q={p.get('q', 0):.3f}, bindings={p.get('binding_count', 0)}")
        
        print()
    
    print("=" * 70)
    print("PATTERN DISCOVERY ANALYSIS")
    print("=" * 70)
    print()
    
    # Analyze what patterns are being discovered
    pattern_usage = defaultdict(int)
    pattern_quality = defaultdict(list)
    pattern_bindings = defaultdict(list)
    
    for e in successful:
        patterns = e['melvin_result'].get('patterns', [])
        for p in patterns:
            pid = p.get('id')
            pattern_usage[pid] += 1
            pattern_quality[pid].append(p.get('q', 0))
            pattern_bindings[pid].append(p.get('binding_count', 0))
    
    print(f"Unique patterns discovered: {len(pattern_usage)}")
    print(f"\nMost frequently used patterns:")
    sorted_patterns = sorted(pattern_usage.items(), key=lambda x: -x[1])[:10]
    for pid, count in sorted_patterns:
        avg_q = sum(pattern_quality[pid]) / len(pattern_quality[pid])
        avg_bindings = sum(pattern_bindings[pid]) / len(pattern_bindings[pid])
        print(f"  Pattern {pid}: used {count} times, avg_q={avg_q:.3f}, avg_bindings={avg_bindings:.1f}")
    
    print()
    print("=" * 70)
    print("RECONSTRUCTION QUALITY ANALYSIS")
    print("=" * 70)
    print()
    
    # Analyze reconstruction quality
    perfect_reconstructions = [e for e in successful 
                              if e['melvin_result'].get('reconstruction_error', 1.0) == 0.0]
    imperfect = [e for e in successful 
                if e['melvin_result'].get('reconstruction_error', 0.0) > 0.0]
    
    print(f"Perfect reconstructions (error=0.0): {len(perfect_reconstructions)}")
    print(f"Imperfect reconstructions: {len(imperfect)}")
    
    if successful:
        errors = [e['melvin_result'].get('reconstruction_error', 0) for e in successful]
        compressions = [e['melvin_result'].get('compression_ratio', 0) for e in successful]
        
        print(f"\nReconstruction error stats:")
        print(f"  Min: {min(errors):.3f}")
        print(f"  Max: {max(errors):.3f}")
        print(f"  Avg: {sum(errors)/len(errors):.3f}")
        
        print(f"\nCompression ratio stats:")
        print(f"  Min: {min(compressions):.3f}")
        print(f"  Max: {max(compressions):.3f}")
        print(f"  Avg: {sum(compressions)/len(compressions):.3f}")
        print(f"  < 1.0 (compressed): {sum(1 for c in compressions if c < 1.0)}")
        print(f"  >= 1.0 (not compressed): {sum(1 for c in compressions if c >= 1.0)}")
    
    print()
    print("=" * 70)
    print("INPUT DIVERSITY")
    print("=" * 70)
    print()
    
    unique_inputs = set(e['task']['input_str'] for e in successful)
    print(f"Unique input strings: {len(unique_inputs)}")
    print(f"\nSample unique inputs:")
    for inp in sorted(list(unique_inputs))[:20]:
        count = sum(1 for e in successful if e['task']['input_str'] == inp)
        print(f"  '{inp}' (seen {count} times)")


def show_detailed_example(entries: List[Dict], input_str: str = None):
    """Show detailed breakdown of a specific input."""
    successful = [e for e in entries if 'error' not in e.get('melvin_result', {})]
    
    if not input_str:
        # Pick first successful entry
        if successful:
            input_str = successful[0]['task']['input_str']
        else:
            print("No successful entries to analyze")
            return
    
    matching = [e for e in successful if e['task']['input_str'] == input_str]
    
    if not matching:
        print(f"No successful entries found for input: '{input_str}'")
        return
    
    print("=" * 70)
    print(f"DETAILED ANALYSIS: '{input_str}'")
    print("=" * 70)
    print()
    
    # Show all occurrences
    for i, e in enumerate(matching[:5], 1):  # Show first 5
        print(f"Occurrence {i} (Round {e['round']}):")
        mr = e['melvin_result']
        
        print(f"  Input length: {len(input_str)} bytes")
        print(f"  Patterns created: {mr.get('num_patterns', 0)}")
        print(f"  Explanation applications: {mr.get('explanation_apps', 0)}")
        print(f"  Compression: {mr.get('compression_ratio', 0):.3f}")
        print(f"  Error: {mr.get('reconstruction_error', 0):.3f}")
        
        patterns = mr.get('patterns', [])
        if patterns:
            print(f"\n  Patterns discovered:")
            for j, p in enumerate(patterns[:10], 1):  # Show first 10
                print(f"    {j}. Pattern {p.get('id', '?')}: q={p.get('q', 0):.3f}, bindings={p.get('binding_count', 0)}")
        
        print()


def show_pattern_to_input_mapping(entries: List[Dict]):
    """Show which patterns are used for which inputs."""
    print("=" * 70)
    print("PATTERN-TO-INPUT MAPPING")
    print("=" * 70)
    print()
    
    successful = [e for e in entries if 'error' not in e.get('melvin_result', {})]
    
    # For each pattern, show which inputs it was used for
    pattern_inputs = defaultdict(set)
    
    for e in successful:
        input_str = e['task']['input_str']
        patterns = e['melvin_result'].get('patterns', [])
        for p in patterns:
            pid = p.get('id')
            pattern_inputs[pid].add(input_str)
    
    print(f"Patterns that appear across multiple inputs:")
    multi_input_patterns = {pid: inputs for pid, inputs in pattern_inputs.items() 
                           if len(inputs) > 1}
    
    sorted_multi = sorted(multi_input_patterns.items(), 
                         key=lambda x: -len(x[1]))[:10]
    
    for pid, inputs in sorted_multi:
        print(f"\nPattern {pid}:")
        print(f"  Used for {len(inputs)} different inputs:")
        for inp in sorted(list(inputs))[:5]:
            print(f"    - '{inp}'")
        if len(inputs) > 5:
            print(f"    ... and {len(inputs) - 5} more")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Investigate Melvin inputs and outputs")
    parser.add_argument("--log-file", type=str, default="teacher_log.jsonl",
                       help="Path to teacher log file")
    parser.add_argument("--input", type=str, default=None,
                       help="Show detailed analysis for specific input string")
    parser.add_argument("--pattern-mapping", action="store_true",
                       help="Show pattern-to-input mapping")
    
    args = parser.parse_args()
    
    entries = load_log_entries(args.log_file)
    
    if not entries:
        print("No log entries found")
        return
    
    show_input_output_analysis(entries)
    
    if args.input:
        show_detailed_example(entries, args.input)
    
    if args.pattern_mapping:
        show_pattern_to_input_mapping(entries)


if __name__ == "__main__":
    main()


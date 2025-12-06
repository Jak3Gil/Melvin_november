#!/usr/bin/env python3
"""
Study pattern formation and value dynamics
"""

import csv
import sys
from collections import defaultdict
import statistics

def load_metrics(filename):
    """Load metrics CSV."""
    data = defaultdict(list)
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    try:
                        data[key].append(float(value))
                    except (ValueError, TypeError):
                        pass
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None

def load_samples(filename):
    """Load node samples CSV."""
    samples = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(row)
        return samples
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None

def analyze_pattern_formation(metrics_data, samples_data):
    """Analyze pattern formation dynamics."""
    print("=== PATTERN FORMATION ANALYSIS ===\n")
    
    if not metrics_data:
        print("No metrics data")
        return
    
    print("1. PATTERN CREATION RATE")
    print("-" * 50)
    if 'num_patterns' in metrics_data:
        pattern_counts = metrics_data['num_patterns']
        if pattern_counts:
            initial = pattern_counts[0]
            final = pattern_counts[-1]
            total_created = final - initial
            
            print(f"  Initial patterns: {initial:.0f}")
            print(f"  Final patterns: {final:.0f}")
            print(f"  Total created: {total_created:.0f}")
            
            if total_created > 0:
                # Find when patterns started forming
                for i, count in enumerate(pattern_counts):
                    if count > initial:
                        print(f"  First pattern created at step: {i * 10}")
                        break
                
                # Growth rate
                growth_rate = total_created / len(pattern_counts) if len(pattern_counts) > 0 else 0
                print(f"  Growth rate: {growth_rate:.2f} patterns per logged step")
                
                # Check if growth is accelerating or decelerating
                if len(pattern_counts) > 10:
                    early_growth = (pattern_counts[len(pattern_counts)//3] - initial) / (len(pattern_counts)//3)
                    late_growth = (final - pattern_counts[2*len(pattern_counts)//3]) / (len(pattern_counts)//3)
                    if late_growth > early_growth * 1.2:
                        print(f"  ✓ ACCELERATING growth (late > early)")
                    elif late_growth < early_growth * 0.8:
                        print(f"  ↓ DECELERATING growth (late < early)")
                    else:
                        print(f"  → STEADY growth")
            else:
                print(f"  ⚠ No patterns created during test")
    print()
    
    print("2. PATTERN ACTIVATION")
    print("-" * 50)
    if 'num_active_patterns' in metrics_data and 'num_patterns' in metrics_data:
        active_counts = metrics_data['num_active_patterns']
        total_counts = metrics_data['num_patterns']
        
        if total_counts[-1] > 0:
            # Compute activation rate
            activation_rates = []
            for i in range(len(active_counts)):
                if total_counts[i] > 0:
                    activation_rates.append(active_counts[i] / total_counts[i])
            
            if activation_rates:
                mean_activation_rate = statistics.mean(activation_rates)
                print(f"  Mean activation rate: {mean_activation_rate:.2%}")
                print(f"  (Fraction of patterns that are active)")
                
                if mean_activation_rate > 0.5:
                    print(f"  ✓ HIGH activation (patterns are being used)")
                elif mean_activation_rate > 0.1:
                    print(f"  → MODERATE activation")
                else:
                    print(f"  ⚠ LOW activation (patterns created but not used)")
        else:
            print(f"  No patterns to analyze")
    print()
    
    print("3. PATTERN VALUE DYNAMICS")
    print("-" * 50)
    if 'mean_pattern_value' in metrics_data:
        pattern_values = [v for v in metrics_data['mean_pattern_value'] if v != 0]
        if pattern_values:
            mean_value = statistics.mean(pattern_values)
            max_value = max(pattern_values)
            print(f"  Mean pattern value: {mean_value:.4f}")
            print(f"  Max pattern value: {max_value:.4f}")
            
            # Check if values are increasing (patterns getting better)
            if len(pattern_values) > 5:
                early_values = [v for v in pattern_values[:len(pattern_values)//3] if v != 0]
                late_values = [v for v in pattern_values[-len(pattern_values)//3:] if v != 0]
                if early_values and late_values:
                    early_mean = statistics.mean(early_values)
                    late_mean = statistics.mean(late_values)
                    change = ((late_mean - early_mean) / abs(early_mean)) * 100 if early_mean != 0 else 0
                    print(f"  Early mean: {early_mean:.4f}, Late mean: {late_mean:.4f}")
                    print(f"  Change: {change:+.1f}%")
                    if change > 10:
                        print(f"  ✓ Pattern values INCREASING (patterns improving)")
                    elif change < -10:
                        print(f"  ⚠ Pattern values DECREASING")
                    else:
                        print(f"  → Pattern values STABLE")
        else:
            print(f"  No pattern values recorded")
    
    if 'max_pattern_value' in metrics_data:
        max_values = [v for v in metrics_data['max_pattern_value'] if v != 0]
        if max_values:
            print(f"  Max pattern value over time: {max(max_values):.4f}")
    print()
    
    print("4. PATTERN SAMPLES ANALYSIS")
    print("-" * 50)
    if samples_data:
        pattern_samples = [s for s in samples_data if s.get('type') == 'PATTERN']
        if pattern_samples:
            print(f"  Pattern node samples: {len(pattern_samples)}")
            
            # Analyze value distribution
            values = [float(s.get('value', 0) or 0) for s in pattern_samples]
            energies = [float(s.get('energy', 0) or 0) for s in pattern_samples]
            pred_errors = [abs(float(s.get('prediction_error', 0) or 0)) for s in pattern_samples if float(s.get('prediction_error', 0) or 0) != 0]
            
            if values:
                print(f"  Value range: {min(values):.4f} to {max(values):.4f}")
                print(f"  Mean value: {statistics.mean(values):.4f}")
            
            if energies:
                print(f"  Energy range: {min(energies):.4f} to {max(energies):.4f}")
                print(f"  Mean energy: {statistics.mean(energies):.4f}")
            
            if pred_errors:
                print(f"  Prediction errors: mean={statistics.mean(pred_errors):.4f}")
            
            # Check correlation between value and energy
            if len(values) > 5 and len(energies) > 5:
                # Simple correlation check
                high_value_samples = [i for i, v in enumerate(values) if v > statistics.mean(values)]
                high_energy_samples = [i for i, e in enumerate(energies) if e > statistics.mean(energies)]
                overlap = len(set(high_value_samples) & set(high_energy_samples))
                if overlap > len(high_value_samples) * 0.5:
                    print(f"  ✓ High-value patterns tend to have high energy")
        else:
            print(f"  No pattern samples in dataset")
    print()
    
    print("5. PATTERN-TO-EXEC RELATIONSHIP")
    print("-" * 50)
    if samples_data:
        pattern_samples = [s for s in samples_data if s.get('type') == 'PATTERN']
        exec_samples = [s for s in samples_data if s.get('type') == 'EXEC']
        
        if pattern_samples and exec_samples:
            # Check if patterns with high control_value correlate with EXEC activity
            pattern_control = [float(s.get('recent_control_value', 0) or 0) for s in pattern_samples]
            exec_control = [float(s.get('recent_control_value', 0) or 0) for s in exec_samples]
            
            if pattern_control and exec_control:
                pattern_mean_control = statistics.mean(pattern_control)
                exec_mean_control = statistics.mean(exec_control)
                print(f"  Pattern mean control_value: {pattern_mean_control:.4f}")
                print(f"  EXEC mean control_value: {exec_mean_control:.4f}")
                
                if exec_mean_control > pattern_mean_control:
                    print(f"  → EXEC nodes have higher control_value (expected)")
                
                # Check EXEC firing
                exec_fires = [int(s.get('exec_count', 0) or 0) for s in exec_samples]
                total_fires = sum(exec_fires)
                if total_fires > 0:
                    print(f"  ✓ EXEC nodes have fired ({total_fires} total)")
                else:
                    print(f"  ⚠ EXEC nodes not firing (may need pattern routing)")
    print()
    
    print("6. COMPRESSION GAIN (PATTERN EFFICIENCY)")
    print("-" * 50)
    if samples_data:
        pattern_samples = [s for s in samples_data if s.get('type') == 'PATTERN']
        if pattern_samples:
            compression_gains = [float(s.get('recent_compression_gain', 0) or 0) for s in pattern_samples]
            if compression_gains:
                positive_gains = [g for g in compression_gains if g > 0]
                negative_gains = [g for g in compression_gains if g < 0]
                
                print(f"  Patterns with positive compression: {len(positive_gains)}/{len(compression_gains)}")
                if positive_gains:
                    print(f"  Mean positive gain: {statistics.mean(positive_gains):.4f}")
                if negative_gains:
                    print(f"  Patterns with negative gain: {len(negative_gains)} (may be inefficient)")
                
                if len(positive_gains) > len(negative_gains):
                    print(f"  ✓ Most patterns show compression gain (reducing active_count)")
                else:
                    print(f"  ⚠ Many patterns not showing compression benefit")
    print()

def main():
    print("=== PATTERN FORMATION & VALUE DYNAMICS ANALYSIS ===\n")
    
    metrics_data = load_metrics('unified_metrics.csv')
    samples_data = load_samples('unified_node_samples.csv')
    
    if metrics_data or samples_data:
        analyze_pattern_formation(metrics_data, samples_data)
        
        # Save summary
        with open('pattern_analysis_summary.txt', 'w') as f:
            import io
            old_stdout = sys.stdout
            sys.stdout = f
            analyze_pattern_formation(metrics_data, samples_data)
            sys.stdout = old_stdout
        print("✓ Summary saved to pattern_analysis_summary.txt")
    else:
        print("Error: Could not load data files")
        sys.exit(1)

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Compare prediction errors vs actual outcomes
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

def analyze_predictions(metrics_data, samples_data):
    """Compare prediction errors vs actual outcomes."""
    print("=== PREDICTION ERROR ANALYSIS ===\n")
    
    if not metrics_data:
        print("No metrics data")
        return
    
    print("1. INTERNAL NEXT-STATE PREDICTION")
    print("-" * 50)
    if 'avg_pred_error' in metrics_data:
        pred_errors = [e for e in metrics_data['avg_pred_error'] if e > 0]
        if pred_errors:
            mean_error = statistics.mean(pred_errors)
            median_error = statistics.median(pred_errors)
            max_error = max(pred_errors)
            print(f"  Mean prediction error: {mean_error:.4f}")
            print(f"  Median: {median_error:.4f}")
            print(f"  Max: {max_error:.4f}")
            
            # Compare early vs late
            n = len(pred_errors)
            if n > 10:
                early = statistics.mean(pred_errors[:n//3])
                late = statistics.mean(pred_errors[-n//3:])
                improvement = ((early - late) / early) * 100 if early > 0 else 0
                print(f"  Early vs Late: {early:.4f} → {late:.4f} ({improvement:+.1f}% change)")
                if improvement > 10:
                    print(f"  ✓ Predictions IMPROVING over time")
                elif improvement < -10:
                    print(f"  ⚠ Predictions WORSENING")
                else:
                    print(f"  → Predictions STABLE")
        else:
            print("  No prediction errors recorded")
    print()
    
    print("2. SENSORY NEXT-INPUT PREDICTION")
    print("-" * 50)
    if 'avg_sensory_error' in metrics_data:
        sensory_errors = [e for e in metrics_data['avg_sensory_error'] if e > 0]
        if sensory_errors:
            mean_error = statistics.mean(sensory_errors)
            print(f"  Mean sensory error: {mean_error:.4f}")
            
            # Check if error decreases over time (learning)
            n = len(sensory_errors)
            if n > 10:
                early = statistics.mean(sensory_errors[:n//3])
                late = statistics.mean(sensory_errors[-n//3:])
                improvement = ((early - late) / early) * 100 if early > 0 else 0
                print(f"  Early vs Late: {early:.4f} → {late:.4f} ({improvement:+.1f}% change)")
                if improvement > 10:
                    print(f"  ✓ Sensory predictions IMPROVING (learning patterns)")
                elif improvement < -10:
                    print(f"  ⚠ Sensory predictions WORSENING")
                else:
                    print(f"  → Sensory predictions STABLE")
        else:
            print("  No sensory errors recorded")
    print()
    
    print("3. VALUE-DELTA PREDICTION")
    print("-" * 50)
    if 'predicted_value_delta' in metrics_data and 'global_value' in metrics_data:
        predicted_deltas = metrics_data['predicted_value_delta']
        actual_values = metrics_data['global_value']
        
        # Compute actual deltas
        actual_deltas = []
        for i in range(1, len(actual_values)):
            actual_deltas.append(actual_values[i] - actual_values[i-1])
        
        # Compare predicted vs actual
        if len(predicted_deltas) > 1 and len(actual_deltas) > 0:
            # Align arrays (predicted is for next step)
            min_len = min(len(predicted_deltas) - 1, len(actual_deltas))
            if min_len > 0:
                errors = []
                for i in range(min_len):
                    pred = predicted_deltas[i]
                    actual = actual_deltas[i]
                    errors.append(abs(pred - actual))
                
                if errors:
                    mean_error = statistics.mean(errors)
                    print(f"  Mean value-delta prediction error: {mean_error:.4f}")
                    print(f"  Predicted deltas: mean={statistics.mean(predicted_deltas[:min_len]):.4f}")
                    print(f"  Actual deltas: mean={statistics.mean(actual_deltas[:min_len]):.4f}")
                    
                    # Check correlation
                    if min_len > 5:
                        pred_mean = statistics.mean(predicted_deltas[:min_len])
                        actual_mean = statistics.mean(actual_deltas[:min_len])
                        if abs(pred_mean) > 0.001 and abs(actual_mean) > 0.001:
                            if (pred_mean > 0 and actual_mean > 0) or (pred_mean < 0 and actual_mean < 0):
                                print(f"  ✓ Predictions match direction (both {('positive' if actual_mean > 0 else 'negative')})")
                            else:
                                print(f"  ⚠ Predictions opposite direction")
    print()
    
    print("4. PREDICTION ERROR BY NODE TYPE")
    print("-" * 50)
    if samples_data:
        by_type = defaultdict(list)
        for sample in samples_data:
            node_type = sample.get('type', '')
            pred_error = float(sample.get('prediction_error', 0) or 0)
            if abs(pred_error) > 0.001:
                by_type[node_type].append(abs(pred_error))
        
        for node_type, errors in by_type.items():
            if errors:
                print(f"  {node_type}:")
                print(f"    Mean error: {statistics.mean(errors):.4f}")
                print(f"    Max error: {max(errors):.4f}")
                print(f"    Samples: {len(errors)}")
    print()
    
    print("5. PREDICTION ACCURACY ASSESSMENT")
    print("-" * 50)
    # Overall assessment
    if 'avg_pred_error' in metrics_data:
        all_pred_errors = [e for e in metrics_data['avg_pred_error'] if e > 0]
        if all_pred_errors:
            overall_mean = statistics.mean(all_pred_errors)
            if overall_mean < 0.1:
                print(f"  ✓ EXCELLENT: Mean error {overall_mean:.4f} < 0.1")
            elif overall_mean < 0.5:
                print(f"  ✓ GOOD: Mean error {overall_mean:.4f} < 0.5")
            elif overall_mean < 1.0:
                print(f"  → MODERATE: Mean error {overall_mean:.4f} < 1.0")
            else:
                print(f"  ⚠ HIGH: Mean error {overall_mean:.4f} >= 1.0")
    print()

def main():
    print("=== PREDICTION ERROR ANALYSIS ===\n")
    
    metrics_data = load_metrics('unified_metrics.csv')
    samples_data = load_samples('unified_node_samples.csv')
    
    if metrics_data or samples_data:
        analyze_predictions(metrics_data, samples_data)
        
        # Save summary
        with open('prediction_analysis_summary.txt', 'w') as f:
            import io
            old_stdout = sys.stdout
            sys.stdout = f
            analyze_predictions(metrics_data, samples_data)
            sys.stdout = old_stdout
        print("✓ Summary saved to prediction_analysis_summary.txt")
    else:
        print("Error: Could not load data files")
        sys.exit(1)

if __name__ == '__main__':
    main()


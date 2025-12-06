#!/usr/bin/env python3
"""
Analyze trends in metrics over time from unified_metrics.csv
"""

import csv
import sys
from collections import defaultdict
import statistics

def load_metrics(filename):
    """Load metrics CSV into structured data."""
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

def analyze_trends(data):
    """Analyze trends over time."""
    print("=== TREND ANALYSIS ===\n")
    
    if not data or 'step' not in data:
        print("No data to analyze")
        return
    
    steps = data['step']
    n = len(steps)
    
    if n < 2:
        print("Not enough data points for trend analysis")
        return
    
    # Split into early, middle, late thirds
    third = n // 3
    early = slice(0, third)
    late = slice(n - third, n)
    
    print("1. ACTIVE COUNT TRENDS")
    print("-" * 50)
    if 'active_count' in data:
        early_avg = statistics.mean(data['active_count'][early])
        late_avg = statistics.mean(data['active_count'][late])
        trend = "↑ INCREASING" if late_avg > early_avg * 1.1 else "↓ DECREASING" if late_avg < early_avg * 0.9 else "→ STABLE"
        print(f"  Early (steps 0-{third}): {early_avg:.1f}")
        print(f"  Late (steps {n-third}-{n-1}): {late_avg:.1f}")
        print(f"  Trend: {trend}")
        print(f"  Min: {min(data['active_count']):.0f}, Max: {max(data['active_count']):.0f}")
    print()
    
    print("2. ENERGY TRENDS")
    print("-" * 50)
    if 'total_energy' in data:
        early_avg = statistics.mean(data['total_energy'][early])
        late_avg = statistics.mean(data['total_energy'][late])
        trend = "↑ INCREASING" if late_avg > early_avg * 1.1 else "↓ DECREASING" if late_avg < early_avg * 0.9 else "→ STABLE"
        print(f"  Early: {early_avg:.4f}")
        print(f"  Late: {late_avg:.4f}")
        print(f"  Trend: {trend}")
        print(f"  Min: {min(data['total_energy']):.4f}, Max: {max(data['total_energy']):.4f}")
    print()
    
    print("3. CHAOS/ERROR TRENDS")
    print("-" * 50)
    if 'avg_chaos' in data:
        early_avg = statistics.mean(data['avg_chaos'][early])
        late_avg = statistics.mean(data['avg_chaos'][late])
        trend = "↑ INCREASING" if late_avg > early_avg * 1.1 else "↓ DECREASING" if late_avg < early_avg * 0.9 else "→ STABLE"
        print(f"  Early: {early_avg:.4f}")
        print(f"  Late: {late_avg:.4f}")
        print(f"  Trend: {trend} (decreasing = more stable)")
        print(f"  Min: {min(data['avg_chaos']):.4f}, Max: {max(data['avg_chaos']):.4f}")
    print()
    
    print("4. PATTERN FORMATION")
    print("-" * 50)
    if 'num_patterns' in data:
        early_patterns = statistics.mean(data['num_patterns'][early])
        late_patterns = statistics.mean(data['num_patterns'][late])
        print(f"  Early: {early_patterns:.1f} patterns")
        print(f"  Late: {late_patterns:.1f} patterns")
        if late_patterns > early_patterns:
            print(f"  ✓ Patterns CREATED: +{late_patterns - early_patterns:.1f}")
        else:
            print(f"  → Pattern count stable")
        
        if 'num_active_patterns' in data:
            early_active = statistics.mean(data['num_active_patterns'][early])
            late_active = statistics.mean(data['num_active_patterns'][late])
            print(f"  Active patterns - Early: {early_active:.1f}, Late: {late_active:.1f}")
        
        if 'mean_pattern_value' in data:
            early_values = [v for v in data['mean_pattern_value'][early] if v != 0]
            late_values = [v for v in data['mean_pattern_value'][late] if v != 0]
            if early_values and late_values:
                early_value = statistics.mean(early_values)
                late_value = statistics.mean(late_values)
                print(f"  Mean pattern value - Early: {early_value:.4f}, Late: {late_value:.4f}")
    print()
    
    print("5. EXEC NODE ACTIVITY")
    print("-" * 50)
    if 'num_exec' in data:
        exec_count = data['num_exec'][0] if data['num_exec'] else 0
        print(f"  EXEC nodes: {exec_count:.0f}")
        
        if 'exec_fires' in data:
            total_fires = sum(data['exec_fires'])
            early_fires = sum(data['exec_fires'][early])
            late_fires = sum(data['exec_fires'][late])
            print(f"  Total fires: {total_fires:.0f}")
            print(f"  Early fires: {early_fires:.0f}")
            print(f"  Late fires: {late_fires:.0f}")
            if total_fires > 0:
                print(f"  ✓ EXEC nodes are firing")
            else:
                print(f"  ⚠ EXEC nodes not firing (may need pattern matching)")
    print()
    
    print("6. PREDICTION ERROR TRENDS")
    print("-" * 50)
    if 'avg_pred_error' in data:
        early_pred = statistics.mean([v for v in data['avg_pred_error'][early] if v > 0])
        late_pred = statistics.mean([v for v in data['avg_pred_error'][late] if v > 0])
        if early_pred and late_pred:
            improvement = ((early_pred - late_pred) / early_pred) * 100
            trend = "✓ IMPROVING" if improvement > 10 else "→ STABLE" if improvement > -10 else "⚠ WORSENING"
            print(f"  Early: {early_pred:.4f}")
            print(f"  Late: {late_pred:.4f}")
            print(f"  Change: {improvement:+.1f}% ({trend})")
    
    if 'avg_sensory_error' in data:
        early_sensory = statistics.mean([v for v in data['avg_sensory_error'][early] if v > 0])
        late_sensory = statistics.mean([v for v in data['avg_sensory_error'][late] if v > 0])
        if early_sensory and late_sensory:
            improvement = ((early_sensory - late_sensory) / early_sensory) * 100
            trend = "✓ IMPROVING" if improvement > 10 else "→ STABLE" if improvement > -10 else "⚠ WORSENING"
            print(f"  Sensory error - Early: {early_sensory:.4f}, Late: {late_sensory:.4f}")
            print(f"  Change: {improvement:+.1f}% ({trend})")
    print()
    
    print("7. VALUE DYNAMICS")
    print("-" * 50)
    if 'global_value' in data:
        early_value = statistics.mean(data['global_value'][early])
        late_value = statistics.mean(data['global_value'][late])
        print(f"  Global value - Early: {early_value:.4f}, Late: {late_value:.4f}")
        if late_value > early_value:
            print(f"  ✓ Value increasing (system learning)")
        elif late_value < early_value:
            print(f"  ⚠ Value decreasing")
        else:
            print(f"  → Value stable")
    
    if 'predicted_value_delta' in data:
        early_delta = statistics.mean(data['predicted_value_delta'][early])
        late_delta = statistics.mean(data['predicted_value_delta'][late])
        print(f"  Predicted delta - Early: {early_delta:.4f}, Late: {late_delta:.4f}")
    print()

def main():
    print("=== METRICS TREND ANALYSIS ===\n")
    
    data = load_metrics('unified_metrics.csv')
    if data:
        analyze_trends(data)
        
        # Save summary
        with open('trend_analysis_summary.txt', 'w') as f:
            import io
            old_stdout = sys.stdout
            sys.stdout = f
            analyze_trends(data)
            sys.stdout = old_stdout
        print("✓ Summary saved to trend_analysis_summary.txt")
    else:
        print("Error: Could not load metrics data")
        sys.exit(1)

if __name__ == '__main__':
    main()


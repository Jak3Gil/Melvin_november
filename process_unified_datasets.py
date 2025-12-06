#!/usr/bin/env python3
"""
Process unified test CSV files into datasets for analysis.

Converts unified_metrics.csv and unified_node_samples.csv into:
- Time series datasets
- Statistical summaries
- Visualization-ready formats
"""

import csv
import sys
import json
from collections import defaultdict
import statistics

def process_metrics_csv(filename):
    """Process unified_metrics.csv into structured dataset."""
    print(f"Processing {filename}...")
    
    data = {
        'steps': [],
        'active_count': [],
        'total_energy': [],
        'avg_chaos': [],
        'global_value': [],
        'predicted_value_delta': [],
        'num_patterns': [],
        'num_active_patterns': [],
        'mean_pattern_value': [],
        'max_pattern_value': [],
        'num_exec': [],
        'exec_fires': [],
        'mean_exec_control': [],
        'mean_exec_value_error': [],
        'avg_pred_error': [],
        'avg_sensory_error': [],
        'avg_active_count': []
    }
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in data.keys():
                    try:
                        data[key].append(float(row[key]))
                    except (ValueError, KeyError):
                        pass
        
        # Compute statistics
        stats = {}
        for key, values in data.items():
            if values:
                stats[key] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        return data, stats
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None, None

def process_samples_csv(filename):
    """Process unified_node_samples.csv into structured dataset."""
    print(f"Processing {filename}...")
    
    data_by_type = {
        'DATA': [],
        'PATTERN': [],
        'EXEC': []
    }
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_type = row.get('type', '')
                if node_type in data_by_type:
                    node_data = {
                        'step': int(row.get('step', 0)),
                        'node_id': int(row.get('node_id', 0)),
                        'energy': float(row.get('energy', 0)),
                        'value': float(row.get('value', 0)),
                        'prediction_error': float(row.get('prediction_error', 0)),
                        'sensory_pred_error': float(row.get('sensory_pred_error', 0)),
                        'recent_error_delta': float(row.get('recent_error_delta', 0)),
                        'recent_compression_gain': float(row.get('recent_compression_gain', 0)),
                        'recent_control_value': float(row.get('recent_control_value', 0))
                    }
                    if node_type == 'EXEC':
                        node_data['exec_count'] = int(row.get('exec_count', 0))
                        node_data['exec_success_rate'] = float(row.get('exec_success_rate', 0))
                    data_by_type[node_type].append(node_data)
        
        return data_by_type
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None

def generate_summary(metrics_data, metrics_stats, samples_data):
    """Generate summary report."""
    print("\n=== Dataset Summary ===")
    
    if metrics_stats and 'step' in metrics_stats:
        print("\nGlobal Metrics Statistics:")
        print(f"  Total steps: {metrics_stats['step']['count']}")
        if 'active_count' in metrics_stats:
            print(f"  Active count: mean={metrics_stats['active_count']['mean']:.1f}, "
                  f"min={metrics_stats['active_count']['min']:.0f}, "
                  f"max={metrics_stats['active_count']['max']:.0f}")
        if 'total_energy' in metrics_stats:
            print(f"  Total energy: mean={metrics_stats['total_energy']['mean']:.4f}, "
                  f"max={metrics_stats['total_energy']['max']:.4f}")
        if 'avg_chaos' in metrics_stats:
            print(f"  Avg chaos: mean={metrics_stats['avg_chaos']['mean']:.4f}")
        if 'global_value' in metrics_stats:
            print(f"  Global value: mean={metrics_stats['global_value']['mean']:.4f}, "
                  f"max={metrics_stats['global_value']['max']:.4f}")
        if 'num_patterns' in metrics_stats:
            print(f"  Patterns: mean={metrics_stats['num_patterns']['mean']:.1f}, "
                  f"active={metrics_stats['num_active_patterns']['mean']:.1f}")
        if 'num_exec' in metrics_stats:
            exec_fires_sum = sum(metrics_data.get('exec_fires', [0]))
            print(f"  EXEC nodes: {metrics_stats['num_exec']['mean']:.1f}, "
                  f"total fires={exec_fires_sum:.0f}")
    
    if samples_data:
        print("\nNode Samples:")
        for node_type, nodes in samples_data.items():
            if nodes:
                print(f"  {node_type}: {len(nodes)} samples")
                if nodes:
                    avg_energy = statistics.mean([n['energy'] for n in nodes])
                    avg_value = statistics.mean([n['value'] for n in nodes])
                    print(f"    Avg energy: {avg_energy:.4f}, Avg value: {avg_value:.4f}")

def save_json_dataset(metrics_data, samples_data, output_prefix):
    """Save datasets as JSON for easy analysis."""
    if metrics_data:
        with open(f'{output_prefix}_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\n✓ Saved {output_prefix}_metrics.json")
    
    if samples_data:
        with open(f'{output_prefix}_samples.json', 'w') as f:
            json.dump(samples_data, f, indent=2)
        print(f"✓ Saved {output_prefix}_samples.json")

def main():
    print("=== Unified Test Dataset Processor ===\n")
    
    # Process metrics
    metrics_data, metrics_stats = process_metrics_csv('unified_metrics.csv')
    
    # Process samples
    samples_data = process_samples_csv('unified_node_samples.csv')
    
    # Generate summary
    if metrics_data or samples_data:
        generate_summary(metrics_data, metrics_stats, samples_data)
        
        # Save JSON datasets
        save_json_dataset(metrics_data, samples_data, 'unified_dataset')
        
        print("\n✓ Dataset processing complete!")
        print("\nGenerated files:")
        print("  - unified_dataset_metrics.json")
        print("  - unified_dataset_samples.json")
    else:
        print("Error: No data files found")
        sys.exit(1)

if __name__ == '__main__':
    main()


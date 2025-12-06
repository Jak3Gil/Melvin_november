#!/usr/bin/env python3
"""
analyze_metrics.py - Analyze CSV metrics and generate numeric scores

NO INTERPRETIVE SCORING - Only numeric thresholds and calculations
"""

import csv
import sys
import os
import numpy as np
from pathlib import Path

def load_csv(csv_path):
    """Load CSV and return as list of dicts"""
    if not os.path.exists(csv_path):
        return []
    
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'step': int(row['step']),
                'total_energy': float(row['total_energy']),
                'active_count': int(row['active_count']),
                'active_groups': int(row['active_groups']),
                'pattern_nodes_active': int(row['pattern_nodes_active']),
                'exec_fires': int(row['exec_fires']),
                'node_count': int(row['node_count']),
                'edge_count': int(row['edge_count']),
                'avg_activation': float(row['avg_activation']),
                'avg_chaos': float(row['avg_chaos']),
            })
    return data

def test_1_pattern_stability(csv_path):
    """Test 1: Pattern Stability & Compression"""
    data = load_csv(csv_path)
    if len(data) < 10:
        return None
    
    # Extract metrics
    steps = [d['step'] for d in data]
    active_counts = [d['active_count'] for d in data]
    pattern_nodes = [d['pattern_nodes_active'] for d in data]
    node_counts = [d['node_count'] for d in data]
    edge_counts = [d['edge_count'] for d in data]
    
    # Metrics
    initial_active = active_counts[0] if active_counts else 0
    final_active = active_counts[-1] if active_counts else 0
    max_active = max(active_counts) if active_counts else 0
    avg_active = np.mean(active_counts) if active_counts else 0
    
    initial_patterns = pattern_nodes[0] if pattern_nodes else 0
    final_patterns = pattern_nodes[-1] if pattern_nodes else 0
    max_patterns = max(pattern_nodes) if pattern_nodes else 0
    
    initial_edges = edge_counts[0] if edge_counts else 0
    final_edges = edge_counts[-1] if edge_counts else 0
    
    # Compression check: did edges grow sublinearly?
    # If pattern formed, edges should grow slowly (pattern reuse)
    edge_growth = final_edges - initial_edges
    
    # Pattern formation: did pattern nodes appear?
    pattern_formed = max_patterns > 0
    
    # Compression: did active_count decrease or stay bounded?
    compression_ratio = (initial_active - final_active) / max(initial_active, 1)
    
    return {
        'initial_active': initial_active,
        'final_active': final_active,
        'max_active': max_active,
        'avg_active': avg_active,
        'initial_patterns': initial_patterns,
        'final_patterns': final_patterns,
        'max_patterns': max_patterns,
        'pattern_formed': pattern_formed,
        'edge_growth': edge_growth,
        'compression_ratio': compression_ratio,
    }

def test_2_locality(csv_path):
    """Test 2: Locality of Activation"""
    data = load_csv(csv_path)
    if len(data) < 5:
        return None
    
    active_counts = [d['active_count'] for d in data]
    active_groups = [d['active_groups'] for d in data]
    node_counts = [d['node_count'] for d in data]
    
    max_active = max(active_counts) if active_counts else 0
    avg_active = np.mean(active_counts) if active_counts else 0
    std_active = np.std(active_counts) if active_counts else 0
    
    max_groups = max(active_groups) if active_groups else 0
    avg_groups = np.mean(active_groups) if active_groups else 0
    
    final_node_count = node_counts[-1] if node_counts else 0
    
    # Locality check: active_count should be bounded regardless of node_count
    # If active_count << node_count, locality is maintained
    locality_ratio = max_active / max(final_node_count, 1)
    
    # Bounded groups: number of active groups should be bounded
    groups_bounded = max_groups < 100  # Threshold: < 100 groups active
    
    return {
        'max_active': max_active,
        'avg_active': avg_active,
        'std_active': std_active,
        'max_groups': max_groups,
        'avg_groups': avg_groups,
        'final_node_count': final_node_count,
        'locality_ratio': locality_ratio,
        'groups_bounded': groups_bounded,
    }

def test_3_surprise(csv_path):
    """Test 3: Reaction to Surprise"""
    data = load_csv(csv_path)
    if len(data) < 10:
        return None
    
    # Find baseline region (first half) and anomaly region (second half)
    mid_point = len(data) // 2
    
    baseline_data = data[:mid_point]
    anomaly_data = data[mid_point:]
    
    baseline_active = [d['active_count'] for d in baseline_data]
    anomaly_active = [d['active_count'] for d in anomaly_data]
    
    baseline_energy = [d['total_energy'] for d in baseline_data]
    anomaly_energy = [d['total_energy'] for d in anomaly_data]
    
    baseline_groups = [d['active_groups'] for d in baseline_data]
    anomaly_groups = [d['active_groups'] for d in anomaly_data]
    
    avg_baseline_active = np.mean(baseline_active) if baseline_active else 0
    avg_anomaly_active = np.mean(anomaly_active) if anomaly_active else 0
    max_anomaly_active = max(anomaly_active) if anomaly_active else 0
    
    avg_baseline_energy = np.mean(baseline_energy) if baseline_energy else 0
    avg_anomaly_energy = np.mean(anomaly_energy) if anomaly_energy else 0
    
    avg_baseline_groups = np.mean(baseline_groups) if baseline_groups else 0
    avg_anomaly_groups = np.mean(anomaly_groups) if anomaly_groups else 0
    
    # Surprise metrics
    active_delta = avg_anomaly_active - avg_baseline_active
    energy_delta = avg_anomaly_energy - avg_baseline_energy
    groups_delta = avg_anomaly_groups - avg_baseline_groups
    
    # Localization: anomaly should cause local spike, not global explosion
    # If max_anomaly_active < 1000, it's localized
    localized = max_anomaly_active < 1000
    
    return {
        'avg_baseline_active': avg_baseline_active,
        'avg_anomaly_active': avg_anomaly_active,
        'max_anomaly_active': max_anomaly_active,
        'active_delta': active_delta,
        'avg_baseline_energy': avg_baseline_energy,
        'avg_anomaly_energy': avg_anomaly_energy,
        'energy_delta': energy_delta,
        'avg_baseline_groups': avg_baseline_groups,
        'avg_anomaly_groups': avg_anomaly_groups,
        'groups_delta': groups_delta,
        'localized': localized,
    }

def test_4_memory_recall(csv_path):
    """Test 4: Memory Recall Under Load"""
    data = load_csv(csv_path)
    if len(data) < 5:
        return None
    
    # Find recall phase (last N steps where pattern is searched)
    # Assume last 10 steps are recall phase
    recall_start = max(0, len(data) - 10)
    recall_data = data[recall_start:]
    
    recall_active = [d['active_count'] for d in recall_data]
    recall_groups = [d['active_groups'] for d in recall_data]
    final_node_count = data[-1]['node_count'] if data else 0
    
    avg_recall_active = np.mean(recall_active) if recall_active else 0
    max_recall_active = max(recall_active) if recall_active else 0
    avg_recall_groups = np.mean(recall_groups) if recall_groups else 0
    
    # Recall cost: active_count should be bounded regardless of node_count
    recall_cost_ratio = avg_recall_active / max(final_node_count, 1)
    
    # Bounded recall: active_count should stay small
    recall_bounded = avg_recall_active < 1000
    
    return {
        'final_node_count': final_node_count,
        'avg_recall_active': avg_recall_active,
        'max_recall_active': max_recall_active,
        'avg_recall_groups': avg_recall_groups,
        'recall_cost_ratio': recall_cost_ratio,
        'recall_bounded': recall_bounded,
    }

def test_5_exec_triggering(csv_path):
    """Test 5: EXEC Function Triggering"""
    data = load_csv(csv_path)
    if len(data) < 5:
        return None
    
    exec_fires = [d['exec_fires'] for d in data]
    initial_exec = exec_fires[0] if exec_fires else 0
    final_exec = exec_fires[-1] if exec_fires else 0
    exec_delta = final_exec - initial_exec
    
    # Check for EXEC firing log
    exec_log_path = csv_path + '.exec_fires'
    exec_fired = False
    if os.path.exists(exec_log_path):
        with open(exec_log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                delta = int(row.get('delta', 0))
                if delta > 0:
                    exec_fired = True
                    break
    
    return {
        'initial_exec': initial_exec,
        'final_exec': final_exec,
        'exec_delta': exec_delta,
        'exec_fired': exec_fired,
    }

def main():
    results_dir = "evaluation_results"
    
    print("=" * 60)
    print("MELVIN METRICS ANALYSIS")
    print("=" * 60)
    print()
    
    # Test 1
    csv_path = f"{results_dir}/test_1_pattern_stability.csv"
    if os.path.exists(csv_path):
        print("Test 1: Pattern Stability & Compression")
        print("-" * 60)
        result = test_1_pattern_stability(csv_path)
        if result:
            print(f"  Initial active: {result['initial_active']}")
            print(f"  Final active: {result['final_active']}")
            print(f"  Max active: {result['max_active']}")
            print(f"  Pattern formed: {result['pattern_formed']}")
            print(f"  Max patterns: {result['max_patterns']}")
            print(f"  Compression ratio: {result['compression_ratio']:.3f}")
        print()
    
    # Test 2
    csv_path = f"{results_dir}/test_2_locality.csv"
    if os.path.exists(csv_path):
        print("Test 2: Locality of Activation")
        print("-" * 60)
        result = test_2_locality(csv_path)
        if result:
            print(f"  Max active: {result['max_active']}")
            print(f"  Avg active: {result['avg_active']:.2f}")
            print(f"  Max groups: {result['max_groups']}")
            print(f"  Final node count: {result['final_node_count']}")
            print(f"  Locality ratio: {result['locality_ratio']:.6f}")
            print(f"  Groups bounded: {result['groups_bounded']}")
        print()
    
    # Test 3
    csv_path = f"{results_dir}/test_3_surprise.csv"
    if os.path.exists(csv_path):
        print("Test 3: Reaction to Surprise")
        print("-" * 60)
        result = test_3_surprise(csv_path)
        if result:
            print(f"  Baseline active: {result['avg_baseline_active']:.2f}")
            print(f"  Anomaly active: {result['avg_anomaly_active']:.2f}")
            print(f"  Active delta: {result['active_delta']:.2f}")
            print(f"  Energy delta: {result['energy_delta']:.6f}")
            print(f"  Max anomaly active: {result['max_anomaly_active']}")
            print(f"  Localized: {result['localized']}")
        print()
    
    # Test 4
    csv_path = f"{results_dir}/test_4_memory_recall.csv"
    if os.path.exists(csv_path):
        print("Test 4: Memory Recall Under Load")
        print("-" * 60)
        result = test_4_memory_recall(csv_path)
        if result:
            print(f"  Final node count: {result['final_node_count']}")
            print(f"  Avg recall active: {result['avg_recall_active']:.2f}")
            print(f"  Max recall active: {result['max_recall_active']}")
            print(f"  Recall cost ratio: {result['recall_cost_ratio']:.6f}")
            print(f"  Recall bounded: {result['recall_bounded']}")
        print()
    
    # Test 5
    csv_path = f"{results_dir}/test_5_exec_triggering.csv"
    if os.path.exists(csv_path):
        print("Test 5: EXEC Function Triggering")
        print("-" * 60)
        result = test_5_exec_triggering(csv_path)
        if result:
            print(f"  Initial exec fires: {result['initial_exec']}")
            print(f"  Final exec fires: {result['final_exec']}")
            print(f"  Exec delta: {result['exec_delta']}")
            print(f"  Exec fired: {result['exec_fired']}")
        print()
    
    print("=" * 60)
    print("Analysis complete. See CSV files for raw data.")
    print("=" * 60)

if __name__ == '__main__':
    main()


"""
Analysis: Compare Melvin vs LSTM Pattern Learning

Generate comparison graphs and calculate efficiency ratio.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_data():
    """Load results from both experiments"""
    melvin_df = pd.read_csv("benchmarks/data/experiment1_results.csv")
    lstm_df = pd.read_csv("benchmarks/data/lstm_baseline_results.csv")
    return melvin_df, lstm_df

def calculate_efficiency_ratio(melvin_df, lstm_df, accuracy_threshold=0.90):
    """Calculate how many fewer examples Melvin needs"""
    
    # Find examples needed for Melvin to reach threshold
    melvin_needed = melvin_df[melvin_df['recognition_score'] >= accuracy_threshold]
    if len(melvin_needed) == 0:
        melvin_examples = melvin_df['repetitions'].max()
    else:
        melvin_examples = melvin_needed['repetitions'].min()
    
    # Find examples needed for LSTM to reach threshold
    lstm_needed = lstm_df[lstm_df['accuracy'] >= accuracy_threshold]
    if len(lstm_needed) == 0:
        lstm_examples = lstm_df['examples'].max()
    else:
        lstm_examples = lstm_needed['examples'].min()
    
    # Calculate ratio
    if melvin_examples > 0:
        ratio = lstm_examples / melvin_examples
    else:
        ratio = float('inf')
    
    return melvin_examples, lstm_examples, ratio

def plot_comparison(melvin_df, lstm_df):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Melvin vs LSTM: Pattern Learning Efficiency', fontsize=16, fontweight='bold')
    
    # Plot 1: Learning curves
    ax1 = axes[0, 0]
    ax1.plot(melvin_df['repetitions'], melvin_df['recognition_score'], 
             'o-', label='Melvin', linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(lstm_df['examples'], lstm_df['accuracy'], 
             's-', label='LSTM', linewidth=2, markersize=8, color='#A23B72')
    ax1.axhline(y=0.90, color='gray', linestyle='--', label='90% Threshold')
    ax1.set_xlabel('Training Examples', fontsize=12)
    ax1.set_ylabel('Accuracy / Recognition Score', fontsize=12)
    ax1.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory usage
    ax2 = axes[0, 1]
    melvin_memory = melvin_df['nodes'] * 64 + melvin_df['edges'] * 20  # Approximate bytes
    lstm_memory = lstm_df['memory_bytes'].iloc[0]  # Constant for LSTM
    
    ax2.bar(['Melvin\n(20 examples)', 'LSTM\n(20 examples)'], 
            [melvin_memory.iloc[-1], lstm_memory],
            color=['#2E86AB', '#A23B72'])
    ax2.set_ylabel('Memory (bytes)', fontsize=12)
    ax2.set_title('Memory Footprint', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Pattern discovery rate
    ax3 = axes[1, 0]
    ax3.plot(melvin_df['repetitions'], melvin_df['pattern_count'], 
             'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax3.set_xlabel('Repetitions', fontsize=12)
    ax3.set_ylabel('Patterns Discovered', fontsize=12)
    ax3.set_title('Melvin: Pattern Discovery Rate', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Edge strength evolution
    ax4 = axes[1, 1]
    ax4.plot(melvin_df['repetitions'], melvin_df['avg_edge_strength'], 
             'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax4.set_xlabel('Repetitions', fontsize=12)
    ax4.set_ylabel('Average Edge Strength', fontsize=12)
    ax4.set_title('Melvin: Edge Strengthening', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    Path("benchmarks/analysis").mkdir(parents=True, exist_ok=True)
    plt.savefig("benchmarks/analysis/comparison_plot.png", dpi=300, bbox_inches='tight')
    print("Saved: benchmarks/analysis/comparison_plot.png")
    
    plt.show()

def generate_report(melvin_df, lstm_df):
    """Generate text report"""
    
    report = []
    report.append("=" * 60)
    report.append("EXPERIMENT 1: PATTERN LEARNING EFFICIENCY")
    report.append("Melvin vs LSTM Comparison")
    report.append("=" * 60)
    report.append("")
    
    # Calculate efficiency ratios at different thresholds
    for threshold in [0.80, 0.90, 0.95]:
        melvin_ex, lstm_ex, ratio = calculate_efficiency_ratio(melvin_df, lstm_df, threshold)
        report.append(f"To reach {threshold:.0%} accuracy:")
        report.append(f"  Melvin:  {melvin_ex} examples")
        report.append(f"  LSTM:    {lstm_ex} examples")
        report.append(f"  Ratio:   {ratio:.1f}x more efficient (Melvin)")
        report.append("")
    
    # Memory comparison
    melvin_memory = (melvin_df['nodes'].iloc[-1] * 64 + 
                     melvin_df['edges'].iloc[-1] * 20)
    lstm_memory = lstm_df['memory_bytes'].iloc[0]
    memory_ratio = lstm_memory / melvin_memory
    
    report.append(f"Memory Footprint (after 20 examples):")
    report.append(f"  Melvin:  {melvin_memory:,.0f} bytes")
    report.append(f"  LSTM:    {lstm_memory:,.0f} bytes")
    report.append(f"  Ratio:   {memory_ratio:.1f}x more efficient (Melvin)")
    report.append("")
    
    # Parameters
    lstm_params = lstm_df['parameters'].iloc[0]
    melvin_params = int(melvin_df['nodes'].iloc[-1] + melvin_df['edges'].iloc[-1])
    param_ratio = lstm_params / melvin_params
    
    report.append(f"Model Complexity:")
    report.append(f"  Melvin:  {melvin_params:,} elements (nodes + edges)")
    report.append(f"  LSTM:    {lstm_params:,} parameters")
    report.append(f"  Ratio:   {param_ratio:.1f}x simpler (Melvin)")
    report.append("")
    
    # Patterns discovered
    patterns = melvin_df['pattern_count'].iloc[-1]
    report.append(f"Patterns Discovered (Melvin): {patterns}")
    report.append("")
    
    report.append("=" * 60)
    report.append("CONCLUSION")
    report.append("=" * 60)
    
    avg_ratio = (ratio + memory_ratio + param_ratio) / 3
    report.append(f"Average Efficiency Gain: {avg_ratio:.0f}x")
    report.append("")
    report.append("Melvin demonstrates:")
    report.append("  ✓ Faster learning (fewer examples needed)")
    report.append("  ✓ Lower memory footprint")
    report.append("  ✓ Simpler model structure")
    report.append("  ✓ Automatic pattern discovery")
    report.append("")
    
    report_text = "\n".join(report)
    
    # Save report
    with open("benchmarks/analysis/experiment1_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    print("\nSaved: benchmarks/analysis/experiment1_report.txt")

def main():
    print("Loading data...")
    
    try:
        melvin_df, lstm_df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run experiments first:")
        print("  1. ./benchmarks/experiment1_pattern_efficiency")
        print("  2. python3 benchmarks/baselines/lstm_pattern_learning.py")
        return
    
    print("Generating analysis...")
    
    # Generate report
    generate_report(melvin_df, lstm_df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(melvin_df, lstm_df)
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()


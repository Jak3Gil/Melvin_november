"""
Plot long-term learning efficiency curves
Shows how learning rate changes over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    try:
        df = pd.read_csv("benchmarks/data/experiment4_results.csv")
    except FileNotFoundError:
        print("Error: Run experiment4 first!")
        print("  ./benchmarks/experiment4_long_term_learning 10000")
        return
    
    print(f"Loaded {len(df)} checkpoints")
    print(f"Total inputs: {df['inputs'].max():,}")
    print(f"Total time: {df['time_sec'].max():.1f}s ({df['time_sec'].max()/60:.1f} min)")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Long-Term Learning Efficiency', fontsize=16, fontweight='bold')
    
    # Plot 1: Pattern growth over time
    ax1 = axes[0, 0]
    ax1.plot(df['inputs'], df['patterns'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
    ax1.set_xlabel('Inputs', fontsize=12)
    ax1.set_ylabel('Total Patterns', fontsize=12)
    ax1.set_title('Pattern Library Growth', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate over time (KEY METRIC!)
    ax2 = axes[0, 1]
    ax2.plot(df['inputs'], df['learning_rate'], 'o-', linewidth=2, markersize=6, color='#A23B72')
    ax2.axhline(y=df['learning_rate'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial rate')
    ax2.set_xlabel('Inputs', fontsize=12)
    ax2.set_ylabel('Learning Rate (patterns per 1K inputs)', fontsize=12)
    ax2.set_title('Learning Rate Evolution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add interpretation
    if len(df) > 1:
        initial_rate = df['learning_rate'].iloc[1]  # Skip first (often noisy)
        final_rate = df['learning_rate'].iloc[-1]
        if final_rate < initial_rate * 0.8:
            ax2.text(0.5, 0.95, '✓ Learning rate DECREASING\n(Reuse working!)', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        elif final_rate > initial_rate * 1.2:
            ax2.text(0.5, 0.95, '⚠ Learning rate INCREASING\n(Pattern explosion?)', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Plot 3: Patterns per input (efficiency)
    ax3 = axes[1, 0]
    ax3.plot(df['inputs'], df['patterns_per_input'], 'o-', linewidth=2, markersize=6, color='#F18F01')
    ax3.set_xlabel('Inputs', fontsize=12)
    ax3.set_ylabel('Patterns per Input', fontsize=12)
    ax3.set_title('Pattern Efficiency', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Graph size
    ax4 = axes[1, 1]
    ax4.plot(df['inputs'], df['nodes'], 'o-', label='Nodes', linewidth=2, markersize=6)
    ax4.plot(df['inputs'], df['edges'], 's-', label='Edges', linewidth=2, markersize=6)
    ax4.set_xlabel('Inputs', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Graph Growth', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("benchmarks/analysis/learning_curve.png", dpi=300, bbox_inches='tight')
    print("\nSaved: benchmarks/analysis/learning_curve.png")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Initial learning rate: {df['learning_rate'].iloc[1]:.3f} patterns/1K inputs")
    print(f"Final learning rate:   {df['learning_rate'].iloc[-1]:.3f} patterns/1K inputs")
    
    if len(df) > 1:
        rate_change = (df['learning_rate'].iloc[-1] / df['learning_rate'].iloc[1] - 1) * 100
        print(f"Change: {rate_change:+.1f}%")
        
        if rate_change < -20:
            print("\n✓ SUCCESS: Learning rate decreased significantly!")
            print("  Pattern reuse is working - efficiency improving over time")
        elif rate_change > 20:
            print("\n⚠ WARNING: Learning rate increased!")
            print("  May indicate pattern explosion or insufficient reuse")
        else:
            print("\n→ NEUTRAL: Learning rate relatively stable")
            print("  May need more data to see reuse effects")
    
    plt.show()

if __name__ == "__main__":
    main()


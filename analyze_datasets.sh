#!/bin/bash
# Quick analysis of unified test datasets

echo "=== Unified Test Dataset Analysis ==="
echo ""

if [ -f unified_metrics.csv ]; then
    echo "unified_metrics.csv:"
    echo "  Total rows: $(wc -l < unified_metrics.csv)"
    echo "  Columns: $(head -1 unified_metrics.csv | tr ',' '\n' | wc -l)"
    echo ""
    echo "  Sample values (last 3 steps):"
    tail -3 unified_metrics.csv
    echo ""
fi

if [ -f unified_node_samples.csv ]; then
    echo "unified_node_samples.csv:"
    echo "  Total rows: $(wc -l < unified_node_samples.csv)"
    echo "  Node types sampled:"
    tail -n +2 unified_node_samples.csv | cut -d',' -f2 | sort | uniq -c
    echo ""
fi

if [ -f unified_dataset_metrics.json ]; then
    echo "✓ Processed JSON datasets available"
    echo "  - unified_dataset_metrics.json"
    echo "  - unified_dataset_samples.json"
fi

echo ""
echo "To visualize, use Python with matplotlib/pandas:"
echo "  python3 -c \"import pandas as pd; df = pd.read_csv('unified_metrics.csv'); print(df.describe())\""

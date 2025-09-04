#!/usr/bin/env python3
"""
🧠 TESTING OPTIMIZED NODE SYSTEM WITH HUGGING FACE DATA
========================================================
Feeds real Hugging Face datasets to the optimized node system.
"""

import json
import time
from pathlib import Path
from test_optimized_nodes_python import OptimizedDynamicNodeSizer

def load_hf_datasets():
    """Load collected Hugging Face datasets"""
    datasets = {}
    
    # Check for collected data files
    data_files = [
        "melvin_datasets/collection_results.json",
        "melvin_datasets/collection_metadata.json",
        "melvin_datasets/final_report.json"
    ]
    
    for file_path in data_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    datasets[Path(file_path).stem] = data
                print(f"📁 Loaded {file_path}")
            except Exception as e:
                print(f"⚠️  Could not load {file_path}: {e}")
    
    return datasets

def extract_text_samples(datasets):
    """Extract text samples from datasets"""
    samples = []
    
    for dataset_name, dataset in datasets.items():
        if isinstance(dataset, dict):
            # Look for text content in various fields
            text_fields = ['text', 'content', 'sentence', 'paragraph', 'description', 'title']
            
            for field in text_fields:
                if field in dataset:
                    content = dataset[field]
                    if isinstance(content, str) and len(content) > 10:
                        samples.append({
                            'source': dataset_name,
                            'field': field,
                            'text': content,
                            'length': len(content)
                        })
            
            # Look for nested data
            if 'data' in dataset and isinstance(dataset['data'], list):
                for item in dataset['data'][:10]:  # Limit to first 10 items
                    if isinstance(item, dict):
                        for field in text_fields:
                            if field in item:
                                content = item[field]
                                if isinstance(content, str) and len(content) > 10:
                                    samples.append({
                                        'source': dataset_name,
                                        'field': field,
                                        'text': content,
                                        'length': len(content)
                                    })
    
    return samples

def test_with_hf_data():
    """Test the optimized system with Hugging Face data"""
    print("🧠 TESTING WITH HUGGING FACE DATA")
    print("=" * 50)
    
    # Load datasets
    datasets = load_hf_datasets()
    if not datasets:
        print("⚠️  No Hugging Face datasets found. Using sample data instead.")
        # Use sample data if no HF datasets available
        sample_data = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "Natural language processing focuses on enabling computers to understand, interpret, and generate human language.",
            "Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information.",
            "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment."
        ]
        
        samples = []
        for i, text in enumerate(sample_data):
            samples.append({
                'source': 'sample_data',
                'field': 'text',
                'text': text,
                'length': len(text)
            })
    else:
        samples = extract_text_samples(datasets)
    
    print(f"📊 Found {len(samples)} text samples")
    
    # Create optimized sizer
    sizer = OptimizedDynamicNodeSizer()
    
    # Test different processing strategies
    strategies = [
        ('tiny', 'Word-level processing'),
        ('small', 'Phrase-level processing'),
        ('medium', 'Concept-level processing'),
        ('large', 'Section-level processing'),
        ('extra_large', 'Document-level processing')
    ]
    
    results = {}
    
    for strategy, description in strategies:
        print(f"\n🔧 {description.upper()}")
        print("-" * 40)
        
        start_time = time.time()
        total_nodes = 0
        processed_samples = 0
        
        for sample in samples:
            nodes = sizer.create_dynamic_nodes(sample['text'], strategy)
            total_nodes += len(nodes)
            processed_samples += 1
        
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        memory_usage = sizer.get_memory_usage()
        stats = sizer.get_statistics()
        
        results[strategy] = {
            'duration': duration,
            'total_nodes': total_nodes,
            'memory_usage': memory_usage,
            'stats': stats.copy(),
            'processed_samples': processed_samples
        }
        
        print(f"⏱️  Processing time: {duration:.2f} ms")
        print(f"📊 Total nodes created: {total_nodes}")
        print(f"📄 Samples processed: {processed_samples}")
        print(f"💾 Memory usage: {memory_usage:,} bytes ({memory_usage/1024:.2f} KB)")
        
        if total_nodes > 0:
            avg_bytes_per_node = memory_usage / total_nodes
            nodes_per_ms = total_nodes / duration if duration > 0 else 0
            print(f"📊 Average bytes per node: {avg_bytes_per_node:.1f}")
            print(f"🚀 Processing rate: {nodes_per_ms:.1f} nodes/ms")
        
        print(f"📈 Node distribution:")
        for key, value in stats.items():
            if value > 0:
                print(f"   🔹 {key}: {value}")
    
    # Performance comparison
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    print(f"{'Strategy':<15} {'Time (ms)':<12} {'Nodes':<8} {'Memory (KB)':<12} {'Rate (nodes/ms)':<15}")
    print("-" * 70)
    
    for strategy, result in results.items():
        rate = result['total_nodes'] / result['duration'] if result['duration'] > 0 else 0
        memory_kb = result['memory_usage'] / 1024
        print(f"{strategy:<15} {result['duration']:<12.2f} {result['total_nodes']:<8} {memory_kb:<12.2f} {rate:<15.1f}")
    
    # Memory efficiency analysis
    print(f"\n💾 MEMORY EFFICIENCY ANALYSIS")
    print("=" * 50)
    
    for strategy, result in results.items():
        if result['total_nodes'] > 0:
            efficiency = result['memory_usage'] / result['total_nodes']
            print(f"🔹 {strategy}: {efficiency:.1f} bytes per node")
    
    # Content analysis
    print(f"\n📝 CONTENT ANALYSIS")
    print("=" * 50)
    
    if samples:
        total_length = sum(sample['length'] for sample in samples)
        avg_length = total_length / len(samples)
        print(f"📊 Total text length: {total_length:,} characters")
        print(f"📊 Average sample length: {avg_length:.1f} characters")
        print(f"📊 Number of samples: {len(samples)}")
        
        # Show sample distribution
        length_ranges = [(0, 50), (51, 100), (101, 200), (201, 500), (501, 1000), (1001, float('inf'))]
        print(f"📊 Length distribution:")
        for min_len, max_len in length_ranges:
            count = sum(1 for sample in samples if min_len <= sample['length'] < max_len)
            if count > 0:
                range_name = f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
                print(f"   🔹 {range_name} chars: {count} samples")
    
    # Final system state
    print(f"\n🎯 FINAL SYSTEM STATE")
    print("=" * 50)
    
    final_stats = sizer.get_statistics()
    final_memory = sizer.get_memory_usage()
    
    total_nodes = sum(final_stats.values())
    print(f"📊 Total nodes in system: {total_nodes}")
    print(f"💾 Total memory usage: {final_memory:,} bytes ({final_memory/1024/1024:.2f} MB)")
    
    if total_nodes > 0:
        print(f"📈 Node distribution:")
        for key, value in final_stats.items():
            if value > 0:
                percentage = (value / total_nodes) * 100
                print(f"   🔹 {key}: {value} ({percentage:.1f}%)")
    
    print(f"\n✅ HUGGING FACE DATA TEST COMPLETED!")
    print(f"🎉 Optimized system successfully processed real-world data!")

if __name__ == "__main__":
    test_with_hf_data()

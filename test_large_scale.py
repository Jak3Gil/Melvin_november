#!/usr/bin/env python3
"""
🧠 LARGE-SCALE OPTIMIZED NODE SYSTEM TEST
=========================================
Tests the optimized node system with large-scale data processing.
"""

import time
import random
import string
from test_optimized_nodes_python import OptimizedDynamicNodeSizer

def generate_large_dataset(num_samples=1000, min_length=50, max_length=500):
    """Generate a large dataset for testing"""
    samples = []
    
    # AI/ML related vocabulary
    vocabulary = [
        "artificial intelligence", "machine learning", "deep learning", "neural networks",
        "computer vision", "natural language processing", "reinforcement learning",
        "supervised learning", "unsupervised learning", "transfer learning",
        "convolutional neural networks", "recurrent neural networks", "transformers",
        "backpropagation", "gradient descent", "optimization algorithms",
        "feature extraction", "pattern recognition", "data preprocessing",
        "model training", "validation", "testing", "overfitting", "underfitting",
        "regularization", "dropout", "batch normalization", "activation functions",
        "loss functions", "accuracy", "precision", "recall", "f1 score",
        "cross validation", "hyperparameter tuning", "ensemble methods",
        "random forests", "support vector machines", "k-means clustering",
        "principal component analysis", "dimensionality reduction"
    ]
    
    # Technical terms
    technical_terms = [
        "algorithm", "implementation", "architecture", "framework", "library",
        "API", "interface", "protocol", "standard", "specification",
        "deployment", "production", "scaling", "performance", "optimization",
        "efficiency", "throughput", "latency", "bandwidth", "capacity",
        "reliability", "availability", "fault tolerance", "redundancy",
        "monitoring", "logging", "debugging", "profiling", "benchmarking"
    ]
    
    # Action words
    actions = [
        "processes", "analyzes", "computes", "calculates", "evaluates",
        "optimizes", "trains", "learns", "adapts", "evolves",
        "classifies", "regresses", "clusters", "segments", "detects",
        "recognizes", "identifies", "extracts", "transforms", "generates",
        "predicts", "forecasts", "estimates", "approximates", "simulates"
    ]
    
    all_words = vocabulary + technical_terms + actions
    
    for i in range(num_samples):
        # Generate random length
        length = random.randint(min_length, max_length)
        
        # Create sentence
        sentence_parts = []
        current_length = 0
        
        while current_length < length:
            # Add random words
            num_words = random.randint(3, 8)
            words = random.sample(all_words, min(num_words, len(all_words)))
            phrase = " ".join(words)
            
            if current_length + len(phrase) + 1 <= length:
                sentence_parts.append(phrase)
                current_length += len(phrase) + 1
            else:
                break
        
        # Create sentence
        sentence = " ".join(sentence_parts) + "."
        
        # Ensure minimum length
        if len(sentence) < min_length:
            sentence += " " + " ".join(random.sample(all_words, 3)) + "."
        
        samples.append(sentence)
    
    return samples

def test_large_scale_processing():
    """Test large-scale data processing"""
    print("🧠 LARGE-SCALE OPTIMIZED NODE SYSTEM TEST")
    print("=" * 60)
    
    # Generate test datasets of different sizes
    dataset_sizes = [100, 500, 1000, 2000]
    
    for size in dataset_sizes:
        print(f"\n📊 TESTING DATASET SIZE: {size} SAMPLES")
        print("=" * 50)
        
        # Generate dataset
        print(f"🔧 Generating {size} samples...")
        start_gen = time.time()
        samples = generate_large_dataset(size)
        end_gen = time.time()
        gen_time = (end_gen - start_gen) * 1000
        
        print(f"⏱️  Generation time: {gen_time:.2f} ms")
        print(f"📝 Average sample length: {sum(len(s) for s in samples) / len(samples):.1f} chars")
        
        # Test different processing strategies
        strategies = [
            ('tiny', 'Word-level'),
            ('small', 'Phrase-level'),
            ('medium', 'Concept-level'),
            ('large', 'Section-level')
        ]
        
        strategy_results = {}
        
        for strategy, description in strategies:
            print(f"\n🔧 {description.upper()} PROCESSING")
            print("-" * 40)
            
            # Create fresh sizer for each strategy
            sizer = OptimizedDynamicNodeSizer()
            
            start_time = time.time()
            total_nodes = 0
            
            for sample in samples:
                nodes = sizer.create_dynamic_nodes(sample, strategy)
                total_nodes += len(nodes)
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            memory_usage = sizer.get_memory_usage()
            stats = sizer.get_statistics()
            
            strategy_results[strategy] = {
                'duration': duration,
                'total_nodes': total_nodes,
                'memory_usage': memory_usage,
                'stats': stats.copy()
            }
            
            print(f"⏱️  Processing time: {duration:.2f} ms")
            print(f"📊 Total nodes created: {total_nodes}")
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
        
        # Performance comparison for this dataset size
        print(f"\n📊 PERFORMANCE COMPARISON ({size} samples)")
        print("=" * 50)
        
        print(f"{'Strategy':<12} {'Time (ms)':<12} {'Nodes':<8} {'Memory (KB)':<12} {'Rate (nodes/ms)':<15}")
        print("-" * 70)
        
        for strategy, result in strategy_results.items():
            rate = result['total_nodes'] / result['duration'] if result['duration'] > 0 else 0
            memory_kb = result['memory_usage'] / 1024
            print(f"{strategy:<12} {result['duration']:<12.2f} {result['total_nodes']:<8} {memory_kb:<12.2f} {rate:<15.1f}")
    
    # Test memory efficiency with very large dataset
    print(f"\n💾 MEMORY EFFICIENCY STRESS TEST")
    print("=" * 50)
    
    large_samples = generate_large_dataset(5000, 100, 300)
    print(f"📊 Generated {len(large_samples)} samples for stress test")
    
    sizer = OptimizedDynamicNodeSizer()
    
    # Process in batches to test memory management
    batch_size = 500
    total_nodes = 0
    total_memory = 0
    
    for i in range(0, len(large_samples), batch_size):
        batch = large_samples[i:i + batch_size]
        
        start_time = time.time()
        batch_nodes = 0
        
        for sample in batch:
            nodes = sizer.create_dynamic_nodes(sample, 'medium')
            batch_nodes += len(nodes)
        
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        
        total_nodes += batch_nodes
        total_memory = sizer.get_memory_usage()
        
        print(f"📦 Batch {i//batch_size + 1}: {batch_nodes} nodes in {duration:.2f} ms")
        print(f"   💾 Cumulative memory: {total_memory:,} bytes ({total_memory/1024/1024:.2f} MB)")
    
    final_stats = sizer.get_statistics()
    print(f"\n🎯 STRESS TEST RESULTS")
    print(f"📊 Total nodes: {total_nodes}")
    print(f"💾 Final memory: {total_memory:,} bytes ({total_memory/1024/1024:.2f} MB)")
    print(f"📊 Average bytes per node: {total_memory/total_nodes:.1f}")
    
    print(f"📈 Final node distribution:")
    for key, value in final_stats.items():
        if value > 0:
            percentage = (value / total_nodes) * 100
            print(f"   🔹 {key}: {value} ({percentage:.1f}%)")
    
    print(f"\n✅ LARGE-SCALE TEST COMPLETED!")
    print(f"🎉 Optimized system successfully handled large-scale data processing!")

def test_memory_optimization():
    """Test memory optimization features"""
    print(f"\n💾 MEMORY OPTIMIZATION TEST")
    print("=" * 50)
    
    sizer = OptimizedDynamicNodeSizer()
    
    # Test content deduplication
    print("🔄 Testing content deduplication...")
    
    duplicate_texts = [
        "artificial intelligence machine learning",
        "artificial intelligence machine learning",  # Duplicate
        "deep learning neural networks",
        "deep learning neural networks",  # Duplicate
        "computer vision natural language processing",
        "computer vision natural language processing"  # Duplicate
    ]
    
    total_nodes = 0
    for text in duplicate_texts:
        nodes = sizer.create_dynamic_nodes(text, 'tiny')
        total_nodes += len(nodes)
        print(f"   📝 '{text[:30]}...': {len(nodes)} nodes")
    
    memory_usage = sizer.get_memory_usage()
    stats = sizer.get_statistics()
    
    print(f"📊 Total nodes with duplicates: {total_nodes}")
    print(f"💾 Memory usage: {memory_usage:,} bytes")
    print(f"📊 Average bytes per node: {memory_usage/total_nodes:.1f}")
    
    # Test with unique content
    print(f"\n🆕 Testing with unique content...")
    
    sizer_unique = OptimizedDynamicNodeSizer()
    unique_texts = [
        "artificial intelligence machine learning",
        "deep learning neural networks",
        "computer vision natural language processing",
        "reinforcement learning optimization",
        "supervised learning classification",
        "unsupervised learning clustering"
    ]
    
    total_unique_nodes = 0
    for text in unique_texts:
        nodes = sizer_unique.create_dynamic_nodes(text, 'tiny')
        total_unique_nodes += len(nodes)
    
    memory_unique = sizer_unique.get_memory_usage()
    
    print(f"📊 Total nodes without duplicates: {total_unique_nodes}")
    print(f"💾 Memory usage: {memory_unique:,} bytes")
    print(f"📊 Average bytes per node: {memory_unique/total_unique_nodes:.1f}")
    
    # Calculate deduplication savings
    if total_nodes > 0 and total_unique_nodes > 0:
        memory_savings = ((memory_usage - memory_unique) / memory_usage) * 100
        print(f"💾 Memory savings from deduplication: {memory_savings:.1f}%")

if __name__ == "__main__":
    test_large_scale_processing()
    test_memory_optimization()

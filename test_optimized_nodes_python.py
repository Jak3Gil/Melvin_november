#!/usr/bin/env python3
"""
🧠 OPTIMIZED NODE SYSTEM TEST
==============================
Tests the optimized node system concepts with real data.
This simulates the C++ optimizations in Python for testing.
"""

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re
from pathlib import Path

# Simulate the optimized C++ structures
@dataclass
class OptimizedNode:
    id: int
    content: str
    size_category: str
    node_type: str
    complexity_score: float
    parent_id: int = 0
    creation_time: int = 0
    connection_count: int = 0
    
    def __post_init__(self):
        if self.creation_time == 0:
            self.creation_time = int(time.time() * 1000)

@dataclass
class NodeConnection:
    source_id: int
    target_id: int
    weight: float
    connection_type: str

@dataclass
class NodeConfig:
    size_category: str
    node_type: str
    connection_strategy: str
    max_connections: int
    similarity_threshold: float
    min_size: int
    max_size: int

class OptimizedNodeStorage:
    """Simulates the optimized C++ storage system"""
    
    def __init__(self):
        self.nodes: Dict[int, OptimizedNode] = {}
        self.connections: List[NodeConnection] = []
        self.content_to_id: Dict[str, int] = {}  # Content deduplication
        self.stats = {
            'tiny_nodes': 0,
            'small_nodes': 0,
            'medium_nodes': 0,
            'large_nodes': 0,
            'extra_large_nodes': 0,
            'total_connections': 0
        }
        
    def create_node(self, content: str, config: NodeConfig) -> int:
        """Create a node with content deduplication"""
        # Check for existing content (simulates C++ optimization)
        if content in self.content_to_id:
            return self.content_to_id[content]
        
        # Generate hash-based ID
        node_id = int(hashlib.md5(content.encode()).hexdigest()[:16], 16)
        
        # Create node
        node = OptimizedNode(
            id=node_id,
            content=content,
            size_category=config.size_category,
            node_type=config.node_type,
            complexity_score=0.0
        )
        
        # Store node and update indexes
        self.nodes[node_id] = node
        self.content_to_id[content] = node_id
        
        # Update statistics
        self.stats[f"{config.size_category}_nodes"] += 1
        
        return node_id
    
    def create_connection(self, source_id: int, target_id: int, 
                         weight: float, connection_type: str) -> int:
        """Create a connection"""
        conn = NodeConnection(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            connection_type=connection_type
        )
        
        self.connections.append(conn)
        self.stats['total_connections'] += 1
        
        # Update node connection counts
        if source_id in self.nodes:
            self.nodes[source_id].connection_count += 1
        if target_id in self.nodes:
            self.nodes[target_id].connection_count += 1
            
        return len(self.connections) - 1
    
    def get_memory_usage(self) -> int:
        """Calculate approximate memory usage"""
        total = 0
        for node in self.nodes.values():
            total += 60  # Simulated C++ node size
            total += len(node.content)
        total += len(self.connections) * 16  # Simulated connection size
        return total

class OptimizedDynamicNodeSizer:
    """Simulates the optimized C++ dynamic node sizer"""
    
    def __init__(self):
        self.storage = OptimizedNodeStorage()
        self.complexity_cache = {}
        
        # Size configurations (simulating C++ static data)
        self.size_configs = {
            'tiny': NodeConfig('tiny', 'word', 'similarity', 5, 0.8, 1, 10),
            'small': NodeConfig('small', 'phrase', 'similarity', 10, 0.6, 11, 50),
            'medium': NodeConfig('medium', 'concept', 'hierarchical', 20, 0.4, 51, 200),
            'large': NodeConfig('large', 'section', 'temporal', 50, 0.3, 201, 1000),
            'extra_large': NodeConfig('extra_large', 'document', 'all', 100, 0.2, 1001, 10000)
        }
    
    def create_dynamic_nodes(self, text: str, preferred_size: str = 'medium', 
                           complexity_threshold: float = 0.5) -> List[int]:
        """Create nodes with dynamic sizing"""
        if preferred_size == 'medium':
            return self.create_auto_sized_nodes(text, complexity_threshold)
        
        size_methods = {
            'tiny': self.create_tiny_nodes,
            'small': self.create_small_nodes,
            'medium': self.create_medium_nodes,
            'large': self.create_large_nodes,
            'extra_large': self.create_extra_large_nodes
        }
        
        return size_methods.get(preferred_size, self.create_medium_nodes)(text)
    
    def create_tiny_nodes(self, text: str) -> List[int]:
        """Create tiny word-level nodes"""
        node_ids = []
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        for word in words:
            if len(word) >= 3:
                config = self.size_configs['tiny']
                node_id = self.storage.create_node(word, config)
                if node_id not in node_ids:
                    node_ids.append(node_id)
        
        return node_ids
    
    def create_small_nodes(self, text: str) -> List[int]:
        """Create small phrase-level nodes"""
        node_ids = []
        words = text.split()
        
        # Create 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            if 11 <= len(phrase) <= 50:
                config = self.size_configs['small']
                node_id = self.storage.create_node(phrase, config)
                if node_id not in node_ids:
                    node_ids.append(node_id)
        
        # Create 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            if 11 <= len(phrase) <= 50:
                config = self.size_configs['small']
                node_id = self.storage.create_node(phrase, config)
                if node_id not in node_ids:
                    node_ids.append(node_id)
        
        return node_ids
    
    def create_medium_nodes(self, text: str) -> List[int]:
        """Create medium concept-level nodes"""
        node_ids = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if 51 <= len(sentence) <= 200:
                config = self.size_configs['medium']
                node_id = self.storage.create_node(sentence, config)
                if node_id not in node_ids:
                    node_ids.append(node_id)
        
        return node_ids
    
    def create_large_nodes(self, text: str) -> List[int]:
        """Create large section-level nodes"""
        node_ids = []
        chunks = self.split_into_chunks(text, 500)
        
        for chunk in chunks:
            if 201 <= len(chunk) <= 1000:
                config = self.size_configs['large']
                node_id = self.storage.create_node(chunk, config)
                if node_id not in node_ids:
                    node_ids.append(node_id)
        
        return node_ids
    
    def create_extra_large_nodes(self, text: str) -> List[int]:
        """Create extra-large document-level nodes"""
        node_ids = []
        chunks = self.split_into_chunks(text, 2000)
        
        for chunk in chunks:
            if 1001 <= len(chunk) <= 10000:
                config = self.size_configs['extra_large']
                node_id = self.storage.create_node(chunk, config)
                if node_id not in node_ids:
                    node_ids.append(node_id)
        
        return node_ids
    
    def create_auto_sized_nodes(self, text: str, complexity_threshold: float) -> List[int]:
        """Create auto-sized nodes based on content complexity"""
        complexity = self.calculate_complexity(text)
        length = len(text)
        
        # Determine optimal size
        if length <= 10:
            return self.create_tiny_nodes(text)
        elif length <= 50:
            return self.create_small_nodes(text)
        elif length <= 200:
            return self.create_medium_nodes(text)
        elif length <= 1000:
            return self.create_large_nodes(text)
        else:
            return self.create_extra_large_nodes(text)
    
    def calculate_complexity(self, text: str) -> float:
        """Calculate text complexity"""
        if text in self.complexity_cache:
            return self.complexity_cache[text]
        
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Vocabulary diversity
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words)
        
        # Combine metrics
        complexity = (avg_word_length * 0.4) + (vocabulary_diversity * 0.6)
        complexity = min(1.0, complexity)
        
        self.complexity_cache[text] = complexity
        return complexity
    
    def split_into_chunks(self, text: str, target_size: int) -> List[str]:
        """Split text into chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip() + "."
            if len(current_chunk) + len(sentence) <= target_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_statistics(self) -> Dict[str, int]:
        """Get system statistics"""
        return self.storage.stats.copy()
    
    def get_memory_usage(self) -> int:
        """Get memory usage"""
        return self.storage.get_memory_usage()

def test_with_real_data():
    """Test the optimized system with real data"""
    print("🧠 TESTING OPTIMIZED NODE SYSTEM WITH REAL DATA")
    print("=" * 60)
    
    # Test data from various sources
    test_data = {
        "AI Concepts": [
            "artificial intelligence machine learning deep learning neural networks",
            "computer vision natural language processing reinforcement learning",
            "supervised learning unsupervised learning transfer learning",
            "convolutional neural networks recurrent neural networks transformers",
            "backpropagation gradient descent optimization algorithms"
        ],
        "Technical Documentation": [
            "The neural network architecture consists of multiple layers including input, hidden, and output layers. Each layer contains neurons that process information and pass it to subsequent layers.",
            "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning paradigms. Each approach has distinct characteristics and applications.",
            "Deep learning models require significant computational resources and large datasets for training. The training process involves optimizing millions of parameters through iterative updates."
        ],
        "Scientific Papers": [
            "Recent advances in artificial intelligence have demonstrated remarkable capabilities in various domains including computer vision, natural language processing, and robotics. These developments have been driven by improvements in neural network architectures, training algorithms, and computational resources.",
            "The transformer architecture has revolutionized natural language processing by introducing attention mechanisms that allow models to focus on relevant parts of input sequences. This has led to significant improvements in machine translation, text generation, and other language tasks.",
            "Reinforcement learning has shown promise in complex decision-making scenarios such as game playing, autonomous driving, and robotic control. The key challenge lies in balancing exploration and exploitation while learning optimal policies."
        ],
        "Code Snippets": [
            "def train_model(model, data, epochs): for epoch in range(epochs): loss = model.train_step(data) print(f'Epoch {epoch}: Loss {loss}')",
            "class NeuralNetwork: def __init__(self, layers): self.layers = layers def forward(self, x): for layer in self.layers: x = layer.forward(x) return x",
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.001) criterion = nn.CrossEntropyLoss()"
        ]
    }
    
    sizer = OptimizedDynamicNodeSizer()
    
    # Test different size categories
    size_tests = [
        ('tiny', 'Tiny (Word-level)'),
        ('small', 'Small (Phrase-level)'),
        ('medium', 'Medium (Concept-level)'),
        ('large', 'Large (Section-level)'),
        ('extra_large', 'Extra Large (Document-level)')
    ]
    
    for size_category, label in size_tests:
        print(f"\n📏 TESTING {label.upper()}")
        print("-" * 40)
        
        total_nodes = 0
        total_memory = 0
        start_time = time.time()
        
        for category, texts in test_data.items():
            for text in texts:
                nodes = sizer.create_dynamic_nodes(text, size_category)
                total_nodes += len(nodes)
        
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        
        memory_usage = sizer.get_memory_usage()
        stats = sizer.get_statistics()
        
        print(f"⏱️  Processing time: {duration:.2f} ms")
        print(f"📊 Total nodes created: {total_nodes}")
        print(f"💾 Memory usage: {memory_usage:,} bytes ({memory_usage/1024:.2f} KB)")
        print(f"📈 Node distribution:")
        for key, value in stats.items():
            if value > 0:
                print(f"   🔹 {key}: {value}")
        
        # Calculate efficiency metrics
        if total_nodes > 0:
            avg_bytes_per_node = memory_usage / total_nodes
            nodes_per_ms = total_nodes / duration if duration > 0 else 0
            print(f"📊 Average bytes per node: {avg_bytes_per_node:.1f}")
            print(f"🚀 Processing rate: {nodes_per_ms:.1f} nodes/ms")
    
    # Test content deduplication
    print(f"\n🔄 CONTENT DEDUPLICATION TEST")
    print("-" * 40)
    
    duplicate_text = "artificial intelligence machine learning"
    nodes1 = sizer.create_dynamic_nodes(duplicate_text, 'tiny')
    nodes2 = sizer.create_dynamic_nodes(duplicate_text, 'tiny')
    
    print(f"Original nodes: {len(nodes1)}")
    print(f"Duplicate nodes: {len(nodes2)}")
    print(f"Deduplication working: {len(nodes2) == 0}")
    
    # Test complexity analysis
    print(f"\n🧠 COMPLEXITY ANALYSIS TEST")
    print("-" * 40)
    
    complexity_texts = [
        ("Simple", "AI ML"),
        ("Medium", "artificial intelligence machine learning"),
        ("Complex", "Artificial intelligence encompasses machine learning, deep learning, neural networks, and various computational approaches to mimic human cognitive functions.")
    ]
    
    for complexity_label, text in complexity_texts:
        complexity = sizer.calculate_complexity(text)
        nodes = sizer.create_dynamic_nodes(text, 'medium')
        print(f"🔹 {complexity_label}: Complexity {complexity:.3f}, Nodes {len(nodes)}")
    
    # Final statistics
    print(f"\n📊 FINAL SYSTEM STATISTICS")
    print("-" * 40)
    
    final_stats = sizer.get_statistics()
    final_memory = sizer.get_memory_usage()
    
    total_nodes = sum(final_stats.values())
    print(f"📊 Total nodes in system: {total_nodes}")
    print(f"💾 Total memory usage: {final_memory:,} bytes ({final_memory/1024/1024:.2f} MB)")
    print(f"📈 Node distribution:")
    for key, value in final_stats.items():
        if value > 0:
            percentage = (value / total_nodes) * 100 if total_nodes > 0 else 0
            print(f"   🔹 {key}: {value} ({percentage:.1f}%)")
    
    print(f"\n✅ OPTIMIZED NODE SYSTEM TEST COMPLETED!")
    print(f"🎉 System ready for high-performance data processing!")

if __name__ == "__main__":
    test_with_real_data()

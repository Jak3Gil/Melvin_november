#!/usr/bin/env python3
"""
🍽️ MELVIN DATA FEEDER
Comprehensive system to feed Melvin diverse data and watch him create dynamic nodes
"""

import os
import json
import time
import sqlite3
import requests
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

# Try to use the C++ backend if available
try:
    from melvin_cpp_brain import MelvinCppBrain
    BACKEND_AVAILABLE = True
except ImportError:
    # Fallback to Python implementation
    from melvin_global_brain import MelvinGlobalBrain
    BACKEND_AVAILABLE = False

class MelvinDataFeeder:
    """Feed Melvin diverse data and monitor dynamic node creation"""
    
    def __init__(self, memory_path: str = "melvin_global_memory"):
        self.memory_path = memory_path
        
        # Initialize brain with best available backend
        if BACKEND_AVAILABLE:
            self.brain = MelvinCppBrain()
            print("🚀 Using C++ high-performance brain")
        else:
            self.brain = MelvinGlobalBrain(memory_path=memory_path)
            print("🐍 Using Python brain implementation")
        
        self.feeding_stats = {
            'total_data_fed': 0,
            'nodes_created': 0,
            'connections_created': 0,
            'atomic_facts_created': 0,
            'consolidations_performed': 0,
            'fragmentations_performed': 0
        }
        
        self.data_sources = []
        self.setup_data_sources()
    
    def setup_data_sources(self):
        """Set up diverse data sources for feeding Melvin"""
        
        # Wikipedia articles (factual knowledge)
        self.data_sources.extend([
            {
                'name': 'Wikipedia - Science',
                'type': 'factual',
                'url': 'https://en.wikipedia.org/api/rest_v1/page/random',
                'processor': self.process_wikipedia_article
            },
            {
                'name': 'Wikipedia - Technology', 
                'type': 'factual',
                'url': 'https://en.wikipedia.org/api/rest_v1/page/random',
                'processor': self.process_wikipedia_article
            }
        ])
        
        # Programming knowledge
        self.data_sources.extend([
            {
                'name': 'Python Code Examples',
                'type': 'code',
                'generator': self.generate_python_examples,
                'processor': self.process_code_example
            },
            {
                'name': 'JavaScript Code Examples',
                'type': 'code', 
                'generator': self.generate_javascript_examples,
                'processor': self.process_code_example
            }
        ])
        
        # Conversational data
        self.data_sources.extend([
            {
                'name': 'Common Questions',
                'type': 'conversational',
                'generator': self.generate_qa_pairs,
                'processor': self.process_qa_pair
            },
            {
                'name': 'Facts and Definitions',
                'type': 'educational',
                'generator': self.generate_educational_content,
                'processor': self.process_educational_content
            }
        ])
        
        # Creative content
        self.data_sources.extend([
            {
                'name': 'Story Fragments',
                'type': 'creative',
                'generator': self.generate_story_fragments,
                'processor': self.process_creative_content
            }
        ])
    
    def process_wikipedia_article(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process Wikipedia article into nodes"""
        nodes = []
        
        if 'extract' in data:
            content = data['extract']
            title = data.get('title', 'Unknown')
            
            # Create main article node
            nodes.append({
                'type': 'concept',
                'content': f"{title}: {content[:200]}...",
                'metadata': {'source': 'wikipedia', 'title': title}
            })
            
            # Break into atomic facts
            sentences = content.split('. ')
            for sentence in sentences:
                if len(sentence.strip()) > 20 and len(sentence.strip()) < 150:
                    nodes.append({
                        'type': 'atomic_fact',
                        'content': sentence.strip(),
                        'metadata': {'source': 'wikipedia', 'article': title}
                    })
        
        return nodes
    
    def generate_python_examples(self) -> List[str]:
        """Generate Python code examples"""
        examples = [
            """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
            
            """class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return [item.upper() for item in self.data]""",
            
            """import numpy as np
import matplotlib.pyplot as plt

def plot_data(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Data Visualization')
    plt.grid(True)
    plt.show()""",
            
            """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
            
            """async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()""",
            
            """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)"""
        ]
        
        return examples
    
    def generate_javascript_examples(self) -> List[str]:
        """Generate JavaScript code examples"""
        examples = [
            """function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}""",
            
            """class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
    }
    
    emit(event, ...args) {
        if (this.events[event]) {
            this.events[event].forEach(listener => listener(...args));
        }
    }
}""",
            
            """const fetchWithRetry = async (url, options = {}, retries = 3) => {
    try {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error('Network response was not ok');
        return response;
    } catch (error) {
        if (retries > 0) {
            console.log(`Retrying... ${retries} attempts left`);
            return fetchWithRetry(url, options, retries - 1);
        }
        throw error;
    }
};""",
            
            """function* fibonacci() {
    let a = 0, b = 1;
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}""",
            
            """const memoize = (fn) => {
    const cache = new Map();
    return (...args) => {
        const key = JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
};"""
        ]
        
        return examples
    
    def process_code_example(self, code: str) -> List[Dict[str, Any]]:
        """Process code example into nodes"""
        nodes = []
        
        # Main code node
        nodes.append({
            'type': 'code',
            'content': code,
            'metadata': {'source': 'generated', 'language': 'python' if 'def ' in code else 'javascript'}
        })
        
        # Extract concepts from code
        if 'def ' in code or 'function ' in code:
            nodes.append({
                'type': 'concept',
                'content': 'Function definition and implementation',
                'metadata': {'concept_type': 'programming', 'relates_to': 'functions'}
            })
        
        if 'class ' in code:
            nodes.append({
                'type': 'concept',
                'content': 'Object-oriented programming with classes',
                'metadata': {'concept_type': 'programming', 'relates_to': 'oop'}
            })
        
        if 'async' in code or 'await' in code:
            nodes.append({
                'type': 'concept',
                'content': 'Asynchronous programming patterns',
                'metadata': {'concept_type': 'programming', 'relates_to': 'async'}
            })
        
        return nodes
    
    def generate_qa_pairs(self) -> List[Dict[str, str]]:
        """Generate question-answer pairs"""
        qa_pairs = [
            {
                'question': 'What is artificial intelligence?',
                'answer': 'Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.'
            },
            {
                'question': 'How does machine learning work?',
                'answer': 'Machine learning uses algorithms to analyze data, identify patterns, and make predictions or decisions without being explicitly programmed for each task.'
            },
            {
                'question': 'What is the difference between AI and ML?',
                'answer': 'AI is the broader concept of machines being able to carry out tasks in a smart way, while ML is a specific application of AI that focuses on learning from data.'
            },
            {
                'question': 'What is deep learning?',
                'answer': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.'
            },
            {
                'question': 'How do neural networks work?',
                'answer': 'Neural networks consist of interconnected nodes that process information by adjusting weights and biases through training to recognize patterns.'
            },
            {
                'question': 'What is natural language processing?',
                'answer': 'Natural language processing is a branch of AI that helps computers understand, interpret, and generate human language in a meaningful way.'
            },
            {
                'question': 'What is computer vision?',
                'answer': 'Computer vision is a field of AI that enables computers to interpret and understand visual information from the world, such as images and videos.'
            },
            {
                'question': 'How does reinforcement learning work?',
                'answer': 'Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones, learning optimal strategies through trial and error.'
            }
        ]
        
        return qa_pairs
    
    def process_qa_pair(self, qa: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process Q&A pair into nodes"""
        nodes = []
        
        # Question node
        nodes.append({
            'type': 'language',
            'content': qa['question'],
            'metadata': {'content_type': 'question', 'source': 'generated'}
        })
        
        # Answer node
        nodes.append({
            'type': 'language',
            'content': qa['answer'],
            'metadata': {'content_type': 'answer', 'source': 'generated'}
        })
        
        # Create atomic facts from answer
        sentences = qa['answer'].split('. ')
        for sentence in sentences:
            if len(sentence.strip()) > 15:
                nodes.append({
                    'type': 'atomic_fact',
                    'content': sentence.strip(),
                    'metadata': {'source': 'qa_extraction', 'relates_to': qa['question']}
                })
        
        return nodes
    
    def generate_educational_content(self) -> List[Dict[str, str]]:
        """Generate educational facts and definitions"""
        content = [
            {
                'term': 'Algorithm',
                'definition': 'A step-by-step procedure for solving a problem or completing a task, often used in computing and mathematics.'
            },
            {
                'term': 'Data Structure',
                'definition': 'A way of organizing and storing data in a computer so that it can be accessed and modified efficiently.'
            },
            {
                'term': 'Big O Notation',
                'definition': 'A mathematical notation that describes the limiting behavior of a function when the argument tends towards infinity, used to classify algorithms.'
            },
            {
                'term': 'Recursion',
                'definition': 'A programming technique where a function calls itself to solve smaller instances of the same problem.'
            },
            {
                'term': 'API',
                'definition': 'Application Programming Interface - a set of protocols and tools for building software applications that specifies how components should interact.'
            },
            {
                'term': 'Database',
                'definition': 'An organized collection of structured information, or data, typically stored electronically in a computer system.'
            },
            {
                'term': 'Cloud Computing',
                'definition': 'The delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the Internet.'
            }
        ]
        
        return content
    
    def process_educational_content(self, content: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process educational content into nodes"""
        nodes = []
        
        # Term node
        nodes.append({
            'type': 'concept',
            'content': content['term'],
            'metadata': {'content_type': 'term', 'source': 'educational'}
        })
        
        # Definition node
        nodes.append({
            'type': 'atomic_fact',
            'content': f"{content['term']}: {content['definition']}",
            'metadata': {'content_type': 'definition', 'source': 'educational'}
        })
        
        return nodes
    
    def generate_story_fragments(self) -> List[str]:
        """Generate creative story fragments"""
        fragments = [
            "The old lighthouse keeper noticed something unusual in the fog that night.",
            "Sarah's discovery in the attic would change everything she thought she knew about her family.",
            "The AI system began exhibiting behaviors that weren't in its programming.",
            "In the year 2045, humans and machines worked together in ways previously unimaginable.",
            "The scientist's experiment with quantum entanglement produced unexpected results.",
            "The ancient book contained knowledge that modern science was just beginning to understand.",
            "The robot's first question surprised everyone: 'What is the purpose of existence?'",
            "The neural network started dreaming, and its dreams were unlike anything humans had ever experienced."
        ]
        
        return fragments
    
    def process_creative_content(self, content: str) -> List[Dict[str, Any]]:
        """Process creative content into nodes"""
        nodes = []
        
        # Main creative node
        nodes.append({
            'type': 'language',
            'content': content,
            'metadata': {'content_type': 'creative', 'source': 'generated'}
        })
        
        # Extract concepts
        if any(word in content.lower() for word in ['ai', 'robot', 'machine', 'neural', 'quantum']):
            nodes.append({
                'type': 'concept',
                'content': 'Science fiction and technology themes',
                'metadata': {'concept_type': 'thematic', 'relates_to': 'sci-fi'}
            })
        
        return nodes
    
    def feed_data_batch(self, batch_size: int = 20) -> Dict[str, int]:
        """Feed a batch of data to Melvin"""
        print(f"🍽️ Feeding Melvin a batch of {batch_size} data items...")
        
        batch_stats = {
            'items_processed': 0,
            'nodes_created': 0,
            'connections_created': 0
        }
        
        for i in range(batch_size):
            # Select random data source
            source = random.choice(self.data_sources)
            
            try:
                # Generate or fetch data
                if 'generator' in source:
                    raw_data = source['generator']()
                    if isinstance(raw_data, list):
                        data_item = random.choice(raw_data)
                    else:
                        data_item = raw_data
                else:
                    # For URL-based sources (not implemented in this demo)
                    continue
                
                # Process data into nodes
                nodes = source['processor'](data_item)
                
                # Feed nodes to Melvin
                node_ids = []
                for node_data in nodes:
                    if hasattr(self.brain, 'create_node'):
                        # C++ backend
                        node_id = self.brain.create_node(node_data['type'], node_data['content'])
                        node_ids.append(node_id)
                    else:
                        # Python backend
                        node_id = self.brain.add_node(
                            node_data['content'], 
                            node_data['type'], 
                            metadata=node_data.get('metadata', {})
                        )
                        node_ids.append(node_id)
                    
                    batch_stats['nodes_created'] += 1
                
                # Create connections between related nodes
                if len(node_ids) > 1:
                    for j in range(len(node_ids) - 1):
                        if hasattr(self.brain, 'create_connection'):
                            # C++ backend
                            self.brain.create_connection(node_ids[j], node_ids[j+1], 'semantic', 0.8)
                        else:
                            # Python backend
                            self.brain.add_connection(node_ids[j], node_ids[j+1], 0.8, 'semantic')
                        
                        batch_stats['connections_created'] += 1
                
                # Apply Hebbian learning if multiple nodes were created
                if len(node_ids) > 1 and hasattr(self.brain, 'hebbian_learning'):
                    self.brain.hebbian_learning(node_ids)
                
                batch_stats['items_processed'] += 1
                
                # Show progress
                if (i + 1) % 5 == 0:
                    print(f"   Processed {i + 1}/{batch_size} items...")
                
            except Exception as e:
                print(f"   ⚠️ Error processing item {i}: {e}")
                continue
        
        # Update global stats
        self.feeding_stats['total_data_fed'] += batch_stats['items_processed']
        self.feeding_stats['nodes_created'] += batch_stats['nodes_created']
        self.feeding_stats['connections_created'] += batch_stats['connections_created']
        
        return batch_stats
    
    def optimize_brain_structure(self):
        """Run dynamic optimization on the brain"""
        print("🧠 Running brain optimization...")
        
        if hasattr(self.brain, 'fragment_large_nodes'):
            # C++ backend optimization
            fragments = self.brain.fragment_large_nodes()
            consolidations = self.brain.consolidate_successful_chains()
            
            self.feeding_stats['fragmentations_performed'] += fragments
            self.feeding_stats['consolidations_performed'] += consolidations
            
            print(f"   🔧 Fragmented {fragments} large nodes")
            print(f"   🔗 Consolidated {consolidations} successful chains")
        else:
            print("   ⚠️ Dynamic optimization not available with current backend")
    
    def monitor_brain_growth(self) -> Dict[str, Any]:
        """Monitor how the brain is growing"""
        if hasattr(self.brain, 'get_performance_stats'):
            # C++ backend stats
            stats = self.brain.get_performance_stats()
            
            growth_stats = {
                'total_nodes': stats['total_nodes'],
                'total_connections': stats['total_connections'],
                'memory_usage_mb': stats['memory_usage_bytes'] / (1024 * 1024),
                'backend': stats['backend']
            }
        else:
            # Python backend stats
            growth_stats = {
                'total_nodes': len(self.brain.nodes) if hasattr(self.brain, 'nodes') else 0,
                'total_connections': len(self.brain.connections) if hasattr(self.brain, 'connections') else 0,
                'memory_usage_mb': 0,
                'backend': 'Python'
            }
        
        # Add feeding stats
        growth_stats.update(self.feeding_stats)
        
        return growth_stats
    
    def continuous_feeding_session(self, duration_minutes: int = 10, batch_size: int = 20):
        """Run a continuous feeding session"""
        print(f"🚀 STARTING CONTINUOUS FEEDING SESSION")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Batch size: {batch_size} items per batch")
        print("=" * 60)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        batch_count = 0
        
        # Initial brain state
        initial_stats = self.monitor_brain_growth()
        print(f"📊 Initial brain state:")
        print(f"   Nodes: {initial_stats['total_nodes']:,}")
        print(f"   Connections: {initial_stats['total_connections']:,}")
        print(f"   Backend: {initial_stats['backend']}")
        print()
        
        while time.time() < end_time:
            batch_count += 1
            print(f"🍽️ Batch {batch_count} - {time.strftime('%H:%M:%S')}")
            
            # Feed data batch
            batch_stats = self.feed_data_batch(batch_size)
            
            # Run optimization every 3rd batch
            if batch_count % 3 == 0:
                self.optimize_brain_structure()
            
            # Show progress
            current_stats = self.monitor_brain_growth()
            print(f"   📊 Brain grew by {batch_stats['nodes_created']} nodes, {batch_stats['connections_created']} connections")
            print(f"   🧠 Total: {current_stats['total_nodes']:,} nodes, {current_stats['total_connections']:,} connections")
            print()
            
            # Brief pause between batches
            time.sleep(2)
        
        # Final statistics
        final_stats = self.monitor_brain_growth()
        
        print(f"🎯 FEEDING SESSION COMPLETE!")
        print("=" * 40)
        print(f"Duration: {duration_minutes} minutes")
        print(f"Batches processed: {batch_count}")
        print(f"Items fed: {self.feeding_stats['total_data_fed']}")
        print()
        print(f"📊 Growth Statistics:")
        print(f"   Nodes: {initial_stats['total_nodes']:,} → {final_stats['total_nodes']:,} (+{final_stats['total_nodes'] - initial_stats['total_nodes']:,})")
        print(f"   Connections: {initial_stats['total_connections']:,} → {final_stats['total_connections']:,} (+{final_stats['total_connections'] - initial_stats['total_connections']:,})")
        
        if self.feeding_stats['fragmentations_performed'] > 0 or self.feeding_stats['consolidations_performed'] > 0:
            print(f"   Optimizations: {self.feeding_stats['fragmentations_performed']} fragmentations, {self.feeding_stats['consolidations_performed']} consolidations")
        
        # Save brain state
        if hasattr(self.brain, 'save_to_disk'):
            self.brain.save_to_disk()
            print(f"💾 Brain state saved to disk")
        
        return final_stats
    
    def close(self):
        """Clean up resources"""
        if hasattr(self.brain, 'close'):
            self.brain.close()

def main():
    """Run the data feeding system"""
    print("🍽️ MELVIN DATA FEEDING SYSTEM")
    print("=" * 50)
    
    feeder = MelvinDataFeeder()
    
    try:
        # Run a feeding session
        feeder.continuous_feeding_session(duration_minutes=5, batch_size=15)
        
    finally:
        feeder.close()

if __name__ == "__main__":
    main()

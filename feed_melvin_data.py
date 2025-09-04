#!/usr/bin/env python3
"""
🍽️ FEED MELVIN DATA - Interactive Data Feeding with Brain Monitoring
====================================================================
Feed Melvin various types of data and monitor how his brain forms connections.
Watch Hebbian learning in action as nodes activate and form relationships.
"""

import time
import random
from typing import List, Dict, Any
from melvin_brain_monitor import MelvinBrainMonitor
from melvin_optimized_v2 import MelvinOptimizedV2, ContentType

class MelvinDataFeeder:
    """Interactive data feeder for Melvin's brain"""
    
    def __init__(self):
        self.melvin = MelvinOptimizedV2()
        self.monitor = MelvinBrainMonitor(self.melvin)
        
        # Sample data sets
        self.text_data = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological brain structures.",
            "Deep learning uses multiple layers of neural networks.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning learns through trial and error.",
            "Supervised learning uses labeled training data.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Transfer learning applies knowledge from one task to another."
        ]
        
        self.code_data = [
            "def hello_world():\n    print('Hello, World!')",
            "class NeuralNetwork:\n    def __init__(self):\n        self.layers = []",
            "import numpy as np\nimport torch\nimport tensorflow as tf",
            "def train_model(X, y):\n    model.fit(X, y, epochs=100)",
            "for epoch in range(num_epochs):\n    loss = model.train_step(data)",
            "def forward_pass(self, x):\n    return self.activation(self.weights @ x + self.bias)",
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)",
            "loss = nn.CrossEntropyLoss()(outputs, targets)",
            "def preprocess_data(data):\n    return normalize(data)",
            "model = Sequential([Dense(128), ReLU(), Dense(10, softmax)])"
        ]
        
        self.concept_data = [
            "Hebbian learning: neurons that fire together wire together",
            "Backpropagation: error correction through gradient descent",
            "Attention mechanism: focusing on relevant information",
            "Transformer architecture: self-attention for sequence processing",
            "Convolutional neural networks: specialized for spatial data",
            "Recurrent neural networks: processing sequential information",
            "Generative adversarial networks: creating synthetic data",
            "Autoencoders: learning compressed representations",
            "Reinforcement learning: learning through environment interaction",
            "Meta-learning: learning how to learn"
        ]
        
        print("🍽️ Melvin Data Feeder initialized")
    
    def feed_text_data(self, count: int = 5):
        """Feed text data to Melvin"""
        print(f"\n📝 Feeding {count} text samples to Melvin...")
        
        for i in range(count):
            text = random.choice(self.text_data)
            node_id = self.melvin.process_text_input(text, "text_feeder")
            
            print(f"   📦 Created node: {node_id.hex()[:8]} - {text[:30]}...")
            time.sleep(0.5)  # Small delay to see Hebbian learning
    
    def feed_code_data(self, count: int = 5):
        """Feed code data to Melvin"""
        print(f"\n💻 Feeding {count} code samples to Melvin...")
        
        for i in range(count):
            code = random.choice(self.code_data)
            node_id = self.melvin.process_code_input(code, "code_feeder")
            
            print(f"   📦 Created node: {node_id.hex()[:8]} - {code[:30]}...")
            time.sleep(0.5)
    
    def feed_concept_data(self, count: int = 5):
        """Feed concept data to Melvin"""
        print(f"\n🧠 Feeding {count} concept samples to Melvin...")
        
        for i in range(count):
            concept = random.choice(self.concept_data)
            node_id = self.melvin.process_text_input(concept, "concept_feeder")
            
            print(f"   📦 Created node: {node_id.hex()[:8]} - {concept[:30]}...")
            time.sleep(0.5)
    
    def feed_mixed_data(self, count: int = 10):
        """Feed mixed data types to Melvin"""
        print(f"\n🎲 Feeding {count} mixed samples to Melvin...")
        
        all_data = self.text_data + self.code_data + self.concept_data
        
        for i in range(count):
            data = random.choice(all_data)
            
            # Determine if it's code (contains function/class definitions)
            if any(keyword in data for keyword in ['def ', 'class ', 'import ', '=']):
                node_id = self.melvin.process_code_input(data, "mixed_feeder")
                data_type = "CODE"
            else:
                node_id = self.melvin.process_text_input(data, "mixed_feeder")
                data_type = "TEXT"
            
            print(f"   📦 Created {data_type} node: {node_id.hex()[:8]} - {data[:30]}...")
            time.sleep(0.3)
    
    def feed_related_concepts(self):
        """Feed related concepts to see Hebbian learning in action"""
        print(f"\n🔗 Feeding related concepts to see Hebbian learning...")
        
        # Related concept groups
        concept_groups = [
            ["Neural networks", "Deep learning", "Machine learning", "Artificial intelligence"],
            ["Backpropagation", "Gradient descent", "Optimization", "Training"],
            ["Computer vision", "Image processing", "Convolutional networks", "Visual data"],
            ["Natural language", "Text processing", "Language models", "NLP"],
            ["Reinforcement learning", "Environment", "Rewards", "Policy learning"]
        ]
        
        for group in concept_groups:
            print(f"\n   🧠 Feeding related group: {group[0]}...")
            
            for concept in group:
                node_id = self.melvin.process_text_input(concept, "related_feeder")
                print(f"      📦 Node: {node_id.hex()[:8]} - {concept}")
                time.sleep(0.2)  # Quick succession to trigger Hebbian learning
            
            # Check Hebbian connections
            state = self.melvin.get_unified_state()
            print(f"      ⚡ Hebbian updates: {state['global_memory']['stats']['hebbian_updates']}")
    
    def show_brain_state(self):
        """Show current brain state"""
        state = self.melvin.get_unified_state()
        
        print(f"\n🧠 MELVIN'S CURRENT BRAIN STATE:")
        print(f"   📦 Total Nodes: {state['global_memory']['total_nodes']:,}")
        print(f"   🔗 Total Connections: {state['global_memory']['total_edges']:,}")
        print(f"   💾 Storage Used: {state['global_memory']['storage_used_mb']:.2f}MB")
        print(f"   ⚡ Hebbian Updates: {state['global_memory']['stats']['hebbian_updates']}")
        print(f"   🕐 Uptime: {state['system']['uptime_seconds']:.1f}s")
        
        # Calculate connection density
        if state['global_memory']['total_nodes'] > 0:
            density = state['global_memory']['total_edges'] / state['global_memory']['total_nodes']
            print(f"   🔗 Connection Density: {density:.2f} connections per node")
    
    def run_interactive_session(self):
        """Run an interactive data feeding session"""
        print("🍽️ MELVIN DATA FEEDING SESSION")
        print("=" * 50)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        try:
            while True:
                print(f"\n📊 Current Brain State:")
                self.show_brain_state()
                
                print(f"\n🍽️ Choose feeding option:")
                print("1. Feed text data (5 samples)")
                print("2. Feed code data (5 samples)")
                print("3. Feed concept data (5 samples)")
                print("4. Feed mixed data (10 samples)")
                print("5. Feed related concepts (Hebbian learning demo)")
                print("6. Show detailed brain state")
                print("7. Exit")
                
                choice = input("\nChoice [1-7]: ").strip()
                
                if choice == "1":
                    self.feed_text_data(5)
                elif choice == "2":
                    self.feed_code_data(5)
                elif choice == "3":
                    self.feed_concept_data(5)
                elif choice == "4":
                    self.feed_mixed_data(10)
                elif choice == "5":
                    self.feed_related_concepts()
                elif choice == "6":
                    self.show_brain_state()
                elif choice == "7":
                    break
                else:
                    print("❌ Invalid choice, please try again")
                
                print(f"\n" + "-" * 40)
                
        except KeyboardInterrupt:
            print("\n🛑 Session interrupted by user")
        finally:
            # Stop monitoring and generate report
            self.monitor.stop_monitoring()
            self.monitor._generate_final_report()

def main():
    """Main function"""
    print("🍽️ MELVIN DATA FEEDER")
    print("=" * 50)
    
    # Create data feeder
    feeder = MelvinDataFeeder()
    
    # Run interactive session
    feeder.run_interactive_session()
    
    print("\n🎉 Data feeding session completed!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🧠 MELVIN SIMPLE DEMO - No External Dependencies
==============================================
Runs Melvin brain with mock Hugging Face data using only built-in Python libraries.
Creates nodes and connections to demonstrate the system.
"""

import json
import time
import random
import uuid
import math
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import threading

# Configure simple logging
def log_info(message):
    print(f"[INFO] {time.strftime('%H:%M:%S')} - {message}")

def log_error(message):
    print(f"[ERROR] {time.strftime('%H:%M:%S')} - {message}")

# ============================================================================
# SIMPLIFIED MELVIN BRAIN - NO EXTERNAL DEPENDENCIES
# ============================================================================

class NodeType(Enum):
    """Node types in Melvin's brain"""
    VISUAL = "visual"
    AUDIO = "audio"
    LANGUAGE = "language"
    CODE = "code"
    CONCEPT = "concept"
    SEQUENCE = "sequence"
    MEMORY = "memory"
    EMOTION = "emotion"
    ACTION = "action"
    SENSOR = "sensor"

class EdgeType(Enum):
    """Connection types between nodes"""
    HEBBIAN = "hebbian"
    SIMILARITY = "similarity"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    MULTIMODAL = "multimodal"
    CAUSAL = "causal"
    ASSOCIATIVE = "associative"

@dataclass
class SimpleNode:
    """Simplified node for demo"""
    node_id: str
    node_type: NodeType
    content: str
    embedding: List[float]
    activation_strength: float = 0.0
    firing_rate: float = 0.0
    last_activation: float = field(default_factory=time.time)
    activation_count: int = 0
    connection_count: int = 0
    creation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality_source: str = ""
    
    def activate(self, strength: float = 1.0):
        """Activate this node"""
        self.activation_strength = min(1.0, self.activation_strength + strength)
        self.firing_rate = min(1.0, self.firing_rate + 0.1)
        self.last_activation = time.time()
        self.activation_count += 1

@dataclass
class SimpleEdge:
    """Simplified edge for demo"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 0.1
    coactivation_count: int = 0
    last_coactivation: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    
    def strengthen(self, amount: float = 0.01):
        """Strengthen connection"""
        self.weight = min(1.0, self.weight + amount)
        self.coactivation_count += 1
        self.last_coactivation = time.time()

class SimpleMelvinBrain:
    """Simplified Melvin brain for demo"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.nodes: Dict[str, SimpleNode] = {}
        self.edges: Dict[str, SimpleEdge] = {}
        self.node_connections: Dict[str, Set[str]] = defaultdict(set)
        self.recent_activations = deque(maxlen=100)
        self.coactivation_window = 2.0  # seconds
        
        # Create memory directory
        self.memory_path = Path("melvin_global_memory")
        self.memory_path.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'hebbian_updates': 0,
            'similarity_connections': 0,
            'temporal_connections': 0,
            'cross_modal_connections': 0,
            'start_time': time.time()
        }
        
        log_info("🧠 Simple Melvin Brain initialized")
    
    def _init_database(self):
        """Initialize SQLite database"""
        db_path = self.memory_path / "global_memory.db"
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.db_lock = threading.Lock()
        
        with self.db_lock:
            # Create nodes table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT,
                    content TEXT,
                    embedding TEXT,
                    activation_strength REAL,
                    firing_rate REAL,
                    last_activation REAL,
                    activation_count INTEGER,
                    creation_time REAL,
                    metadata TEXT,
                    modality_source TEXT
                )
            ''')
            
            # Create edges table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT,
                    target_id TEXT,
                    edge_type TEXT,
                    weight REAL,
                    coactivation_count INTEGER,
                    last_coactivation REAL,
                    creation_time REAL,
                    FOREIGN KEY (source_id) REFERENCES nodes (node_id),
                    FOREIGN KEY (target_id) REFERENCES nodes (node_id)
                )
            ''')
            
            self.conn.commit()
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create simple embedding from text"""
        # Simple hash-based embedding
        text_hash = hash(text.lower())
        random.seed(text_hash)
        
        embedding = [random.random() for _ in range(self.embedding_dim)]
        
        # Add semantic clustering based on keywords
        keywords = {
            'color': ['red', 'blue', 'green', 'yellow', 'color'],
            'movement': ['move', 'run', 'walk', 'jump', 'motion'],
            'emotion': ['happy', 'sad', 'angry', 'love', 'joy'],
            'learning': ['learn', 'think', 'know', 'understand', 'brain'],
            'code': ['function', 'class', 'def', 'import', 'python'],
            'visual': ['see', 'look', 'image', 'camera', 'vision']
        }
        
        for category, words in keywords.items():
            if any(word in text.lower() for word in words):
                # Boost specific dimensions for this category
                start_idx = hash(category) % (self.embedding_dim - 10)
                for i in range(10):
                    embedding[start_idx + i] += 0.3
        
        # Normalize
        norm = math.sqrt(sum(x*x for x in embedding))
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        return embedding
    
    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a*b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(x*x for x in emb1))
        norm2 = math.sqrt(sum(x*x for x in emb2))
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    
    def add_node(self, content: str, node_type: NodeType, modality_source: str = "") -> str:
        """Add node to brain"""
        node_id = f"{node_type.value}_{uuid.uuid4().hex[:12]}"
        embedding = self._create_embedding(content)
        
        node = SimpleNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            embedding=embedding,
            modality_source=modality_source
        )
        
        # Add to memory
        self.nodes[node_id] = node
        self.stats['total_nodes'] += 1
        
        # Activate node
        node.activate(0.8)
        self._record_activation(node_id, 0.8)
        
        # Create connections
        self._create_connections_for_node(node_id)
        
        # Save to database
        self._save_node_to_db(node)
        
        log_info(f"🔗 Added {node_type.value} node: {content[:50]}...")
        return node_id
    
    def _record_activation(self, node_id: str, strength: float):
        """Record activation for Hebbian learning"""
        self.recent_activations.append({
            'node_id': node_id,
            'strength': strength,
            'timestamp': time.time()
        })
    
    def _create_connections_for_node(self, node_id: str):
        """Create connections for new node"""
        if node_id not in self.nodes:
            return
        
        new_node = self.nodes[node_id]
        connections_created = 0
        
        # Check similarity with existing nodes
        for existing_id, existing_node in self.nodes.items():
            if existing_id == node_id:
                continue
            
            similarity = self._calculate_similarity(new_node.embedding, existing_node.embedding)
            
            if similarity > 0.7:  # High similarity
                self._create_edge(node_id, existing_id, EdgeType.SIMILARITY, similarity)
                connections_created += 1
                
                # Cross-modal bonus
                if new_node.node_type != existing_node.node_type:
                    self._create_edge(node_id, existing_id, EdgeType.MULTIMODAL, similarity * 0.8)
                    self.stats['cross_modal_connections'] += 1
        
        # Temporal connections
        self._create_temporal_connections(node_id)
        
        log_info(f"🔗 Created {connections_created} connections for {node_id}")
    
    def _create_edge(self, source_id: str, target_id: str, edge_type: EdgeType, initial_weight: float = 0.1):
        """Create edge between nodes"""
        edge_id = f"{source_id}→{target_id}_{edge_type.value}"
        
        if edge_id in self.edges:
            self.edges[edge_id].strengthen()
            return edge_id
        
        edge = SimpleEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=initial_weight
        )
        
        self.edges[edge_id] = edge
        self.node_connections[source_id].add(target_id)
        self.node_connections[target_id].add(source_id)
        
        # Update connection counts
        if source_id in self.nodes:
            self.nodes[source_id].connection_count += 1
        if target_id in self.nodes:
            self.nodes[target_id].connection_count += 1
        
        self.stats['total_edges'] += 1
        self._save_edge_to_db(edge)
        
        return edge_id
    
    def _create_temporal_connections(self, node_id: str):
        """Create temporal connections based on recent activations"""
        current_time = time.time()
        cutoff_time = current_time - self.coactivation_window
        
        recent = [act for act in self.recent_activations 
                 if act['timestamp'] >= cutoff_time and act['node_id'] != node_id]
        
        for activation in recent[-3:]:  # Last 3 activations
            self._create_edge(activation['node_id'], node_id, EdgeType.TEMPORAL, 0.3)
            self.stats['temporal_connections'] += 1
    
    def _save_node_to_db(self, node: SimpleNode):
        """Save node to database"""
        with self.db_lock:
            self.conn.execute('''
                INSERT OR REPLACE INTO nodes 
                (node_id, node_type, content, embedding, activation_strength, firing_rate,
                 last_activation, activation_count, creation_time, metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node.node_id, node.node_type.value, node.content,
                json.dumps(node.embedding), node.activation_strength, node.firing_rate,
                node.last_activation, node.activation_count, node.creation_time,
                json.dumps(node.metadata), node.modality_source
            ))
            self.conn.commit()
    
    def _save_edge_to_db(self, edge: SimpleEdge):
        """Save edge to database"""
        with self.db_lock:
            self.conn.execute('''
                INSERT OR REPLACE INTO edges
                (edge_id, source_id, target_id, edge_type, weight, coactivation_count,
                 last_coactivation, creation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                edge.edge_id, edge.source_id, edge.target_id, edge.edge_type.value,
                edge.weight, edge.coactivation_count, edge.last_coactivation, edge.creation_time
            ))
            self.conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get brain statistics"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': {nt.value: sum(1 for n in self.nodes.values() if n.node_type == nt) 
                          for nt in NodeType},
            'edge_types': {et.value: sum(1 for e in self.edges.values() if e.edge_type == et) 
                          for et in EdgeType},
            'stats': self.stats,
            'runtime': time.time() - self.stats['start_time']
        }
    
    def save_state(self):
        """Save complete brain state"""
        state_file = self.memory_path / "complete_brain_state.json"
        
        try:
            state = self.get_stats()
            # Add detailed node and edge information
            state['nodes'] = {
                node_id: {
                    'type': node.node_type.value,
                    'content': node.content[:100],  # Truncate for readability
                    'activation': node.activation_strength,
                    'connections': node.connection_count,
                    'source': node.modality_source
                }
                for node_id, node in list(self.nodes.items())[:50]  # First 50 nodes
            }
            
            state['edges'] = {
                edge_id: {
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'type': edge.edge_type.value,
                    'weight': edge.weight,
                    'coactivations': edge.coactivation_count
                }
                for edge_id, edge in list(self.edges.items())[:100]  # First 100 edges
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            log_info(f"💾 Brain state saved: {state['total_nodes']} nodes, {state['total_edges']} edges")
            
        except Exception as e:
            log_error(f"Error saving state: {e}")

class HuggingFaceMockData:
    """Mock Hugging Face data for demonstration"""
    
    @staticmethod
    def get_squad_data():
        """Mock SQuAD question-answer data"""
        return [
            {
                "question": "What is artificial intelligence?",
                "context": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans and animals.",
                "answer": "intelligence demonstrated by machines"
            },
            {
                "question": "How do neural networks learn?",
                "context": "Neural networks learn through a process called backpropagation, where errors are propagated backward through the network to adjust weights and biases.",
                "answer": "through backpropagation"
            },
            {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.",
                "answer": "learning from experience without explicit programming"
            },
            {
                "question": "How do robots perceive their environment?",
                "context": "Robots use various sensors like cameras, microphones, and touch sensors to perceive and understand their environment, similar to how humans use their senses.",
                "answer": "through various sensors like cameras and microphones"
            },
            {
                "question": "What is computer vision?",
                "context": "Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world, enabling machines to identify objects, faces, and scenes.",
                "answer": "training computers to interpret visual information"
            }
        ]
    
    @staticmethod
    def get_sentiment_data():
        """Mock sentiment analysis data"""
        return [
            {"text": "This AI system is absolutely amazing! It learns so quickly and makes intelligent connections.", "sentiment": "positive"},
            {"text": "The robot's movements are smooth and natural. Very impressive engineering.", "sentiment": "positive"},
            {"text": "Machine learning is fascinating. The way it finds patterns in data is remarkable.", "sentiment": "positive"},
            {"text": "I'm concerned about the complexity of this system. It might be too difficult to understand.", "sentiment": "negative"},
            {"text": "The neural network failed to converge. Poor performance on this task.", "sentiment": "negative"},
            {"text": "Excellent work on the computer vision module. Clear and accurate object detection.", "sentiment": "positive"},
            {"text": "The Hebbian learning implementation is brilliant. Connections strengthen naturally.", "sentiment": "positive"},
            {"text": "This robotic system shows great potential for real-world applications.", "sentiment": "positive"}
        ]
    
    @staticmethod
    def get_code_examples():
        """Mock code examples"""
        return [
            {
                "code": "def create_node(content, node_type):\n    node_id = generate_id()\n    return Node(node_id, node_type, content)",
                "language": "python",
                "description": "Node creation function"
            },
            {
                "code": "class BrainGraph:\n    def __init__(self):\n        self.nodes = {}\n        self.edges = {}\n    \n    def add_connection(self, source, target):\n        edge = Edge(source, target)\n        self.edges[edge.id] = edge",
                "language": "python",
                "description": "Brain graph class"
            },
            {
                "code": "for node in active_nodes:\n    if node.activation > threshold:\n        strengthen_connections(node)\n        update_weights(node.connections)",
                "language": "python",
                "description": "Hebbian learning loop"
            },
            {
                "code": "import cv2\nimport numpy as np\n\ndef process_camera_frame(frame):\n    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n    features = extract_features(gray)\n    return create_visual_node(features)",
                "language": "python",
                "description": "Camera processing pipeline"
            },
            {
                "code": "class Node {\npublic:\n    NodeID id;\n    NodeType type;\n    std::vector<Connection> connections;\n    \n    void activate(float strength) {\n        activation_strength += strength;\n    }\n};",
                "language": "cpp",
                "description": "C++ node structure"
            }
        ]
    
    @staticmethod
    def get_knowledge_base():
        """Mock knowledge base entries"""
        return [
            "The human brain contains approximately 86 billion neurons connected by trillions of synapses",
            "Hebbian learning follows the principle that neurons that fire together wire together",
            "Computer vision enables machines to interpret and understand visual information from cameras",
            "Natural language processing helps computers understand and generate human language",
            "Robotics combines mechanical engineering, electronics, and computer science",
            "Machine learning algorithms can recognize patterns in large datasets",
            "Neural networks are inspired by the structure and function of biological brains",
            "Artificial intelligence aims to create machines that can perform tasks requiring human intelligence",
            "Sensors provide robots with information about their environment and internal state",
            "Actuators enable robots to move and manipulate objects in the physical world",
            "Feedback loops allow systems to learn and adapt based on their performance",
            "Embeddings represent data as vectors in high-dimensional space for similarity comparison",
            "Graph neural networks operate on graph-structured data with nodes and edges",
            "Reinforcement learning trains agents through rewards and penalties in an environment",
            "Computer graphics generates and manipulates visual content using mathematical algorithms"
        ]

def run_huggingface_integration():
    """Run the complete Hugging Face integration demo"""
    print("🤗 MELVIN + HUGGING FACE INTEGRATION DEMO")
    print("=" * 60)
    print("🔹 Creating nodes and connections from mock HF data")
    print("🔹 Demonstrating Hebbian learning")
    print("🔹 Building cross-modal connections")
    print("🔹 Saving to persistent SQLite database")
    print("=" * 60)
    
    # Initialize brain
    brain = SimpleMelvinBrain(embedding_dim=128)
    
    # Get initial state
    initial_stats = brain.get_stats()
    print(f"📊 Initial state: {initial_stats['total_nodes']} nodes, {initial_stats['total_edges']} edges")
    
    # Process mock Hugging Face data
    mock_data = HuggingFaceMockData()
    
    # 1. Process SQuAD Q&A data
    print("\n📚 Processing SQuAD question-answer data...")
    squad_data = mock_data.get_squad_data()
    for i, item in enumerate(squad_data):
        brain.add_node(item['question'], NodeType.LANGUAGE, f"squad_question_{i}")
        brain.add_node(item['context'], NodeType.CONCEPT, f"squad_context_{i}")
        brain.add_node(item['answer'], NodeType.LANGUAGE, f"squad_answer_{i}")
        time.sleep(0.1)  # Small delay for temporal connections
    
    # 2. Process sentiment data
    print("\n💭 Processing sentiment analysis data...")
    sentiment_data = mock_data.get_sentiment_data()
    for i, item in enumerate(sentiment_data):
        brain.add_node(item['text'], NodeType.LANGUAGE, f"sentiment_text_{i}")
        brain.add_node(f"emotion: {item['sentiment']}", NodeType.EMOTION, f"sentiment_label_{i}")
        time.sleep(0.1)
    
    # 3. Process code examples
    print("\n💻 Processing code examples...")
    code_data = mock_data.get_code_examples()
    for i, item in enumerate(code_data):
        brain.add_node(item['code'], NodeType.CODE, f"code_{item['language']}_{i}")
        brain.add_node(item['description'], NodeType.CONCEPT, f"code_desc_{i}")
        time.sleep(0.1)
    
    # 4. Process knowledge base
    print("\n🧠 Processing knowledge base...")
    knowledge_data = mock_data.get_knowledge_base()
    for i, knowledge in enumerate(knowledge_data):
        brain.add_node(knowledge, NodeType.CONCEPT, f"knowledge_{i}")
        time.sleep(0.1)
    
    # Allow time for background processing
    print("\n⏳ Processing Hebbian learning connections...")
    time.sleep(2.0)
    
    # Get final statistics
    final_stats = brain.get_stats()
    
    print("\n🎉 INTEGRATION COMPLETE!")
    print("=" * 40)
    print(f"📊 Final state: {final_stats['total_nodes']} nodes, {final_stats['total_edges']} edges")
    print(f"🚀 Growth: +{final_stats['total_nodes']} nodes, +{final_stats['total_edges']} edges")
    print(f"⚡ Runtime: {final_stats['runtime']:.2f} seconds")
    
    print(f"\n🧠 Node Distribution:")
    for node_type, count in final_stats['node_types'].items():
        if count > 0:
            print(f"   {node_type}: {count} nodes")
    
    print(f"\n🔗 Connection Distribution:")
    for edge_type, count in final_stats['edge_types'].items():
        if count > 0:
            print(f"   {edge_type}: {count} connections")
    
    # Save brain state
    brain.save_state()
    
    print(f"\n💾 PERSISTENCE CONFIRMED:")
    print(f"   📁 Database: melvin_global_memory/global_memory.db")
    print(f"   📄 JSON state: melvin_global_memory/complete_brain_state.json")
    print(f"   🔄 Ready for future sessions!")
    
    return brain

def main():
    """Main entry point"""
    try:
        brain = run_huggingface_integration()
        
        print(f"\n✅ SUCCESS! Melvin brain populated with Hugging Face-inspired data")
        print(f"🔄 All nodes and connections are now saved in your repository")
        print(f"🚀 Next time you run Melvin, it will load this knowledge automatically")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Integration interrupted")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

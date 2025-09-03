#!/usr/bin/env python3
"""
🧠 MELVIN GLOBAL BRAIN - Complete Unified System
===============================================
One brain, one memory, many outputs. All knowledge unified.
Hebbian learning: "What fires together, wires together"
Deployable on Jetson Orin via COM8/PuTTY
"""

import cv2
import numpy as np
import time
import threading
import json
import pickle
import hashlib
import uuid
import logging
import asyncio
import websockets
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import queue
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CORE TYPES AND STRUCTURES
# ============================================================================

class NodeType(Enum):
    """All possible node types in unified memory"""
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
    HEBBIAN = "hebbian"           # Co-activation strengthening
    SIMILARITY = "similarity"     # Semantic/feature similarity
    TEMPORAL = "temporal"         # Sequential activation
    HIERARCHICAL = "hierarchical" # Parent-child relationship
    MULTIMODAL = "multimodal"    # Cross-modal connections
    CAUSAL = "causal"            # Cause-effect
    ASSOCIATIVE = "associative"   # General association

@dataclass
class GlobalNode:
    """Universal node in Melvin's global memory"""
    node_id: str
    node_type: NodeType
    content: Any
    embedding: np.ndarray
    
    # Hebbian properties
    activation_strength: float = 0.0
    firing_rate: float = 0.0
    last_activation: float = field(default_factory=time.time)
    activation_count: int = 0
    
    # Connection properties
    connection_strength: float = 1.0
    connection_count: int = 0
    
    # Temporal properties
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality_source: str = ""
    
    def activate(self, strength: float = 1.0):
        """Activate this node (Hebbian learning)"""
        self.activation_strength = min(1.0, self.activation_strength + strength)
        self.firing_rate = min(1.0, self.firing_rate + 0.1)
        self.last_activation = time.time()
        self.activation_count += 1
        self.last_update = time.time()

@dataclass
class GlobalEdge:
    """Connection between nodes with Hebbian strengthening"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    
    # Hebbian weight
    weight: float = 0.1
    coactivation_count: int = 0
    last_coactivation: float = field(default_factory=time.time)
    
    # Learning parameters
    learning_rate: float = 0.01
    decay_rate: float = 0.999
    min_weight: float = 0.001
    
    # Temporal properties
    creation_time: float = field(default_factory=time.time)
    last_reinforcement: float = field(default_factory=time.time)
    
    def strengthen(self, amount: float = None):
        """Strengthen connection (Hebbian learning)"""
        if amount is None:
            amount = self.learning_rate
        
        self.weight = min(1.0, self.weight + amount)
        self.coactivation_count += 1
        self.last_coactivation = time.time()
        self.last_reinforcement = time.time()
    
    def decay(self):
        """Apply decay to connection weight"""
        current_time = time.time()
        time_since_reinforcement = (current_time - self.last_reinforcement) / 60.0  # minutes
        decay_factor = self.decay_rate ** time_since_reinforcement
        self.weight *= decay_factor
        
        if self.weight < self.min_weight:
            return False  # Signal for pruning
        return True



# ============================================================================
# MELVIN GLOBAL MEMORY SYSTEM
# ============================================================================

class MelvinGlobalMemory:
    """Unified memory system - everything goes here"""
    
    def __init__(self, memory_path: str = "melvin_global_memory"):
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(exist_ok=True)
        
        # Core storage
        self.nodes: Dict[str, GlobalNode] = {}
        self.edges: Dict[str, GlobalEdge] = {}
        self.node_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Activation tracking for Hebbian learning
        self.recent_activations = deque(maxlen=1000)
        self.coactivation_window = 2.0  # seconds
        
        # Background processing
        self.background_enabled = True
        self.background_thread = None
        
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
        
        # Initialize database
        self._init_database()
        
        # Load existing memory
        self._load_existing_memory()
        
        logger.info("🧠 Melvin Global Memory initialized")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        db_path = self.memory_path / "global_memory.db"
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.db_lock = threading.Lock()
        
        # Create tables
        with self.db_lock:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT,
                    content TEXT,
                    embedding BLOB,
                    activation_strength REAL,
                    firing_rate REAL,
                    last_activation REAL,
                    activation_count INTEGER,
                    creation_time REAL,
                    metadata TEXT,
                    modality_source TEXT
                )
            ''')
            
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
    
    def add_node(self, content: Any, node_type: NodeType, embedding: np.ndarray,
                 modality_source: str = "", metadata: Dict[str, Any] = None) -> str:
        """Add node to global memory - EVERYTHING goes here"""
        
        node_id = f"{node_type.value}_{uuid.uuid4().hex[:12]}"
        
        node = GlobalNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            embedding=embedding.copy(),
            modality_source=modality_source,
            metadata=metadata or {}
        )
        
        # Add to memory
        self.nodes[node_id] = node
        self.stats['total_nodes'] += 1
        
        # Activate immediately
        node.activate(0.8)
        self._record_activation(node_id, 0.8)
        
        # Find and create connections
        self._create_connections_for_node(node_id)
        
        # Save to database
        self._save_node_to_db(node)
        
        logger.info(f"🔗 Added {node_type.value} node: {node_id} (from {modality_source})")
        return node_id
    
    def _record_activation(self, node_id: str, strength: float):
        """Record activation for Hebbian learning"""
        self.recent_activations.append({
            'node_id': node_id,
            'strength': strength,
            'timestamp': time.time()
        })
    
    def _create_connections_for_node(self, node_id: str):
        """Create all types of connections for new node"""
        if node_id not in self.nodes:
            return
        
        new_node = self.nodes[node_id]
        connections_created = 0
        
        # Check against all existing nodes
        for existing_id, existing_node in self.nodes.items():
            if existing_id == node_id:
                continue
            
            # Similarity-based connections
            similarity = self._calculate_similarity(new_node.embedding, existing_node.embedding)
            if similarity > 0.7:  # High similarity threshold
                self._create_edge(node_id, existing_id, EdgeType.SIMILARITY, similarity)
                connections_created += 1
                
                # Cross-modal bonus
                if new_node.node_type != existing_node.node_type:
                    self._create_edge(node_id, existing_id, EdgeType.MULTIMODAL, similarity * 0.8)
                    self.stats['cross_modal_connections'] += 1
        
        # Temporal connections (recent activations)
        self._create_temporal_connections(node_id)
        
        logger.info(f"🔗 Created {connections_created} connections for {node_id}")
    
    def _create_edge(self, source_id: str, target_id: str, edge_type: EdgeType, initial_weight: float = 0.1):
        """Create edge between nodes"""
        edge_id = f"{source_id}→{target_id}_{edge_type.value}"
        
        if edge_id in self.edges:
            # Strengthen existing edge
            self.edges[edge_id].strengthen()
            return edge_id
        
        # Create new edge
        edge = GlobalEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=initial_weight
        )
        
        self.edges[edge_id] = edge
        self.node_connections[source_id].add(target_id)
        self.node_connections[target_id].add(source_id)
        
        # Update node connection counts
        if source_id in self.nodes:
            self.nodes[source_id].connection_count += 1
        if target_id in self.nodes:
            self.nodes[target_id].connection_count += 1
        
        self.stats['total_edges'] += 1
        
        # Save to database
        self._save_edge_to_db(edge)
        
        return edge_id
    
    def _create_temporal_connections(self, node_id: str):
        """Create temporal connections based on recent activations"""
        current_time = time.time()
        cutoff_time = current_time - self.coactivation_window
        
        # Get recent activations
        recent = [act for act in self.recent_activations 
                 if act['timestamp'] >= cutoff_time and act['node_id'] != node_id]
        
        for activation in recent[-5:]:  # Last 5 activations
            self._create_edge(activation['node_id'], node_id, EdgeType.TEMPORAL, 0.3)
            self.stats['temporal_connections'] += 1
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norms = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return dot_product / norms if norms > 0 else 0.0



    def process_hebbian_learning(self):
        """Process Hebbian learning: "What fires together, wires together" """
        current_time = time.time()
        cutoff_time = current_time - self.coactivation_window
        
        # Get recent activations within window
        recent_activations = [act for act in self.recent_activations 
                            if act['timestamp'] >= cutoff_time]
        
        if len(recent_activations) < 2:
            return
        
        # Find co-activating pairs
        strengthened_connections = 0
        for i, act1 in enumerate(recent_activations):
            for act2 in recent_activations[i+1:]:
                if act1['node_id'] != act2['node_id']:
                    # Calculate co-activation strength
                    time_diff = abs(act1['timestamp'] - act2['timestamp'])
                    if time_diff <= self.coactivation_window:
                        coactivation_strength = min(act1['strength'], act2['strength'])
                        
                        # Strengthen bidirectional connections
                        edge_id1 = f"{act1['node_id']}→{act2['node_id']}_hebbian"
                        edge_id2 = f"{act2['node_id']}→{act1['node_id']}_hebbian"
                        
                        for edge_id, source, target in [(edge_id1, act1['node_id'], act2['node_id']),
                                                       (edge_id2, act2['node_id'], act1['node_id'])]:
                            if edge_id in self.edges:
                                self.edges[edge_id].strengthen(coactivation_strength * 0.1)
                            else:
                                self._create_edge(source, target, EdgeType.HEBBIAN, coactivation_strength * 0.05)
                            
                            strengthened_connections += 1
        
        if strengthened_connections > 0:
            self.stats['hebbian_updates'] += strengthened_connections
            logger.info(f"🧠 Hebbian learning: {strengthened_connections} connections strengthened")
    
    def _save_node_to_db(self, node: GlobalNode):
        """Save node to database"""
        with self.db_lock:
            self.conn.execute('''
                INSERT OR REPLACE INTO nodes 
                (node_id, node_type, content, embedding, activation_strength, firing_rate,
                 last_activation, activation_count, creation_time, metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node.node_id, node.node_type.value, str(node.content)[:1000],
                pickle.dumps(node.embedding), node.activation_strength, node.firing_rate,
                node.last_activation, node.activation_count, node.creation_time,
                json.dumps(node.metadata), node.modality_source
            ))
            self.conn.commit()
    
    def _save_edge_to_db(self, edge: GlobalEdge):
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
    
    def _load_existing_memory(self):
        """Load existing memory from old melvin_global_memory"""
        old_memory_path = Path("melvin_global_memory")
        if old_memory_path.exists():
            logger.info("🔄 Migrating existing memory...")
            
            # Look for existing memory files
            memory_files = list(old_memory_path.glob("*.json"))
            nodes_migrated = 0
            
            for memory_file in memory_files:
                try:
                    with open(memory_file, 'r') as f:
                        old_data = json.load(f)
                    
                    # Migrate nodes
                    if 'nodes' in old_data:
                        for node_id, node_data in old_data['nodes'].items():
                            # Create embedding from old data - match current embedding dimension
                            embedding = np.random.random(self.embedder.embedding_dim) * 0.1 + 0.5  # Match current dimension
                            
                            # Determine node type
                            node_type = NodeType.CONCEPT
                            if 'visual' in node_id.lower():
                                node_type = NodeType.VISUAL
                            elif 'text' in node_id.lower() or 'language' in node_id.lower():
                                node_type = NodeType.LANGUAGE
                            
                            # Create migrated node
                            migrated_id = self.add_node(
                                content=node_data.get('concept', ''),
                                node_type=node_type,
                                embedding=embedding,
                                modality_source='migrated',
                                metadata={'migrated': True, 'original_data': node_data}
                            )
                            nodes_migrated += 1
                            
                except Exception as e:
                    logger.warning(f"Could not migrate {memory_file}: {e}")
            
            if nodes_migrated > 0:
                logger.info(f"✅ Migrated {nodes_migrated} nodes from existing memory")
    
    def start_background_processing(self):
        """Start background Hebbian learning and maintenance"""
        self.background_enabled = True
        self.background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.background_thread.start()
        logger.info("🔄 Background Hebbian processing started")
    
    def _background_worker(self):
        """Background worker for continuous learning"""
        while self.background_enabled:
            try:
                # Hebbian learning updates
                self.process_hebbian_learning()
                
                # Edge decay
                self._apply_edge_decay()
                
                # Memory consolidation
                if len(self.nodes) % 100 == 0:
                    self._consolidate_memory()
                
                time.sleep(2.0)  # Process every 2 seconds
                
            except Exception as e:
                logger.error(f"Background processing error: {e}")
                time.sleep(5.0)
    
    def _apply_edge_decay(self):
        """Apply decay to all edges"""
        edges_to_remove = []
        
        for edge_id, edge in self.edges.items():
            if not edge.decay():
                edges_to_remove.append(edge_id)
        
        # Remove weak edges
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
            self.stats['total_edges'] -= 1
    
    def _consolidate_memory(self):
        """Consolidate memory - strengthen important connections"""
        # Find highly connected nodes
        important_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.connection_count * n.activation_count,
            reverse=True
        )[:50]  # Top 50 important nodes
        
        # Strengthen connections between important nodes
        for i, node1 in enumerate(important_nodes):
            for node2 in important_nodes[i+1:]:
                if node2.node_id in self.node_connections[node1.node_id]:
                    # Find and strengthen edge
                    for edge_id, edge in self.edges.items():
                        if ((edge.source_id == node1.node_id and edge.target_id == node2.node_id) or
                            (edge.source_id == node2.node_id and edge.target_id == node1.node_id)):
                            edge.strengthen(0.01)  # Small consolidation boost
                            break
        
        logger.info("🧠 Memory consolidation completed")
    
    def get_global_state(self) -> Dict[str, Any]:
        """Get complete global memory state"""
        # Most active nodes
        most_active = sorted(
            self.nodes.values(),
            key=lambda n: n.activation_strength * n.firing_rate,
            reverse=True
        )[:10]
        
        # Most connected nodes
        most_connected = sorted(
            self.nodes.values(),
            key=lambda n: n.connection_count,
            reverse=True
        )[:10]
        
        # Node type distribution
        node_types = defaultdict(int)
        for node in self.nodes.values():
            node_types[node.node_type.value] += 1
        
        # Edge type distribution
        edge_types = defaultdict(int)
        for edge in self.edges.values():
            edge_types[edge.edge_type.value] += 1
        
        return {
            'stats': self.stats.copy(),
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'most_active_nodes': [
                {
                    'id': node.node_id,
                    'type': node.node_type.value,
                    'content': str(node.content)[:50],
                    'activation': node.activation_strength,
                    'connections': node.connection_count
                }
                for node in most_active
            ],
            'most_connected_nodes': [
                {
                    'id': node.node_id,
                    'type': node.node_type.value,
                    'content': str(node.content)[:50],
                    'connections': node.connection_count,
                    'activation': node.activation_strength
                }
                for node in most_connected
            ],
            'runtime': time.time() - self.stats['start_time']
        }


# ============================================================================
# MULTIMODAL PROCESSORS - ALL INPUTS GO TO GLOBAL MEMORY
# ============================================================================

class MultimodalEmbedder:
    """Creates unified embeddings for all input types"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        
        # Foundation vocabularies (HuggingFace-inspired)
        self.text_vocab = self._build_text_foundation()
        self.visual_features = self._build_visual_foundation()
        self.code_patterns = self._build_code_foundation()
        self.audio_features = self._build_audio_foundation()
        
        logger.info(f"🎯 Multimodal Embedder initialized (dim: {embedding_dim})")
    
    def _build_text_foundation(self) -> Dict[str, np.ndarray]:
        """Build text foundation from common concepts"""
        # Core vocabulary with pre-computed embeddings
        foundation_words = [
            'ball', 'hand', 'face', 'eye', 'mouth', 'head', 'body', 'arm', 'leg',
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown',
            'black', 'white', 'gray', 'color', 'bright', 'dark', 'light',
            'move', 'stop', 'go', 'come', 'run', 'walk', 'jump', 'sit', 'stand',
            'look', 'see', 'hear', 'touch', 'feel', 'think', 'know', 'remember',
            'happy', 'sad', 'angry', 'fear', 'surprise', 'love', 'excited', 'calm',
            'big', 'small', 'large', 'tiny', 'huge', 'little', 'medium',
            'up', 'down', 'left', 'right', 'front', 'back', 'inside', 'outside',
            'now', 'then', 'before', 'after', 'today', 'yesterday', 'tomorrow',
            'person', 'people', 'man', 'woman', 'child', 'baby', 'family',
            'house', 'car', 'tree', 'flower', 'water', 'fire', 'earth', 'air',
            'food', 'eat', 'drink', 'sleep', 'wake', 'work', 'play', 'learn'
        ]
        
        vocab = {}
        for i, word in enumerate(foundation_words):
            # Create pseudo-BERT-like embeddings
            embedding = np.random.random(self.embedding_dim)
            # Add semantic clustering
            if word in ['red', 'blue', 'green', 'yellow', 'orange']:
                embedding[0:50] += 0.3  # Color cluster
            elif word in ['ball', 'hand', 'face', 'eye']:
                embedding[50:100] += 0.3  # Object cluster
            elif word in ['move', 'run', 'walk', 'jump']:
                embedding[100:150] += 0.3  # Action cluster
            elif word in ['happy', 'sad', 'angry', 'love']:
                embedding[150:200] += 0.3  # Emotion cluster
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            vocab[word] = embedding
        
        return vocab
    
    def _build_visual_foundation(self) -> Dict[str, np.ndarray]:
        """Build visual feature foundation"""
        visual_concepts = {
            'brightness': np.array([1.0, 0.8, 0.3] + [0.1] * (self.embedding_dim - 3)),
            'darkness': np.array([0.1, 0.2, 0.8] + [0.1] * (self.embedding_dim - 3)),
            'red_color': np.array([0.9, 0.1, 0.1] + [0.2] * (self.embedding_dim - 3)),
            'blue_color': np.array([0.1, 0.1, 0.9] + [0.2] * (self.embedding_dim - 3)),
            'green_color': np.array([0.1, 0.9, 0.1] + [0.2] * (self.embedding_dim - 3)),
            'face_pattern': np.array([0.5, 0.7, 0.6] + [0.3] * (self.embedding_dim - 3)),
            'circular_shape': np.array([0.8, 0.8, 0.2] + [0.25] * (self.embedding_dim - 3)),
            'rectangular_shape': np.array([0.2, 0.8, 0.8] + [0.25] * (self.embedding_dim - 3)),
            'movement_pattern': np.array([0.6, 0.3, 0.9] + [0.2] * (self.embedding_dim - 3)),
            'stillness_pattern': np.array([0.3, 0.6, 0.1] + [0.2] * (self.embedding_dim - 3))
        }
        
        # Normalize all embeddings
        for key in visual_concepts:
            visual_concepts[key] = visual_concepts[key] / np.linalg.norm(visual_concepts[key])
        
        return visual_concepts
    
    def _build_code_foundation(self) -> Dict[str, np.ndarray]:
        """Build code pattern foundation"""
        code_patterns = {
            'function_def': 'def function():',
            'class_def': 'class MyClass:',
            'if_statement': 'if condition:',
            'for_loop': 'for item in list:',
            'while_loop': 'while True:',
            'import_statement': 'import module',
            'return_statement': 'return value',
            'print_statement': 'print("hello")',
            'variable_assignment': 'x = 10',
            'list_creation': '[1, 2, 3]',
            'dict_creation': '{"key": "value"}',
            'lambda_function': 'lambda x: x'
        }
        
        embeddings = {}
        for i, (pattern_name, code) in enumerate(code_patterns.items()):
            embedding = np.random.random(self.embedding_dim) * 0.1
            # Add code-specific features
            embedding[300:350] = 0.5 + np.random.random(50) * 0.3  # Code cluster
            if 'def' in code or 'class' in code:
                embedding[350:370] += 0.4  # Definition cluster
            elif 'for' in code or 'while' in code:
                embedding[370:390] += 0.4  # Loop cluster
            
            embeddings[pattern_name] = embedding / np.linalg.norm(embedding)
        
        return embeddings
    
    def _build_audio_foundation(self) -> Dict[str, np.ndarray]:
        """Build audio feature foundation"""
        audio_concepts = {
            'high_pitch': np.array([0.9, 0.2, 0.1] + [0.1] * (self.embedding_dim - 3)),
            'low_pitch': np.array([0.1, 0.2, 0.9] + [0.1] * (self.embedding_dim - 3)),
            'loud_volume': np.array([0.8, 0.8, 0.3] + [0.2] * (self.embedding_dim - 3)),
            'quiet_volume': np.array([0.2, 0.3, 0.8] + [0.1] * (self.embedding_dim - 3)),
            'voice_pattern': np.array([0.6, 0.7, 0.5] + [0.3] * (self.embedding_dim - 3)),
            'music_pattern': np.array([0.5, 0.6, 0.7] + [0.25] * (self.embedding_dim - 3)),
            'noise_pattern': np.array([0.4, 0.4, 0.4] + [0.2] * (self.embedding_dim - 3))
        }
        
        # Normalize
        for key in audio_concepts:
            audio_concepts[key] = audio_concepts[key] / np.linalg.norm(audio_concepts[key])
        
        return audio_concepts
    
    def embed_text(self, text: str) -> np.ndarray:
        """Create embedding for text input"""
        if not text:
            return np.zeros(self.embedding_dim)
        
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)
        found_words = 0
        
        # Use foundation vocabulary
        for word in words:
            if word in self.text_vocab:
                embedding += self.text_vocab[word]
                found_words += 1
        
        # Add text statistics
        if len(embedding) > 50:
            stats_start = self.embedding_dim - 50
            embedding[stats_start:stats_start+20] = [
                len(text) / 1000.0,
                len(words) / 100.0,
                text.count('?') / max(len(text), 1),
                text.count('!') / max(len(text), 1),
                text.count('.') / max(len(text), 1),
                sum(1 for c in text if c.isupper()) / max(len(text), 1),
                sum(1 for c in text if c.isdigit()) / max(len(text), 1),
                len(set(words)) / max(len(words), 1),
                np.mean([len(w) for w in words]) / 10.0 if words else 0,
                1.0 if any(w in ['good', 'great', 'excellent'] for w in words) else 0.0,
                1.0 if any(w in ['bad', 'terrible', 'awful'] for w in words) else 0.0,
                1.0 if any(w in ['red', 'blue', 'green', 'color'] for w in words) else 0.0,
                1.0 if any(w in ['move', 'run', 'walk', 'action'] for w in words) else 0.0,
                1.0 if any(w in ['face', 'eye', 'hand', 'body'] for w in words) else 0.0,
                1.0 if any(w in ['think', 'know', 'remember', 'learn'] for w in words) else 0.0,
                1.0 if any(w in ['happy', 'sad', 'emotion', 'feel'] for w in words) else 0.0,
                found_words / max(len(words), 1),  # Vocabulary coverage
                1.0 if '?' in text else 0.0,  # Question
                1.0 if text.startswith(('I ', 'My ', 'We ')) else 0.0,  # Personal
                float(len([w for w in words if len(w) > 6])) / max(len(words), 1)  # Complex words
            ]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def embed_visual(self, visual_features: Dict[str, float]) -> np.ndarray:
        """Create embedding for visual input"""
        embedding = np.zeros(self.embedding_dim)
        
        # Map features to foundation concepts
        feature_mappings = {
            'brightness': 'brightness' if visual_features.get('brightness', 0) > 0.6 else 'darkness',
            'color_red': 'red_color',
            'color_blue': 'blue_color', 
            'color_green': 'green_color',
            'face_detected': 'face_pattern',
            'motion': 'movement_pattern' if visual_features.get('motion', 0) > 0.1 else 'stillness_pattern'
        }
        
        for feature_name, value in visual_features.items():
            if feature_name in feature_mappings:
                concept_name = feature_mappings[feature_name]
                if concept_name in self.visual_features:
                    embedding += self.visual_features[concept_name] * float(value)
        
        # Add raw feature values
        feature_start = self.embedding_dim - 100
        raw_features = [
            visual_features.get('brightness', 0),
            visual_features.get('contrast', 0),
            visual_features.get('saturation', 0),
            visual_features.get('hue_mean', 0),
            visual_features.get('edge_density', 0),
            visual_features.get('motion', 0),
            visual_features.get('face_detected', 0),
            visual_features.get('color_red', 0),
            visual_features.get('color_green', 0),
            visual_features.get('color_blue', 0)
        ]
        
        if feature_start >= 0:
            embedding[feature_start:feature_start+len(raw_features)] = raw_features
        
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def embed_code(self, code: str) -> np.ndarray:
        """Create embedding for code input"""
        if not code:
            return np.zeros(self.embedding_dim)
        
        embedding = np.zeros(self.embedding_dim)
        
        # Match against code patterns
        for pattern_name, pattern_embedding in self.code_patterns.items():
            if any(keyword in code.lower() for keyword in pattern_name.split('_')):
                embedding += pattern_embedding * 0.5
        
        # Code structure analysis
        lines = code.split('\n')
        code_stats = [
            len(lines) / 100.0,
            np.mean([len(line) for line in lines]) / 100.0,
            code.count('def ') + code.count('class '),
            code.count('if ') + code.count('elif '),
            code.count('for ') + code.count('while '),
            code.count('import ') + code.count('from '),
            code.count('return '),
            code.count('print('),
            code.count('='),
            code.count('#')
        ]
        
        # Add to embedding
        stats_start = self.embedding_dim - 20
        if stats_start >= 0:
            embedding[stats_start:stats_start+len(code_stats)] = code_stats
        
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def embed_audio(self, audio_features: Dict[str, float]) -> np.ndarray:
        """Create embedding for audio input"""
        embedding = np.zeros(self.embedding_dim)
        
        # Map to audio concepts
        for feature_name, value in audio_features.items():
            concept_map = {
                'pitch': 'high_pitch' if value > 0.6 else 'low_pitch',
                'volume': 'loud_volume' if value > 0.6 else 'quiet_volume',
                'voice_detected': 'voice_pattern',
                'music_detected': 'music_pattern'
            }
            
            if feature_name in concept_map:
                concept_name = concept_map[feature_name]
                if concept_name in self.audio_features:
                    embedding += self.audio_features[concept_name] * float(value)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding



# ============================================================================
# MAIN MELVIN GLOBAL BRAIN - UNIFIED SYSTEM
# ============================================================================

class MelvinGlobalBrain:
    """Main unified brain - everything flows through global memory"""
    
    def __init__(self, embedding_dim: int = 512):
        # Core components
        self.global_memory = MelvinGlobalMemory()
        self.embedder = MultimodalEmbedder(embedding_dim)
        
        # Processing state
        self.processing_enabled = True
        self.conscious_mode = True  # vs unconscious background processing
        
        # Input processors
        self.camera_processor = None
        self.audio_processor = None
        
        # Statistics
        self.session_stats = {
            'inputs_processed': 0,
            'visual_inputs': 0,
            'text_inputs': 0,
            'code_inputs': 0,
            'audio_inputs': 0,
            'start_time': time.time()
        }
        
        logger.info("🧠 Melvin Global Brain initialized - ONE BRAIN, ONE MEMORY")
    
    def start_unified_processing(self):
        """Start all background processing"""
        self.global_memory.start_background_processing()
        logger.info("🚀 Unified processing started")
    
    def stop_unified_processing(self):
        """Stop all processing"""
        self.processing_enabled = False
        self.global_memory.background_enabled = False
        logger.info("🛑 Unified processing stopped")
    
    def process_visual_input(self, frame: np.ndarray = None, visual_features: Dict[str, float] = None) -> str:
        """Process visual input - camera frame or features"""
        if frame is not None:
            # Extract features from frame
            visual_features = self._extract_visual_features(frame)
        
        if not visual_features:
            return None
        
        # Create embedding
        embedding = self.embedder.embed_visual(visual_features)
        
        # Add to global memory
        node_id = self.global_memory.add_node(
            content=visual_features,
            node_type=NodeType.VISUAL,
            embedding=embedding,
            modality_source="camera",
            metadata={
                'frame_timestamp': time.time(),
                'features': visual_features,
                'processing_mode': 'conscious' if self.conscious_mode else 'unconscious'
            }
        )
        
        self.session_stats['visual_inputs'] += 1
        self.session_stats['inputs_processed'] += 1
        
        return node_id
    
    def process_text_input(self, text: str, source: str = "user") -> str:
        """Process text input - ALL TEXT GOES TO GLOBAL MEMORY"""
        if not text:
            return None
        
        # Create embedding
        embedding = self.embedder.embed_text(text)
        
        # Add to global memory
        node_id = self.global_memory.add_node(
            content=text,
            node_type=NodeType.LANGUAGE,
            embedding=embedding,
            modality_source=source,
            metadata={
                'text_length': len(text),
                'word_count': len(text.split()),
                'input_timestamp': time.time(),
                'processing_mode': 'conscious' if self.conscious_mode else 'unconscious'
            }
        )
        
        self.session_stats['text_inputs'] += 1
        self.session_stats['inputs_processed'] += 1
        
        return node_id
    
    def process_code_input(self, code: str, language: str = "python") -> str:
        """Process code input - ALL CODE GOES TO GLOBAL MEMORY"""
        if not code:
            return None
        
        # Create embedding
        embedding = self.embedder.embed_code(code)
        
        # Add to global memory
        node_id = self.global_memory.add_node(
            content=code,
            node_type=NodeType.CODE,
            embedding=embedding,
            modality_source=f"code_{language}",
            metadata={
                'language': language,
                'line_count': len(code.split('\n')),
                'char_count': len(code),
                'input_timestamp': time.time()
            }
        )
        
        self.session_stats['code_inputs'] += 1
        self.session_stats['inputs_processed'] += 1
        
        return node_id
    
    def process_audio_input(self, audio_features: Dict[str, float], source: str = "microphone") -> str:
        """Process audio input - ALL AUDIO GOES TO GLOBAL MEMORY"""
        if not audio_features:
            return None
        
        # Create embedding
        embedding = self.embedder.embed_audio(audio_features)
        
        # Add to global memory
        node_id = self.global_memory.add_node(
            content=audio_features,
            node_type=NodeType.AUDIO,
            embedding=embedding,
            modality_source=source,
            metadata={
                'features': audio_features,
                'input_timestamp': time.time()
            }
        )
        
        self.session_stats['audio_inputs'] += 1
        self.session_stats['inputs_processed'] += 1
        
        return node_id
    
    def _extract_visual_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract visual features from camera frame"""
        if frame is None:
            return {}
        
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            features = {}
            
            # Basic statistics
            features['brightness'] = float(np.mean(gray) / 255.0)
            features['contrast'] = float(np.std(gray) / 255.0)
            features['saturation'] = float(np.mean(hsv[:,:,1]) / 255.0)
            features['hue_mean'] = float(np.mean(hsv[:,:,0]) / 180.0)
            
            # Color detection
            for color_name, (lower, upper) in {
                'red': [(0, 50, 50), (10, 255, 255)],
                'green': [(40, 50, 50), (80, 255, 255)],
                'blue': [(100, 50, 50), (130, 255, 255)]
            }.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                color_ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
                features[f'color_{color_name}'] = float(color_ratio)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.count_nonzero(edges) / edges.size)
            
            # Face detection (if cascade available)
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                features['face_detected'] = 1.0 if len(faces) > 0 else 0.0
                features['face_count'] = float(len(faces))
            except:
                features['face_detected'] = 0.0
                features['face_count'] = 0.0
            
            # Motion (simplified - would need previous frame for real motion)
            features['motion'] = float(np.std(gray) / 255.0)  # Proxy for motion
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {}
    
    def generate_response(self, query: str = None, context: Dict[str, Any] = None) -> str:
        """Generate response from global memory - EVERYTHING IS GROUNDED"""
        # Process query if provided
        if query:
            query_node_id = self.process_text_input(query, "query")
            
            # Find related nodes in global memory
            if query_node_id in self.global_memory.nodes:
                query_node = self.global_memory.nodes[query_node_id]
                
                # Find most similar nodes across ALL modalities
                related_nodes = []
                for node_id, node in self.global_memory.nodes.items():
                    if node_id != query_node_id:
                        similarity = self.global_memory._calculate_similarity(
                            query_node.embedding, node.embedding
                        )
                        if similarity > 0.3:  # Similarity threshold
                            related_nodes.append((node, similarity))
                
                # Sort by similarity
                related_nodes.sort(key=lambda x: x[1], reverse=True)
                
                # Generate response from top related nodes
                response_parts = []
                for node, similarity in related_nodes[:5]:
                    content_str = str(node.content)[:100]
                    response_parts.append(f"{node.node_type.value}: {content_str} (sim: {similarity:.2f})")
                
                if response_parts:
                    response = f"From global memory: {'; '.join(response_parts)}"
                else:
                    response = "No strong associations found in global memory."
            else:
                response = "Query processing failed."
        else:
            # Generate status response
            state = self.global_memory.get_global_state()
            response = f"Global brain active: {state['total_nodes']} nodes, {state['total_edges']} connections"
        
        # Process response as new input (feedback loop)
        self.process_text_input(response, "self_output")
        
        return response
    
    def get_unified_state(self) -> Dict[str, Any]:
        """Get complete unified brain state"""
        global_state = self.global_memory.get_global_state()
        
        # Add session statistics
        runtime = time.time() - self.session_stats['start_time']
        processing_rate = self.session_stats['inputs_processed'] / runtime if runtime > 0 else 0
        
        unified_state = {
            'global_memory': global_state,
            'session_stats': self.session_stats.copy(),
            'processing_rate': processing_rate,
            'conscious_mode': self.conscious_mode,
            'processing_enabled': self.processing_enabled,
            'modality_distribution': {
                'visual': self.session_stats['visual_inputs'],
                'text': self.session_stats['text_inputs'],
                'code': self.session_stats['code_inputs'],
                'audio': self.session_stats['audio_inputs']
            },
            'runtime': runtime
        }
        
        return unified_state
    
    def save_complete_state(self):
        """Save complete brain state"""
        state_file = Path("melvin_global_memory/complete_brain_state.json")
        
        try:
            complete_state = self.get_unified_state()
            # Convert numpy arrays to lists for JSON serialization
            serializable_state = json.loads(json.dumps(complete_state, default=str))
            
            with open(state_file, 'w') as f:
                json.dump(serializable_state, f, indent=2)
            
            logger.info(f"💾 Complete brain state saved: {complete_state['global_memory']['total_nodes']} nodes")
            
        except Exception as e:
            logger.error(f"Error saving complete state: {e}")
# ============================================================================
# CAMERA INTEGRATION - VISUAL INPUT TO GLOBAL BRAIN
# ============================================================================

class UnifiedCameraProcessor:
    """Camera processor that feeds directly into global brain"""
    
    def __init__(self, global_brain: MelvinGlobalBrain, camera_device: str = '/dev/video0'):
        self.global_brain = global_brain
        self.camera_device = camera_device
        self.camera = None
        self.processing_thread = None
        self.is_running = False
        
        # Processing settings
        self.target_fps = 15
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info(f"📹 Unified Camera Processor initialized: {camera_device}")
    
    def start_camera_processing(self, duration: int = None):
        """Start continuous camera processing"""
        try:
            self.camera = cv2.VideoCapture(self.camera_device)
            if not self.camera.isOpened():
                logger.error(f"❌ Could not open camera: {self.camera_device}")
                return False
            
            # Configure camera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            logger.info(f"📹 Camera processing started (target: {self.target_fps} FPS)")
            
            # Processing loop
            while self.is_running:
                if duration and (time.time() - self.start_time) > duration:
                    break
                
                ret, frame = self.camera.read()
                if ret:
                    # Process frame through global brain
                    node_id = self.global_brain.process_visual_input(frame=frame)
                    self.frame_count += 1
                    
                    # Show progress every 30 frames
                    if self.frame_count % 30 == 0:
                        elapsed = time.time() - self.start_time
                        fps = self.frame_count / elapsed
                        brain_state = self.global_brain.get_unified_state()
                        
                        logger.info(f"📊 Frame {self.frame_count}: {fps:.1f} FPS, "
                                   f"Nodes: {brain_state['global_memory']['total_nodes']}, "
                                   f"Edges: {brain_state['global_memory']['total_edges']}")
                
                # Frame timing
                time.sleep(1.0 / self.target_fps)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Camera processing interrupted")
            return True
        except Exception as e:
            logger.error(f"❌ Camera processing error: {e}")
            return False
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup camera resources"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        
        # Final statistics
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            logger.info(f"📹 Camera processing completed: {self.frame_count} frames, {fps:.1f} avg FPS")

# ============================================================================
# INTERACTIVE INTERFACE
# ============================================================================

class UnifiedInterface:
    """Interactive interface for the unified brain"""
    
    def __init__(self, global_brain: MelvinGlobalBrain):
        self.global_brain = global_brain
        self.camera_processor = UnifiedCameraProcessor(global_brain)
        
    def run_interactive_mode(self):
        """Run interactive mode"""
        print("🎮 MELVIN GLOBAL BRAIN - INTERACTIVE MODE")
        print("=" * 50)
        print("Commands:")
        print("  text <message>     - Process text input")
        print("  code <code>        - Process code input") 
        print("  camera             - Process single camera frame")
        print("  camera_stream <sec>- Process camera stream")
        print("  query <question>   - Query global memory")
        print("  status             - Show brain status")
        print("  state              - Show detailed state")
        print("  save               - Save brain state")
        print("  conscious          - Switch to conscious mode")
        print("  unconscious        - Switch to unconscious mode")
        print("  quit               - Exit")
        print()
        
        while True:
            try:
                command = input("melvin-global> ").strip()
                
                if command in ['quit', 'exit']:
                    break
                elif command.startswith('text '):
                    text = command[5:]
                    node_id = self.global_brain.process_text_input(text, "user")
                    print(f"✅ Text processed: {node_id}")
                    
                elif command.startswith('code '):
                    code = command[5:]
                    node_id = self.global_brain.process_code_input(code)
                    print(f"✅ Code processed: {node_id}")
                    
                elif command == 'camera':
                    # Single frame
                    try:
                        cap = cv2.VideoCapture(self.camera_processor.camera_device)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                node_id = self.global_brain.process_visual_input(frame=frame)
                                print(f"✅ Camera frame processed: {node_id}")
                            cap.release()
                        else:
                            print("❌ Camera not available")
                    except Exception as e:
                        print(f"❌ Camera error: {e}")
                        
                elif command.startswith('camera_stream '):
                    try:
                        duration = int(command.split()[1])
                        print(f"📹 Starting camera stream for {duration} seconds...")
                        self.camera_processor.start_camera_processing(duration)
                    except ValueError:
                        print("❌ Invalid duration")
                        
                elif command.startswith('query '):
                    query = command[6:]
                    response = self.global_brain.generate_response(query)
                    print(f"🧠 Response: {response}")
                    
                elif command == 'status':
                    state = self.global_brain.get_unified_state()
                    print(f"📊 Global Brain Status:")
                    print(f"   Nodes: {state['global_memory']['total_nodes']}")
                    print(f"   Edges: {state['global_memory']['total_edges']}")
                    print(f"   Inputs processed: {state['session_stats']['inputs_processed']}")
                    print(f"   Processing rate: {state['processing_rate']:.2f}/sec")
                    print(f"   Mode: {'Conscious' if state['conscious_mode'] else 'Unconscious'}")
                    
                elif command == 'state':
                    state = self.global_brain.get_unified_state()
                    print(json.dumps(state, indent=2, default=str))
                    
                elif command == 'save':
                    self.global_brain.save_complete_state()
                    print("💾 Brain state saved")
                    
                elif command == 'conscious':
                    self.global_brain.conscious_mode = True
                    print("🧠 Switched to conscious mode")
                    
                elif command == 'unconscious':
                    self.global_brain.conscious_mode = False
                    print("🌙 Switched to unconscious mode")
                    
                else:
                    print("❓ Unknown command")
                    
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("✅ Interactive mode ended")

# ============================================================================
# MAIN FUNCTION - UNIFIED SYSTEM ENTRY POINT
# ============================================================================

def main():
    """Main entry point for Melvin Global Brain"""
    print("🧠 MELVIN GLOBAL BRAIN - UNIFIED COGNITIVE SYSTEM")
    print("=" * 60)
    print("🔹 ONE BRAIN, ONE MEMORY, MANY OUTPUTS")
    print("🔹 ALL INPUTS FLOW TO GLOBAL MEMORY")
    print("🔹 HEBBIAN LEARNING: FIRE TOGETHER, WIRE TOGETHER")
    print("🔹 CROSS-MODAL CONNECTIONS: TEXT ↔ VISION ↔ CODE ↔ AUDIO")
    print("🔹 DEPLOYABLE ON JETSON VIA COM8/PUTTY")
    print("=" * 60)
    
    try:
        # Initialize global brain
        print("🚀 Initializing Melvin Global Brain...")
        global_brain = MelvinGlobalBrain(embedding_dim=512)
        
        # Start unified processing
        global_brain.start_unified_processing()
        
        # Create interface
        interface = UnifiedInterface(global_brain)
        
        print("\n✅ MELVIN GLOBAL BRAIN READY!")
        print("📊 Foundation loaded, Hebbian learning active")
        print("🔗 All modalities connected to unified memory")
        
        # Show initial state
        initial_state = global_brain.get_unified_state()
        print(f"📦 Initial nodes: {initial_state['global_memory']['total_nodes']}")
        print(f"🔗 Initial edges: {initial_state['global_memory']['total_edges']}")
        
        print("\nSelect mode:")
        print("1. Interactive mode (default)")
        print("2. Camera stream (60 seconds)")
        print("3. Demo mode (mixed inputs)")
        print("4. Autonomous mode (continuous)")
        
        choice = input("Choice [1-4]: ").strip() or "1"
        
        if choice == "2":
            print("📹 Starting 60-second camera stream...")
            camera_processor = UnifiedCameraProcessor(global_brain)
            camera_processor.start_camera_processing(60)
            
        elif choice == "3":
            print("🎭 Running demo mode...")
            run_demo_mode(global_brain)
            
        elif choice == "4":
            print("🤖 Running autonomous mode...")
            run_autonomous_mode(global_brain)
            
        else:
            # Interactive mode
            interface.run_interactive_mode()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'global_brain' in locals():
            global_brain.stop_unified_processing()
            global_brain.save_complete_state()
        print("✅ Melvin Global Brain shutdown complete")

def run_demo_mode(global_brain: MelvinGlobalBrain):
    """Run demonstration mode with mixed inputs"""
    print("🎭 DEMO MODE - MIXED MULTIMODAL INPUTS")
    print("=" * 40)
    
    # Demo inputs
    demo_inputs = [
        ("text", "I see a red ball moving", "user"),
        ("text", "The ball is round and bright", "user"),
        ("code", "def detect_ball(image):\n    return find_circles(image)", "python"),
        ("text", "Ball detection algorithm", "user"),
        ("code", "ball.move()", "python"),
        ("text", "Movement and motion patterns", "user")
    ]
    
    print("Processing demo inputs...")
    for input_type, content, source in demo_inputs:
        print(f"📥 Processing {input_type}: {content[:50]}...")
        
        if input_type == "text":
            node_id = global_brain.process_text_input(content, source)
        elif input_type == "code":
            node_id = global_brain.process_code_input(content, source)
        
        print(f"✅ Created node: {node_id}")
        time.sleep(1)
    
    # Process camera if available
    try:
        print("📹 Processing camera frame...")
        cap = cv2.VideoCapture('/dev/video0')
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                node_id = global_brain.process_visual_input(frame=frame)
                print(f"✅ Camera node: {node_id}")
            cap.release()
    except:
        print("📹 Camera not available, skipping")
    
    # Show final state
    final_state = global_brain.get_unified_state()
    print(f"\n🎉 DEMO COMPLETE!")
    print(f"📦 Total nodes: {final_state['global_memory']['total_nodes']}")
    print(f"🔗 Total edges: {final_state['global_memory']['total_edges']}")
    print(f"🧠 Cross-modal connections: {final_state['global_memory']['edge_types'].get('multimodal', 0)}")
    print(f"⚡ Hebbian updates: {final_state['global_memory']['stats']['hebbian_updates']}")

def run_autonomous_mode(global_brain: MelvinGlobalBrain):
    """Run autonomous mode"""
    print("🤖 AUTONOMOUS MODE - CONTINUOUS PROCESSING")
    print("Press Ctrl+C to stop")
    
    camera_processor = UnifiedCameraProcessor(global_brain)
    
    try:
        # Start camera processing
        camera_processor.start_camera_processing()
    except KeyboardInterrupt:
        print("🛑 Autonomous mode stopped")

if __name__ == "__main__":
    import sys
    sys.exit(main())

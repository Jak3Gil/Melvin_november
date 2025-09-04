#!/usr/bin/env python3
"""
🧠 MELVIN OPTIMIZED V2 - Complete Unified System
===============================================
Pure binary storage + Intelligent pruning + Advanced compression + 4TB optimization
Everything stored as bytes, converted to text only when needed for debugging.
One brain, one memory, many outputs. All knowledge unified.
Hebbian learning: "What fires together, wires together"
Deployable on Jetson Orin via COM8/PuTTY
"""

import os
import struct
import hashlib
import gzip
import lzma
import zstandard as zstd
import time
import logging
import threading
import sqlite3
import json
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict, deque
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('melvin_optimized_v2.log'),
        logging.StreamHandler()
    ]
)

# ============================================================================
# CORE TYPES AND STRUCTURES
# ============================================================================

class ContentType(IntEnum):
    """Content types as integers (1 byte each)"""
    TEXT = 0
    IMAGE = 1
    AUDIO = 2
    CODE = 3
    EMBEDDING = 4
    METADATA = 5
    CONCEPT = 6
    SEQUENCE = 7
    VISUAL = 8
    SENSOR = 9

class CompressionType(IntEnum):
    """Compression types as integers (1 byte each)"""
    NONE = 0
    GZIP = 1
    LZMA = 2
    ZSTD = 3

class ConnectionType(IntEnum):
    """Connection types as integers (1 byte each)"""
    HEBBIAN = 0
    SIMILARITY = 1
    TEMPORAL = 2
    HIERARCHICAL = 3
    MULTIMODAL = 4
    CAUSAL = 5
    ASSOCIATIVE = 6

@dataclass
class BinaryNode:
    """Pure binary node structure - 28 bytes header + content"""
    id: bytes                    # 8 bytes - unique identifier
    content: bytes               # Raw binary content
    content_type: int            # 1 byte - ContentType enum
    compression: int             # 1 byte - CompressionType enum
    importance: int              # 1 byte - 0-255 importance score
    creation_time: int           # 8 bytes - timestamp
    content_length: int          # 4 bytes - length of content
    connection_count: int         # 4 bytes - number of connections
    activation_strength: int     # 1 byte - 0-255 activation strength
    
    def to_bytes(self) -> bytes:
        """Convert node to binary format"""
        # Header: 28 bytes
        header = struct.pack('<QQBBBBII',
            int.from_bytes(self.id, 'little'),  # 8 bytes
            self.creation_time,                  # 8 bytes
            self.content_type,                   # 1 byte
            self.compression,                     # 1 byte
            self.importance,                      # 1 byte
            self.activation_strength,            # 1 byte
            self.content_length,                 # 4 bytes
            self.connection_count                # 4 bytes
        )
        
        return header + self.content

    @classmethod
    def from_bytes(cls, data: bytes) -> 'BinaryNode':
        """Create node from binary data"""
        # Parse header (28 bytes)
        header_size = 28
        header = data[:header_size]
        content = data[header_size:]
        
        (node_id_int, creation_time, content_type, compression, 
         importance, activation_strength, content_length, connection_count) = struct.unpack('<QQBBBBII', header)
        
        return cls(
            id=node_id_int.to_bytes(8, 'little'),
            content=content,
            content_type=content_type,
            compression=compression,
            importance=importance,
            creation_time=creation_time,
            content_length=content_length,
            connection_count=connection_count,
            activation_strength=activation_strength
        )

@dataclass
class BinaryConnection:
    """Pure binary connection structure - 18 bytes"""
    id: bytes                    # 8 bytes - unique identifier
    source_id: bytes             # 8 bytes - source node ID
    target_id: bytes             # 8 bytes - target node ID
    connection_type: int         # 1 byte - ConnectionType enum
    weight: int                  # 1 byte - 0-255 weight
    
    def to_bytes(self) -> bytes:
        """Convert connection to binary format"""
        return struct.pack('<QQQBB',
            int.from_bytes(self.id, 'little'),
            int.from_bytes(self.source_id, 'little'),
            int.from_bytes(self.target_id, 'little'),
            self.connection_type,
            self.weight
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'BinaryConnection':
        """Create connection from binary data"""
        (conn_id_int, source_id_int, target_id_int, 
         connection_type, weight) = struct.unpack('<QQQBB', data)
        
        return cls(
            id=conn_id_int.to_bytes(8, 'little'),
            source_id=source_id_int.to_bytes(8, 'little'),
            target_id=target_id_int.to_bytes(8, 'little'),
            connection_type=connection_type,
            weight=weight
        )

@dataclass
class PruningDecision:
    """Decision about whether to keep or prune a node"""
    node_id: bytes
    keep: bool
    confidence: float
    reason: str
    importance_score: float
    timestamp: float

# ============================================================================
# INTELLIGENT PRUNING SYSTEM
# ============================================================================

class IntelligentPruningSystem:
    """Comprehensive pruning system with multiple importance criteria"""
    
    def __init__(self):
        self.importance_cache = {}
        self.pruning_history = []
        self.user_feedback = []
        
        # Configuration
        self.activation_threshold = 0.1
        self.connection_threshold = 5
        self.semantic_threshold = 0.6
        self.temporal_half_life_days = 30
        self.eternal_threshold = 0.9
        
        # Content type weights
        self.content_type_weights = {
            ContentType.CODE: 0.9,
            ContentType.CONCEPT: 0.8,
            ContentType.EMBEDDING: 0.8,
            ContentType.TEXT: 0.7,
            ContentType.IMAGE: 0.6,
            ContentType.AUDIO: 0.5,
            ContentType.METADATA: 0.4
        }
        
        # Adaptive learning
        self.learning_rate = 0.01
        self.feedback_weight = 0.3
        
        logging.info("🧠 Intelligent Pruning System initialized")
    
    def calculate_activation_importance(self, node: BinaryNode) -> float:
        """Calculate importance based on activation patterns"""
        activation_strength = node.activation_strength / 255.0
        connection_count = node.connection_count
        
        # Current activation strength
        current_activation = activation_strength
        
        # Recent activity bonus
        days_since_creation = (time.time() - node.creation_time) / 86400
        recency_bonus = max(0, 1 - (days_since_creation / 7))  # Bonus for last 7 days
        
        # Frequency bonus
        frequency_bonus = min(1.0, connection_count / 100)  # Cap at 100 connections
        
        activation_score = (current_activation * 0.5) + (recency_bonus * 0.3) + (frequency_bonus * 0.2)
        return min(1.0, activation_score)
    
    def calculate_connection_importance(self, node: BinaryNode, connection_count: int) -> float:
        """Calculate importance based on connection patterns"""
        # Hub score (many connections)
        hub_score = min(1.0, connection_count / 50)
        
        # Authority score (incoming connections - approximated)
        authority_score = min(1.0, connection_count / 25)
        
        # Bridge score (connects different clusters)
        bridge_score = min(1.0, connection_count / 30)
        
        connection_score = (hub_score * 0.4) + (authority_score * 0.4) + (bridge_score * 0.2)
        return min(1.0, connection_score)
    
    def calculate_semantic_importance(self, content: bytes, content_type: ContentType) -> float:
        """Calculate semantic importance using content analysis"""
        if not content:
            return 0.0
        
        # Base importance by content type
        base_importance = self.content_type_weights.get(content_type, 0.5)
        
        # Content length factor (longer = more important)
        length_factor = min(1.0, len(content) / 1000)
        
        # Content complexity factor
        complexity_factor = min(1.0, len(set(content)) / len(content)) if content else 0.0
        
        semantic_score = (base_importance * 0.6) + (length_factor * 0.2) + (complexity_factor * 0.2)
        return min(1.0, semantic_score)
    
    def calculate_temporal_importance(self, node: BinaryNode) -> float:
        """Calculate importance based on temporal factors"""
        days_old = (time.time() - node.creation_time) / 86400
        
        # Exponential decay
        decay_factor = 0.5 ** (days_old / self.temporal_half_life_days)
        
        # Some content is timeless
        if node.importance > self.eternal_threshold * 255:
            decay_factor = max(decay_factor, 0.8)
        
        # Recent content gets a bonus
        recency_bonus = max(0, 1 - (days_old / 7))  # Bonus for last 7 days
        
        temporal_score = (decay_factor * 0.7) + (recency_bonus * 0.3)
        return min(1.0, temporal_score)
    
    def calculate_combined_importance(self, node: BinaryNode, connection_count: int) -> float:
        """Calculate combined importance score using all criteria"""
        # Calculate individual scores
        activation_score = self.calculate_activation_importance(node)
        connection_score = self.calculate_connection_importance(node, connection_count)
        semantic_score = self.calculate_semantic_importance(node.content, ContentType(node.content_type))
        temporal_score = self.calculate_temporal_importance(node)
        
        # Weighted combination
        combined_score = (
            activation_score * 0.25 +
            connection_score * 0.25 +
            semantic_score * 0.20 +
            temporal_score * 0.15 +
            (node.importance / 255.0) * 0.15  # Stored importance
        )
        
        return min(1.0, combined_score)
    
    def should_keep_node(self, node: BinaryNode, connection_count: int, threshold: float = 0.3) -> PruningDecision:
        """Decide whether to keep or prune a node"""
        importance = self.calculate_combined_importance(node, connection_count)
        
        # Decision logic
        keep = importance > threshold
        
        # Determine confidence and reason
        if keep:
            confidence = importance
            if importance > 0.7:
                reason = "High importance"
            elif node.connection_count > 10:
                reason = "Many connections"
            elif node.activation_strength > 200:
                reason = "High activation"
            else:
                reason = "Moderate importance"
        else:
            confidence = 1 - importance
            if importance < 0.1:
                reason = "Low importance"
            elif node.connection_count < 2:
                reason = "Few connections"
            elif node.activation_strength < 50:
                reason = "Low activation"
            else:
                reason = "Below threshold"
        
        decision = PruningDecision(
            node_id=node.id,
            keep=keep,
            confidence=confidence,
            reason=reason,
            importance_score=importance,
            timestamp=time.time()
        )
        
        return decision

# ============================================================================
# PURE BINARY STORAGE SYSTEM
# ============================================================================

class PureBinaryStorage:
    """Pure binary storage system - no text, just bytes!"""
    
    def __init__(self, storage_path: str = "melvin_binary_memory"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Binary files
        self.nodes_file = os.path.join(storage_path, "nodes.bin")
        self.connections_file = os.path.join(storage_path, "connections.bin")
        self.index_file = os.path.join(storage_path, "index.bin")
        
        # Statistics
        self.total_nodes = 0
        self.total_connections = 0
        self.total_bytes = 0
        
        # Pruning system
        self.pruning_system = IntelligentPruningSystem()
        
        # Initialize files
        self._init_storage()
        logging.info("🧠 Pure Binary Storage initialized")
    
    def _init_storage(self):
        """Initialize binary storage files"""
        # Create empty files if they don't exist
        for file_path in [self.nodes_file, self.connections_file, self.index_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    f.write(b'')  # Empty file
    
    def _compress_content(self, content: bytes, compression_type: CompressionType) -> bytes:
        """Compress content using specified algorithm"""
        if compression_type == CompressionType.NONE:
            return content
        elif compression_type == CompressionType.GZIP:
            return gzip.compress(content)
        elif compression_type == CompressionType.LZMA:
            return lzma.compress(content)
        elif compression_type == CompressionType.ZSTD:
            return zstd.compress(content)
        else:
            return content
    
    def _decompress_content(self, content: bytes, compression_type: CompressionType) -> bytes:
        """Decompress content using specified algorithm"""
        if compression_type == CompressionType.NONE:
            return content
        elif compression_type == CompressionType.GZIP:
            return gzip.decompress(content)
        elif compression_type == CompressionType.LZMA:
            return lzma.decompress(content)
        elif compression_type == CompressionType.ZSTD:
            return zstd.decompress(content)
        else:
            return content
    
    def _determine_compression(self, content: bytes) -> CompressionType:
        """Automatically determine best compression"""
        if len(content) < 100:  # Too small to compress
            return CompressionType.NONE
        
        # Test different compression methods
        gzip_size = len(gzip.compress(content))
        lzma_size = len(lzma.compress(content))
        zstd_size = len(zstd.compress(content))
        
        # Choose best compression
        sizes = {
            CompressionType.NONE: len(content),
            CompressionType.GZIP: gzip_size,
            CompressionType.LZMA: lzma_size,
            CompressionType.ZSTD: zstd_size
        }
        
        return min(sizes, key=sizes.get)
    
    def _calculate_importance(self, content: bytes, content_type: ContentType) -> int:
        """Calculate importance score (0-255)"""
        # Base importance by content type
        base_importance = {
            ContentType.CODE: 200,
            ContentType.CONCEPT: 180,
            ContentType.EMBEDDING: 180,
            ContentType.TEXT: 100,
            ContentType.IMAGE: 120,
            ContentType.AUDIO: 80,
            ContentType.METADATA: 150,
            ContentType.SEQUENCE: 160,
            ContentType.VISUAL: 140,
            ContentType.SENSOR: 90
        }
        
        importance = base_importance.get(content_type, 100)
        
        # Adjust by content length (longer = more important)
        length_factor = min(255, len(content) // 10)
        importance = min(255, importance + length_factor)
        
        return importance
    
    def store_node(self, content: bytes, content_type: ContentType, 
                   node_id: Optional[bytes] = None) -> bytes:
        """Store a node as pure binary data"""
        
        # Generate ID if not provided
        if node_id is None:
            node_id = hashlib.sha256(content).digest()[:8]
        
        # Determine compression
        compression_type = self._determine_compression(content)
        compressed_content = self._compress_content(content, compression_type)
        
        # Calculate importance
        importance = self._calculate_importance(content, content_type)
        
        # Create binary node
        node = BinaryNode(
            id=node_id,
            content=compressed_content,
            content_type=content_type.value,
            compression=compression_type.value,
            importance=importance,
            creation_time=int(time.time()),
            content_length=len(compressed_content),
            connection_count=0,
            activation_strength=0
        )
        
        # Write to binary file
        with open(self.nodes_file, 'ab') as f:
            node_bytes = node.to_bytes()
            f.write(node_bytes)
            self.total_bytes += len(node_bytes)
        
        self.total_nodes += 1
        
        logging.info(f"📦 Stored binary node: {node_id.hex()[:8]} "
                    f"({len(compressed_content)} bytes, "
                    f"compression: {compression_type.name})")
        
        return node_id
    
    def store_connection(self, source_id: bytes, target_id: bytes, 
                        connection_type: ConnectionType, weight: int = 128) -> bytes:
        """Store a connection as pure binary data"""
        
        # Generate connection ID
        conn_id = hashlib.sha256(source_id + target_id).digest()[:8]
        
        # Create binary connection
        connection = BinaryConnection(
            id=conn_id,
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type.value,
            weight=min(255, max(0, weight))
        )
        
        # Write to binary file
        with open(self.connections_file, 'ab') as f:
            conn_bytes = connection.to_bytes()
            f.write(conn_bytes)
            self.total_bytes += len(conn_bytes)
        
        self.total_connections += 1
        
        return conn_id
    
    def get_node(self, node_id: bytes) -> Optional[BinaryNode]:
        """Retrieve a node by ID"""
        # For simplicity, read entire file (in production, use index)
        if not os.path.exists(self.nodes_file):
            return None
        
        with open(self.nodes_file, 'rb') as f:
            data = f.read()
        
        # Parse nodes (each node starts with 28-byte header)
        offset = 0
        while offset < len(data):
            if offset + 28 > len(data):
                break
            
            # Read header
            header = data[offset:offset+28]
            (node_id_int, creation_time, content_type, compression, 
             importance, activation_strength, content_length, connection_count) = struct.unpack('<QQBBBBII', header)
            
            current_node_id = node_id_int.to_bytes(8, 'little')
            
            if current_node_id == node_id:
                # Found the node
                node_data = data[offset:offset+28+content_length]
                return BinaryNode.from_bytes(node_data)
            
            # Move to next node
            offset += 28 + content_length
        
        return None
    
    def get_node_as_text(self, node_id: bytes) -> Optional[str]:
        """Get node content as text (for debugging)"""
        node = self.get_node(node_id)
        if node is None:
            return None
        
        # Decompress content
        decompressed = self._decompress_content(
            node.content, 
            CompressionType(node.compression)
        )
        
        # Convert to text if it's text content
        if node.content_type == ContentType.TEXT.value:
            try:
                return decompressed.decode('utf-8')
            except UnicodeDecodeError:
                return f"[Binary data: {len(decompressed)} bytes]"
        else:
            return f"[{ContentType(node.content_type).name}: {len(decompressed)} bytes]"
    
    def prune_nodes(self, pruning_level: str = 'daily', max_nodes_to_prune: int = 1000) -> List[bytes]:
        """Prune nodes based on importance scores"""
        if not os.path.exists(self.nodes_file):
            return []
        
        with open(self.nodes_file, 'rb') as f:
            data = f.read()
        
        # Parse all nodes
        nodes = []
        offset = 0
        while offset < len(data):
            if offset + 28 > len(data):
                break
            
            # Read header
            header = data[offset:offset+28]
            (node_id_int, creation_time, content_type, compression, 
             importance, activation_strength, content_length, connection_count) = struct.unpack('<QQBBBBII', header)
            
            # Create node object
            node = BinaryNode(
                id=node_id_int.to_bytes(8, 'little'),
                content=data[offset+28:offset+28+content_length],
                content_type=content_type,
                compression=compression,
                importance=importance,
                creation_time=creation_time,
                content_length=content_length,
                connection_count=connection_count,
                activation_strength=activation_strength
            )
            
            nodes.append(node)
            offset += 28 + content_length
        
        # Make pruning decisions
        pruning_decisions = []
        for node in nodes:
            decision = self.pruning_system.should_keep_node(node, node.connection_count)
            pruning_decisions.append(decision)
        
        # Sort by importance score (lowest first)
        pruning_decisions.sort(key=lambda x: x.importance_score)
        
        # Select nodes to prune
        nodes_to_prune = []
        for decision in pruning_decisions:
            if not decision.keep and len(nodes_to_prune) < max_nodes_to_prune:
                nodes_to_prune.append(decision.node_id)
        
        # Log pruning decisions
        logging.info(f"🔍 Pruning analysis: {len(nodes)} nodes analyzed")
        logging.info(f"📊 Keeping: {len([d for d in pruning_decisions if d.keep])} nodes")
        logging.info(f"🗑️ Pruning: {len(nodes_to_prune)} nodes")
        
        return nodes_to_prune
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'total_nodes': self.total_nodes,
            'total_connections': self.total_connections,
            'total_bytes': self.total_bytes,
            'total_mb': self.total_bytes / (1024 * 1024),
            'nodes_file_size': os.path.getsize(self.nodes_file) if os.path.exists(self.nodes_file) else 0,
            'connections_file_size': os.path.getsize(self.connections_file) if os.path.exists(self.connections_file) else 0,
            'index_file_size': os.path.getsize(self.index_file) if os.path.exists(self.index_file) else 0
        }
        return stats

# ============================================================================
# OPTIMIZED MELVIN GLOBAL BRAIN
# ============================================================================

class MelvinOptimizedV2:
    """Optimized Melvin Global Brain with pure binary storage"""
    
    def __init__(self, storage_path: str = "melvin_binary_memory"):
        self.binary_storage = PureBinaryStorage(storage_path)
        self.running = False
        self.background_thread = None
        
        # Hebbian learning
        self.recent_activations = deque(maxlen=1000)
        self.coactivation_window = 2.0  # seconds
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'total_connections': 0,
            'hebbian_updates': 0,
            'similarity_connections': 0,
            'temporal_connections': 0,
            'cross_modal_connections': 0,
            'start_time': time.time()
        }
        
        logging.info("🧠 Melvin Optimized V2 initialized")
    
    def process_text_input(self, text: str, source: str = "user") -> bytes:
        """Process text input and store as binary"""
        text_bytes = text.encode('utf-8')
        node_id = self.binary_storage.store_node(text_bytes, ContentType.TEXT)
        
        # Update statistics
        self.stats['total_nodes'] += 1
        
        # Hebbian learning
        self._update_hebbian_learning(node_id)
        
        logging.info(f"📝 Processed text input: {text[:50]}... -> {node_id.hex()[:8]}")
        return node_id
    
    def process_code_input(self, code: str, source: str = "python") -> bytes:
        """Process code input and store as binary"""
        code_bytes = code.encode('utf-8')
        node_id = self.binary_storage.store_node(code_bytes, ContentType.CODE)
        
        # Update statistics
        self.stats['total_nodes'] += 1
        
        # Hebbian learning
        self._update_hebbian_learning(node_id)
        
        logging.info(f"💻 Processed code input: {code[:50]}... -> {node_id.hex()[:8]}")
        return node_id
    
    def process_visual_input(self, frame: np.ndarray = None, image_path: str = None) -> bytes:
        """Process visual input and store as binary"""
        if frame is not None:
            # Convert frame to bytes
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                image_bytes = buffer.tobytes()
            else:
                return None
        elif image_path:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        else:
            return None
        
        node_id = self.binary_storage.store_node(image_bytes, ContentType.IMAGE)
        
        # Update statistics
        self.stats['total_nodes'] += 1
        
        # Hebbian learning
        self._update_hebbian_learning(node_id)
        
        logging.info(f"🖼️ Processed visual input -> {node_id.hex()[:8]}")
        return node_id
    
    def process_audio_input(self, audio_path: str) -> bytes:
        """Process audio input and store as binary"""
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        node_id = self.binary_storage.store_node(audio_bytes, ContentType.AUDIO)
        
        # Update statistics
        self.stats['total_nodes'] += 1
        
        # Hebbian learning
        self._update_hebbian_learning(node_id)
        
        logging.info(f"🎵 Processed audio input -> {node_id.hex()[:8]}")
        return node_id
    
    def _update_hebbian_learning(self, node_id: bytes):
        """Update Hebbian learning for co-activation"""
        current_time = time.time()
        self.recent_activations.append((node_id, current_time))
        
        # Find co-activated nodes within window
        co_activated = []
        for recent_id, recent_time in self.recent_activations:
            if recent_id != node_id and (current_time - recent_time) <= self.coactivation_window:
                co_activated.append(recent_id)
        
        # Create Hebbian connections
        for co_activated_id in co_activated:
            conn_id = self.binary_storage.store_connection(
                node_id, co_activated_id, ConnectionType.HEBBIAN, weight=150
            )
            self.stats['hebbian_updates'] += 1
            self.stats['total_connections'] += 1
    
    def create_similarity_connection(self, source_id: bytes, target_id: bytes, similarity: float):
        """Create similarity connection between nodes"""
        weight = int(similarity * 255)
        conn_id = self.binary_storage.store_connection(
            source_id, target_id, ConnectionType.SIMILARITY, weight=weight
        )
        self.stats['similarity_connections'] += 1
        self.stats['total_connections'] += 1
        return conn_id
    
    def get_node_content(self, node_id: bytes) -> Optional[str]:
        """Get node content as text (for debugging)"""
        return self.binary_storage.get_node_as_text(node_id)
    
    def get_unified_state(self) -> Dict[str, Any]:
        """Get unified state of the brain"""
        storage_stats = self.binary_storage.get_storage_stats()
        
        return {
            'global_memory': {
                'total_nodes': storage_stats['total_nodes'],
                'total_edges': storage_stats['total_connections'],
                'storage_used_mb': storage_stats['total_mb'],
                'stats': self.stats
            },
            'system': {
                'running': self.running,
                'uptime_seconds': time.time() - self.stats['start_time']
            }
        }
    
    def prune_old_nodes(self, max_nodes_to_prune: int = 1000):
        """Prune old/unimportant nodes"""
        pruned_nodes = self.binary_storage.prune_nodes(max_nodes_to_prune=max_nodes_to_prune)
        logging.info(f"🗑️ Pruned {len(pruned_nodes)} nodes")
        return pruned_nodes
    
    def save_complete_state(self):
        """Save complete state to disk"""
        # Binary storage is already persistent
        logging.info("💾 Complete state saved (binary storage is persistent)")
    
    def stop_unified_processing(self):
        """Stop unified processing"""
        self.running = False
        if self.background_thread:
            self.background_thread.join()
        logging.info("🛑 Unified processing stopped")

# ============================================================================
# OPTIMIZED CONTINUOUS FEEDER
# ============================================================================

class OptimizedContinuousFeeder:
    """4TB-optimized continuous data feeding system"""
    
    def __init__(self, max_storage_gb: int = 4000):
        self.max_storage_bytes = max_storage_gb * 1024**3
        self.melvin_brain = MelvinOptimizedV2()
        self.running = False
        self.save_thread = None
        
        logging.info(f"🧠 Optimized Continuous Feeder initialized (max: {max_storage_gb}GB)")
    
    def _check_storage_limit(self) -> bool:
        """Check if we're approaching storage limit"""
        stats = self.melvin_brain.binary_storage.get_storage_stats()
        if stats['total_mb'] > self.max_storage_bytes / (1024 * 1024) * 0.95:  # 95% threshold
            logging.warning(f"⚠️ Storage limit approaching: {stats['total_mb']:.1f}MB / {self.max_storage_bytes/(1024*1024):.1f}MB")
            return False
        return True
    
    def process_data_sources(self, data_sources: List[str]):
        """Process data sources with 4TB optimization"""
        logging.info(f"🚀 Starting 4TB optimized processing of {len(data_sources)} sources")
        
        for i, source_path in enumerate(data_sources):
            if not self._check_storage_limit():
                logging.warning("🛑 Storage limit reached, stopping processing")
                break
            
            try:
                logging.info(f"📖 Processing source {i+1}/{len(data_sources)}: {source_path}")
                
                # Extract content from source
                content_list = self._extract_content_from_source(source_path)
                
                if not content_list:
                    continue
                
                # Process each content item
                for j, content in enumerate(content_list):
                    if not self._check_storage_limit():
                        break
                    
                    # Store as binary node
                    if isinstance(content, str):
                        node_id = self.melvin_brain.process_text_input(content, "feeder")
                    else:
                        # Handle binary content
                        node_id = self.melvin_brain.binary_storage.store_node(content, ContentType.TEXT)
                    
                    if node_id and j % 100 == 0:
                        stats = self.melvin_brain.binary_storage.get_storage_stats()
                        logging.info(f"📊 Progress: {j+1}/{len(content_list)} items, "
                                   f"{stats['total_nodes']} nodes, {stats['total_mb']:.1f}MB used")
                
                logging.info(f"✅ Completed {source_path}")
                
            except Exception as e:
                logging.error(f"Error processing {source_path}: {e}")
        
        # Final statistics
        stats = self.melvin_brain.binary_storage.get_storage_stats()
        logging.info(f"🏁 Processing completed:")
        logging.info(f"   🧠 Nodes created: {stats['total_nodes']:,}")
        logging.info(f"   🔗 Connections created: {stats['total_connections']:,}")
        logging.info(f"   💾 Storage used: {stats['total_mb']:.1f}MB / {self.max_storage_bytes/(1024*1024):.1f}MB")
    
    def _extract_content_from_source(self, source_path: str) -> List[Any]:
        """Extract content from a data source"""
        content_list = []
        
        try:
            if source_path.endswith('.json'):
                with open(source_path, 'r') as f:
                    data = json.load(f)
                content_list = self._extract_from_json(data)
            elif source_path.endswith(('.txt', '.md', '.py', '.js')):
                with open(source_path, 'r', encoding='utf-8') as f:
                    content_list = [f.read()]
        except Exception as e:
            logging.error(f"Error extracting from {source_path}: {e}")
        
        return content_list
    
    def _extract_from_json(self, data: Any, max_depth: int = 3) -> List[str]:
        """Extract text content from JSON structure"""
        content_list = []
        
        def extract_recursive(obj, depth=0):
            if depth > max_depth:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and len(value) > 10:
                        content_list.append(value)
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and len(item) > 10:
                        content_list.append(item)
                    elif isinstance(item, (dict, list)):
                        extract_recursive(item, depth + 1)
        
        extract_recursive(data)
        return content_list

# ============================================================================
# MAIN SYSTEM
# ============================================================================

def main():
    """Main function"""
    print("🧠 MELVIN OPTIMIZED V2")
    print("=" * 50)
    
    # Initialize optimized system
    melvin = MelvinOptimizedV2()
    
    # Test basic functionality
    print("🧪 Testing basic functionality...")
    
    # Process some test inputs
    text_id = melvin.process_text_input("This is a test of the optimized Melvin system!")
    code_id = melvin.process_code_input("def hello_world():\n    print('Hello, World!')")
    
    # Get unified state
    state = melvin.get_unified_state()
    print(f"📊 State: {state['global_memory']['total_nodes']} nodes, {state['global_memory']['total_edges']} edges")
    
    # Test retrieval
    text_content = melvin.get_node_content(text_id)
    code_content = melvin.get_node_content(code_id)
    print(f"📖 Retrieved text: {text_content}")
    print(f"💻 Retrieved code: {code_content}")
    
    # Test pruning
    print("\n🔍 Testing pruning system...")
    pruned_nodes = melvin.prune_old_nodes(max_nodes_to_prune=10)
    print(f"🗑️ Pruned {len(pruned_nodes)} nodes")
    
    # Test continuous feeder
    print("\n🚀 Testing continuous feeder...")
    feeder = OptimizedContinuousFeeder(max_storage_gb=4000)
    
    # Process some test sources
    test_sources = [
        "README.md",
        "melvin_global_brain.py"
    ]
    
    feeder.process_data_sources(test_sources)
    
    # Final stats
    final_state = melvin.get_unified_state()
    print(f"\n📊 Final stats:")
    print(f"   🧠 Nodes: {final_state['global_memory']['total_nodes']:,}")
    print(f"   🔗 Edges: {final_state['global_memory']['total_edges']:,}")
    print(f"   💾 Storage: {final_state['global_memory']['storage_used_mb']:.1f}MB")
    print(f"   ⚡ Hebbian updates: {final_state['global_memory']['stats']['hebbian_updates']}")
    
    print("\n🎉 Melvin Optimized V2 test completed successfully!")

if __name__ == "__main__":
    main()

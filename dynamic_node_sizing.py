#!/usr/bin/env python3
"""
🧠 DYNAMIC NODE SIZING SYSTEM
==============================
Creates nodes of dynamic sizes based on content complexity and context.
Can create small nodes (words), medium nodes (phrases), large nodes (concepts), 
and extra-large nodes (documents) based on the input and requirements.

Features:
- Adaptive node sizing based on content complexity
- Context-aware node creation
- Multi-level granularity
- Smart size selection algorithms
- Hierarchical node organization
"""

import os
import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
import math

# Import Melvin's systems
from melvin_global_brain import MelvinGlobalBrain, NodeType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NodeSize:
    """Node size configuration"""
    name: str
    min_size: int  # Minimum content length
    max_size: int  # Maximum content length
    granularity: str  # 'word', 'phrase', 'concept', 'document'
    connection_strategy: str  # 'similarity', 'hierarchical', 'temporal', 'all'
    max_connections: int
    similarity_threshold: float

@dataclass
class DynamicNode:
    """Dynamic node representation"""
    id: str
    content: str
    node_type: str
    size_category: str
    content_length: int
    complexity_score: float
    parent_id: Optional[str] = None
    creation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DynamicNodeSizer:
    """Creates nodes of dynamic sizes based on content complexity"""
    
    def __init__(self, brain: MelvinGlobalBrain = None):
        self.brain = brain or MelvinGlobalBrain(embedding_dim=512)
        
        # Define node size categories
        self.node_sizes = {
            'tiny': NodeSize(
                name='tiny',
                min_size=1,
                max_size=10,
                granularity='word',
                connection_strategy='similarity',
                max_connections=5,
                similarity_threshold=0.8
            ),
            'small': NodeSize(
                name='small',
                min_size=11,
                max_size=50,
                granularity='phrase',
                connection_strategy='similarity',
                max_connections=10,
                similarity_threshold=0.6
            ),
            'medium': NodeSize(
                name='medium',
                min_size=51,
                max_size=200,
                granularity='concept',
                connection_strategy='hierarchical',
                max_connections=20,
                similarity_threshold=0.4
            ),
            'large': NodeSize(
                name='large',
                min_size=201,
                max_size=1000,
                granularity='section',
                connection_strategy='temporal',
                max_connections=50,
                similarity_threshold=0.3
            ),
            'extra_large': NodeSize(
                name='extra_large',
                min_size=1001,
                max_size=10000,
                granularity='document',
                connection_strategy='all',
                max_connections=100,
                similarity_threshold=0.2
            )
        }
        
        # Statistics
        self.stats = {
            'total_nodes_created': 0,
            'tiny_nodes': 0,
            'small_nodes': 0,
            'medium_nodes': 0,
            'large_nodes': 0,
            'extra_large_nodes': 0,
            'connections_created': 0
        }
        
        logger.info("🧠 Dynamic Node Sizer initialized")
    
    def create_dynamic_nodes(self, text: str, preferred_size: str = 'auto', 
                           complexity_threshold: float = 0.5) -> List[DynamicNode]:
        """Create nodes of dynamic sizes based on content"""
        logger.info(f"📝 Creating dynamic nodes from text: {text[:50]}...")
        
        nodes = []
        
        if preferred_size == 'auto':
            # Auto-determine node sizes based on content complexity
            nodes = self._create_auto_sized_nodes(text, complexity_threshold)
        else:
            # Create nodes of specific size
            nodes = self._create_specific_sized_nodes(text, preferred_size)
        
        # Add to brain and create connections
        self._add_nodes_to_brain(nodes)
        
        return nodes
    
    def _create_auto_sized_nodes(self, text: str, complexity_threshold: float) -> List[DynamicNode]:
        """Automatically determine node sizes based on content complexity"""
        nodes = []
        
        # Analyze text complexity
        complexity_score = self._calculate_complexity(text)
        content_length = len(text)
        
        # Determine appropriate node sizes
        if content_length <= 10:
            # Very short text - create tiny nodes
            nodes.extend(self._create_tiny_nodes(text))
        elif content_length <= 50:
            # Short text - create small nodes
            nodes.extend(self._create_small_nodes(text))
        elif content_length <= 200:
            # Medium text - create medium nodes
            nodes.extend(self._create_medium_nodes(text))
        elif content_length <= 1000:
            # Long text - create large nodes
            nodes.extend(self._create_large_nodes(text))
        else:
            # Very long text - create extra large nodes
            nodes.extend(self._create_extra_large_nodes(text))
        
        # If complexity is high, also create smaller granular nodes
        if complexity_score > complexity_threshold:
            logger.info(f"🔍 High complexity detected ({complexity_score:.2f}), creating additional granular nodes")
            granular_nodes = self._create_granular_nodes(text)
            nodes.extend(granular_nodes)
        
        return nodes
    
    def _create_specific_sized_nodes(self, text: str, size: str) -> List[DynamicNode]:
        """Create nodes of a specific size category"""
        if size == 'tiny':
            return self._create_tiny_nodes(text)
        elif size == 'small':
            return self._create_small_nodes(text)
        elif size == 'medium':
            return self._create_medium_nodes(text)
        elif size == 'large':
            return self._create_large_nodes(text)
        elif size == 'extra_large':
            return self._create_extra_large_nodes(text)
        else:
            raise ValueError(f"Unknown size category: {size}")
    
    def _create_tiny_nodes(self, text: str) -> List[DynamicNode]:
        """Create tiny nodes (word-level)"""
        nodes = []
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        for i, word in enumerate(words):
            if len(word) >= 3:  # Skip very short words
                node_id = f"tiny_{hashlib.md5(f'{word}_{i}'.encode()).hexdigest()[:8]}"
                
                node = DynamicNode(
                    id=node_id,
                    content=word,
                    node_type="word",
                    size_category="tiny",
                    content_length=len(word),
                    complexity_score=self._calculate_word_complexity(word),
                    metadata={
                        'position': i,
                        'word_length': len(word)
                    }
                )
                
                nodes.append(node)
                self.stats['tiny_nodes'] += 1
        
        logger.info(f"📝 Created {len(nodes)} tiny nodes")
        return nodes
    
    def _create_small_nodes(self, text: str) -> List[DynamicNode]:
        """Create small nodes (phrase-level)"""
        nodes = []
        sentences = re.split(r'[.!?]+', text)
        
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Extract phrases
            phrases = self._extract_phrases(sentence)
            
            for phrase_idx, phrase in enumerate(phrases):
                if 11 <= len(phrase) <= 50:  # Small node size range
                    node_id = f"small_{hashlib.md5(f'{phrase}_{sent_idx}_{phrase_idx}'.encode()).hexdigest()[:8]}"
                    
                    node = DynamicNode(
                        id=node_id,
                        content=phrase,
                        node_type="phrase",
                        size_category="small",
                        content_length=len(phrase),
                        complexity_score=self._calculate_phrase_complexity(phrase),
                        metadata={
                            'sentence_index': sent_idx,
                            'phrase_index': phrase_idx,
                            'word_count': len(phrase.split())
                        }
                    )
                    
                    nodes.append(node)
                    self.stats['small_nodes'] += 1
        
        logger.info(f"📝 Created {len(nodes)} small nodes")
        return nodes
    
    def _create_medium_nodes(self, text: str) -> List[DynamicNode]:
        """Create medium nodes (concept-level)"""
        nodes = []
        
        # Split into medium-sized chunks
        chunks = self._split_into_chunks(text, target_size=100)
        
        for chunk_idx, chunk in enumerate(chunks):
            if 51 <= len(chunk) <= 200:  # Medium node size range
                node_id = f"medium_{hashlib.md5(f'{chunk}_{chunk_idx}'.encode()).hexdigest()[:8]}"
                
                node = DynamicNode(
                    id=node_id,
                    content=chunk,
                    node_type="concept",
                    size_category="medium",
                    content_length=len(chunk),
                    complexity_score=self._calculate_concept_complexity(chunk),
                    metadata={
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks)
                    }
                )
                
                nodes.append(node)
                self.stats['medium_nodes'] += 1
        
        logger.info(f"📝 Created {len(nodes)} medium nodes")
        return nodes
    
    def _create_large_nodes(self, text: str) -> List[DynamicNode]:
        """Create large nodes (section-level)"""
        nodes = []
        
        # Split into large chunks
        chunks = self._split_into_chunks(text, target_size=500)
        
        for chunk_idx, chunk in enumerate(chunks):
            if 201 <= len(chunk) <= 1000:  # Large node size range
                node_id = f"large_{hashlib.md5(f'{chunk}_{chunk_idx}'.encode()).hexdigest()[:8]}"
                
                node = DynamicNode(
                    id=node_id,
                    content=chunk,
                    node_type="section",
                    size_category="large",
                    content_length=len(chunk),
                    complexity_score=self._calculate_section_complexity(chunk),
                    metadata={
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks)
                    }
                )
                
                nodes.append(node)
                self.stats['large_nodes'] += 1
        
        logger.info(f"📝 Created {len(nodes)} large nodes")
        return nodes
    
    def _create_extra_large_nodes(self, text: str) -> List[DynamicNode]:
        """Create extra large nodes (document-level)"""
        nodes = []
        
        # Split into extra large chunks
        chunks = self._split_into_chunks(text, target_size=2000)
        
        for chunk_idx, chunk in enumerate(chunks):
            if 1001 <= len(chunk) <= 10000:  # Extra large node size range
                node_id = f"extra_large_{hashlib.md5(f'{chunk}_{chunk_idx}'.encode()).hexdigest()[:8]}"
                
                node = DynamicNode(
                    id=node_id,
                    content=chunk,
                    node_type="document",
                    size_category="extra_large",
                    content_length=len(chunk),
                    complexity_score=self._calculate_document_complexity(chunk),
                    metadata={
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks)
                    }
                )
                
                nodes.append(node)
                self.stats['extra_large_nodes'] += 1
        
        logger.info(f"📝 Created {len(nodes)} extra large nodes")
        return nodes
    
    def _create_granular_nodes(self, text: str) -> List[DynamicNode]:
        """Create additional granular nodes for complex content"""
        nodes = []
        
        # Create word-level nodes for complex content
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        complex_words = [w for w in words if len(w) > 5]  # Longer words are more complex
        
        for word in complex_words[:20]:  # Limit to top 20 complex words
            node_id = f"granular_{hashlib.md5(word.encode()).hexdigest()[:8]}"
            
            node = DynamicNode(
                id=node_id,
                content=word,
                node_type="complex_word",
                size_category="tiny",
                content_length=len(word),
                complexity_score=1.0,  # High complexity
                metadata={
                    'complexity_type': 'long_word',
                    'word_length': len(word)
                }
            )
            
            nodes.append(node)
        
        logger.info(f"📝 Created {len(nodes)} granular nodes for complex content")
        return nodes
    
    def _extract_phrases(self, sentence: str) -> List[str]:
        """Extract meaningful phrases from sentence"""
        phrases = []
        words = sentence.split()
        
        # Extract 2-3 word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if self._is_meaningful_phrase(phrase):
                phrases.append(phrase)
        
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if self._is_meaningful_phrase(phrase):
                phrases.append(phrase)
        
        return list(set(phrases))
    
    def _split_into_chunks(self, text: str, target_size: int) -> List[str]:
        """Split text into chunks of target size"""
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) <= target_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if phrase is meaningful"""
        words = phrase.split()
        if len(words) < 2:
            return False
        
        # Check if all words are substantial
        return all(len(word) > 2 for word in words)
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate overall text complexity"""
        # Simple complexity metrics
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / max(1, len(words))
        
        # Combine metrics
        complexity = (avg_word_length * 0.4) + (vocabulary_diversity * 0.6)
        return min(1.0, complexity)
    
    def _calculate_word_complexity(self, word: str) -> float:
        """Calculate word complexity"""
        return min(1.0, len(word) / 10.0)
    
    def _calculate_phrase_complexity(self, phrase: str) -> float:
        """Calculate phrase complexity"""
        words = phrase.split()
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        return min(1.0, avg_word_length / 8.0)
    
    def _calculate_concept_complexity(self, concept: str) -> float:
        """Calculate concept complexity"""
        return self._calculate_complexity(concept)
    
    def _calculate_section_complexity(self, section: str) -> float:
        """Calculate section complexity"""
        return self._calculate_complexity(section)
    
    def _calculate_document_complexity(self, document: str) -> float:
        """Calculate document complexity"""
        return self._calculate_complexity(document)
    
    def _add_nodes_to_brain(self, nodes: List[DynamicNode]):
        """Add nodes to brain and create connections"""
        for node in nodes:
            # Add to brain
            brain_node = self.brain.process_text_input(
                node.content, 
                node.id
            )
            
            # Create connections based on node size
            self._create_size_appropriate_connections(node)
            
            self.stats['total_nodes_created'] += 1
        
        logger.info(f"🧠 Added {len(nodes)} dynamic nodes to brain")
    
    def _create_size_appropriate_connections(self, node: DynamicNode):
        """Create connections appropriate for the node size"""
        size_config = self.node_sizes[node.size_category]
        
        if size_config.connection_strategy == 'similarity':
            self._create_similarity_connections(node, size_config)
        elif size_config.connection_strategy == 'hierarchical':
            self._create_hierarchical_connections(node, size_config)
        elif size_config.connection_strategy == 'temporal':
            self._create_temporal_connections(node, size_config)
        elif size_config.connection_strategy == 'all':
            self._create_all_connections(node, size_config)
    
    def _create_similarity_connections(self, node: DynamicNode, size_config: NodeSize):
        """Create similarity-based connections"""
        # This would connect to similar nodes based on content similarity
        # Implementation depends on brain's connection methods
        pass
    
    def _create_hierarchical_connections(self, node: DynamicNode, size_config: NodeSize):
        """Create hierarchical connections"""
        # This would connect to parent/child nodes
        pass
    
    def _create_temporal_connections(self, node: DynamicNode, size_config: NodeSize):
        """Create temporal connections"""
        # This would connect to recently created nodes
        pass
    
    def _create_all_connections(self, node: DynamicNode, size_config: NodeSize):
        """Create all types of connections"""
        # This would create similarity, hierarchical, and temporal connections
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get creation statistics"""
        return {
            'total_nodes_created': self.stats['total_nodes_created'],
            'tiny_nodes': self.stats['tiny_nodes'],
            'small_nodes': self.stats['small_nodes'],
            'medium_nodes': self.stats['medium_nodes'],
            'large_nodes': self.stats['large_nodes'],
            'extra_large_nodes': self.stats['extra_large_nodes'],
            'connections_created': self.stats['connections_created'],
            'size_distribution': {
                'tiny': self.stats['tiny_nodes'],
                'small': self.stats['small_nodes'],
                'medium': self.stats['medium_nodes'],
                'large': self.stats['large_nodes'],
                'extra_large': self.stats['extra_large_nodes']
            }
        }

def main():
    """Main entry point for dynamic node sizing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Dynamic Nodes for Melvin Brain")
    parser.add_argument('--text', type=str, required=True, 
                       help='Text to create dynamic nodes from')
    parser.add_argument('--size', type=str, default='auto',
                       choices=['auto', 'tiny', 'small', 'medium', 'large', 'extra_large'],
                       help='Preferred node size (auto for automatic sizing)')
    parser.add_argument('--complexity-threshold', type=float, default=0.5,
                       help='Complexity threshold for additional granular nodes')
    
    args = parser.parse_args()
    
    print("🧠 DYNAMIC NODE SIZING SYSTEM")
    print("=" * 50)
    print(f"🔹 Text: {args.text[:50]}...")
    print(f"🔹 Size: {args.size}")
    print(f"🔹 Complexity threshold: {args.complexity_threshold}")
    print("=" * 50)
    
    # Initialize sizer
    sizer = DynamicNodeSizer()
    
    try:
        # Create dynamic nodes
        nodes = sizer.create_dynamic_nodes(
            args.text, 
            preferred_size=args.size,
            complexity_threshold=args.complexity_threshold
        )
        
        # Display results
        print("\n🎉 DYNAMIC NODE CREATION COMPLETE!")
        print("=" * 40)
        
        stats = sizer.get_statistics()
        print(f"📊 Total nodes created: {stats['total_nodes_created']}")
        print(f"🔹 Tiny nodes: {stats['tiny_nodes']}")
        print(f"🔹 Small nodes: {stats['small_nodes']}")
        print(f"🔹 Medium nodes: {stats['medium_nodes']}")
        print(f"🔹 Large nodes: {stats['large_nodes']}")
        print(f"🔹 Extra large nodes: {stats['extra_large_nodes']}")
        print(f"🔗 Total connections: {stats['connections_created']}")
        
        # Show size distribution
        print(f"\n📈 SIZE DISTRIBUTION:")
        for size, count in stats['size_distribution'].items():
            if count > 0:
                percentage = (count / stats['total_nodes_created']) * 100
                print(f"   {size}: {count} nodes ({percentage:.1f}%)")
        
        # Show sample nodes
        print(f"\n📋 SAMPLE NODES CREATED:")
        for i, node in enumerate(nodes[:10]):
            print(f"  {i+1}. {node.id}: {node.content[:30]}... ({node.size_category}, complexity: {node.complexity_score:.2f})")
        
        if len(nodes) > 10:
            print(f"  ... and {len(nodes) - 10} more nodes")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating dynamic nodes: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

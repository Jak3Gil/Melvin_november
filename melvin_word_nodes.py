#!/usr/bin/env python3
"""
🔤 MELVIN WORD-BASED NODE ARCHITECTURE
Creates tiny word nodes that connect to larger concept nodes through shared bytes
"""

import sqlite3
import time
import json
import re
from typing import Set, List, Dict, Tuple
from collections import defaultdict, Counter

class MelvinWordNodes:
    def __init__(self):
        self.conn = sqlite3.connect('melvin_global_memory/global_memory.db')
        self.cursor = self.conn.cursor()
        
        # Common words to skip (too frequent, not meaningful)
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'a', 'an', 'as',
            'if', 'then', 'than', 'when', 'where', 'why', 'how', 'what', 'which',
            'who', 'whom', 'whose', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'also', 'just', 'now', 'very', 'too', 'well'
        }
        
        print("🔤 MELVIN WORD-BASED NODE ARCHITECTURE")
        print("=" * 50)
    
    def extract_meaningful_words(self, text: str) -> Set[str]:
        """Extract meaningful words from text (3+ chars, not stop words)"""
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words and very common words
        meaningful_words = set()
        for word in words:
            if word not in self.stop_words and len(word) >= 3:
                meaningful_words.add(word)
        
        return meaningful_words
    
    def create_word_node(self, word: str) -> int:
        """Create or get a word node (tiny 3-10 byte nodes)"""
        # Check if word node already exists
        self.cursor.execute("""
            SELECT rowid FROM nodes 
            WHERE node_type = 'word' AND content = ?
        """, (word,))
        
        existing = self.cursor.fetchone()
        if existing:
            return existing[0]
        
        # Create new word node (minimal storage)
        try:
            self.cursor.execute("""
                INSERT INTO nodes 
                (node_type, content, embedding, activation_strength, firing_rate, 
                 last_activation, activation_count, creation_time, metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'word', word, '[]', 0.1, 0.0, time.time(), 0, time.time(),
                json.dumps({"word_length": len(word), "type": "atomic_word"}),
                'word_extraction'
            ))
            return self.cursor.lastrowid
        except Exception as e:
            print(f"Error creating word node '{word}': {e}")
            return None
    
    def connect_word_to_concept(self, word_node_id: int, concept_node_id: int, word: str) -> bool:
        """Create lightweight connection between word and concept node"""
        if not word_node_id or not concept_node_id:
            return False
        
        try:
            # Minimal edge ID (just numbers and word)
            edge_id = f"w{word_node_id}c{concept_node_id}"
            
            self.cursor.execute("""
                INSERT OR IGNORE INTO edges 
                (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                 last_coactivation, creation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge_id, word_node_id, concept_node_id, 'contains_word',
                1.0, 1, time.time(), time.time()
            ))
            return True
        except Exception as e:
            print(f"Error connecting word '{word}' to concept: {e}")
            return False
    
    def connect_shared_words(self, word_node_id: int, concept_nodes: List[int]) -> int:
        """Connect word node to all concept nodes that contain it"""
        connections_made = 0
        
        for concept_id in concept_nodes:
            if self.connect_word_to_concept(word_node_id, concept_id, "shared"):
                connections_made += 1
        
        return connections_made
    
    def decompose_concepts_to_words(self, batch_size: int = 100) -> Dict[str, int]:
        """Break down concept nodes into word nodes with connections"""
        
        # Get recent concept nodes
        self.cursor.execute("""
            SELECT rowid, content, node_type FROM nodes 
            WHERE modality_source = 'educational_feed'
            AND LENGTH(content) > 10
            ORDER BY creation_time DESC 
            LIMIT ?
        """, (batch_size,))
        
        concept_nodes = self.cursor.fetchall()
        
        if not concept_nodes:
            return {"error": "No concept nodes found"}
        
        print(f"🔍 Decomposing {len(concept_nodes)} concept nodes into words...")
        
        # Track word occurrences across concepts
        word_to_concepts = defaultdict(list)
        word_nodes_created = 0
        connections_made = 0
        
        # First pass: extract all words and map to concepts
        for concept_id, content, node_type in concept_nodes:
            words = self.extract_meaningful_words(content)
            
            for word in words:
                word_to_concepts[word].append(concept_id)
        
        print(f"📝 Found {len(word_to_concepts)} unique meaningful words")
        
        # Second pass: create word nodes and connections
        for word, concept_list in word_to_concepts.items():
            # Only create word nodes for words that appear in multiple concepts
            # (shared words are more valuable for connections)
            if len(concept_list) >= 2:
                word_node_id = self.create_word_node(word)
                
                if word_node_id:
                    word_nodes_created += 1
                    
                    # Connect word to all concepts containing it
                    for concept_id in concept_list:
                        if self.connect_word_to_concept(word_node_id, concept_id, word):
                            connections_made += 1
        
        return {
            "word_nodes_created": word_nodes_created,
            "connections_made": connections_made,
            "unique_words": len(word_to_concepts),
            "concepts_processed": len(concept_nodes)
        }
    
    def create_word_to_word_connections(self) -> int:
        """Connect words that frequently appear together"""
        connections_made = 0
        
        # Get all word nodes
        self.cursor.execute("""
            SELECT rowid, content FROM nodes 
            WHERE node_type = 'word'
            ORDER BY activation_count DESC
            LIMIT 200
        """)
        
        word_nodes = self.cursor.fetchall()
        print(f"🔗 Creating word-to-word connections for {len(word_nodes)} words...")
        
        # Find words that co-occur in the same concepts
        word_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for word1_id, word1 in word_nodes:
            # Get concepts containing this word
            self.cursor.execute("""
                SELECT target_id FROM edges 
                WHERE source_id = ? AND edge_type = 'contains_word'
            """, (word1_id,))
            
            concepts1 = [row[0] for row in self.cursor.fetchall()]
            
            for word2_id, word2 in word_nodes:
                if word1_id >= word2_id:  # Avoid duplicates
                    continue
                
                # Get concepts containing second word
                self.cursor.execute("""
                    SELECT target_id FROM edges 
                    WHERE source_id = ? AND edge_type = 'contains_word'
                """, (word2_id,))
                
                concepts2 = [row[0] for row in self.cursor.fetchall()]
                
                # Count shared concepts
                shared_concepts = len(set(concepts1) & set(concepts2))
                
                if shared_concepts >= 2:  # Words appear together in 2+ concepts
                    try:
                        edge_id = f"ww{word1_id}_{word2_id}"
                        weight = min(shared_concepts / 10.0, 1.0)  # Normalize weight
                        
                        self.cursor.execute("""
                            INSERT OR IGNORE INTO edges 
                            (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                             last_coactivation, creation_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            edge_id, word1_id, word2_id, 'word_cooccurrence',
                            weight, shared_concepts, time.time(), time.time()
                        ))
                        
                        connections_made += 1
                        
                    except Exception as e:
                        continue
        
        return connections_made
    
    def analyze_word_network(self) -> Dict[str, any]:
        """Analyze the efficiency of the word-based network"""
        
        # Count different node types
        self.cursor.execute("""
            SELECT node_type, COUNT(*), AVG(LENGTH(content)) 
            FROM nodes 
            GROUP BY node_type
        """)
        node_stats = self.cursor.fetchall()
        
        # Count different edge types
        self.cursor.execute("""
            SELECT edge_type, COUNT(*), AVG(LENGTH(edge_id))
            FROM edges 
            GROUP BY edge_type
        """)
        edge_stats = self.cursor.fetchall()
        
        # Calculate storage efficiency
        self.cursor.execute('SELECT COUNT(*) FROM nodes WHERE node_type = "word"')
        word_nodes = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM edges WHERE edge_type = "contains_word"')
        word_edges = self.cursor.fetchone()[0]
        
        return {
            "node_stats": node_stats,
            "edge_stats": edge_stats,
            "word_nodes": word_nodes,
            "word_connections": word_edges,
            "storage_efficiency": "Word nodes are ~5-10 bytes vs 50-200 bytes for concepts"
        }
    
    def run_full_word_decomposition(self) -> Dict[str, any]:
        """Run complete word-based decomposition"""
        print("🚀 RUNNING FULL WORD DECOMPOSITION")
        print("=" * 40)
        
        # Step 1: Decompose concepts to words
        decomposition_results = self.decompose_concepts_to_words(batch_size=200)
        
        # Step 2: Create word-to-word connections
        word_connections = self.create_word_to_word_connections()
        
        # Step 3: Commit changes
        self.conn.commit()
        
        # Step 4: Analyze results
        analysis = self.analyze_word_network()
        
        results = {
            **decomposition_results,
            "word_to_word_connections": word_connections,
            "analysis": analysis
        }
        
        print(f"\\n✅ WORD DECOMPOSITION COMPLETE!")
        print(f"🔤 Word nodes created: {results.get('word_nodes_created', 0)}")
        print(f"🔗 Word-concept connections: {results.get('connections_made', 0)}")
        print(f"🔗 Word-word connections: {word_connections}")
        
        return results
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    word_system = MelvinWordNodes()
    try:
        results = word_system.run_full_word_decomposition()
        
        print(f"\\n📊 FINAL WORD NETWORK STATS:")
        for node_type, count, avg_size in results['analysis']['node_stats']:
            print(f"   {node_type.capitalize()}: {count:,} nodes (avg {avg_size:.1f} bytes)")
        
        print(f"\\n🔗 CONNECTION TYPES:")
        for edge_type, count, avg_id_size in results['analysis']['edge_stats']:
            print(f"   {edge_type}: {count:,} connections (avg {avg_id_size:.1f} byte IDs)")
        
    finally:
        word_system.close()

if __name__ == "__main__":
    main()

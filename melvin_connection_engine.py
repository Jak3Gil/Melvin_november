#!/usr/bin/env python3
"""
🔗 MELVIN CONNECTION ENGINE
Creates intelligent connections between nodes using multiple strategies
"""

import sqlite3
import time
import json
import re
from typing import List, Tuple, Set, Dict
from collections import defaultdict

class MelvinConnectionEngine:
    def __init__(self):
        self.conn = sqlite3.connect('melvin_global_memory/global_memory.db')
        self.cursor = self.conn.cursor()
        
        # Key concepts for connection matching
        self.concept_keywords = {
            'computation': ['algorithm', 'compute', 'calculation', 'process', 'data', 'program', 'code'],
            'learning': ['learn', 'knowledge', 'understand', 'education', 'study', 'memory', 'intelligence'],
            'science': ['theory', 'experiment', 'research', 'discovery', 'analysis', 'method', 'principle'],
            'mathematics': ['number', 'equation', 'formula', 'calculate', 'mathematical', 'solve', 'problem'],
            'physics': ['energy', 'force', 'motion', 'particle', 'wave', 'matter', 'quantum', 'field'],
            'chemistry': ['atom', 'molecule', 'reaction', 'element', 'compound', 'bond', 'chemical'],
            'biology': ['cell', 'organism', 'life', 'evolution', 'genetic', 'species', 'biological'],
            'systems': ['system', 'network', 'structure', 'organization', 'relationship', 'interaction'],
            'technology': ['machine', 'device', 'tool', 'invention', 'innovation', 'digital', 'electronic']
        }
        
        print("🔗 MELVIN CONNECTION ENGINE")
        print("=" * 40)
    
    def extract_key_concepts(self, text: str) -> Set[str]:
        """Extract key concepts from text"""
        text_lower = text.lower()
        concepts = set()
        
        # Find concept categories
        for category, keywords in self.concept_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    concepts.add(category)
                    concepts.add(keyword)
        
        # Extract important nouns (simple heuristic)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)
        
        # Filter out common words
        stop_words = {'that', 'with', 'from', 'they', 'have', 'this', 'will', 'been', 'their', 'said', 'each', 'which', 'what', 'there', 'when', 'more', 'very', 'about', 'other', 'many', 'some', 'time', 'into', 'only', 'than', 'also', 'then', 'them', 'these', 'two', 'may', 'first', 'after', 'back', 'other', 'many', 'where', 'much', 'should', 'well', 'without', 'through', 'being', 'during', 'before', 'under', 'while', 'above', 'between', 'both', 'since', 'until', 'such', 'because', 'most', 'every', 'same', 'different', 'following', 'around', 'however', 'within', 'along', 'among', 'across', 'against', 'instead', 'another', 'several', 'including', 'according', 'although', 'whether', 'example', 'general', 'particular', 'especially', 'therefore', 'sometimes', 'usually', 'often', 'always', 'never', 'still', 'already', 'yet', 'again', 'further', 'once', 'together', 'possible', 'important', 'large', 'small', 'great', 'high', 'low', 'long', 'short', 'good', 'better', 'best', 'bad', 'worse', 'worst', 'old', 'new', 'young', 'early', 'late', 'right', 'wrong', 'true', 'false', 'real', 'sure', 'clear', 'full', 'free', 'open', 'close', 'hard', 'easy', 'strong', 'weak', 'light', 'dark', 'fast', 'slow', 'hot', 'cold', 'warm', 'cool'}
        
        for word in words:
            if len(word) >= 4 and word not in stop_words:
                concepts.add(word)
        
        return concepts
    
    def calculate_similarity(self, concepts1: Set[str], concepts2: Set[str]) -> float:
        """Calculate similarity score between two concept sets"""
        if not concepts1 or not concepts2:
            return 0.0
        
        intersection = concepts1 & concepts2
        union = concepts1 | concepts2
        
        # Jaccard similarity with bonuses for important matches
        base_similarity = len(intersection) / len(union)
        
        # Bonus for concept category matches
        category_bonus = 0.0
        for category in self.concept_keywords.keys():
            if category in intersection:
                category_bonus += 0.2
        
        return min(base_similarity + category_bonus, 1.0)
    
    def create_conceptual_connections(self, batch_size: int = 50) -> int:
        """Create connections between conceptually similar nodes"""
        connections_created = 0
        
        # Get recent nodes
        self.cursor.execute("""
            SELECT rowid, content, node_type FROM nodes 
            WHERE modality_source = 'educational_feed'
            ORDER BY creation_time DESC 
            LIMIT ?
        """, (batch_size,))
        
        recent_nodes = self.cursor.fetchall()
        print(f"🔍 Analyzing {len(recent_nodes)} recent nodes for connections...")
        
        # Analyze each pair
        for i in range(len(recent_nodes)):
            for j in range(i + 1, len(recent_nodes)):
                node1_id, content1, type1 = recent_nodes[i]
                node2_id, content2, type2 = recent_nodes[j]
                
                # Extract concepts
                concepts1 = self.extract_key_concepts(content1)
                concepts2 = self.extract_key_concepts(content2)
                
                # Calculate similarity
                similarity = self.calculate_similarity(concepts1, concepts2)
                
                if similarity > 0.15:  # Lower threshold for more connections
                    try:
                        edge_id = f"conceptual_{node1_id}_{node2_id}"
                        
                        self.cursor.execute("""
                            INSERT OR IGNORE INTO edges 
                            (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                             last_coactivation, creation_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            edge_id, node1_id, node2_id, 'conceptual_similarity',
                            similarity, 1, time.time(), time.time()
                        ))
                        
                        if self.cursor.rowcount > 0:
                            connections_created += 1
                            print(f"   🔗 Connected nodes {node1_id}-{node2_id} (similarity: {similarity:.2f})")
                    
                    except Exception as e:
                        continue
        
        return connections_created
    
    def create_domain_connections(self) -> int:
        """Create connections between nodes in the same domain"""
        connections_created = 0
        
        # Group nodes by domain
        domains = ['computer_science', 'mathematics', 'physics', 'chemistry', 'biology', 'history', 'philosophy']
        
        for domain in domains:
            self.cursor.execute("""
                SELECT rowid, content FROM nodes 
                WHERE metadata LIKE ?
                AND modality_source = 'educational_feed'
                ORDER BY creation_time DESC 
                LIMIT 20
            """, (f'%{domain}%',))
            
            domain_nodes = self.cursor.fetchall()
            
            if len(domain_nodes) < 2:
                continue
            
            print(f"📚 Creating {domain} domain connections...")
            
            # Connect nodes within the same domain
            for i in range(len(domain_nodes)):
                for j in range(i + 1, min(i + 4, len(domain_nodes))):  # Connect to next 3 nodes
                    node1_id, content1 = domain_nodes[i]
                    node2_id, content2 = domain_nodes[j]
                    
                    try:
                        edge_id = f"domain_{domain}_{node1_id}_{node2_id}"
                        
                        self.cursor.execute("""
                            INSERT OR IGNORE INTO edges 
                            (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                             last_coactivation, creation_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            edge_id, node1_id, node2_id, 'domain_relationship',
                            0.7, 1, time.time(), time.time()
                        ))
                        
                        if self.cursor.rowcount > 0:
                            connections_created += 1
                    
                    except Exception as e:
                        continue
        
        return connections_created
    
    def create_sequential_connections(self) -> int:
        """Create connections between sequentially added nodes"""
        connections_created = 0
        
        # Get the most recent nodes
        self.cursor.execute("""
            SELECT rowid, content FROM nodes 
            WHERE modality_source = 'educational_feed'
            ORDER BY creation_time DESC 
            LIMIT 30
        """)
        
        recent_nodes = self.cursor.fetchall()
        
        print(f"⏭️ Creating sequential connections...")
        
        # Connect each node to the next few nodes (temporal locality)
        for i in range(len(recent_nodes) - 1):
            for j in range(i + 1, min(i + 4, len(recent_nodes))):
                node1_id, content1 = recent_nodes[i]
                node2_id, content2 = recent_nodes[j]
                
                try:
                    edge_id = f"sequential_{node1_id}_{node2_id}"
                    
                    # Weight decreases with distance
                    weight = 0.5 / (j - i)
                    
                    self.cursor.execute("""
                        INSERT OR IGNORE INTO edges 
                        (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                         last_coactivation, creation_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        edge_id, node1_id, node2_id, 'temporal_sequence',
                        weight, 1, time.time(), time.time()
                    ))
                    
                    if self.cursor.rowcount > 0:
                        connections_created += 1
                
                except Exception as e:
                    continue
        
        return connections_created
    
    def create_keyword_connections(self) -> int:
        """Create connections based on shared important keywords"""
        connections_created = 0
        
        # Build keyword index
        keyword_nodes = defaultdict(list)
        
        self.cursor.execute("""
            SELECT rowid, content FROM nodes 
            WHERE modality_source = 'educational_feed'
            ORDER BY creation_time DESC 
            LIMIT 100
        """)
        
        nodes = self.cursor.fetchall()
        
        print(f"🔑 Building keyword connections...")
        
        # Index nodes by keywords
        for node_id, content in nodes:
            concepts = self.extract_key_concepts(content)
            for concept in concepts:
                if len(concept) > 3:  # Skip very short words
                    keyword_nodes[concept].append(node_id)
        
        # Create connections between nodes sharing important keywords
        for keyword, node_list in keyword_nodes.items():
            if len(node_list) > 1 and len(node_list) < 20:  # Not too common, not too rare
                for i in range(len(node_list)):
                    for j in range(i + 1, min(i + 5, len(node_list))):
                        node1_id = node_list[i]
                        node2_id = node_list[j]
                        
                        try:
                            edge_id = f"keyword_{keyword}_{node1_id}_{node2_id}"
                            
                            self.cursor.execute("""
                                INSERT OR IGNORE INTO edges 
                                (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                                 last_coactivation, creation_time)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                edge_id, node1_id, node2_id, 'keyword_similarity',
                                0.6, 1, time.time(), time.time()
                            ))
                            
                            if self.cursor.rowcount > 0:
                                connections_created += 1
                        
                        except Exception as e:
                            continue
        
        return connections_created
    
    def run_full_connection_pass(self) -> Dict[str, int]:
        """Run all connection strategies"""
        results = {}
        
        print("🚀 RUNNING FULL CONNECTION PASS")
        print("=" * 40)
        
        # Strategy 1: Conceptual similarity
        results['conceptual'] = self.create_conceptual_connections()
        
        # Strategy 2: Domain relationships
        results['domain'] = self.create_domain_connections()
        
        # Strategy 3: Sequential connections
        results['sequential'] = self.create_sequential_connections()
        
        # Strategy 4: Keyword connections
        results['keyword'] = self.create_keyword_connections()
        
        # Commit all changes
        self.conn.commit()
        
        total_connections = sum(results.values())
        print(f"\n✅ CONNECTION PASS COMPLETE!")
        print(f"🔗 Total connections created: {total_connections}")
        for strategy, count in results.items():
            print(f"   {strategy.capitalize()}: {count}")
        
        return results
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    engine = MelvinConnectionEngine()
    try:
        results = engine.run_full_connection_pass()
        
        # Show final stats
        engine.cursor.execute('SELECT COUNT(*) FROM edges')
        total_edges = engine.cursor.fetchone()[0]
        
        engine.cursor.execute('SELECT COUNT(*) FROM nodes')
        total_nodes = engine.cursor.fetchone()[0]
        
        print(f"\n📊 FINAL BRAIN STATS:")
        print(f"🧠 Total nodes: {total_nodes:,}")
        print(f"🔗 Total connections: {total_edges:,}")
        print(f"📈 Connections per node: {total_edges/total_nodes:.1f}")
        
    finally:
        engine.close()

if __name__ == "__main__":
    main()

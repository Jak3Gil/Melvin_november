#!/usr/bin/env python3
"""
🧠 MELVIN NODE OPTIMIZER
Break large nodes into smaller, more connected atomic facts
"""

import sqlite3
import json
import re
from typing import List, Dict, Any, Tuple
import hashlib

class MelvinNodeOptimizer:
    """Optimize Melvin's brain by creating smaller, more connected nodes"""
    
    def __init__(self, db_path: str = "melvin_global_memory/global_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.new_nodes = []
        self.new_connections = []
        
    def analyze_current_structure(self):
        """Analyze current node structure"""
        print("🔍 ANALYZING CURRENT BRAIN STRUCTURE")
        print("=" * 50)
        
        # Large nodes that need breaking down
        self.cursor.execute("""
            SELECT node_id, node_type, content, LENGTH(content) as size 
            FROM nodes 
            WHERE LENGTH(content) > 200 
            ORDER BY size DESC 
            LIMIT 20
        """)
        
        large_nodes = self.cursor.fetchall()
        
        print(f"📊 Found {len(large_nodes)} nodes > 200 bytes that need optimization")
        
        total_chars = sum(size for _, _, _, size in large_nodes)
        print(f"💾 Total content to optimize: {total_chars:,} characters")
        
        return large_nodes
    
    def extract_atomic_facts(self, content: str, node_type: str) -> List[str]:
        """Break content into atomic facts"""
        facts = []
        
        if node_type == 'language':
            # Split by sentences
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and len(sentence) < 150:  # Good atomic size
                    # Clean up the sentence
                    sentence = re.sub(r'<br\s*/?>|<[^>]+>', '', sentence)  # Remove HTML
                    sentence = re.sub(r'\s+', ' ', sentence)  # Normalize whitespace
                    if sentence and not sentence.startswith('http'):
                        facts.append(sentence)
        
        elif node_type == 'code':
            # Split by lines for code
            lines = content.split('\n')
            current_block = []
            
            for line in lines:
                line = line.strip()
                if line:
                    current_block.append(line)
                    
                    # Create blocks for functions, classes, etc.
                    if (line.endswith(':') or 
                        line.startswith('def ') or 
                        line.startswith('class ') or
                        len(current_block) >= 3):
                        
                        block_content = '\n'.join(current_block)
                        if len(block_content) < 200:
                            facts.append(block_content)
                        current_block = []
            
            # Add remaining block
            if current_block:
                block_content = '\n'.join(current_block)
                if len(block_content) < 200:
                    facts.append(block_content)
        
        else:
            # For other types, try to split intelligently
            if len(content) > 100:
                # Split by commas or semicolons for structured data
                parts = re.split(r'[,;]', content)
                for part in parts:
                    part = part.strip()
                    if len(part) > 5 and len(part) < 100:
                        facts.append(part)
            else:
                facts.append(content)  # Keep as is if already small
        
        return facts
    
    def create_optimized_nodes(self, large_nodes: List[Tuple]) -> Dict[str, Any]:
        """Create smaller, more connected nodes"""
        print("\n🔧 CREATING OPTIMIZED NODES")
        print("-" * 30)
        
        stats = {
            'original_nodes': len(large_nodes),
            'new_nodes_created': 0,
            'connections_created': 0,
            'bytes_saved': 0
        }
        
        for original_id, node_type, content, size in large_nodes:
            print(f"Processing {node_type} node ({size} bytes)...")
            
            # Extract atomic facts
            facts = self.extract_atomic_facts(content, node_type)
            
            if len(facts) <= 1:
                continue  # Skip if can't be broken down
            
            print(f"  → Broke into {len(facts)} atomic facts")
            
            # Create new nodes for each fact
            new_node_ids = []
            for i, fact in enumerate(facts):
                # Create unique node ID
                fact_hash = hashlib.md5(fact.encode()).hexdigest()[:12]
                new_node_id = f"{node_type}_atomic_{fact_hash}"
                
                # Store new node
                self.new_nodes.append({
                    'node_id': new_node_id,
                    'node_type': node_type,
                    'content': fact,
                    'original_parent': original_id,
                    'atomic_index': i
                })
                
                new_node_ids.append(new_node_id)
                stats['new_nodes_created'] += 1
            
            # Create connections between related facts
            for i, node_id_1 in enumerate(new_node_ids):
                for j, node_id_2 in enumerate(new_node_ids):
                    if i != j:
                        # Connect related facts with stronger weights for closer facts
                        weight = 1.0 / (abs(i - j) + 1)  # Closer facts have higher weight
                        
                        connection_id = f"atomic_{hashlib.md5(f'{node_id_1}_{node_id_2}'.encode()).hexdigest()[:12]}"
                        
                        self.new_connections.append({
                            'edge_id': connection_id,
                            'source_id': node_id_1,
                            'target_id': node_id_2,
                            'edge_type': 'atomic_relation',
                            'weight': weight,
                            'coactivation_count': 1,
                            'creation_time': 1756918000.0
                        })
                        
                        stats['connections_created'] += 1
            
            stats['bytes_saved'] += size - sum(len(fact) for fact in facts)
        
        return stats
    
    def save_optimized_brain(self):
        """Save the optimized nodes and connections"""
        print("\n💾 SAVING OPTIMIZED BRAIN")
        print("-" * 30)
        
        # Insert new nodes
        for node in self.new_nodes:
            try:
                self.cursor.execute("""
                    INSERT OR REPLACE INTO nodes 
                    (node_id, node_type, content, activation, embedding, creation_time)
                    VALUES (?, ?, ?, 0.5, '[]', ?)
                """, (
                    node['node_id'], 
                    node['node_type'], 
                    node['content'],
                    1756918000.0
                ))
            except Exception as e:
                print(f"Error inserting node: {e}")
        
        # Insert new connections
        for conn in self.new_connections:
            try:
                self.cursor.execute("""
                    INSERT OR REPLACE INTO edges 
                    (edge_id, source_id, target_id, edge_type, weight, coactivation_count, last_coactivation, creation_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conn['edge_id'],
                    conn['source_id'],
                    conn['target_id'],
                    conn['edge_type'],
                    conn['weight'],
                    conn['coactivation_count'],
                    1756918000.0,
                    conn['creation_time']
                ))
            except Exception as e:
                print(f"Error inserting connection: {e}")
        
        self.conn.commit()
        print(f"✅ Saved {len(self.new_nodes)} new nodes and {len(self.new_connections)} connections")
    
    def optimize_brain(self):
        """Run the full optimization process"""
        print("🚀 MELVIN BRAIN OPTIMIZATION")
        print("=" * 60)
        
        # Analyze current structure
        large_nodes = self.analyze_current_structure()
        
        if not large_nodes:
            print("✅ Brain is already well-optimized!")
            return
        
        # Create optimized nodes
        stats = self.create_optimized_nodes(large_nodes[:10])  # Process first 10 for demo
        
        # Save to database
        self.save_optimized_brain()
        
        # Show results
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print("=" * 30)
        print(f"📊 Original large nodes: {stats['original_nodes']}")
        print(f"🔧 New atomic nodes created: {stats['new_nodes_created']}")
        print(f"🔗 New connections created: {stats['connections_created']}")
        print(f"💾 Bytes optimized: {stats['bytes_saved']:,}")
        
        # Test the improvement
        self.test_improved_responses()
    
    def test_improved_responses(self):
        """Test how the optimization improves responses"""
        print(f"\n🧪 TESTING IMPROVED RESPONSES")
        print("-" * 30)
        
        test_queries = ["machine learning", "artificial intelligence", "computer"]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Search atomic nodes
            self.cursor.execute("""
                SELECT node_type, content FROM nodes 
                WHERE node_id LIKE '%atomic%' AND LOWER(content) LIKE ? 
                LIMIT 3
            """, (f'%{query}%',))
            
            results = self.cursor.fetchall()
            
            if results:
                print("✅ Atomic facts found:")
                for node_type, content in results:
                    print(f"   • {content}")
            else:
                print("❌ No atomic facts found")
    
    def close(self):
        """Close database connection"""
        self.conn.close()

if __name__ == "__main__":
    optimizer = MelvinNodeOptimizer()
    try:
        optimizer.optimize_brain()
    finally:
        optimizer.close()

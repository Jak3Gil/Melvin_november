#!/usr/bin/env python3
"""
🧠 MELVIN DYNAMIC NODE SYSTEM
Adaptive node sizing based on success patterns and usage
"""

import sqlite3
import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class NodeStrategy(Enum):
    FRAGMENT = "fragment"      # Break into smaller pieces
    CONSOLIDATE = "consolidate"  # Merge successful chains
    MAINTAIN = "maintain"      # Keep current size
    SPECIALIZE = "specialize"  # Create focused variants

@dataclass
class NodeMetrics:
    """Track node performance metrics"""
    node_id: str
    access_count: int = 0
    success_rate: float = 0.0
    connection_strength: float = 0.0
    last_accessed: float = 0.0
    chain_success: int = 0  # How often this node leads to successful chains
    optimal_size: Optional[int] = None

class MelvinDynamicNodes:
    """Dynamic node management system"""
    
    def __init__(self, db_path: str = "melvin_global_memory/global_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Node size constraints
        self.MIN_NODE_SIZE = 1      # Single character (like punctuation, operators)
        self.MAX_NODE_SIZE = 2000   # Complex concepts or successful chains
        self.OPTIMAL_SMALL = 50     # Sweet spot for atomic facts
        self.OPTIMAL_MEDIUM = 200   # Good for concepts
        self.OPTIMAL_LARGE = 800    # For successful knowledge chains
        
        # Performance thresholds
        self.SUCCESS_THRESHOLD = 0.7    # When to consolidate
        self.FRAGMENT_THRESHOLD = 0.3   # When to break apart
        self.MIN_CHAIN_SUCCESS = 5      # Minimum successful chains to consolidate
        
        print("🧠 Melvin Dynamic Node System Initialized")
        self.analyze_current_performance()
    
    def analyze_current_performance(self):
        """Analyze current node performance patterns"""
        print("\n📊 ANALYZING NODE PERFORMANCE")
        print("-" * 40)
        
        # Get nodes with their usage patterns
        self.cursor.execute("""
            SELECT 
                n.node_id,
                n.node_type,
                LENGTH(n.content) as size,
                n.content,
                COUNT(e.edge_id) as connections,
                n.activation_count,
                n.last_activation
            FROM nodes n
            LEFT JOIN edges e ON n.node_id = e.source_id OR n.node_id = e.target_id
            GROUP BY n.node_id
            ORDER BY connections DESC
            LIMIT 20
        """)
        
        top_nodes = self.cursor.fetchall()
        
        print("🌟 Top Connected Nodes:")
        for node_id, node_type, size, content, connections, activations, last_active in top_nodes[:5]:
            content_preview = content[:40] + "..." if len(content) > 40 else content
            strategy = self.determine_strategy(size, connections, activations or 0)
            print(f"   {size:3d}b | {connections:3d} conn | {strategy.value:11s} | {content_preview}")
    
    def determine_strategy(self, size: int, connections: int, activations: int) -> NodeStrategy:
        """Determine what to do with a node based on its metrics"""
        
        # Calculate success indicators
        connection_density = connections / max(size, 1)  # connections per byte
        usage_frequency = activations
        
        # Very successful small nodes - consider consolidating
        if (size <= self.OPTIMAL_SMALL and 
            connections > 200 and 
            usage_frequency > 10):
            return NodeStrategy.CONSOLIDATE
        
        # Large nodes with poor performance - fragment
        elif (size > self.OPTIMAL_MEDIUM and 
              connection_density < 1.0 and 
              usage_frequency < 5):
            return NodeStrategy.FRAGMENT
        
        # Medium nodes performing well - maintain
        elif (self.OPTIMAL_SMALL <= size <= self.OPTIMAL_MEDIUM and 
              connection_density > 2.0):
            return NodeStrategy.MAINTAIN
        
        # High-performing nodes - create specialized variants
        elif (connections > 100 and usage_frequency > 20):
            return NodeStrategy.SPECIALIZE
        
        else:
            return NodeStrategy.MAINTAIN
    
    def fragment_large_node(self, node_id: str, content: str) -> List[str]:
        """Break a large node into optimal smaller pieces"""
        print(f"🔧 Fragmenting large node: {node_id}")
        
        fragments = []
        
        # Strategy 1: Sentence-based fragmentation
        import re
        sentences = re.split(r'[.!?]+', content)
        
        current_fragment = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Would adding this sentence exceed optimal size?
            potential_size = len(current_fragment) + len(sentence) + 1
            
            if potential_size <= self.OPTIMAL_SMALL:
                current_fragment += sentence + ". "
            else:
                # Save current fragment if it has content
                if current_fragment.strip():
                    fragments.append(current_fragment.strip())
                current_fragment = sentence + ". "
        
        # Add final fragment
        if current_fragment.strip():
            fragments.append(current_fragment.strip())
        
        # Strategy 2: If sentences are too long, split by concepts
        if not fragments or max(len(f) for f in fragments) > self.OPTIMAL_MEDIUM:
            # Split by commas, semicolons, or conjunctions
            concept_splits = re.split(r'[,;]|\\band\\b|\\bor\\b|\\bbut\\b', content)
            fragments = [s.strip() for s in concept_splits if len(s.strip()) > 5]
        
        return [f for f in fragments if self.MIN_NODE_SIZE <= len(f) <= self.OPTIMAL_MEDIUM]
    
    def consolidate_successful_chain(self, node_ids: List[str]) -> str:
        """Consolidate a successful chain of nodes into a larger, more efficient node"""
        print(f"🔗 Consolidating successful chain: {len(node_ids)} nodes")
        
        # Get content from all nodes in the chain
        contents = []
        for node_id in node_ids:
            self.cursor.execute("SELECT content FROM nodes WHERE node_id = ?", (node_id,))
            result = self.cursor.fetchone()
            if result:
                contents.append(result[0])
        
        # Create consolidated content
        consolidated_content = " → ".join(contents)
        
        # Ensure it doesn't exceed max size
        if len(consolidated_content) > self.MAX_NODE_SIZE:
            # Truncate intelligently
            consolidated_content = consolidated_content[:self.MAX_NODE_SIZE-3] + "..."
        
        return consolidated_content
    
    def create_specialized_variant(self, base_node_id: str, specialization_context: str) -> str:
        """Create a specialized variant of a successful node"""
        print(f"⚡ Creating specialized variant of {base_node_id}")
        
        self.cursor.execute("SELECT content, node_type FROM nodes WHERE node_id = ?", (base_node_id,))
        result = self.cursor.fetchone()
        
        if not result:
            return ""
        
        base_content, node_type = result
        
        # Create specialized content based on context
        specialized_content = f"{base_content} [{specialization_context}]"
        
        return specialized_content
    
    def optimize_node_sizes(self, max_operations: int = 50):
        """Run dynamic optimization on nodes"""
        print(f"\n🚀 RUNNING DYNAMIC NODE OPTIMIZATION")
        print("=" * 50)
        
        operations_performed = 0
        
        # Get nodes that need optimization
        self.cursor.execute("""
            SELECT 
                n.node_id,
                n.node_type,
                n.content,
                LENGTH(n.content) as size,
                COUNT(e.edge_id) as connections,
                COALESCE(n.activation_count, 0) as activations
            FROM nodes n
            LEFT JOIN edges e ON n.node_id = e.source_id OR n.node_id = e.target_id
            GROUP BY n.node_id
            HAVING size > 0
            ORDER BY connections DESC, activations DESC
            LIMIT ?
        """, (max_operations * 2,))
        
        candidates = self.cursor.fetchall()
        
        for node_id, node_type, content, size, connections, activations in candidates:
            if operations_performed >= max_operations:
                break
                
            strategy = self.determine_strategy(size, connections, activations)
            
            if strategy == NodeStrategy.FRAGMENT and size > self.OPTIMAL_MEDIUM:
                fragments = self.fragment_large_node(node_id, content)
                if len(fragments) > 1:
                    self.implement_fragmentation(node_id, node_type, fragments)
                    operations_performed += 1
            
            elif strategy == NodeStrategy.CONSOLIDATE:
                # Find related nodes to consolidate with
                related_nodes = self.find_consolidation_candidates(node_id)
                if len(related_nodes) >= 2:
                    consolidated_content = self.consolidate_successful_chain(related_nodes)
                    self.implement_consolidation(related_nodes, node_type, consolidated_content)
                    operations_performed += 1
            
            elif strategy == NodeStrategy.SPECIALIZE and connections > 150:
                specialized_content = self.create_specialized_variant(node_id, f"high_performance")
                if specialized_content:
                    self.implement_specialization(node_id, node_type, specialized_content)
                    operations_performed += 1
        
        print(f"✅ Optimization complete: {operations_performed} operations performed")
        return operations_performed
    
    def find_consolidation_candidates(self, node_id: str, max_candidates: int = 3) -> List[str]:
        """Find nodes that could be consolidated with the given node"""
        
        # Find highly connected nodes of similar type and size
        self.cursor.execute("""
            SELECT DISTINCT n2.node_id
            FROM edges e
            JOIN nodes n1 ON (e.source_id = n1.node_id OR e.target_id = n1.node_id)
            JOIN nodes n2 ON (e.source_id = n2.node_id OR e.target_id = n2.node_id)
            WHERE n1.node_id = ? 
            AND n2.node_id != ?
            AND LENGTH(n2.content) <= ?
            AND n1.node_type = n2.node_type
            ORDER BY e.weight DESC
            LIMIT ?
        """, (node_id, node_id, self.OPTIMAL_SMALL, max_candidates))
        
        candidates = [row[0] for row in self.cursor.fetchall()]
        return [node_id] + candidates  # Include original node
    
    def implement_fragmentation(self, original_id: str, node_type: str, fragments: List[str]):
        """Actually create the fragmented nodes"""
        print(f"   Creating {len(fragments)} fragments...")
        
        for i, fragment in enumerate(fragments):
            fragment_id = f"{original_id}_frag_{i}_{hashlib.md5(fragment.encode()).hexdigest()[:6]}"
            
            try:
                self.cursor.execute("""
                    INSERT OR IGNORE INTO nodes 
                    (node_id, node_type, content, embedding, activation_strength, 
                     firing_rate, last_activation, activation_count, creation_time, 
                     metadata, modality_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fragment_id, f"{node_type}_fragment", fragment, '[]', 0.9,
                    0.0, time.time(), 1, time.time(),
                    f'{{"parent": "{original_id}", "fragment_index": {i}}}', 'dynamic_optimization'
                ))
            except Exception as e:
                print(f"Error creating fragment: {e}")
    
    def implement_consolidation(self, node_ids: List[str], node_type: str, consolidated_content: str):
        """Create consolidated node"""
        consolidated_id = f"consolidated_{hashlib.md5('_'.join(node_ids).encode()).hexdigest()[:8]}"
        
        try:
            self.cursor.execute("""
                INSERT OR IGNORE INTO nodes 
                (node_id, node_type, content, embedding, activation_strength, 
                 firing_rate, last_activation, activation_count, creation_time, 
                 metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                consolidated_id, f"{node_type}_consolidated", consolidated_content, '[]', 1.0,
                0.0, time.time(), len(node_ids), time.time(),
                f'{{"parents": {json.dumps(node_ids)}}}', 'dynamic_optimization'
            ))
            print(f"   Created consolidated node: {len(consolidated_content)} bytes")
        except Exception as e:
            print(f"Error creating consolidated node: {e}")
    
    def implement_specialization(self, base_id: str, node_type: str, specialized_content: str):
        """Create specialized variant"""
        specialized_id = f"{base_id}_specialized_{hashlib.md5(specialized_content.encode()).hexdigest()[:6]}"
        
        try:
            self.cursor.execute("""
                INSERT OR IGNORE INTO nodes 
                (node_id, node_type, content, embedding, activation_strength, 
                 firing_rate, last_activation, activation_count, creation_time, 
                 metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                specialized_id, f"{node_type}_specialized", specialized_content, '[]', 1.1,
                0.0, time.time(), 1, time.time(),
                f'{{"base_node": "{base_id}"}}', 'dynamic_optimization'
            ))
            print(f"   Created specialized variant: {len(specialized_content)} bytes")
        except Exception as e:
            print(f"Error creating specialized node: {e}")
    
    def get_optimization_report(self) -> Dict:
        """Generate report on node optimization"""
        
        # Count nodes by type and strategy
        self.cursor.execute("""
            SELECT 
                CASE 
                    WHEN node_type LIKE '%fragment' THEN 'fragmented'
                    WHEN node_type LIKE '%consolidated' THEN 'consolidated' 
                    WHEN node_type LIKE '%specialized' THEN 'specialized'
                    WHEN node_type LIKE '%atomic%' THEN 'atomic'
                    ELSE 'original'
                END as category,
                COUNT(*) as count,
                AVG(LENGTH(content)) as avg_size,
                MIN(LENGTH(content)) as min_size,
                MAX(LENGTH(content)) as max_size
            FROM nodes
            GROUP BY category
            ORDER BY count DESC
        """)
        
        results = self.cursor.fetchall()
        
        report = {
            'categories': {},
            'total_nodes': 0,
            'optimization_coverage': 0
        }
        
        for category, count, avg_size, min_size, max_size in results:
            report['categories'][category] = {
                'count': count,
                'avg_size': round(avg_size, 1),
                'size_range': f"{min_size}-{max_size}"
            }
            report['total_nodes'] += count
        
        # Calculate optimization coverage
        optimized = sum(data['count'] for cat, data in report['categories'].items() 
                       if cat in ['fragmented', 'consolidated', 'specialized', 'atomic'])
        report['optimization_coverage'] = round(optimized / report['total_nodes'] * 100, 1)
        
        return report
    
    def run_full_optimization(self):
        """Run complete dynamic optimization"""
        print("🚀 MELVIN DYNAMIC NODE OPTIMIZATION")
        print("=" * 60)
        
        # Run optimization
        operations = self.optimize_node_sizes(max_operations=30)
        
        # Commit changes
        self.conn.commit()
        
        # Generate report
        report = self.get_optimization_report()
        
        print(f"\n📊 OPTIMIZATION REPORT:")
        print("-" * 30)
        print(f"Total Operations: {operations}")
        print(f"Total Nodes: {report['total_nodes']:,}")
        print(f"Optimization Coverage: {report['optimization_coverage']}%")
        print()
        print("Node Categories:")
        for category, data in report['categories'].items():
            print(f"   {category.capitalize()}: {data['count']:,} nodes (avg: {data['avg_size']} bytes)")
        
        print(f"\n🎯 DYNAMIC SIZING RULES:")
        print(f"   Minimum: {self.MIN_NODE_SIZE} bytes (single concepts)")
        print(f"   Optimal Small: {self.OPTIMAL_SMALL} bytes (atomic facts)")
        print(f"   Optimal Medium: {self.OPTIMAL_MEDIUM} bytes (concepts)")  
        print(f"   Optimal Large: {self.OPTIMAL_LARGE} bytes (successful chains)")
        print(f"   Maximum: {self.MAX_NODE_SIZE} bytes (complex knowledge)")
        
        return report
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    optimizer = MelvinDynamicNodes()
    try:
        optimizer.run_full_optimization()
    finally:
        optimizer.close()

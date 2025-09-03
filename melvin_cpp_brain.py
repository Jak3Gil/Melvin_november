#!/usr/bin/env python3
"""
🚀 MELVIN C++ BRAIN INTEGRATION
High-performance brain operations using C++ backend
"""

import sys
import time
from typing import List, Dict, Any, Tuple, Optional
import json
import sqlite3

# Try to import the C++ module
try:
    import fast_brain_core as cpp_brain
    CPP_AVAILABLE = True
    print("✅ C++ Fast Brain Core loaded successfully!")
except ImportError as e:
    CPP_AVAILABLE = False
    print(f"⚠️ C++ Fast Brain Core not available: {e}")
    print("   Falling back to Python implementation")

class MelvinCppBrain:
    """High-performance brain using C++ backend with Python fallback"""
    
    def __init__(self, db_path: str = "melvin_global_memory/global_memory.db"):
        self.db_path = db_path
        self.use_cpp = CPP_AVAILABLE
        
        if self.use_cpp:
            self.cpp_core = cpp_brain.FastBrainCore()
            print("🚀 Using C++ Fast Brain Core")
        else:
            # Fallback to Python SQLite
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            print("🐍 Using Python fallback implementation")
        
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing data from SQLite into C++ core"""
        if not self.use_cpp:
            return
        
        print("📊 Loading existing data into C++ core...")
        
        try:
            # Load from SQLite database
            success = self.cpp_core.load_from_sqlite(self.db_path)
            if success:
                stats = self.cpp_core.get_performance_stats()
                print(f"✅ Loaded {stats.total_nodes:,} nodes and {stats.total_connections:,} connections")
            else:
                print("⚠️ Could not load from SQLite, starting with empty brain")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def create_node(self, node_type: str, content: str) -> int:
        """Create a new node"""
        if self.use_cpp:
            # Map string types to C++ enum
            type_mapping = {
                'language': cpp_brain.NodeType.LANGUAGE,
                'code': cpp_brain.NodeType.CODE,
                'visual': cpp_brain.NodeType.VISUAL,
                'audio': cpp_brain.NodeType.AUDIO,
                'concept': cpp_brain.NodeType.CONCEPT,
                'emotion': cpp_brain.NodeType.EMOTION,
                'atomic_fact': cpp_brain.NodeType.ATOMIC_FACT,
                'consolidated': cpp_brain.NodeType.CONSOLIDATED,
                'specialized': cpp_brain.NodeType.SPECIALIZED
            }
            
            cpp_type = type_mapping.get(node_type, cpp_brain.NodeType.LANGUAGE)
            return self.cpp_core.create_node(cpp_type, content)
        else:
            # Fallback Python implementation
            self.cursor.execute("""
                INSERT INTO nodes (node_type, content, creation_time)
                VALUES (?, ?, ?)
            """, (node_type, content, time.time()))
            return self.cursor.lastrowid
    
    def create_connection(self, source_id: int, target_id: int, connection_type: str = "similarity", weight: float = 1.0) -> int:
        """Create a connection between nodes"""
        if self.use_cpp:
            type_mapping = {
                'similarity': cpp_brain.ConnectionType.SIMILARITY,
                'temporal': cpp_brain.ConnectionType.TEMPORAL,
                'hebbian': cpp_brain.ConnectionType.HEBBIAN,
                'multimodal': cpp_brain.ConnectionType.MULTIMODAL,
                'atomic_relation': cpp_brain.ConnectionType.ATOMIC_RELATION,
                'consolidation': cpp_brain.ConnectionType.CONSOLIDATION
            }
            
            cpp_type = type_mapping.get(connection_type, cpp_brain.ConnectionType.SIMILARITY)
            return self.cpp_core.create_connection(source_id, target_id, cpp_type, weight)
        else:
            # Fallback Python implementation
            self.cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, weight, creation_time)
                VALUES (?, ?, ?, ?, ?)
            """, (source_id, target_id, connection_type, weight, time.time()))
            return self.cursor.lastrowid
    
    def search_nodes(self, query: str, max_results: int = 10) -> List[int]:
        """High-speed node search"""
        if self.use_cpp:
            return self.cpp_core.search_nodes_simd(query, max_results)
        else:
            # Fallback Python search
            self.cursor.execute("""
                SELECT rowid FROM nodes 
                WHERE content LIKE ? 
                LIMIT ?
            """, (f'%{query}%', max_results))
            return [row[0] for row in self.cursor.fetchall()]
    
    def get_node_content(self, node_id: int) -> Optional[str]:
        """Get node content"""
        if self.use_cpp:
            # Would need to implement getter in C++
            # For now, fallback to SQLite
            pass
        
        # SQLite fallback
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM nodes WHERE rowid = ?", (node_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def activate_node(self, node_id: int, strength: float = 1.0) -> bool:
        """Activate a node"""
        if self.use_cpp:
            return self.cpp_core.activate_node(node_id, strength)
        else:
            # Fallback - update activation in SQLite
            self.cursor.execute("""
                UPDATE nodes SET 
                    activation_strength = ?,
                    last_activation = ?,
                    activation_count = activation_count + 1
                WHERE rowid = ?
            """, (strength, time.time(), node_id))
            return self.cursor.rowcount > 0
    
    def get_connected_nodes(self, node_id: int, max_results: int = 100) -> List[int]:
        """Get nodes connected to the given node"""
        if self.use_cpp:
            return self.cpp_core.get_connected_nodes(node_id, max_results)
        else:
            # Fallback SQLite query
            self.cursor.execute("""
                SELECT CASE 
                    WHEN source_id = ? THEN target_id 
                    ELSE source_id 
                END as connected_id
                FROM edges 
                WHERE source_id = ? OR target_id = ?
                ORDER BY weight DESC
                LIMIT ?
            """, (node_id, node_id, node_id, max_results))
            return [row[0] for row in self.cursor.fetchall()]
    
    def hebbian_learning(self, active_node_ids: List[int]):
        """Apply Hebbian learning to active nodes"""
        if self.use_cpp:
            self.cpp_core.hebbian_update_batch(active_node_ids)
        else:
            # Fallback - strengthen connections between active nodes
            for i, node1 in enumerate(active_node_ids):
                for node2 in active_node_ids[i+1:]:
                    self.cursor.execute("""
                        UPDATE edges SET 
                            weight = weight + 0.01,
                            coactivation_count = coactivation_count + 1
                        WHERE (source_id = ? AND target_id = ?) 
                           OR (source_id = ? AND target_id = ?)
                    """, (node1, node2, node2, node1))
    
    def fragment_large_nodes(self) -> int:
        """Fragment large nodes into smaller pieces"""
        if self.use_cpp:
            candidates = self.cpp_core.find_fragmentation_candidates()
            fragments_created = 0
            
            for node_id in candidates[:10]:  # Process first 10
                content = self.get_node_content(node_id)
                if content and len(content) > 200:
                    # Simple fragmentation by sentences
                    import re
                    sentences = re.split(r'[.!?]+', content)
                    fragments = [s.strip() for s in sentences if len(s.strip()) > 10]
                    
                    if len(fragments) > 1:
                        success = self.cpp_core.fragment_node(node_id, fragments)
                        if success:
                            fragments_created += len(fragments)
            
            return fragments_created
        else:
            return 0  # Not implemented in fallback
    
    def consolidate_successful_chains(self) -> int:
        """Consolidate successful node chains"""
        if self.use_cpp:
            candidate_groups = self.cpp_core.find_consolidation_candidates()
            consolidated_count = 0
            
            for group in candidate_groups[:5]:  # Process first 5 groups
                if len(group) >= 2:
                    # Get content from all nodes
                    contents = []
                    for node_id in group:
                        content = self.get_node_content(node_id)
                        if content:
                            contents.append(content)
                    
                    if contents:
                        consolidated_content = " → ".join(contents)
                        new_node_id = self.cpp_core.consolidate_nodes(group, consolidated_content)
                        if new_node_id > 0:
                            consolidated_count += 1
            
            return consolidated_count
        else:
            return 0  # Not implemented in fallback
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.use_cpp:
            stats = self.cpp_core.get_performance_stats()
            return {
                'total_nodes': stats.total_nodes,
                'total_connections': stats.total_connections,
                'total_activations': stats.total_activations,
                'total_searches': stats.total_searches,
                'avg_search_time_ms': stats.avg_search_time_ms,
                'avg_activation_time_ms': stats.avg_activation_time_ms,
                'memory_usage_bytes': stats.memory_usage_bytes,
                'cache_hit_rate': stats.cache_hit_rate,
                'backend': 'C++ Fast Core'
            }
        else:
            # Fallback stats from SQLite
            self.cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = self.cursor.fetchone()[0]
            
            return {
                'total_nodes': node_count,
                'total_connections': edge_count,
                'total_activations': 0,
                'total_searches': 0,
                'avg_search_time_ms': 0.0,
                'avg_activation_time_ms': 0.0,
                'memory_usage_bytes': 0,
                'cache_hit_rate': 0.0,
                'backend': 'Python SQLite Fallback'
            }
    
    def benchmark_performance(self, num_operations: int = 1000) -> Dict[str, float]:
        """Benchmark the brain performance"""
        results = {}
        
        print(f"🏃 Running performance benchmark ({num_operations} operations)...")
        
        # Benchmark node creation
        start_time = time.time()
        node_ids = []
        for i in range(num_operations // 10):
            node_id = self.create_node('atomic_fact', f'Test fact {i}')
            node_ids.append(node_id)
        
        results['node_creation_time'] = time.time() - start_time
        results['nodes_per_second'] = len(node_ids) / results['node_creation_time']
        
        # Benchmark search
        start_time = time.time()
        for i in range(100):
            self.search_nodes('test', max_results=10)
        
        results['search_time'] = time.time() - start_time
        results['searches_per_second'] = 100 / results['search_time']
        
        # Benchmark activation
        if node_ids:
            start_time = time.time()
            for node_id in node_ids:
                self.activate_node(node_id, 0.8)
            
            results['activation_time'] = time.time() - start_time
            results['activations_per_second'] = len(node_ids) / results['activation_time']
        
        return results
    
    def save_to_disk(self):
        """Save brain state to disk"""
        if self.use_cpp:
            success = self.cpp_core.save_to_sqlite(self.db_path)
            return success
        else:
            self.conn.commit()
            return True
    
    def close(self):
        """Close the brain and cleanup resources"""
        if self.use_cpp:
            # C++ objects will be cleaned up automatically
            pass
        else:
            if hasattr(self, 'conn'):
                self.conn.close()

def main():
    """Demo of the high-performance brain"""
    print("🚀 MELVIN C++ BRAIN PERFORMANCE DEMO")
    print("=" * 60)
    
    brain = MelvinCppBrain()
    
    try:
        # Show current stats
        stats = brain.get_performance_stats()
        print(f"📊 Current Brain State:")
        print(f"   Backend: {stats['backend']}")
        print(f"   Nodes: {stats['total_nodes']:,}")
        print(f"   Connections: {stats['total_connections']:,}")
        print(f"   Memory Usage: {stats['memory_usage_bytes']:,} bytes")
        
        # Run benchmark
        if CPP_AVAILABLE:
            benchmark_results = brain.benchmark_performance(1000)
            print(f"\n🏃 Performance Benchmark Results:")
            print(f"   Node Creation: {benchmark_results['nodes_per_second']:.0f} nodes/sec")
            print(f"   Search Speed: {benchmark_results['searches_per_second']:.0f} searches/sec")
            if 'activations_per_second' in benchmark_results:
                print(f"   Activation Speed: {benchmark_results['activations_per_second']:.0f} activations/sec")
        
        # Test search functionality
        print(f"\n🔍 Testing Search:")
        results = brain.search_nodes("machine learning", max_results=5)
        print(f"   Found {len(results)} results for 'machine learning'")
        
        for i, node_id in enumerate(results[:3], 1):
            content = brain.get_node_content(node_id)
            if content:
                preview = content[:60] + "..." if len(content) > 60 else content
                print(f"   {i}. [{node_id}] {preview}")
        
        # Test dynamic optimization
        if CPP_AVAILABLE:
            print(f"\n🔧 Testing Dynamic Optimization:")
            fragments = brain.fragment_large_nodes()
            consolidations = brain.consolidate_successful_chains()
            print(f"   Fragments created: {fragments}")
            print(f"   Consolidations created: {consolidations}")
        
        # Save changes
        brain.save_to_disk()
        print(f"\n💾 Brain state saved to disk")
        
    finally:
        brain.close()

if __name__ == "__main__":
    main()

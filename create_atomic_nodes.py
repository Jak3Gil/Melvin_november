#!/usr/bin/env python3
"""
🔬 CREATE ATOMIC NODES
Break large nodes into smaller, atomic facts for better responses
"""

import sqlite3
import re
import hashlib
from typing import List

def extract_facts_from_text(text: str) -> List[str]:
    """Extract atomic facts from text"""
    # Clean the text
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    facts = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Keep sentences that are informative but not too long
        if 10 <= len(sentence) <= 100 and not sentence.lower().startswith(('this', 'it', 'that')):
            facts.append(sentence)
    
    return facts

def create_atomic_brain():
    """Create atomic nodes from existing large nodes"""
    conn = sqlite3.connect('melvin_global_memory/global_memory.db')
    cursor = conn.cursor()
    
    print("🔬 CREATING ATOMIC KNOWLEDGE NODES")
    print("=" * 50)
    
    # Find large text nodes
    cursor.execute("""
        SELECT node_id, content FROM nodes 
        WHERE node_type = 'language' 
        AND LENGTH(content) > 100 
        ORDER BY LENGTH(content) DESC 
        LIMIT 20
    """)
    
    large_nodes = cursor.fetchall()
    print(f"📊 Processing {len(large_nodes)} large nodes...")
    
    atomic_nodes_created = 0
    connections_created = 0
    
    for original_id, content in large_nodes:
        facts = extract_facts_from_text(content)
        
        if len(facts) < 2:
            continue
            
        print(f"   Breaking down node → {len(facts)} facts")
        
        # Create atomic nodes
        fact_ids = []
        for fact in facts:
            # Create unique ID for this fact
            fact_hash = hashlib.md5(fact.encode()).hexdigest()[:8]
            atomic_id = f"atomic_fact_{fact_hash}"
            
            # Insert atomic node
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO nodes 
                    (node_id, node_type, content, embedding, activation_strength, 
                     firing_rate, last_activation, activation_count, creation_time, 
                     metadata, modality_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    atomic_id, 'atomic_fact', fact, '[]', 0.8,
                    0.0, 1756918000.0, 1, 1756918000.0,
                    f'{{"parent": "{original_id}"}}', 'optimized'
                ))
                
                fact_ids.append(atomic_id)
                atomic_nodes_created += 1
                
            except Exception as e:
                print(f"Error creating atomic node: {e}")
        
        # Create connections between related facts
        for i, fact_id_1 in enumerate(fact_ids):
            for j, fact_id_2 in enumerate(fact_ids[i+1:], i+1):
                conn_id = f"atomic_conn_{hashlib.md5(f'{fact_id_1}_{fact_id_2}'.encode()).hexdigest()[:8]}"
                
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO edges 
                        (edge_id, source_id, target_id, edge_type, weight, 
                         coactivation_count, last_coactivation, creation_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        conn_id, fact_id_1, fact_id_2, 'atomic_relation',
                        1.0 / (abs(i - j) + 1),  # Closer facts have higher weight
                        1, 1756918000.0, 1756918000.0
                    ))
                    
                    connections_created += 1
                    
                except Exception as e:
                    print(f"Error creating connection: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✅ ATOMIC BRAIN CREATION COMPLETE!")
    print(f"   🔬 Atomic facts created: {atomic_nodes_created}")
    print(f"   🔗 Connections created: {connections_created}")
    
    return atomic_nodes_created, connections_created

def test_atomic_responses():
    """Test responses using atomic facts"""
    conn = sqlite3.connect('melvin_global_memory/global_memory.db')
    cursor = conn.cursor()
    
    print(f"\\n🧪 TESTING ATOMIC FACT RETRIEVAL")
    print("-" * 40)
    
    test_queries = [
        "machine learning",
        "artificial intelligence", 
        "neural network",
        "computer",
        "algorithm"
    ]
    
    for query in test_queries:
        print(f"\\n🔍 Query: '{query}'")
        
        # Search atomic facts
        cursor.execute("""
            SELECT content FROM nodes 
            WHERE node_type = 'atomic_fact' 
            AND LOWER(content) LIKE ? 
            LIMIT 3
        """, (f'%{query.lower()}%',))
        
        results = cursor.fetchall()
        
        if results:
            print("✅ Atomic facts found:")
            for (content,) in results:
                print(f"   💡 {content}")
        else:
            print("❌ No atomic facts found")
    
    conn.close()

if __name__ == "__main__":
    # Create atomic nodes
    atomic_count, conn_count = create_atomic_brain()
    
    # Test the results
    if atomic_count > 0:
        test_atomic_responses()
    else:
        print("❌ No atomic nodes were created - check the data source")

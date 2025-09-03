#!/usr/bin/env python3
"""
📊 MELVIN GROWTH MONITOR
Real-time monitoring of Melvin's brain growth and learning
"""

import sqlite3
import time
import json
from datetime import datetime

def monitor_melvin():
    """Monitor Melvin's brain growth in real-time"""
    print("📊 MELVIN BRAIN GROWTH MONITOR")
    print("=" * 50)
    print("Monitoring brain growth every 30 seconds...")
    print("Press Ctrl+C to stop\n")
    
    last_nodes = 0
    last_edges = 0
    start_time = time.time()
    
    try:
        while True:
            conn = sqlite3.connect('melvin_global_memory/global_memory.db')
            cursor = conn.cursor()
            
            # Get current stats
            cursor.execute('SELECT COUNT(*) FROM nodes')
            current_nodes = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM edges')
            current_edges = cursor.fetchone()[0]
            
            # Get domain distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN metadata LIKE '%computer%' THEN 'Computer Science'
                        WHEN metadata LIKE '%mathematics%' THEN 'Mathematics' 
                        WHEN metadata LIKE '%physics%' THEN 'Physics'
                        WHEN metadata LIKE '%chemistry%' THEN 'Chemistry'
                        WHEN metadata LIKE '%biology%' THEN 'Biology'
                        WHEN metadata LIKE '%history%' THEN 'History'
                        WHEN metadata LIKE '%philosophy%' THEN 'Philosophy'
                        ELSE 'Other'
                    END as domain,
                    COUNT(*) as count
                FROM nodes 
                WHERE modality_source = 'educational_feed'
                GROUP BY domain
                ORDER BY count DESC
            """)
            domain_stats = cursor.fetchall()
            
            conn.close()
            
            # Calculate growth
            nodes_growth = current_nodes - last_nodes
            edges_growth = current_edges - last_edges
            runtime = time.time() - start_time
            
            # Display stats
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
            print(f"🧠 Nodes: {current_nodes:,} (+{nodes_growth})")
            print(f"🔗 Edges: {current_edges:,} (+{edges_growth})")
            print(f"📈 Network Density: {(current_edges*2/current_nodes):.1f} connections/node")
            print(f"⚡ Runtime: {runtime/60:.1f} minutes")
            
            if domain_stats:
                print("📚 Knowledge Domains:")
                for domain, count in domain_stats:
                    print(f"   {domain}: {count}")
            
            print("-" * 40)
            
            last_nodes = current_nodes
            last_edges = current_edges
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n📊 Monitoring stopped.")

if __name__ == "__main__":
    monitor_melvin()

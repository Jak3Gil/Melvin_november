#!/usr/bin/env python3
"""
🔄 MELVIN CONTINUOUS DATA FEEDER
Continuously feeds Melvin diverse educational data and saves to repo every few minutes
"""

import sqlite3
import time
import json
import random
import subprocess
from datetime import datetime
from typing import Dict, List, Any

class MelvinContinuousFeeder:
    def __init__(self, save_interval_minutes: int = 3):
        self.conn = sqlite3.connect('melvin_global_memory/global_memory.db')
        self.cursor = self.conn.cursor()
        self.save_interval = save_interval_minutes * 60  # Convert to seconds
        self.last_save = time.time()
        self.nodes_added_since_save = 0
        self.connections_added_since_save = 0
        
        print("🔄 MELVIN CONTINUOUS DATA FEEDER")
        print(f"💾 Auto-save interval: {save_interval_minutes} minutes")
        print("=" * 50)
    
    def get_diverse_educational_data(self) -> Dict[str, List[str]]:
        """Get diverse educational content across multiple domains"""
        return {
            'computer_science': [
                "A compiler translates high-level programming code into machine language that computers can execute directly",
                "Hash tables provide O(1) average-case lookup time by using hash functions to map keys to array indices",
                "Sorting algorithms like quicksort and mergesort organize data efficiently with different time complexities",
                "Binary search trees maintain sorted data and enable efficient searching, insertion, and deletion operations",
                "Graph algorithms like Dijkstra's find shortest paths between nodes in weighted networks",
                "Dynamic programming solves complex problems by breaking them into simpler overlapping subproblems",
                "Regular expressions are powerful patterns for matching and manipulating text strings",
                "Database normalization reduces data redundancy and improves data integrity in relational databases",
                "Concurrency allows multiple processes to execute simultaneously, improving system performance",
                "Memory management involves allocating and deallocating computer memory efficiently during program execution"
            ],
            
            'mathematics': [
                "Calculus studies rates of change and accumulation, fundamental to physics and engineering",
                "Linear algebra deals with vectors, matrices, and systems of linear equations",
                "Statistics analyzes data patterns and helps make predictions based on sample information",
                "Probability theory quantifies uncertainty and randomness in mathematical terms",
                "Number theory studies properties and relationships of integers and prime numbers",
                "Geometry explores shapes, sizes, positions, and properties of space and objects",
                "Discrete mathematics covers logic, set theory, and combinatorics for computer science",
                "Differential equations model how quantities change over time in natural phenomena",
                "Graph theory studies networks of connected nodes and their mathematical properties",
                "Optimization finds the best solution among many possible alternatives using mathematical methods"
            ],
            
            'physics': [
                "Newton's laws of motion describe the relationship between forces and object movement",
                "Einstein's theory of relativity revolutionized understanding of space, time, and gravity",
                "Quantum mechanics explains the behavior of matter and energy at atomic and subatomic scales",
                "Thermodynamics studies heat, temperature, and energy transfer in physical systems",
                "Electromagnetic waves include visible light, radio waves, and X-rays traveling at light speed",
                "Conservation laws state that energy, momentum, and mass cannot be created or destroyed",
                "Wave-particle duality shows that light and matter exhibit both wave and particle properties",
                "Nuclear physics studies atomic nuclei, radioactivity, and nuclear reactions",
                "Fluid dynamics analyzes the motion of liquids and gases in various conditions",
                "Optics investigates the behavior and properties of light and its interaction with matter"
            ],
            
            'chemistry': [
                "The periodic table organizes elements by atomic number and reveals patterns in chemical properties",
                "Chemical bonds form when atoms share or transfer electrons to achieve stable configurations",
                "Acids and bases are substances that donate or accept protons in chemical reactions",
                "Catalysts speed up chemical reactions without being consumed in the process",
                "Organic chemistry studies carbon-based compounds essential to all living organisms",
                "Stoichiometry calculates quantities of reactants and products in chemical equations",
                "Phase transitions occur when matter changes between solid, liquid, and gas states",
                "Electrochemistry involves chemical reactions that produce or consume electrical energy",
                "Molecular geometry determines the three-dimensional arrangement of atoms in molecules",
                "Chemical equilibrium occurs when forward and reverse reaction rates become equal"
            ],
            
            'biology': [
                "DNA contains genetic instructions for the development and function of all living organisms",
                "Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen in plants",
                "Evolution through natural selection explains how species change and adapt over time",
                "Cellular respiration breaks down glucose to produce ATP energy for cellular processes",
                "Ecosystems consist of interconnected living organisms and their physical environment",
                "Protein synthesis involves transcribing DNA to RNA and translating RNA to proteins",
                "The nervous system transmits electrical and chemical signals throughout the body",
                "Genetics studies how traits are inherited from parents to offspring through genes",
                "Homeostasis maintains stable internal conditions in living organisms despite environmental changes",
                "Biodiversity encompasses the variety of life forms and their ecological relationships"
            ],
            
            'history': [
                "The Industrial Revolution transformed society through mechanization and mass production in the 18th-19th centuries",
                "The Renaissance was a period of cultural rebirth and scientific advancement in 14th-17th century Europe",
                "World War II was a global conflict from 1939-1945 that reshaped international relations",
                "The Scientific Revolution established the scientific method and modern scientific thinking",
                "Ancient civilizations like Egypt, Greece, and Rome laid foundations for modern society",
                "The Cold War was a period of geopolitical tension between the US and Soviet Union",
                "The Age of Exploration led to European colonization and global trade networks",
                "Democratic revolutions established principles of individual rights and representative government",
                "The Agricultural Revolution enabled permanent settlements and population growth",
                "Technological innovations throughout history have driven social and economic change"
            ],
            
            'philosophy': [
                "Ethics examines moral principles and what constitutes right and wrong behavior",
                "Logic studies valid reasoning and the principles of correct inference",
                "Epistemology investigates the nature of knowledge and how we acquire understanding",
                "Metaphysics explores fundamental questions about reality, existence, and being",
                "Political philosophy examines justice, rights, and the ideal organization of society",
                "Philosophy of mind studies consciousness, mental states, and the mind-body relationship",
                "Aesthetics investigates beauty, art, and the nature of aesthetic experience",
                "Existentialism emphasizes individual existence, freedom, and the search for meaning",
                "Utilitarianism judges actions by their consequences and overall happiness produced",
                "Skepticism questions the possibility of certain knowledge and absolute truth"
            ]
        }
    
    def add_educational_batch(self, domain: str, facts: List[str]) -> int:
        """Add a batch of educational facts from a specific domain"""
        added_count = 0
        
        for fact in facts:
            try:
                self.cursor.execute("""
                    INSERT INTO nodes 
                    (node_type, content, embedding, activation_strength, firing_rate, 
                     last_activation, activation_count, creation_time, metadata, modality_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'atomic_fact', fact, '[]', 0.8, 0.0, time.time(), 1, time.time(),
                    json.dumps({"domain": domain, "source": "continuous_feed", "priority": "high"}),
                    'educational_feed'
                ))
                added_count += 1
                
            except sqlite3.IntegrityError:
                # Skip duplicates
                continue
            except Exception as e:
                print(f"Error adding fact: {e}")
        
        return added_count
    
    def create_cross_domain_connections(self, batch_size: int = 10) -> int:
        """Create connections between facts from different domains"""
        connections_added = 0
        
        # Get recent facts from different domains
        self.cursor.execute("""
            SELECT node_id, content, metadata FROM nodes 
            WHERE modality_source = 'educational_feed'
            ORDER BY creation_time DESC 
            LIMIT ?
        """, (batch_size * 2,))
        
        recent_nodes = self.cursor.fetchall()
        
        # Create connections between related concepts
        for i in range(len(recent_nodes)):
            for j in range(i + 1, min(i + 3, len(recent_nodes))):  # Connect to next 2 nodes
                node1_id, content1, metadata1 = recent_nodes[i]
                node2_id, content2, metadata2 = recent_nodes[j]
                
                try:
                    # Check if they share common keywords (simple relatedness)
                    words1 = set(content1.lower().split())
                    words2 = set(content2.lower().split())
                    
                    if len(words1 & words2) >= 2:  # At least 2 common words
                        edge_id = f"cross_domain_{node1_id}_{node2_id}"
                        
                        self.cursor.execute("""
                            INSERT OR IGNORE INTO edges 
                            (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                             last_coactivation, creation_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            edge_id, node1_id, node2_id, 'conceptual_similarity',
                            0.6, 1, time.time(), time.time()
                        ))
                        connections_added += 1
                        
                except Exception as e:
                    continue
        
        return connections_added
    
    def save_to_repo(self):
        """Save current brain state and commit to git repository"""
        try:
            # Update brain state JSON
            self.cursor.execute('SELECT COUNT(*) FROM nodes')
            total_nodes = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(*) FROM edges')
            total_edges = self.cursor.fetchone()[0]
            
            brain_state = {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'last_update': datetime.now().isoformat(),
                'nodes_added_this_session': self.nodes_added_since_save,
                'connections_added_this_session': self.connections_added_since_save
            }
            
            with open('melvin_global_memory/complete_brain_state.json', 'w') as f:
                json.dump(brain_state, f, indent=2)
            
            # Commit changes to git
            subprocess.run(['git', 'add', 'melvin_global_memory/'], check=True)
            subprocess.run([
                'git', 'commit', '-m', 
                f'Auto-save: +{self.nodes_added_since_save} nodes, +{self.connections_added_since_save} connections'
            ], check=True)
            
            print(f"💾 SAVED TO REPO: {total_nodes:,} nodes, {total_edges:,} connections")
            
            # Reset counters
            self.nodes_added_since_save = 0
            self.connections_added_since_save = 0
            self.last_save = time.time()
            
        except subprocess.CalledProcessError as e:
            print(f"Git error (continuing anyway): {e}")
        except Exception as e:
            print(f"Save error: {e}")
    
    def feed_continuously(self, max_iterations: int = 100):
        """Continuously feed data and save periodically"""
        print("🚀 Starting continuous data feeding...")
        print(f"📊 Will save every {self.save_interval//60} minutes")
        print("Press Ctrl+C to stop gracefully\n")
        
        educational_data = self.get_diverse_educational_data()
        domains = list(educational_data.keys())
        
        iteration = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                
                # Select random domain and facts
                domain = random.choice(domains)
                available_facts = educational_data[domain]
                batch_size = random.randint(3, 8)  # Random batch size
                selected_facts = random.sample(available_facts, min(batch_size, len(available_facts)))
                
                # Add facts
                added = self.add_educational_batch(domain, selected_facts)
                self.nodes_added_since_save += added
                
                # Create some connections
                if iteration % 3 == 0:  # Every 3rd iteration
                    connections = self.create_cross_domain_connections()
                    self.connections_added_since_save += connections
                
                # Commit changes
                self.conn.commit()
                
                print(f"📝 Iteration {iteration}: Added {added} {domain} facts")
                
                # Check if it's time to save
                if time.time() - self.last_save >= self.save_interval:
                    self.save_to_repo()
                
                # Brief pause
                time.sleep(random.uniform(2, 5))  # 2-5 second pause
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping gracefully...")
            
        finally:
            # Final save
            print(f"💾 Final save...")
            self.save_to_repo()
            
            # Final stats
            self.cursor.execute('SELECT COUNT(*) FROM nodes')
            final_nodes = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(*) FROM edges')  
            final_edges = self.cursor.fetchone()[0]
            
            print(f"\n✅ FEEDING SESSION COMPLETE!")
            print(f"🧠 Final brain state: {final_nodes:,} nodes, {final_edges:,} connections")
            print(f"📈 Session added: {self.nodes_added_since_save} nodes, {self.connections_added_since_save} connections")
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    print("🔄 MELVIN CONTINUOUS DATA FEEDER")
    print("=" * 50)
    
    # Ask for save interval
    try:
        interval = input("Save interval in minutes (default 3): ").strip()
        interval = int(interval) if interval else 3
    except ValueError:
        interval = 3
    
    feeder = MelvinContinuousFeeder(save_interval_minutes=interval)
    
    try:
        feeder.feed_continuously()
    finally:
        feeder.close()

if __name__ == "__main__":
    main()

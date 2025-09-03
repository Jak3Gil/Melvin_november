#!/usr/bin/env python3
"""
📚 ADD QUALITY EDUCATIONAL DATA
Add high-quality educational content to balance out the movie reviews
"""

import sqlite3
import time
import hashlib

def add_quality_educational_data():
    """Add comprehensive educational data to Melvin's brain"""
    
    conn = sqlite3.connect('melvin_global_memory/global_memory.db')
    cursor = conn.cursor()
    
    print("📚 ADDING HIGH-QUALITY EDUCATIONAL DATA")
    print("=" * 50)
    
    # High-quality educational content
    educational_facts = [
        # Programming concepts
        "An algorithm is a step-by-step procedure for solving a problem or completing a task",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data",
        "A data structure is a way of organizing and storing data efficiently in computer memory",
        "Object-oriented programming organizes code into classes and objects for better modularity",
        "Recursion is a programming technique where a function calls itself to solve smaller problems",
        "Big O notation describes the computational complexity and efficiency of algorithms",
        "APIs (Application Programming Interfaces) allow different software systems to communicate",
        "Databases store and organize large amounts of structured information for easy retrieval",
        
        # Computer Science fundamentals
        "Binary code uses only 0s and 1s to represent all information in computers",
        "The CPU (Central Processing Unit) is the brain of the computer that executes instructions",
        "RAM (Random Access Memory) provides temporary storage for programs currently running",
        "Operating systems manage computer hardware and provide services to applications",
        "Networks allow computers to communicate and share resources across distances",
        "Encryption protects data by converting it into unreadable code without the proper key",
        "Compilers translate human-readable code into machine language that computers understand",
        "Version control systems like Git track changes in code and enable collaboration",
        
        # AI and Machine Learning
        "Neural networks are computing systems inspired by biological neural networks in brains",
        "Deep learning uses multiple layers of neural networks to model complex patterns",
        "Supervised learning trains models on labeled data to make predictions on new data",
        "Unsupervised learning finds hidden patterns in data without labeled examples",
        "Natural language processing enables computers to understand and generate human language",
        "Computer vision allows machines to interpret and understand visual information",
        "Reinforcement learning trains agents through rewards and penalties for their actions",
        "Feature engineering involves selecting and transforming variables for machine learning models",
        
        # Data Science
        "Data mining discovers patterns and insights from large datasets",
        "Statistical analysis helps understand relationships and trends in data",
        "Data visualization presents information in graphical formats for better understanding",
        "Big data refers to datasets too large or complex for traditional processing methods",
        "Cloud computing provides on-demand access to computing resources over the internet",
        "Distributed systems spread computation across multiple connected computers",
        "Parallel processing divides tasks among multiple processors to increase speed",
        "Caching stores frequently accessed data in fast memory for quicker retrieval",
        
        # Software Engineering
        "Agile development emphasizes iterative progress and collaboration in software projects",
        "Test-driven development writes tests before implementing functionality",
        "Code refactoring improves code structure without changing its external behavior",
        "Design patterns provide reusable solutions to common programming problems",
        "Software architecture defines the high-level structure and organization of systems",
        "DevOps combines development and operations for faster, more reliable software delivery",
        "Microservices architecture breaks applications into small, independent services",
        "Continuous integration automatically tests and integrates code changes frequently"
    ]
    
    # Programming concepts with examples
    programming_concepts = [
        {
            'concept': 'Variables store data values that can be used and modified in programs',
            'example': 'name = "Alice"\nage = 25\nprint(f"{name} is {age} years old")'
        },
        {
            'concept': 'Functions are reusable blocks of code that perform specific tasks',
            'example': 'def calculate_area(length, width):\n    return length * width\n\narea = calculate_area(5, 3)'
        },
        {
            'concept': 'Loops repeat code execution for efficiency and automation',
            'example': 'for i in range(5):\n    print(f"Count: {i}")\n\nwhile condition:\n    do_something()'
        },
        {
            'concept': 'Conditional statements make decisions based on different conditions',
            'example': 'if temperature > 30:\n    print("Hot day")\nelif temperature > 20:\n    print("Warm day")\nelse:\n    print("Cool day")'
        },
        {
            'concept': 'Lists and arrays store multiple values in ordered collections',
            'example': 'fruits = ["apple", "banana", "orange"]\nfor fruit in fruits:\n    print(fruit)'
        },
        {
            'concept': 'Dictionaries store key-value pairs for efficient data lookup',
            'example': 'person = {"name": "Bob", "age": 30, "city": "New York"}\nprint(person["name"])'
        }
    ]
    
    # Technology definitions
    tech_definitions = [
        ('Artificial Intelligence', 'The simulation of human intelligence in machines that can think and learn'),
        ('Cloud Computing', 'Delivery of computing services over the internet including storage and processing'),
        ('Cybersecurity', 'Protection of digital systems, networks, and data from malicious attacks'),
        ('Internet of Things', 'Network of physical devices connected to the internet that can collect and share data'),
        ('Blockchain', 'Distributed ledger technology that maintains secure, transparent transaction records'),
        ('Quantum Computing', 'Computing using quantum-mechanical phenomena to process information in new ways'),
        ('Virtual Reality', 'Computer-generated simulation of three-dimensional environments for immersive experiences'),
        ('Augmented Reality', 'Technology that overlays digital information onto the real world through devices'),
        ('5G Technology', 'Fifth-generation cellular network technology providing faster speeds and lower latency'),
        ('Edge Computing', 'Processing data closer to where it is generated rather than in centralized data centers')
    ]
    
    nodes_added = 0
    connections_added = 0
    
    # Add educational facts as atomic facts
    print("📝 Adding educational facts...")
    for fact in educational_facts:
        try:
            cursor.execute("""
                INSERT INTO nodes 
                (node_type, content, embedding, activation_strength, firing_rate, 
                 last_activation, activation_count, creation_time, metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'atomic_fact', fact, '[]', 0.9, 0.0, time.time(), 1, time.time(),
                '{"source": "educational", "priority": "high"}', 'curated_educational'
            ))
            nodes_added += 1
        except Exception as e:
            print(f"Error adding fact: {e}")
    
    # Add programming concepts
    print("💻 Adding programming concepts...")
    for concept_data in programming_concepts:
        try:
            # Add concept node
            cursor.execute("""
                INSERT INTO nodes 
                (node_type, content, embedding, activation_strength, firing_rate, 
                 last_activation, activation_count, creation_time, metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'concept', concept_data['concept'], '[]', 0.9, 0.0, time.time(), 1, time.time(),
                '{"source": "programming", "priority": "high"}', 'curated_educational'
            ))
            concept_id = cursor.lastrowid
            
            # Add code example
            cursor.execute("""
                INSERT INTO nodes 
                (node_type, content, embedding, activation_strength, firing_rate, 
                 last_activation, activation_count, creation_time, metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'code', concept_data['example'], '[]', 0.9, 0.0, time.time(), 1, time.time(),
                '{"source": "programming", "priority": "high", "language": "python"}', 'curated_educational'
            ))
            code_id = cursor.lastrowid
            
            # Connect concept to code example
            cursor.execute("""
                INSERT INTO edges 
                (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                 last_coactivation, creation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"concept_code_{concept_id}_{code_id}", concept_id, code_id, 'example_relationship',
                1.0, 1, time.time(), time.time()
            ))
            
            nodes_added += 2
            connections_added += 1
            
        except Exception as e:
            print(f"Error adding programming concept: {e}")
    
    # Add technology definitions
    print("🔬 Adding technology definitions...")
    for term, definition in tech_definitions:
        try:
            # Add term as concept
            cursor.execute("""
                INSERT INTO nodes 
                (node_type, content, embedding, activation_strength, firing_rate, 
                 last_activation, activation_count, creation_time, metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'concept', term, '[]', 0.9, 0.0, time.time(), 1, time.time(),
                '{"source": "technology", "priority": "high", "type": "term"}', 'curated_educational'
            ))
            term_id = cursor.lastrowid
            
            # Add definition as atomic fact
            cursor.execute("""
                INSERT INTO nodes 
                (node_type, content, embedding, activation_strength, firing_rate, 
                 last_activation, activation_count, creation_time, metadata, modality_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'atomic_fact', f"{term}: {definition}", '[]', 0.9, 0.0, time.time(), 1, time.time(),
                '{"source": "technology", "priority": "high", "type": "definition"}', 'curated_educational'
            ))
            def_id = cursor.lastrowid
            
            # Connect term to definition
            cursor.execute("""
                INSERT INTO edges 
                (edge_id, source_id, target_id, edge_type, weight, coactivation_count, 
                 last_coactivation, creation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"term_def_{term_id}_{def_id}", term_id, def_id, 'definition_relationship',
                1.0, 1, time.time(), time.time()
            ))
            
            nodes_added += 2
            connections_added += 1
            
        except Exception as e:
            print(f"Error adding tech definition: {e}")
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print(f"✅ EDUCATIONAL DATA ADDED SUCCESSFULLY!")
    print(f"   📝 Nodes added: {nodes_added}")
    print(f"   🔗 Connections added: {connections_added}")
    
    return nodes_added, connections_added

if __name__ == "__main__":
    add_quality_educational_data()

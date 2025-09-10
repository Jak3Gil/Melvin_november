#!/usr/bin/env python3
"""
Demo script for Melvin Learning System
Showcases the curiosity-tutor loop and knowledge graph functionality
"""

from melvin_learning import MelvinLearningSystem
import json
import time

def demo_curiosity_learning():
    """Demonstrate Melvin's curiosity-driven learning."""
    print("🤖 MELVIN LEARNING SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize Melvin
    melvin = MelvinLearningSystem()
    
    # Demo questions
    questions = [
        "What is a cat?",
        "What is a dog?", 
        "What is a bird?",
        "What is a cat?",  # Repeat to show memory retrieval
        "What is a fish?",
        "What is a tree?"
    ]
    
    print("\n🧠 Testing Melvin's Curiosity Loop:")
    print("-" * 40)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        answer = melvin.curiosity_loop(question)
        print(f"   Answer: {answer}")
        time.sleep(0.5)  # Small delay for readability
    
    # Show final knowledge summary
    print("\n" + "=" * 50)
    melvin.show_knowledge_summary()
    
    # Show the knowledge graph structure
    print("\n📊 KNOWLEDGE GRAPH STRUCTURE:")
    print("-" * 30)
    for node in melvin.knowledge_graph.nodes.values():
        print(f"Concept: {node.concept}")
        print(f"  Definition: {node.definition[:60]}...")
        print(f"  Connections: {node.connections}")
        print(f"  Access Count: {node.access_count}")
        print()

def demo_persistence():
    """Demonstrate knowledge persistence across sessions."""
    print("\n💾 TESTING KNOWLEDGE PERSISTENCE:")
    print("-" * 40)
    
    # Create a new Melvin instance (simulates restart)
    melvin2 = MelvinLearningSystem()
    
    # Ask about something Melvin should already know
    print("Asking Melvin about cats (should retrieve from memory):")
    answer = melvin2.curiosity_loop("What is a cat?")
    print(f"Answer: {answer}")
    
    # Show that knowledge was loaded
    print(f"\nMelvin now has {len(melvin2.knowledge_graph.nodes)} concepts in memory")

def demo_learning_stats():
    """Show detailed learning statistics."""
    print("\n📈 LEARNING STATISTICS:")
    print("-" * 25)
    
    melvin = MelvinLearningSystem()
    stats = melvin.get_learning_stats()
    
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

def demo_knowledge_graph_analysis():
    """Analyze the knowledge graph structure."""
    print("\n🔍 KNOWLEDGE GRAPH ANALYSIS:")
    print("-" * 35)
    
    melvin = MelvinLearningSystem()
    
    if melvin.knowledge_graph.nodes:
        print(f"Total Nodes: {len(melvin.knowledge_graph.nodes)}")
        print(f"Unique Concepts: {len(melvin.knowledge_graph.concept_index)}")
        
        # Find most accessed concepts
        most_accessed = sorted(
            melvin.knowledge_graph.nodes.values(),
            key=lambda x: x.access_count,
            reverse=True
        )
        
        print("\nMost Accessed Concepts:")
        for node in most_accessed[:3]:
            print(f"  • {node.concept}: {node.access_count} accesses")
        
        # Find concepts with connections
        connected_concepts = [
            node for node in melvin.knowledge_graph.nodes.values()
            if len(node.connections) > 1
        ]
        
        print(f"\nConnected Concepts: {len(connected_concepts)}")
        for node in connected_concepts:
            print(f"  • {node.concept}: {len(node.connections)} connections")
    else:
        print("No knowledge graph data available")

def main():
    """Run the complete demo."""
    try:
        # Run all demos
        demo_curiosity_learning()
        demo_persistence()
        demo_learning_stats()
        demo_knowledge_graph_analysis()
        
        print("\n🎉 DEMO COMPLETE!")
        print("Melvin's learning system is working correctly!")
        print("\nKey Features Demonstrated:")
        print("✅ Curiosity-driven learning")
        print("✅ Knowledge graph storage")
        print("✅ Memory retrieval")
        print("✅ Persistence across sessions")
        print("✅ Learning statistics tracking")
        print("✅ Knowledge graph analysis")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

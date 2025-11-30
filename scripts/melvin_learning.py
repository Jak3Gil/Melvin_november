#!/usr/bin/env python3
"""
Melvin Learning Module - Curiosity-Tutor Loop with Knowledge Graph

A humanoid robot AI that learns by curiosity, asking questions when he doesn't know something,
and using Ollama as his tutor to build a persistent knowledge graph.

Author: Melvin AI Development Team
Version: 1.0
"""

import json
import uuid
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import sys


@dataclass
class KnowledgeNode:
    """Represents a single concept in Melvin's knowledge graph."""
    id: str
    concept: str
    definition: str
    connections: List[str]
    source: str
    confidence: float
    created_at: str
    last_accessed: str
    access_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeNode':
        """Create node from dictionary."""
        return cls(**data)


class MelvinKnowledgeGraph:
    """Melvin's persistent knowledge graph system."""
    
    def __init__(self, nodes_file: str = "nodes.json"):
        self.nodes_file = nodes_file
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.concept_index: Dict[str, str] = {}  # concept -> node_id mapping
        self.load_knowledge()
    
    def load_knowledge(self) -> None:
        """Load existing knowledge from nodes.json."""
        if os.path.exists(self.nodes_file):
            try:
                with open(self.nodes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for node_data in data.get('nodes', []):
                        node = KnowledgeNode.from_dict(node_data)
                        self.nodes[node.id] = node
                        self.concept_index[node.concept.lower()] = node.id
                print(f"🧠 Loaded {len(self.nodes)} knowledge nodes from {self.nodes_file}")
            except Exception as e:
                print(f"⚠️ Error loading knowledge: {e}")
                self.nodes = {}
                self.concept_index = {}
    
    def save_knowledge(self) -> None:
        """Save current knowledge to nodes.json."""
        try:
            data = {
                'nodes': [node.to_dict() for node in self.nodes.values()],
                'metadata': {
                    'total_nodes': len(self.nodes),
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            with open(self.nodes_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(self.nodes)} knowledge nodes to {self.nodes_file}")
        except Exception as e:
            print(f"❌ Error saving knowledge: {e}")
    
    def find_concept(self, concept: str) -> Optional[KnowledgeNode]:
        """Find a concept in the knowledge graph."""
        concept_lower = concept.lower().strip()
        node_id = self.concept_index.get(concept_lower)
        if node_id:
            node = self.nodes[node_id]
            node.last_accessed = datetime.now().isoformat()
            node.access_count += 1
            return node
        return None
    
    def add_node(self, node: KnowledgeNode) -> None:
        """Add a new node to the knowledge graph."""
        self.nodes[node.id] = node
        self.concept_index[node.concept.lower()] = node.id
        print(f"➕ Added concept: {node.concept}")
    
    def find_related_concepts(self, concept: str) -> List[KnowledgeNode]:
        """Find concepts related to the given concept."""
        related = []
        concept_lower = concept.lower()
        
        for node in self.nodes.values():
            # Check if concept appears in connections or definition
            if (concept_lower in node.definition.lower() or 
                concept_lower in [conn.lower() for conn in node.connections] or
                concept_lower in node.concept.lower()):
                related.append(node)
        
        return related


class OllamaTutor:
    """Interface to Ollama AI tutor."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama2"  # Default model
    
    def ask_ollama(self, question: str) -> str:
        """
        Ask Ollama a question and get a response.
        This is a placeholder implementation - in production, you'd use the Ollama API.
        """
        # Simulate Ollama responses for demo purposes
        responses = {
            "cat": "A cat is a small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws. Cats are popular pets and are known for their independence and hunting abilities.",
            "dog": "A dog is a domesticated carnivorous mammal that is commonly kept as a pet. Dogs are known for their loyalty, intelligence, and ability to be trained for various tasks.",
            "bird": "A bird is a warm-blooded vertebrate animal with feathers, wings, and a beak. Most birds can fly, and they lay hard-shelled eggs.",
            "fish": "A fish is a cold-blooded aquatic vertebrate animal with gills and fins. Fish live in water and breathe through their gills.",
            "tree": "A tree is a perennial plant with an elongated stem, or trunk, supporting branches and leaves. Trees are important for producing oxygen and providing habitat for wildlife.",
            "car": "A car is a wheeled motor vehicle used for transportation. Cars typically have four wheels and are powered by an internal combustion engine or electric motor.",
            "computer": "A computer is an electronic device that processes data according to instructions. Computers can perform calculations, store information, and communicate with other devices.",
            "book": "A book is a written or printed work consisting of pages bound together. Books contain information, stories, or other content and are used for education and entertainment.",
            "house": "A house is a building designed for people to live in. Houses provide shelter and typically contain rooms for sleeping, cooking, and other activities.",
            "water": "Water is a transparent, odorless, tasteless liquid that is essential for life. Water covers about 71% of Earth's surface and is vital for all living organisms."
        }
        
        # Extract key concept from question
        question_lower = question.lower()
        for concept, definition in responses.items():
            if concept in question_lower:
                return definition
        
        # Default response for unknown concepts
        return f"I don't have specific information about that topic, but I can help you learn more. Could you provide more context about what you'd like to know?"
    
    def extract_concept_and_definition(self, question: str, response: str) -> Tuple[str, str]:
        """Extract the main concept and definition from Ollama's response."""
        # Simple extraction - in production, you'd use more sophisticated NLP
        question_lower = question.lower()
        
        # Extract concept from question
        concept = self._extract_concept_from_question(question)
        
        # Clean up the definition
        definition = response.strip()
        
        return concept, definition
    
    def _extract_concept_from_question(self, question: str) -> str:
        """Extract the main concept from a question."""
        # Remove common question words
        question_words = ['what', 'is', 'a', 'an', 'the', 'are', 'do', 'does', 'how', 'why', 'when', 'where', 'tell', 'me', 'about']
        words = re.findall(r'\b\w+\b', question.lower())
        
        # Look for patterns like "what is X" or "what's X"
        if len(words) >= 3:
            if words[0] in ['what', 'whats'] and words[1] in ['is', 'are']:
                # Take the word after "what is/are"
                concept = words[2]
                # Skip articles
                if concept in ['a', 'an', 'the'] and len(words) > 3:
                    concept = words[3]
                return concept.capitalize()
        
        # Look for "what's X" pattern
        if len(words) >= 2:
            if words[0] in ['whats'] and len(words) > 1:
                concept = words[1]
                if concept in ['a', 'an', 'the'] and len(words) > 2:
                    concept = words[2]
                return concept.capitalize()
        
        # Find the first non-question word that's longer than 2 characters
        for word in words:
            if word not in question_words and len(word) > 2:
                return word.capitalize()
        
        # Fallback: use the last word
        return words[-1].capitalize() if words else "Unknown"


class MelvinLearningSystem:
    """Main learning system that implements the curiosity-tutor loop."""
    
    def __init__(self):
        self.knowledge_graph = MelvinKnowledgeGraph()
        self.ollama_tutor = OllamaTutor()
        self.learning_stats = {
            'questions_asked': 0,
            'new_concepts_learned': 0,
            'concepts_retrieved': 0,
            'total_nodes': 0
        }
    
    def melvin_knows(self, question: str) -> bool:
        """Check if Melvin already knows the answer to a question."""
        concept = self.ollama_tutor._extract_concept_from_question(question)
        return self.knowledge_graph.find_concept(concept) is not None
    
    def melvin_answer(self, question: str) -> str:
        """Retrieve answer from Melvin's knowledge graph."""
        concept = self.ollama_tutor._extract_concept_from_question(question)
        node = self.knowledge_graph.find_concept(concept)
        
        if node:
            self.learning_stats['concepts_retrieved'] += 1
            return node.definition
        
        return "I don't know the answer to that question."
    
    def ask_ollama(self, question: str) -> str:
        """Ask Ollama for information."""
        return self.ollama_tutor.ask_ollama(question)
    
    def create_node(self, concept: str, definition: str, connections: List[str]) -> KnowledgeNode:
        """Create a new knowledge node."""
        node = KnowledgeNode(
            id=str(uuid.uuid4()),
            concept=concept,
            definition=definition,
            connections=connections,
            source="ollama",
            confidence=0.8,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            access_count=0
        )
        return node
    
    def connect_to_graph(self, node: KnowledgeNode) -> None:
        """Add new node to graph and create connections with existing nodes."""
        # Add the node
        self.knowledge_graph.add_node(node)
        
        # Find and create connections with existing nodes
        related_nodes = self.knowledge_graph.find_related_concepts(node.concept)
        
        for related_node in related_nodes:
            # Add bidirectional connections
            if node.concept not in related_node.connections:
                related_node.connections.append(node.concept)
            if related_node.concept not in node.connections:
                node.connections.append(related_node.concept)
        
        self.learning_stats['new_concepts_learned'] += 1
        self.learning_stats['total_nodes'] = len(self.knowledge_graph.nodes)
    
    def curiosity_loop(self, question: str) -> str:
        """
        Main curiosity-tutor loop:
        1. Check if Melvin knows the answer
        2. If not, ask Ollama
        3. Create new knowledge node
        4. Connect to existing knowledge
        5. Return the answer
        """
        self.learning_stats['questions_asked'] += 1
        
        print(f"🤔 Melvin is thinking about: {question}")
        
        # Check if Melvin already knows
        if self.melvin_knows(question):
            print("🧠 Melvin knows this! Retrieving from memory...")
            answer = self.melvin_answer(question)
            return answer
        
        # Melvin doesn't know - ask Ollama
        print("❓ Melvin doesn't know this. Asking Ollama tutor...")
        ollama_response = self.ask_ollama(question)
        
        # Extract concept and definition
        concept, definition = self.ollama_tutor.extract_concept_and_definition(question, ollama_response)
        
        # Create new knowledge node
        print(f"📚 Creating new knowledge node for: {concept}")
        node = self.create_node(concept, definition, [])
        
        # Connect to existing knowledge
        print("🔗 Connecting to existing knowledge...")
        self.connect_to_graph(node)
        
        # Save knowledge
        self.knowledge_graph.save_knowledge()
        
        # Return the answer
        answer = f"A {concept.lower()} is {definition.lower()}"
        print(f"✅ Melvin learned something new!")
        return answer
    
    def get_learning_stats(self) -> Dict:
        """Get current learning statistics."""
        return {
            **self.learning_stats,
            'knowledge_graph_size': len(self.knowledge_graph.nodes),
            'unique_concepts': len(self.knowledge_graph.concept_index)
        }
    
    def show_knowledge_summary(self) -> None:
        """Display a summary of Melvin's current knowledge."""
        print("\n📊 MELVIN'S KNOWLEDGE SUMMARY")
        print("=" * 40)
        print(f"Total Concepts: {len(self.knowledge_graph.nodes)}")
        print(f"Questions Asked: {self.learning_stats['questions_asked']}")
        print(f"New Concepts Learned: {self.learning_stats['new_concepts_learned']}")
        print(f"Concepts Retrieved: {self.learning_stats['concepts_retrieved']}")
        
        if self.knowledge_graph.nodes:
            print("\n🧠 Recent Knowledge:")
            recent_nodes = sorted(
                self.knowledge_graph.nodes.values(),
                key=lambda x: x.created_at,
                reverse=True
            )[:5]
            
            for node in recent_nodes:
                print(f"  • {node.concept}: {node.definition[:50]}...")
        
        print("=" * 40)


def main():
    """CLI demo of Melvin's learning system."""
    if len(sys.argv) < 2:
        print("Usage: python melvin_learning.py \"What is a cat?\"")
        print("Example: python melvin_learning.py \"What is a dog?\"")
        sys.exit(1)
    
    question = sys.argv[1]
    
    # Initialize Melvin's learning system
    melvin = MelvinLearningSystem()
    
    print("🤖 Melvin Learning System Initialized")
    print("=" * 50)
    
    # Run the curiosity loop
    answer = melvin.curiosity_loop(question)
    
    print(f"\n🎯 Answer: {answer}")
    
    # Show learning stats
    melvin.show_knowledge_summary()
    
    # Interactive mode
    print("\n🔄 Interactive Mode (type 'quit' to exit, 'stats' for summary)")
    while True:
        try:
            user_input = input("\nAsk Melvin: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'stats':
                melvin.show_knowledge_summary()
            elif user_input:
                answer = melvin.curiosity_loop(user_input)
                print(f"🎯 Answer: {answer}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Final save
    melvin.knowledge_graph.save_knowledge()
    print("\n💾 Knowledge saved to nodes.json")


if __name__ == "__main__":
    main()

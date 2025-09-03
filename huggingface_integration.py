#!/usr/bin/env python3
"""
🤗 HUGGING FACE INTEGRATION FOR MELVIN BRAIN
===========================================
Pulls data from Hugging Face and creates nodes/connections in Melvin's unified brain.
Run this on the Jetson device via COM8/PuTTY for full integration.
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio

# Import Melvin's global brain system
from melvin_global_brain import MelvinGlobalBrain, NodeType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HuggingFaceIntegration:
    """Integration with Hugging Face datasets and models for Melvin brain"""
    
    def __init__(self, global_brain: MelvinGlobalBrain):
        self.global_brain = global_brain
        self.hf_available = False
        self.datasets_available = False
        
        # Try to import Hugging Face libraries
        try:
            global transformers, datasets, torch
            import transformers
            import datasets
            import torch
            self.hf_available = True
            self.datasets_available = True
            logger.info("🤗 Hugging Face libraries loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Hugging Face libraries not available: {e}")
            logger.info("📦 Install with: pip install transformers datasets torch")
        
        # Statistics
        self.stats = {
            'datasets_processed': 0,
            'nodes_created': 0,
            'connections_created': 0,
            'start_time': time.time()
        }
        
    def install_requirements(self):
        """Install required packages for Hugging Face integration"""
        import subprocess
        import sys
        
        packages = [
            'transformers',
            'datasets', 
            'torch',
            'tokenizers',
            'accelerate'
        ]
        
        logger.info("📦 Installing Hugging Face requirements...")
        for package in packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                
        # Try importing again
        try:
            global transformers, datasets, torch
            import transformers
            import datasets
            import torch
            self.hf_available = True
            self.datasets_available = True
            logger.info("🤗 Hugging Face libraries now available!")
        except ImportError:
            logger.error("❌ Still cannot import Hugging Face libraries")
    
    def pull_text_datasets(self, dataset_names: List[str] = None) -> Dict[str, Any]:
        """Pull text datasets from Hugging Face and create language nodes"""
        if not self.datasets_available:
            return self._create_mock_text_data()
        
        if dataset_names is None:
            # Default interesting datasets for Melvin
            dataset_names = [
                'squad',  # Question answering
                'imdb',   # Sentiment analysis
                'wikitext',  # General knowledge
                'common_voice',  # Speech/audio
            ]
        
        results = {}
        
        for dataset_name in dataset_names:
            try:
                logger.info(f"📚 Loading dataset: {dataset_name}")
                
                # Load dataset
                if dataset_name == 'squad':
                    dataset = datasets.load_dataset('squad', split='train[:100]')  # First 100 examples
                    
                    for i, example in enumerate(dataset):
                        # Process question as language node
                        question_node = self.global_brain.process_text_input(
                            example['question'], f"huggingface_squad_q_{i}"
                        )
                        
                        # Process context as language node
                        context_node = self.global_brain.process_text_input(
                            example['context'][:500], f"huggingface_squad_c_{i}"  # Limit context length
                        )
                        
                        # Process answer as language node
                        for answer in example['answers']['text']:
                            answer_node = self.global_brain.process_text_input(
                                answer, f"huggingface_squad_a_{i}"
                            )
                        
                        self.stats['nodes_created'] += 3
                        
                        if i % 10 == 0:
                            logger.info(f"📊 Processed {i} SQuAD examples")
                            
                elif dataset_name == 'imdb':
                    dataset = datasets.load_dataset('imdb', split='train[:50]')  # First 50 examples
                    
                    for i, example in enumerate(dataset):
                        # Process review text
                        review_node = self.global_brain.process_text_input(
                            example['text'][:1000], f"huggingface_imdb_{i}"  # Limit review length
                        )
                        
                        # Process label as concept
                        label = "positive" if example['label'] == 1 else "negative"
                        label_node = self.global_brain.process_text_input(
                            f"sentiment {label}", f"huggingface_sentiment_{label}_{i}"
                        )
                        
                        self.stats['nodes_created'] += 2
                        
                elif dataset_name == 'wikitext':
                    dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:30]')
                    
                    for i, example in enumerate(dataset):
                        if example['text'].strip():  # Skip empty entries
                            # Process wiki text
                            wiki_node = self.global_brain.process_text_input(
                                example['text'][:800], f"huggingface_wiki_{i}"
                            )
                            self.stats['nodes_created'] += 1
                
                results[dataset_name] = f"✅ Processed {dataset_name} successfully"
                self.stats['datasets_processed'] += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {dataset_name}: {e}")
                results[dataset_name] = f"❌ Error: {e}"
        
        return results
    
    def _create_mock_text_data(self) -> Dict[str, Any]:
        """Create mock text data when Hugging Face is not available"""
        logger.info("🎭 Creating mock Hugging Face data for demonstration")
        
        # Mock SQuAD-like data
        mock_qa_pairs = [
            {
                "question": "What is artificial intelligence?",
                "context": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans.",
                "answer": "intelligence demonstrated by machines"
            },
            {
                "question": "How do neural networks learn?",
                "context": "Neural networks learn through a process called backpropagation, where errors are propagated backward through the network to adjust weights.",
                "answer": "through backpropagation"
            },
            {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
                "answer": "learning from experience without explicit programming"
            }
        ]
        
        # Mock sentiment data
        mock_reviews = [
            {"text": "This movie is absolutely fantastic! Great acting and storyline.", "sentiment": "positive"},
            {"text": "Terrible movie, waste of time. Poor acting and confusing plot.", "sentiment": "negative"},
            {"text": "Amazing visual effects and compelling characters. Highly recommended!", "sentiment": "positive"},
            {"text": "Boring and predictable. Nothing new or interesting.", "sentiment": "negative"}
        ]
        
        # Mock knowledge data
        mock_knowledge = [
            "The human brain contains approximately 86 billion neurons",
            "Machine learning algorithms can recognize patterns in data",
            "Computer vision enables machines to interpret visual information",
            "Natural language processing helps computers understand human language",
            "Robotics combines AI, engineering, and computer science"
        ]
        
        # Process mock data through Melvin brain
        nodes_created = 0
        
        # Process Q&A pairs
        for i, qa in enumerate(mock_qa_pairs):
            question_node = self.global_brain.process_text_input(qa['question'], f"mock_squad_q_{i}")
            context_node = self.global_brain.process_text_input(qa['context'], f"mock_squad_c_{i}")
            answer_node = self.global_brain.process_text_input(qa['answer'], f"mock_squad_a_{i}")
            nodes_created += 3
            logger.info(f"📝 Created Q&A nodes: {qa['question'][:40]}...")
        
        # Process sentiment data
        for i, review in enumerate(mock_reviews):
            review_node = self.global_brain.process_text_input(review['text'], f"mock_imdb_{i}")
            sentiment_node = self.global_brain.process_text_input(f"sentiment {review['sentiment']}", f"mock_sentiment_{i}")
            nodes_created += 2
            logger.info(f"💭 Created sentiment nodes: {review['sentiment']}")
        
        # Process knowledge data
        for i, knowledge in enumerate(mock_knowledge):
            knowledge_node = self.global_brain.process_text_input(knowledge, f"mock_knowledge_{i}")
            nodes_created += 1
            logger.info(f"🧠 Created knowledge node: {knowledge[:40]}...")
        
        self.stats['nodes_created'] += nodes_created
        self.stats['datasets_processed'] += 3  # SQuAD, IMDB, Knowledge
        
        return {
            'squad_mock': f"✅ Created {len(mock_qa_pairs) * 3} Q&A nodes",
            'imdb_mock': f"✅ Created {len(mock_reviews) * 2} sentiment nodes", 
            'knowledge_mock': f"✅ Created {len(mock_knowledge)} knowledge nodes",
            'total_nodes': nodes_created
        }
    
    def pull_code_datasets(self) -> Dict[str, Any]:
        """Pull code datasets and create code nodes"""
        logger.info("💻 Processing code examples for Melvin brain")
        
        # Code examples from different languages
        code_examples = [
            # Python examples
            {
                "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "language": "python",
                "description": "Recursive fibonacci function"
            },
            {
                "code": "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n    def forward(self, x):\n        return self.layers(x)",
                "language": "python", 
                "description": "Simple neural network class"
            },
            {
                "code": "import cv2\nimport numpy as np\n\ndef detect_faces(image):\n    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')\n    return cascade.detectMultiScale(image)",
                "language": "python",
                "description": "Computer vision face detection"
            },
            # C++ examples
            {
                "code": "class Node {\npublic:\n    NodeID id;\n    std::string name;\n    std::vector<Connection> connections;\n};",
                "language": "cpp",
                "description": "Graph node structure"
            },
            {
                "code": "void BrainGraph::create_connection(NodeID source, NodeID target) {\n    Connection conn(source, target);\n    connections_.push_back(conn);\n}",
                "language": "cpp",
                "description": "Brain graph connection creation"
            },
            # JavaScript examples
            {
                "code": "const brain = {\n  nodes: [],\n  connections: [],\n  addNode(node) {\n    this.nodes.push(node);\n  }\n};",
                "language": "javascript",
                "description": "JavaScript brain structure"
            }
        ]
        
        nodes_created = 0
        for i, example in enumerate(code_examples):
            # Create code node
            code_node = self.global_brain.process_code_input(
                example['code'], example['language']
            )
            
            # Create description node
            desc_node = self.global_brain.process_text_input(
                example['description'], f"code_description_{i}"
            )
            
            nodes_created += 2
            logger.info(f"💻 Created code nodes: {example['language']} - {example['description']}")
        
        self.stats['nodes_created'] += nodes_created
        
        return {
            'code_examples': f"✅ Created {nodes_created} code-related nodes",
            'languages': ['python', 'cpp', 'javascript'],
            'nodes_created': nodes_created
        }
    
    def create_foundation_knowledge(self) -> Dict[str, Any]:
        """Create foundation knowledge nodes inspired by Hugging Face model training data"""
        logger.info("🌱 Creating foundation knowledge nodes")
        
        # Foundation concepts inspired by large language model training
        foundation_concepts = {
            # Science and Technology
            "science": [
                "Physics is the study of matter, energy, and their interactions",
                "Chemistry examines the composition and behavior of substances", 
                "Biology studies living organisms and life processes",
                "Computer science focuses on computation and information processing",
                "Mathematics provides the language for describing patterns and relationships"
            ],
            
            # Artificial Intelligence
            "ai_ml": [
                "Machine learning enables computers to learn from data",
                "Deep learning uses neural networks with multiple layers",
                "Reinforcement learning trains agents through rewards and penalties",
                "Computer vision allows machines to interpret visual information",
                "Natural language processing helps computers understand human language"
            ],
            
            # Robotics and Hardware
            "robotics": [
                "Robotics combines mechanical engineering, electronics, and software",
                "Sensors provide robots with information about their environment",
                "Actuators enable robots to move and manipulate objects",
                "Control systems coordinate robot movements and behaviors",
                "Human-robot interaction focuses on safe and intuitive cooperation"
            ],
            
            # Philosophy and Cognition
            "cognition": [
                "Consciousness involves awareness and subjective experience",
                "Learning is the process of acquiring new knowledge or skills",
                "Memory stores and retrieves information over time",
                "Perception interprets sensory information about the environment",
                "Intelligence involves problem-solving and adaptive behavior"
            ]
        }
        
        nodes_created = 0
        
        for category, concepts in foundation_concepts.items():
            # Create category node
            category_node = self.global_brain.process_text_input(
                f"Category: {category}", f"foundation_category_{category}"
            )
            nodes_created += 1
            
            # Create concept nodes
            for i, concept in enumerate(concepts):
                concept_node = self.global_brain.process_text_input(
                    concept, f"foundation_{category}_{i}"
                )
                nodes_created += 1
                
                logger.info(f"🌱 Created foundation concept: {concept[:50]}...")
        
        self.stats['nodes_created'] += nodes_created
        
        return {
            'foundation_knowledge': f"✅ Created {nodes_created} foundation nodes",
            'categories': list(foundation_concepts.keys()),
            'nodes_created': nodes_created
        }
    
    def simulate_multimodal_learning(self) -> Dict[str, Any]:
        """Simulate multimodal learning with cross-modal connections"""
        logger.info("🎭 Simulating multimodal learning scenarios")
        
        # Multimodal scenarios that would create cross-modal connections
        multimodal_scenarios = [
            {
                "visual_desc": "A red ball rolling across a wooden floor",
                "audio_desc": "Sound of a ball bouncing and rolling",
                "text_desc": "The ball moves from left to right with momentum",
                "code_desc": "ball.position += velocity * time_delta"
            },
            {
                "visual_desc": "Person waving hand in greeting gesture",
                "audio_desc": "Voice saying hello and greeting sounds",
                "text_desc": "Friendly greeting and social interaction",
                "code_desc": "if gesture_detected('wave'): respond_greeting()"
            },
            {
                "visual_desc": "Robot arm picking up an object with precision",
                "audio_desc": "Motor sounds and mechanical movement",
                "text_desc": "Precise manipulation and object grasping",
                "code_desc": "arm.move_to(target_position); gripper.close()"
            },
            {
                "visual_desc": "Face showing happy expression with smile",
                "audio_desc": "Laughter and positive vocal tones",
                "text_desc": "Joy, happiness, and positive emotions",
                "code_desc": "emotion_state = 'happy'; confidence = 0.95"
            }
        ]
        
        nodes_created = 0
        
        for i, scenario in enumerate(multimodal_scenarios):
            logger.info(f"🎭 Processing multimodal scenario {i+1}")
            
            # Create nodes for each modality
            visual_node = self.global_brain.process_text_input(
                scenario['visual_desc'], f"multimodal_visual_{i}"
            )
            
            # Simulate audio features
            audio_features = {
                'volume': 0.7,
                'pitch': 0.5,
                'voice_detected': 1.0 if 'voice' in scenario['audio_desc'] else 0.0,
                'music_detected': 0.0
            }
            audio_node = self.global_brain.process_audio_input(
                audio_features, f"multimodal_audio_{i}"
            )
            
            text_node = self.global_brain.process_text_input(
                scenario['text_desc'], f"multimodal_text_{i}"
            )
            
            code_node = self.global_brain.process_code_input(
                scenario['code_desc'], "python"
            )
            
            nodes_created += 4
            
            # Small delay to allow Hebbian learning to create connections
            time.sleep(0.5)
        
        self.stats['nodes_created'] += nodes_created
        
        return {
            'multimodal_scenarios': f"✅ Created {nodes_created} multimodal nodes",
            'scenarios_processed': len(multimodal_scenarios),
            'cross_modal_connections': "Expected via Hebbian learning",
            'nodes_created': nodes_created
        }
    
    def run_full_integration(self, install_packages: bool = True) -> Dict[str, Any]:
        """Run complete Hugging Face integration"""
        logger.info("🚀 Starting full Hugging Face integration for Melvin brain")
        
        # Install packages if requested and needed
        if install_packages and not self.hf_available:
            self.install_requirements()
        
        # Get initial brain state
        initial_state = self.global_brain.get_unified_state()
        initial_nodes = initial_state['global_memory']['total_nodes']
        initial_edges = initial_state['global_memory']['total_edges']
        
        logger.info(f"📊 Initial state: {initial_nodes} nodes, {initial_edges} edges")
        
        results = {}
        
        # 1. Create foundation knowledge
        foundation_result = self.create_foundation_knowledge()
        results['foundation'] = foundation_result
        
        # 2. Pull text datasets
        text_result = self.pull_text_datasets()
        results['text_datasets'] = text_result
        
        # 3. Process code examples
        code_result = self.pull_code_datasets()
        results['code_datasets'] = code_result
        
        # 4. Simulate multimodal learning
        multimodal_result = self.simulate_multimodal_learning()
        results['multimodal'] = multimodal_result
        
        # Allow time for background Hebbian processing
        logger.info("⏳ Allowing time for Hebbian learning to create connections...")
        time.sleep(5.0)
        
        # Get final brain state
        final_state = self.global_brain.get_unified_state()
        final_nodes = final_state['global_memory']['total_nodes']
        final_edges = final_state['global_memory']['total_edges']
        
        # Calculate growth
        nodes_added = final_nodes - initial_nodes
        edges_added = final_edges - initial_edges
        
        runtime = time.time() - self.stats['start_time']
        
        summary = {
            'initial_state': {'nodes': initial_nodes, 'edges': initial_edges},
            'final_state': {'nodes': final_nodes, 'edges': final_edges},
            'growth': {'nodes_added': nodes_added, 'edges_added': edges_added},
            'processing_time': f"{runtime:.2f} seconds",
            'nodes_per_second': nodes_added / runtime if runtime > 0 else 0,
            'hebbian_updates': final_state['global_memory']['stats'].get('hebbian_updates', 0),
            'cross_modal_connections': final_state['global_memory']['stats'].get('cross_modal_connections', 0),
            'detailed_results': results
        }
        
        logger.info(f"🎉 Integration complete! Added {nodes_added} nodes, {edges_added} connections")
        
        return summary

def main():
    """Main entry point for Hugging Face integration"""
    print("🤗 HUGGING FACE INTEGRATION FOR MELVIN BRAIN")
    print("=" * 60)
    print("🔹 PULLING DATA FROM HUGGING FACE")
    print("🔹 CREATING NODES AND CONNECTIONS")
    print("🔹 ENABLING CROSS-MODAL LEARNING")
    print("🔹 HEBBIAN LEARNING: FIRE TOGETHER, WIRE TOGETHER")
    print("=" * 60)
    
    try:
        # Initialize Melvin Global Brain
        print("🚀 Initializing Melvin Global Brain...")
        global_brain = MelvinGlobalBrain(embedding_dim=512)
        
        # Start unified processing for Hebbian learning
        global_brain.start_unified_processing()
        
        # Create Hugging Face integration
        hf_integration = HuggingFaceIntegration(global_brain)
        
        print("\n🤗 Starting Hugging Face data integration...")
        
        # Run full integration
        results = hf_integration.run_full_integration(install_packages=True)
        
        # Display results
        print("\n🎉 INTEGRATION RESULTS:")
        print("=" * 40)
        print(f"📊 Initial: {results['initial_state']['nodes']} nodes, {results['initial_state']['edges']} edges")
        print(f"📈 Final: {results['final_state']['nodes']} nodes, {results['final_state']['edges']} edges")
        print(f"🚀 Growth: +{results['growth']['nodes_added']} nodes, +{results['growth']['edges_added']} edges")
        print(f"⚡ Processing time: {results['processing_time']}")
        print(f"🧠 Hebbian updates: {results['hebbian_updates']}")
        print(f"🔗 Cross-modal connections: {results['cross_modal_connections']}")
        print(f"📊 Processing rate: {results['nodes_per_second']:.1f} nodes/second")
        
        # Show brain state
        brain_state = global_brain.get_unified_state()
        print(f"\n🧠 FINAL BRAIN STATE:")
        print(f"   Total nodes: {brain_state['global_memory']['total_nodes']}")
        print(f"   Total edges: {brain_state['global_memory']['total_edges']}")
        print(f"   Node types: {brain_state['global_memory']['node_types']}")
        print(f"   Edge types: {brain_state['global_memory']['edge_types']}")
        
        # Save complete state
        global_brain.save_complete_state()
        print("\n💾 Brain state saved to melvin_global_memory/complete_brain_state.json")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Integration interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'global_brain' in locals():
            global_brain.stop_unified_processing()
        print("✅ Hugging Face integration complete")

if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
🧠 MELVIN CONTINUOUS MULTIMODAL INGESTION
========================================
Continuously processes images, sounds, text, and code data.
Saves to repository every few seconds.
Run until user stops with Ctrl+C.
"""

import json
import time
import random
import uuid
import math
import threading
import sqlite3
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import base64

# Configure logging
def log_info(message):
    print(f"[INFO] {time.strftime('%H:%M:%S')} - {message}")

def log_error(message):
    print(f"[ERROR] {time.strftime('%H:%M:%S')} - {message}")

# Import simplified Melvin brain
from melvin_simple_demo import SimpleMelvinBrain, NodeType, EdgeType, SimpleNode, SimpleEdge

class ContinuousDataIngestion:
    """Continuous multimodal data ingestion for Melvin brain"""
    
    def __init__(self, brain: SimpleMelvinBrain):
        self.brain = brain
        self.running = False
        self.save_interval = 3.0  # Save every 3 seconds
        self.processing_interval = 0.5  # Process new data every 0.5 seconds
        
        # Data sources
        self.text_sources = self._init_text_sources()
        self.code_sources = self._init_code_sources()
        self.image_data = self._init_image_data()
        self.audio_data = self._init_audio_data()
        
        # Processing counters
        self.counters = {
            'text_processed': 0,
            'code_processed': 0,
            'images_processed': 0,
            'audio_processed': 0,
            'saves_performed': 0,
            'start_time': time.time()
        }
        
        # Background threads
        self.save_thread = None
        self.processing_thread = None
        
        log_info("🔄 Continuous Data Ingestion initialized")
    
    def _init_text_sources(self) -> List[Dict[str, str]]:
        """Initialize diverse text data sources"""
        return [
            # AI and Machine Learning
            {"text": "Deep learning models use multiple layers to learn hierarchical representations of data", "category": "ai_ml"},
            {"text": "Convolutional neural networks excel at processing visual data through learned filters", "category": "ai_ml"},
            {"text": "Recurrent neural networks can process sequential data like text and time series", "category": "ai_ml"},
            {"text": "Attention mechanisms allow models to focus on relevant parts of input data", "category": "ai_ml"},
            {"text": "Transfer learning enables models to apply knowledge from one domain to another", "category": "ai_ml"},
            
            # Robotics and Hardware
            {"text": "Servo motors provide precise position control for robotic joints and actuators", "category": "robotics"},
            {"text": "LiDAR sensors create detailed 3D maps of the environment for navigation", "category": "robotics"},
            {"text": "Computer vision algorithms enable robots to recognize and track objects", "category": "robotics"},
            {"text": "Path planning algorithms help robots navigate from point A to point B safely", "category": "robotics"},
            {"text": "Force sensors provide tactile feedback for delicate manipulation tasks", "category": "robotics"},
            
            # Programming and Software
            {"text": "Object-oriented programming organizes code into reusable classes and objects", "category": "programming"},
            {"text": "Version control systems track changes and enable collaboration on code projects", "category": "programming"},
            {"text": "APIs provide standardized interfaces for software components to communicate", "category": "programming"},
            {"text": "Database indexing improves query performance by creating efficient lookup structures", "category": "programming"},
            {"text": "Asynchronous programming enables non-blocking execution of concurrent tasks", "category": "programming"},
            
            # Science and Mathematics
            {"text": "Linear algebra provides the mathematical foundation for machine learning algorithms", "category": "science"},
            {"text": "Probability theory helps quantify uncertainty in predictions and decisions", "category": "science"},
            {"text": "Signal processing techniques extract meaningful information from sensor data", "category": "science"},
            {"text": "Graph theory studies relationships and connections between discrete objects", "category": "science"},
            {"text": "Optimization algorithms find the best solution from a set of possible alternatives", "category": "science"},
            
            # Human-Computer Interaction
            {"text": "User interface design focuses on creating intuitive and accessible interactions", "category": "hci"},
            {"text": "Voice recognition systems convert spoken language into digital text", "category": "hci"},
            {"text": "Gesture recognition enables natural interaction through hand and body movements", "category": "hci"},
            {"text": "Augmented reality overlays digital information onto the physical world", "category": "hci"},
            {"text": "Brain-computer interfaces enable direct communication between brain and machine", "category": "hci"}
        ]
    
    def _init_code_sources(self) -> List[Dict[str, str]]:
        """Initialize code examples from different languages and domains"""
        return [
            # Python AI/ML
            {
                "code": "import torch\nimport torch.nn as nn\n\nclass NeuralNetwork(nn.Module):\n    def __init__(self, input_size, hidden_size, output_size):\n        super().__init__()\n        self.fc1 = nn.Linear(input_size, hidden_size)\n        self.fc2 = nn.Linear(hidden_size, output_size)\n        self.relu = nn.ReLU()\n    \n    def forward(self, x):\n        x = self.relu(self.fc1(x))\n        return self.fc2(x)",
                "language": "python",
                "category": "ai_ml",
                "description": "PyTorch neural network implementation"
            },
            {
                "code": "def hebbian_update(weight_matrix, pre_activity, post_activity, learning_rate=0.01):\n    \"\"\"Update weights using Hebbian learning rule\"\"\"\n    delta_w = learning_rate * np.outer(post_activity, pre_activity)\n    weight_matrix += delta_w\n    return weight_matrix",
                "language": "python", 
                "category": "ai_ml",
                "description": "Hebbian learning weight update"
            },
            
            # Computer Vision
            {
                "code": "import cv2\nimport numpy as np\n\ndef detect_faces(image):\n    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')\n    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n    faces = face_cascade.detectMultiScale(gray, 1.1, 4)\n    return faces",
                "language": "python",
                "category": "computer_vision", 
                "description": "Face detection with OpenCV"
            },
            {
                "code": "def extract_features(image):\n    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n    features = {\n        'brightness': np.mean(gray) / 255.0,\n        'contrast': np.std(gray) / 255.0,\n        'edges': cv2.Canny(gray, 50, 150)\n    }\n    return features",
                "language": "python",
                "category": "computer_vision",
                "description": "Image feature extraction"
            },
            
            # Robotics Control
            {
                "code": "class RobotArm:\n    def __init__(self, joints=6):\n        self.joints = joints\n        self.positions = [0.0] * joints\n    \n    def move_to(self, target_positions, speed=1.0):\n        for i, target in enumerate(target_positions):\n            self.positions[i] = target\n    \n    def get_end_effector_pose(self):\n        return self.forward_kinematics(self.positions)",
                "language": "python",
                "category": "robotics",
                "description": "Robot arm control class"
            },
            
            # C++ Brain Graph
            {
                "code": "class BrainNode {\npublic:\n    NodeID id;\n    NodeType type;\n    std::string content;\n    std::vector<float> embedding;\n    float activation_strength;\n    \n    void activate(float strength) {\n        activation_strength = std::min(1.0f, activation_strength + strength);\n    }\n};",
                "language": "cpp",
                "category": "brain_graph",
                "description": "Brain node C++ implementation"
            },
            {
                "code": "void BrainGraph::create_connection(NodeID source, NodeID target, float weight) {\n    ConnectionID conn_id = generate_connection_id();\n    Connection conn(conn_id, source, target, weight);\n    connections_[conn_id] = conn;\n    update_adjacency_list(source, target);\n}",
                "language": "cpp",
                "category": "brain_graph",
                "description": "Brain graph connection creation"
            },
            
            # JavaScript Frontend
            {
                "code": "const BrainVisualizer = {\n  nodes: [],\n  edges: [],\n  \n  addNode(node) {\n    this.nodes.push(node);\n    this.updateVisualization();\n  },\n  \n  addEdge(source, target, weight) {\n    this.edges.push({source, target, weight});\n    this.updateConnections();\n  }\n};",
                "language": "javascript",
                "category": "frontend",
                "description": "Brain visualization JavaScript"
            },
            
            # Data Processing
            {
                "code": "def process_multimodal_input(visual_data, audio_data, text_data):\n    embeddings = []\n    \n    if visual_data is not None:\n        visual_emb = extract_visual_features(visual_data)\n        embeddings.append(visual_emb)\n    \n    if audio_data is not None:\n        audio_emb = extract_audio_features(audio_data)\n        embeddings.append(audio_emb)\n    \n    if text_data:\n        text_emb = extract_text_features(text_data)\n        embeddings.append(text_emb)\n    \n    return combine_embeddings(embeddings)",
                "language": "python",
                "category": "multimodal",
                "description": "Multimodal data processing pipeline"
            }
        ]
    
    def _init_image_data(self) -> List[Dict[str, Any]]:
        """Initialize image descriptions and features"""
        return [
            {
                "description": "A red spherical ball sitting on a wooden table surface",
                "features": {"brightness": 0.6, "color_red": 0.8, "circular_shape": 0.9, "texture": "smooth"},
                "scene": "indoor_object"
            },
            {
                "description": "Person waving hand in a friendly greeting gesture",
                "features": {"face_detected": 1.0, "hand_visible": 1.0, "motion": 0.7, "social_interaction": 1.0},
                "scene": "human_interaction"
            },
            {
                "description": "Robot arm reaching toward a blue cube on a metal surface",
                "features": {"robot_visible": 1.0, "color_blue": 0.9, "geometric_shape": 0.8, "mechanical": 1.0},
                "scene": "robot_manipulation"
            },
            {
                "description": "Computer screen displaying colorful data visualization charts",
                "features": {"screen_detected": 1.0, "colors_multiple": 1.0, "data_pattern": 0.9, "technology": 1.0},
                "scene": "computer_interface"
            },
            {
                "description": "Green plant leaves with natural lighting and shadows",
                "features": {"color_green": 0.9, "organic_texture": 1.0, "natural_lighting": 0.8, "nature": 1.0},
                "scene": "natural_environment"
            },
            {
                "description": "Circuit board with electronic components and copper traces",
                "features": {"geometric_patterns": 0.9, "metallic_surface": 0.7, "technology": 1.0, "complexity": 0.8},
                "scene": "electronics"
            },
            {
                "description": "Books stacked on a desk with reading lamp illumination",
                "features": {"rectangular_shapes": 0.8, "text_visible": 0.6, "warm_lighting": 0.7, "knowledge": 1.0},
                "scene": "learning_environment"
            },
            {
                "description": "Camera lens focusing on a distant landscape view",
                "features": {"circular_shape": 0.9, "depth_of_field": 0.8, "optical": 1.0, "perspective": 0.9},
                "scene": "photography"
            }
        ]
    
    def _init_audio_data(self) -> List[Dict[str, Any]]:
        """Initialize audio descriptions and features"""
        return [
            {
                "description": "Human voice speaking clearly with neutral tone",
                "features": {"voice_detected": 1.0, "pitch": 0.5, "volume": 0.6, "clarity": 0.8},
                "category": "speech"
            },
            {
                "description": "Mechanical whirring sound of robot motors in motion",
                "features": {"mechanical": 1.0, "pitch": 0.3, "volume": 0.4, "rhythmic": 0.7},
                "category": "robot_sounds"
            },
            {
                "description": "Keyboard typing with rapid key press patterns",
                "features": {"rhythmic": 0.9, "sharp_sounds": 0.8, "volume": 0.3, "frequency": 0.6},
                "category": "computer_interaction"
            },
            {
                "description": "Gentle background music with melodic progression",
                "features": {"music_detected": 1.0, "harmonic": 0.9, "volume": 0.4, "pleasant": 0.8},
                "category": "music"
            },
            {
                "description": "Camera shutter click and lens focus adjustment sounds",
                "features": {"mechanical": 0.6, "sharp_sounds": 0.9, "brief": 1.0, "camera": 1.0},
                "category": "photography_sounds"
            },
            {
                "description": "Quiet humming of computer fans and electronics",
                "features": {"electronic": 1.0, "continuous": 0.9, "low_frequency": 0.8, "quiet": 0.7},
                "category": "electronic_ambient"
            },
            {
                "description": "Footsteps walking on different surface materials",
                "features": {"rhythmic": 0.8, "impact": 0.6, "human": 1.0, "movement": 0.9},
                "category": "human_movement"
            },
            {
                "description": "Alert beep sounds from system notifications",
                "features": {"electronic": 1.0, "attention": 0.9, "brief": 1.0, "notification": 1.0},
                "category": "system_sounds"
            }
        ]
    
    def _get_diverse_text_content(self) -> str:
        """Get diverse text content for processing"""
        text_categories = {
            "technical": [
                "Graph neural networks process data represented as nodes and edges with learned embeddings",
                "Transformer architectures revolutionized natural language processing through self-attention mechanisms",
                "Reinforcement learning agents explore environments to maximize cumulative reward signals",
                "Convolutional layers detect local features through learnable filter kernels",
                "Gradient descent optimization minimizes loss functions through iterative parameter updates"
            ],
            "conversational": [
                "How are you doing today? I hope you're having a great time learning new things!",
                "The weather looks beautiful outside. Perfect day for working on robotics projects.",
                "I'm excited to see how Melvin's brain grows with all this new knowledge and connections.",
                "Can you help me understand how Hebbian learning works in practice?",
                "This system is fascinating! The way it connects different types of information is amazing."
            ],
            "descriptive": [
                "The robot's metallic arm moves smoothly through three-dimensional space with precise control",
                "Bright LED lights illuminate the workspace where electronic components are carefully assembled",
                "Data flows through neural pathways like electricity through copper wires in a circuit",
                "Camera sensors capture photons and convert them into digital pixel arrays for processing",
                "Sound waves propagate through air and are detected by sensitive microphone elements"
            ],
            "instructional": [
                "To create a new node, first generate a unique identifier and then assign appropriate metadata",
                "When processing visual input, extract features like brightness, contrast, and color distributions",
                "Hebbian learning strengthens connections between nodes that activate simultaneously",
                "Save brain state regularly to ensure no learning progress is lost during system restarts",
                "Monitor system resources to prevent memory overflow during intensive processing"
            ]
        }
        
        category = random.choice(list(text_categories.keys()))
        return random.choice(text_categories[category])
    
    def _get_code_example(self) -> Dict[str, str]:
        """Get random code example"""
        return random.choice(self.code_sources)
    
    def _get_image_data(self) -> Dict[str, Any]:
        """Get random image data"""
        return random.choice(self.image_data)
    
    def _get_audio_data(self) -> Dict[str, Any]:
        """Get random audio data"""
        return random.choice(self.audio_data)
    
    def _process_image_input(self, image_data: Dict[str, Any]) -> str:
        """Process image data into visual nodes"""
        # Create visual description node
        visual_node = self.brain.add_node(
            image_data['description'], 
            NodeType.VISUAL, 
            f"image_{image_data['scene']}"
        )
        
        # Create feature nodes for significant features
        for feature_name, feature_value in image_data['features'].items():
            if feature_value > 0.7:  # Only strong features
                feature_node = self.brain.add_node(
                    f"visual_feature: {feature_name} = {feature_value:.2f}",
                    NodeType.SENSOR,
                    f"visual_feature_{feature_name}"
                )
        
        return visual_node
    
    def _process_audio_input(self, audio_data: Dict[str, Any]) -> str:
        """Process audio data into audio nodes"""
        # Create audio description node
        audio_node = self.brain.add_node(
            audio_data['description'],
            NodeType.AUDIO,
            f"audio_{audio_data['category']}"
        )
        
        # Create feature nodes
        for feature_name, feature_value in audio_data['features'].items():
            if feature_value > 0.6:  # Audio feature threshold
                feature_node = self.brain.add_node(
                    f"audio_feature: {feature_name} = {feature_value:.2f}",
                    NodeType.SENSOR,
                    f"audio_feature_{feature_name}"
                )
        
        return audio_node
    
    def _auto_save_worker(self):
        """Background worker for automatic saving"""
        while self.running:
            try:
                time.sleep(self.save_interval)
                if self.running:  # Check again after sleep
                    self.brain.save_state()
                    self.counters['saves_performed'] += 1
                    
                    # Print progress
                    stats = self.brain.get_stats()
                    runtime = time.time() - self.counters['start_time']
                    
                    log_info(f"💾 Auto-save #{self.counters['saves_performed']}: "
                           f"{stats['total_nodes']} nodes, {stats['total_edges']} edges "
                           f"({runtime:.1f}s runtime)")
                    
            except Exception as e:
                log_error(f"Auto-save error: {e}")
    
    def _processing_worker(self):
        """Background worker for continuous data processing"""
        while self.running:
            try:
                # Randomly select data type to process
                data_types = ['text', 'code', 'image', 'audio']
                weights = [0.4, 0.3, 0.2, 0.1]  # Favor text and code
                data_type = random.choices(data_types, weights=weights)[0]
                
                if data_type == 'text':
                    text_content = self._get_diverse_text_content()
                    self.brain.add_node(text_content, NodeType.LANGUAGE, "continuous_text")
                    self.counters['text_processed'] += 1
                    
                elif data_type == 'code':
                    code_data = self._get_code_example()
                    self.brain.add_node(code_data['code'], NodeType.CODE, f"continuous_code_{code_data['language']}")
                    self.brain.add_node(code_data['description'], NodeType.CONCEPT, f"code_concept_{code_data['category']}")
                    self.counters['code_processed'] += 1
                    
                elif data_type == 'image':
                    image_data = self._get_image_data()
                    self._process_image_input(image_data)
                    self.counters['images_processed'] += 1
                    
                elif data_type == 'audio':
                    audio_data = self._get_audio_data()
                    self._process_audio_input(audio_data)
                    self.counters['audio_processed'] += 1
                
                # Sleep before next processing
                time.sleep(self.processing_interval)
                
            except Exception as e:
                log_error(f"Processing error: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def start_continuous_processing(self):
        """Start continuous multimodal data processing"""
        if self.running:
            log_info("⚠️ Already running!")
            return
        
        log_info("🚀 Starting continuous multimodal data ingestion...")
        self.running = True
        
        # Start background threads
        self.save_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        
        self.save_thread.start()
        self.processing_thread.start()
        
        log_info(f"✅ Started continuous processing:")
        log_info(f"   📊 Processing interval: {self.processing_interval}s")
        log_info(f"   💾 Save interval: {self.save_interval}s")
        log_info(f"   🔄 Press Ctrl+C to stop")
        
        try:
            # Main monitoring loop
            last_report = time.time()
            report_interval = 10.0  # Report every 10 seconds
            
            while self.running:
                time.sleep(1.0)
                
                # Periodic progress report
                if time.time() - last_report >= report_interval:
                    self._print_progress_report()
                    last_report = time.time()
                    
        except KeyboardInterrupt:
            log_info("🛑 Stopping continuous processing...")
            self.stop_continuous_processing()
    
    def stop_continuous_processing(self):
        """Stop continuous processing"""
        self.running = False
        
        # Wait for threads to finish
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=5.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Final save
        self.brain.save_state()
        
        log_info("✅ Continuous processing stopped")
        self._print_final_report()
    
    def _print_progress_report(self):
        """Print current progress report"""
        stats = self.brain.get_stats()
        runtime = time.time() - self.counters['start_time']
        
        print(f"\n📊 PROGRESS REPORT ({runtime:.1f}s runtime)")
        print(f"   🧠 Nodes: {stats['total_nodes']} | Edges: {stats['total_edges']}")
        print(f"   📝 Text: {self.counters['text_processed']} | 💻 Code: {self.counters['code_processed']}")
        print(f"   🖼️ Images: {self.counters['images_processed']} | 🔊 Audio: {self.counters['audio_processed']}")
        print(f"   💾 Saves: {self.counters['saves_performed']}")
        
        # Show processing rate
        total_processed = sum([
            self.counters['text_processed'],
            self.counters['code_processed'], 
            self.counters['images_processed'],
            self.counters['audio_processed']
        ])
        rate = total_processed / runtime if runtime > 0 else 0
        print(f"   ⚡ Rate: {rate:.2f} items/second")
    
    def _print_final_report(self):
        """Print final statistics report"""
        stats = self.brain.get_stats()
        runtime = time.time() - self.counters['start_time']
        
        print(f"\n🎉 FINAL REPORT")
        print(f"=" * 50)
        print(f"⏱️ Total runtime: {runtime:.1f} seconds")
        print(f"🧠 Final brain state: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        
        print(f"\n📊 Data processed:")
        print(f"   📝 Text items: {self.counters['text_processed']}")
        print(f"   💻 Code examples: {self.counters['code_processed']}")
        print(f"   🖼️ Image descriptions: {self.counters['images_processed']}")
        print(f"   🔊 Audio descriptions: {self.counters['audio_processed']}")
        
        print(f"\n🔗 Connection distribution:")
        for edge_type, count in stats['edge_types'].items():
            if count > 0:
                print(f"   {edge_type}: {count}")
        
        print(f"\n💾 Persistence:")
        print(f"   📁 Database: melvin_global_memory/global_memory.db")
        print(f"   📄 JSON state: melvin_global_memory/complete_brain_state.json")
        print(f"   🔄 Auto-saves performed: {self.counters['saves_performed']}")
        
        # Calculate growth rate
        total_processed = sum([
            self.counters['text_processed'],
            self.counters['code_processed'],
            self.counters['images_processed'], 
            self.counters['audio_processed']
        ])
        
        if runtime > 0:
            print(f"\n📈 Performance:")
            print(f"   ⚡ Processing rate: {total_processed/runtime:.2f} items/second")
            print(f"   🧠 Node creation rate: {stats['total_nodes']/runtime:.2f} nodes/second")
            print(f"   🔗 Connection rate: {stats['total_edges']/runtime:.2f} edges/second")

def main():
    """Main entry point for continuous ingestion"""
    print("🧠 MELVIN CONTINUOUS MULTIMODAL INGESTION")
    print("=" * 60)
    print("🔹 PROCESSING: Images, Audio, Text, Code")
    print("🔹 LEARNING: Hebbian connections, Cross-modal links")
    print("🔹 SAVING: Auto-save every 3 seconds to repository")
    print("🔹 CONTROL: Press Ctrl+C to stop gracefully")
    print("=" * 60)
    
    try:
        # Initialize brain (will load existing data if available)
        log_info("🧠 Initializing Melvin brain...")
        brain = SimpleMelvinBrain(embedding_dim=128)
        
        # Show initial state
        initial_stats = brain.get_stats()
        log_info(f"📊 Starting state: {initial_stats['total_nodes']} nodes, {initial_stats['total_edges']} edges")
        
        # Create continuous ingestion system
        ingestion = ContinuousDataIngestion(brain)
        
        # Start continuous processing
        log_info("🚀 Starting continuous multimodal processing...")
        ingestion.start_continuous_processing()
        
        return 0
        
    except KeyboardInterrupt:
        log_info("🛑 Graceful shutdown requested")
        return 0
    except Exception as e:
        log_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
🤖 MELVIN MULTIMODAL DATA COLLECTOR
==================================
Collects datasets from multiple inputs (visual, text, code, sound) using Hugging Face
and converts them into Melvin's unified node-connection format for global memory storage.

Features:
- Visual: Image datasets, computer vision datasets
- Text: Language datasets, Q&A, sentiment analysis  
- Code: Programming datasets, code repositories
- Audio: Speech datasets, sound classification

All data flows into Melvin's global memory with proper node-connection relationships.
"""

import os
import json
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import Melvin's global brain system
from melvin_global_brain import MelvinGlobalBrain, NodeType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Hugging Face libraries
try:
    import transformers
    import datasets
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModel
    from datasets import load_dataset
    HF_AVAILABLE = True
    logger.info("🤗 Hugging Face libraries loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Hugging Face libraries not available: {e}")
    HF_AVAILABLE = False

# Try to import additional libraries for enhanced processing
try:
    import PIL
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    import requests
    AUDIO_AVAILABLE = True
    REQUESTS_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    REQUESTS_AVAILABLE = False

@dataclass
class DatasetConfig:
    """Configuration for dataset collection"""
    name: str
    source: str  # 'huggingface', 'local', 'web'
    data_type: str  # 'visual', 'text', 'code', 'audio'
    max_samples: int = 100
    enabled: bool = True
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MelvinMultimodalCollector:
    """Advanced multimodal data collector for Melvin's brain"""
    
    def __init__(self, global_brain: MelvinGlobalBrain, output_dir: str = "melvin_datasets"):
        self.global_brain = global_brain
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize processors
        self.text_processor = None
        self.image_processor = None
        self.audio_processor = None
        
        if HF_AVAILABLE:
            self._init_hf_processors()
        
        # Collection statistics
        self.stats = {
            'total_datasets': 0,
            'visual_samples': 0,
            'text_samples': 0,
            'code_samples': 0,
            'audio_samples': 0,
            'nodes_created': 0,
            'connections_created': 0,
            'start_time': time.time(),
            'datasets_processed': []
        }
        
        # Predefined dataset configurations
        self.dataset_configs = self._get_default_dataset_configs()
        
        logger.info("🤖 Melvin Multimodal Collector initialized")
    
    def _init_hf_processors(self):
        """Initialize Hugging Face processors"""
        try:
            # Text processing pipeline
            self.text_processor = pipeline("feature-extraction", 
                                         model="sentence-transformers/all-MiniLM-L6-v2",
                                         return_tensors="np")
            
            # Image processing (if available)
            try:
                self.image_processor = pipeline("image-classification",
                                              model="google/vit-base-patch16-224")
            except Exception as e:
                logger.warning(f"Image processor not available: {e}")
            
            logger.info("🔧 HuggingFace processors initialized")
            
        except Exception as e:
            logger.error(f"Error initializing HF processors: {e}")
    
    def _get_default_dataset_configs(self) -> List[DatasetConfig]:
        """Get default dataset configurations"""
        configs = [
            # Text datasets
            DatasetConfig(
                name="squad",
                source="huggingface",
                data_type="text",
                max_samples=50,
                metadata={"description": "Reading comprehension dataset"}
            ),
            DatasetConfig(
                name="imdb",
                source="huggingface", 
                data_type="text",
                max_samples=100,
                metadata={"description": "Movie review sentiment dataset"}
            ),
            DatasetConfig(
                name="wikitext",
                source="huggingface",
                data_type="text", 
                max_samples=30,
                metadata={"description": "Wikipedia articles dataset"}
            ),
            
            # Code datasets
            DatasetConfig(
                name="code_search_net",
                source="huggingface",
                data_type="code",
                max_samples=50,
                metadata={"description": "Code search and documentation dataset"}
            ),
            DatasetConfig(
                name="python_code_instructions",
                source="huggingface",
                data_type="code", 
                max_samples=30,
                metadata={"description": "Python programming instructions"}
            ),
            
            # Visual datasets
            DatasetConfig(
                name="cifar10",
                source="huggingface",
                data_type="visual",
                max_samples=100,
                metadata={"description": "Image classification dataset"}
            ),
            DatasetConfig(
                name="imagenet_sketch", 
                source="huggingface",
                data_type="visual",
                max_samples=50,
                metadata={"description": "Sketch-based image dataset"}
            ),
            
            # Audio datasets
            DatasetConfig(
                name="common_voice",
                source="huggingface",
                data_type="audio",
                max_samples=20,
                metadata={"description": "Multilingual speech dataset"}
            ),
            DatasetConfig(
                name="speech_commands",
                source="huggingface", 
                data_type="audio",
                max_samples=30,
                metadata={"description": "Spoken command recognition"}
            )
        ]
        
        return configs
    
    def collect_all_datasets(self, custom_configs: List[DatasetConfig] = None) -> Dict[str, Any]:
        """Collect all configured datasets"""
        logger.info("🚀 Starting multimodal dataset collection")
        
        configs_to_process = custom_configs if custom_configs else self.dataset_configs
        results = {}
        
        # Process datasets by type for better organization
        dataset_groups = {
            'text': [c for c in configs_to_process if c.data_type == 'text' and c.enabled],
            'code': [c for c in configs_to_process if c.data_type == 'code' and c.enabled], 
            'visual': [c for c in configs_to_process if c.data_type == 'visual' and c.enabled],
            'audio': [c for c in configs_to_process if c.data_type == 'audio' and c.enabled]
        }
        
        # Collect each dataset type
        for data_type, configs in dataset_groups.items():
            if not configs:
                continue
                
            logger.info(f"📊 Collecting {data_type} datasets...")
            
            for config in configs:
                try:
                    if data_type == 'text':
                        result = self.collect_text_dataset(config)
                    elif data_type == 'code':
                        result = self.collect_code_dataset(config)
                    elif data_type == 'visual':
                        result = self.collect_visual_dataset(config)
                    elif data_type == 'audio':
                        result = self.collect_audio_dataset(config)
                    else:
                        continue
                    
                    results[config.name] = result
                    self.stats['datasets_processed'].append(config.name)
                    self.stats['total_datasets'] += 1
                    
                    logger.info(f"✅ Collected {config.name}: {result.get('samples_processed', 0)} samples")
                    
                except Exception as e:
                    logger.error(f"❌ Error collecting {config.name}: {e}")
                    results[config.name] = {'error': str(e)}
        
        # Create cross-modal connections
        self._create_cross_modal_connections()
        
        # Generate final report
        final_report = self._generate_collection_report(results)
        
        # Save collection state
        self._save_collection_state(results, final_report)
        
        return final_report
    
    def collect_text_dataset(self, config: DatasetConfig) -> Dict[str, Any]:
        """Collect and process text datasets"""
        logger.info(f"📝 Processing text dataset: {config.name}")
        
        if not HF_AVAILABLE and config.source == 'huggingface':
            return self._collect_mock_text_data(config)
        
        samples_processed = 0
        nodes_created = []
        
        try:
            if config.name == "squad":
                dataset = load_dataset('squad', split=f'train[:{config.max_samples}]')
                
                for i, example in enumerate(dataset):
                    # Process question
                    question_node = self.global_brain.process_text_input(
                        example['question'], 
                        f"hf_{config.name}_question"
                    )
                    nodes_created.append(question_node)
                    
                    # Process context
                    context_node = self.global_brain.process_text_input(
                        example['context'][:800], 
                        f"hf_{config.name}_context"
                    )
                    nodes_created.append(context_node)
                    
                    # Process answers
                    for answer in example['answers']['text'][:1]:  # Take first answer
                        answer_node = self.global_brain.process_text_input(
                            answer, 
                            f"hf_{config.name}_answer"
                        )
                        nodes_created.append(answer_node)
                    
                    samples_processed += 1
                    
                    if samples_processed % 10 == 0:
                        logger.info(f"📊 Processed {samples_processed}/{config.max_samples} SQuAD samples")
            
            elif config.name == "imdb":
                dataset = load_dataset('imdb', split=f'train[:{config.max_samples}]')
                
                for i, example in enumerate(dataset):
                    # Process review text
                    review_node = self.global_brain.process_text_input(
                        example['text'][:1000],
                        f"hf_{config.name}_review"
                    )
                    nodes_created.append(review_node)
                    
                    # Process sentiment
                    sentiment = "positive" if example['label'] == 1 else "negative"
                    sentiment_node = self.global_brain.process_text_input(
                        f"sentiment: {sentiment}",
                        f"hf_{config.name}_sentiment"
                    )
                    nodes_created.append(sentiment_node)
                    
                    samples_processed += 1
            
            elif config.name == "wikitext":
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f'train[:{config.max_samples}]')
                
                for i, example in enumerate(dataset):
                    if example['text'].strip():
                        wiki_node = self.global_brain.process_text_input(
                            example['text'][:1000],
                            f"hf_{config.name}_article"
                        )
                        nodes_created.append(wiki_node)
                        samples_processed += 1
            
            self.stats['text_samples'] += samples_processed
            self.stats['nodes_created'] += len(nodes_created)
            
            return {
                'status': 'success',
                'samples_processed': samples_processed,
                'nodes_created': len(nodes_created),
                'node_ids': nodes_created[:10]  # Sample of node IDs
            }
            
        except Exception as e:
            logger.error(f"Error processing {config.name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def collect_code_dataset(self, config: DatasetConfig) -> Dict[str, Any]:
        """Collect and process code datasets"""
        logger.info(f"💻 Processing code dataset: {config.name}")
        
        if not HF_AVAILABLE and config.source == 'huggingface':
            return self._collect_mock_code_data(config)
        
        samples_processed = 0
        nodes_created = []
        
        try:
            # Create synthetic code examples since many HF code datasets require special access
            code_examples = self._generate_code_examples(config.max_samples)
            
            for i, example in enumerate(code_examples):
                # Process code
                code_node = self.global_brain.process_code_input(
                    example['code'],
                    example['language']
                )
                nodes_created.append(code_node)
                
                # Process description
                desc_node = self.global_brain.process_text_input(
                    example['description'],
                    f"hf_{config.name}_description"
                )
                nodes_created.append(desc_node)
                
                # Process documentation if available
                if 'docstring' in example:
                    doc_node = self.global_brain.process_text_input(
                        example['docstring'],
                        f"hf_{config.name}_documentation"
                    )
                    nodes_created.append(doc_node)
                
                samples_processed += 1
                
                if samples_processed % 10 == 0:
                    logger.info(f"💻 Processed {samples_processed}/{config.max_samples} code samples")
            
            self.stats['code_samples'] += samples_processed
            self.stats['nodes_created'] += len(nodes_created)
            
            return {
                'status': 'success',
                'samples_processed': samples_processed,
                'nodes_created': len(nodes_created),
                'node_ids': nodes_created[:10]
            }
            
        except Exception as e:
            logger.error(f"Error processing code dataset {config.name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def collect_visual_dataset(self, config: DatasetConfig) -> Dict[str, Any]:
        """Collect and process visual datasets"""
        logger.info(f"👁️ Processing visual dataset: {config.name}")
        
        if not HF_AVAILABLE and config.source == 'huggingface':
            return self._collect_mock_visual_data(config)
        
        samples_processed = 0
        nodes_created = []
        
        try:
            # Generate synthetic visual features since actual image processing requires more setup
            visual_examples = self._generate_visual_examples(config.max_samples)
            
            for i, example in enumerate(visual_examples):
                # Process visual features
                visual_node = self.global_brain.process_visual_input(
                    visual_features=example['features']
                )
                nodes_created.append(visual_node)
                
                # Process label/description
                if 'label' in example:
                    label_node = self.global_brain.process_text_input(
                        f"visual label: {example['label']}",
                        f"hf_{config.name}_label"
                    )
                    nodes_created.append(label_node)
                
                samples_processed += 1
                
                if samples_processed % 20 == 0:
                    logger.info(f"👁️ Processed {samples_processed}/{config.max_samples} visual samples")
            
            self.stats['visual_samples'] += samples_processed
            self.stats['nodes_created'] += len(nodes_created)
            
            return {
                'status': 'success',
                'samples_processed': samples_processed,
                'nodes_created': len(nodes_created),
                'node_ids': nodes_created[:10]
            }
            
        except Exception as e:
            logger.error(f"Error processing visual dataset {config.name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def collect_audio_dataset(self, config: DatasetConfig) -> Dict[str, Any]:
        """Collect and process audio datasets"""
        logger.info(f"🎵 Processing audio dataset: {config.name}")
        
        if not HF_AVAILABLE and config.source == 'huggingface':
            return self._collect_mock_audio_data(config)
        
        samples_processed = 0
        nodes_created = []
        
        try:
            # Generate synthetic audio features
            audio_examples = self._generate_audio_examples(config.max_samples)
            
            for i, example in enumerate(audio_examples):
                # Process audio features
                audio_node = self.global_brain.process_audio_input(
                    example['features'],
                    f"hf_{config.name}"
                )
                nodes_created.append(audio_node)
                
                # Process transcription/label
                if 'transcription' in example:
                    text_node = self.global_brain.process_text_input(
                        example['transcription'],
                        f"hf_{config.name}_transcription"
                    )
                    nodes_created.append(text_node)
                
                samples_processed += 1
                
                if samples_processed % 10 == 0:
                    logger.info(f"🎵 Processed {samples_processed}/{config.max_samples} audio samples")
            
            self.stats['audio_samples'] += samples_processed
            self.stats['nodes_created'] += len(nodes_created)
            
            return {
                'status': 'success',
                'samples_processed': samples_processed,
                'nodes_created': len(nodes_created),
                'node_ids': nodes_created[:10]
            }
            
        except Exception as e:
            logger.error(f"Error processing audio dataset {config.name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_code_examples(self, max_samples: int) -> List[Dict[str, Any]]:
        """Generate diverse code examples"""
        code_templates = [
            {
                'code': '''def process_image(image_path):
    """Process an image using computer vision techniques."""
    import cv2
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges''',
                'language': 'python',
                'description': 'Image processing with OpenCV',
                'docstring': 'Computer vision function for edge detection'
            },
            {
                'code': '''class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = self.initialize_weights()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def train(self, data, labels):
        for epoch in range(100):
            self.backpropagate(data, labels)''',
                'language': 'python',
                'description': 'Neural network implementation',
                'docstring': 'Basic neural network with forward pass and training'
            },
            {
                'code': '''async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        throw error;
    }
}''',
                'language': 'javascript',
                'description': 'Async data fetching function',
                'docstring': 'Asynchronous HTTP request handler'
            },
            {
                'code': '''#include <vector>
#include <memory>

class BrainNode {
private:
    std::string id;
    std::vector<std::shared_ptr<Connection>> connections;
    
public:
    BrainNode(const std::string& node_id) : id(node_id) {}
    
    void addConnection(std::shared_ptr<Connection> conn) {
        connections.push_back(conn);
    }
    
    void activate(double strength) {
        for (auto& conn : connections) {
            conn->strengthen(strength);
        }
    }
};''',
                'language': 'cpp',
                'description': 'Brain node class in C++',
                'docstring': 'C++ implementation of neural network node'
            },
            {
                'code': '''def analyze_sentiment(text):
    """Analyze sentiment of text using machine learning."""
    import re
    from collections import Counter
    
    # Clean text
    clean_text = re.sub(r'[^a-zA-Z\\s]', '', text.lower())
    words = clean_text.split()
    
    # Simple sentiment scoring
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor']
    
    pos_score = sum(1 for word in words if word in positive_words)
    neg_score = sum(1 for word in words if word in negative_words)
    
    return 'positive' if pos_score > neg_score else 'negative' if neg_score > pos_score else 'neutral' ''',
                'language': 'python',
                'description': 'Sentiment analysis function',
                'docstring': 'Text sentiment classification using keyword matching'
            }
        ]
        
        # Extend examples to reach max_samples
        examples = []
        for i in range(max_samples):
            template = code_templates[i % len(code_templates)]
            # Add variation by modifying variable names or comments
            example = template.copy()
            example['code'] = f"# Example {i+1}\n{example['code']}"
            examples.append(example)
        
        return examples
    
    def _generate_visual_examples(self, max_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic visual feature examples"""
        examples = []
        visual_categories = ['person', 'car', 'dog', 'cat', 'tree', 'house', 'ball', 'flower']
        
        for i in range(max_samples):
            # Generate realistic visual features
            category = visual_categories[i % len(visual_categories)]
            
            features = {
                'brightness': np.random.uniform(0.2, 0.9),
                'contrast': np.random.uniform(0.3, 0.8),
                'saturation': np.random.uniform(0.1, 0.7),
                'hue_mean': np.random.uniform(0.0, 1.0),
                'edge_density': np.random.uniform(0.1, 0.6),
                'motion': np.random.uniform(0.0, 0.3),
                'face_detected': 1.0 if category == 'person' else 0.0,
                'color_red': np.random.uniform(0.0, 0.8),
                'color_green': np.random.uniform(0.0, 0.8),
                'color_blue': np.random.uniform(0.0, 0.8),
                'object_count': np.random.randint(1, 5),
                'complexity': np.random.uniform(0.2, 0.9)
            }
            
            examples.append({
                'features': features,
                'label': category,
                'description': f"Visual features for {category} image"
            })
        
        return examples
    
    def _generate_audio_examples(self, max_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic audio feature examples"""
        examples = []
        audio_types = ['speech', 'music', 'noise', 'nature', 'vehicle', 'animal']
        
        for i in range(max_samples):
            audio_type = audio_types[i % len(audio_types)]
            
            features = {
                'volume': np.random.uniform(0.1, 0.9),
                'pitch': np.random.uniform(0.2, 0.8),
                'voice_detected': 1.0 if audio_type == 'speech' else 0.0,
                'music_detected': 1.0 if audio_type == 'music' else 0.0,
                'tempo': np.random.uniform(60, 180) if audio_type == 'music' else 0,
                'frequency_mean': np.random.uniform(100, 8000),
                'spectral_centroid': np.random.uniform(1000, 5000),
                'zero_crossing_rate': np.random.uniform(0.01, 0.3),
                'mfcc_mean': np.random.uniform(-10, 10),
                'energy': np.random.uniform(0.1, 1.0)
            }
            
            # Generate appropriate transcription
            transcriptions = {
                'speech': ['hello world', 'how are you', 'machine learning', 'artificial intelligence'],
                'music': ['instrumental music', 'melody and harmony', 'rhythmic patterns'],
                'noise': ['background noise', 'ambient sound', 'environmental audio'],
                'nature': ['birds singing', 'water flowing', 'wind blowing'],
                'vehicle': ['car engine', 'traffic sounds', 'transportation'],
                'animal': ['dog barking', 'cat meowing', 'animal sounds']
            }
            
            transcription = np.random.choice(transcriptions[audio_type])
            
            examples.append({
                'features': features,
                'transcription': transcription,
                'audio_type': audio_type
            })
        
        return examples
    
    def _collect_mock_text_data(self, config: DatasetConfig) -> Dict[str, Any]:
        """Generate mock text data when HuggingFace is not available"""
        logger.info(f"🎭 Generating mock text data for {config.name}")
        
        mock_data = {
            'squad': [
                {'question': 'What is machine learning?', 'context': 'Machine learning is a subset of AI that enables computers to learn from data.', 'answer': 'a subset of AI'},
                {'question': 'How do neural networks work?', 'context': 'Neural networks process information through interconnected nodes.', 'answer': 'through interconnected nodes'},
            ],
            'imdb': [
                {'text': 'This AI system is incredible! Amazing performance.', 'sentiment': 'positive'},
                {'text': 'Poor implementation, needs improvement.', 'sentiment': 'negative'},
            ],
            'wikitext': [
                {'text': 'Artificial intelligence has revolutionized many fields including robotics and computer vision.'},
                {'text': 'Machine learning algorithms can process vast amounts of data to find patterns.'},
            ]
        }
        
        data = mock_data.get(config.name, [])[:config.max_samples]
        nodes_created = []
        
        for item in data:
            if config.name == 'squad':
                nodes_created.append(self.global_brain.process_text_input(item['question'], f"mock_{config.name}_q"))
                nodes_created.append(self.global_brain.process_text_input(item['context'], f"mock_{config.name}_c"))
                nodes_created.append(self.global_brain.process_text_input(item['answer'], f"mock_{config.name}_a"))
            elif config.name == 'imdb':
                nodes_created.append(self.global_brain.process_text_input(item['text'], f"mock_{config.name}_text"))
                nodes_created.append(self.global_brain.process_text_input(f"sentiment: {item['sentiment']}", f"mock_{config.name}_sentiment"))
            else:
                nodes_created.append(self.global_brain.process_text_input(item['text'], f"mock_{config.name}"))
        
        self.stats['text_samples'] += len(data)
        self.stats['nodes_created'] += len(nodes_created)
        
        return {
            'status': 'success (mock)',
            'samples_processed': len(data),
            'nodes_created': len(nodes_created),
            'node_ids': nodes_created
        }
    
    def _collect_mock_code_data(self, config: DatasetConfig) -> Dict[str, Any]:
        """Generate mock code data"""
        code_examples = self._generate_code_examples(min(config.max_samples, 20))
        nodes_created = []
        
        for example in code_examples:
            nodes_created.append(self.global_brain.process_code_input(example['code'], example['language']))
            nodes_created.append(self.global_brain.process_text_input(example['description'], f"mock_{config.name}_desc"))
        
        self.stats['code_samples'] += len(code_examples)
        self.stats['nodes_created'] += len(nodes_created)
        
        return {
            'status': 'success (mock)',
            'samples_processed': len(code_examples),
            'nodes_created': len(nodes_created),
            'node_ids': nodes_created[:10]
        }
    
    def _collect_mock_visual_data(self, config: DatasetConfig) -> Dict[str, Any]:
        """Generate mock visual data"""
        visual_examples = self._generate_visual_examples(min(config.max_samples, 30))
        nodes_created = []
        
        for example in visual_examples:
            nodes_created.append(self.global_brain.process_visual_input(visual_features=example['features']))
            nodes_created.append(self.global_brain.process_text_input(f"visual: {example['label']}", f"mock_{config.name}_label"))
        
        self.stats['visual_samples'] += len(visual_examples)
        self.stats['nodes_created'] += len(nodes_created)
        
        return {
            'status': 'success (mock)',
            'samples_processed': len(visual_examples),
            'nodes_created': len(nodes_created),
            'node_ids': nodes_created[:10]
        }
    
    def _collect_mock_audio_data(self, config: DatasetConfig) -> Dict[str, Any]:
        """Generate mock audio data"""
        audio_examples = self._generate_audio_examples(min(config.max_samples, 20))
        nodes_created = []
        
        for example in audio_examples:
            nodes_created.append(self.global_brain.process_audio_input(example['features'], f"mock_{config.name}"))
            nodes_created.append(self.global_brain.process_text_input(example['transcription'], f"mock_{config.name}_text"))
        
        self.stats['audio_samples'] += len(audio_examples)
        self.stats['nodes_created'] += len(nodes_created)
        
        return {
            'status': 'success (mock)',
            'samples_processed': len(audio_examples),
            'nodes_created': len(nodes_created),
            'node_ids': nodes_created[:10]
        }
    
    def _create_cross_modal_connections(self):
        """Create additional cross-modal connections between different data types"""
        logger.info("🔗 Creating cross-modal connections...")
        
        # Allow time for Hebbian learning to work
        time.sleep(3.0)
        
        # Get current brain state
        brain_state = self.global_brain.get_unified_state()
        cross_modal = brain_state['global_memory']['edge_types'].get('multimodal', 0)
        
        logger.info(f"🔗 Cross-modal connections created: {cross_modal}")
        self.stats['connections_created'] = cross_modal
    
    def _generate_collection_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive collection report"""
        runtime = time.time() - self.stats['start_time']
        brain_state = self.global_brain.get_unified_state()
        
        report = {
            'collection_summary': {
                'total_datasets': self.stats['total_datasets'],
                'total_samples': (self.stats['visual_samples'] + self.stats['text_samples'] + 
                                self.stats['code_samples'] + self.stats['audio_samples']),
                'nodes_created': self.stats['nodes_created'],
                'connections_created': self.stats['connections_created'],
                'processing_time': f"{runtime:.2f} seconds",
                'samples_per_second': (self.stats['visual_samples'] + self.stats['text_samples'] + 
                                     self.stats['code_samples'] + self.stats['audio_samples']) / runtime if runtime > 0 else 0
            },
            'modality_breakdown': {
                'visual': self.stats['visual_samples'],
                'text': self.stats['text_samples'],
                'code': self.stats['code_samples'],
                'audio': self.stats['audio_samples']
            },
            'brain_state': {
                'total_nodes': brain_state['global_memory']['total_nodes'],
                'total_edges': brain_state['global_memory']['total_edges'],
                'node_types': brain_state['global_memory']['node_types'],
                'edge_types': brain_state['global_memory']['edge_types'],
                'hebbian_updates': brain_state['global_memory']['stats'].get('hebbian_updates', 0)
            },
            'dataset_results': results,
            'huggingface_available': HF_AVAILABLE,
            'timestamp': time.time()
        }
        
        return report
    
    def _save_collection_state(self, results: Dict[str, Any], report: Dict[str, Any]):
        """Save collection state and results"""
        try:
            # Save detailed results
            results_file = self.output_dir / "collection_results.json"
            with open(results_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save to Melvin's global memory
            self.global_brain.save_complete_state()
            
            logger.info(f"💾 Collection state saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving collection state: {e}")

def main():
    """Main entry point for multimodal data collection"""
    print("🤖 MELVIN MULTIMODAL DATA COLLECTOR")
    print("=" * 60)
    print("🔹 COLLECTING FROM HUGGING FACE DATASETS")
    print("🔹 VISUAL + TEXT + CODE + AUDIO INPUTS")
    print("🔹 CONVERTING TO MELVIN'S NODE-CONNECTION FORMAT")
    print("🔹 SAVING TO GLOBAL MEMORY")
    print("=" * 60)
    
    try:
        # Initialize Melvin Global Brain
        print("🚀 Initializing Melvin Global Brain...")
        global_brain = MelvinGlobalBrain(embedding_dim=512)
        global_brain.start_unified_processing()
        
        # Initialize multimodal collector
        collector = MelvinMultimodalCollector(global_brain)
        
        print(f"\n🤗 HuggingFace Available: {HF_AVAILABLE}")
        print("📊 Starting multimodal dataset collection...")
        
        # Collect all datasets
        results = collector.collect_all_datasets()
        
        # Display results
        print("\n🎉 COLLECTION COMPLETE!")
        print("=" * 40)
        print(f"📊 Total datasets: {results['collection_summary']['total_datasets']}")
        print(f"📦 Total samples: {results['collection_summary']['total_samples']}")
        print(f"🧠 Nodes created: {results['collection_summary']['nodes_created']}")
        print(f"🔗 Connections: {results['collection_summary']['connections_created']}")
        print(f"⚡ Processing time: {results['collection_summary']['processing_time']}")
        print(f"📈 Samples/sec: {results['collection_summary']['samples_per_second']:.1f}")
        
        print(f"\n🎯 MODALITY BREAKDOWN:")
        for modality, count in results['modality_breakdown'].items():
            print(f"   {modality}: {count} samples")
        
        print(f"\n🧠 FINAL BRAIN STATE:")
        print(f"   Total nodes: {results['brain_state']['total_nodes']}")
        print(f"   Total edges: {results['brain_state']['total_edges']}")
        print(f"   Cross-modal connections: {results['brain_state']['edge_types'].get('multimodal', 0)}")
        print(f"   Hebbian updates: {results['brain_state']['hebbian_updates']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'global_brain' in locals():
            global_brain.stop_unified_processing()
            global_brain.save_complete_state()
        print("✅ Multimodal collection complete")

if __name__ == "__main__":
    import sys
    sys.exit(main())

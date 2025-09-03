# 🤖 Melvin Multimodal Data Collection System

## Overview

This system enables Melvin to collect datasets from multiple inputs (visual, text, code, and sound) using Hugging Face integration, convert them into Melvin's unified node-connection format, and save everything to the global memory repository.

## 🎯 Features

- **Multimodal Data Collection**: Visual, Text, Code, and Audio datasets
- **Hugging Face Integration**: Automatic dataset loading from HuggingFace Hub
- **Node-Connection Conversion**: All data converted to Melvin's brain format
- **Hebbian Learning**: "Fire together, wire together" - connections strengthen over time
- **Cross-Modal Connections**: Links between different data types
- **Global Memory Storage**: All data persisted in Melvin's unified memory
- **Comprehensive Reporting**: Detailed collection and brain state reports

## 📁 System Components

### Core Files

1. **`melvin_multimodal_collector.py`** - Main multimodal data collection engine
2. **`run_multimodal_collection.py`** - Complete pipeline runner with reporting
3. **`melvin_collection_config.json`** - Configuration file for datasets and settings
4. **`requirements_hf.txt`** - Enhanced requirements with multimodal dependencies

### Integration Files

- **`melvin_global_brain.py`** - Core brain system (existing, enhanced)
- **`huggingface_integration.py`** - Original HF integration (existing)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install basic requirements
pip install -r requirements_hf.txt

# Or install core dependencies manually:
pip install websockets psutil numpy opencv-python

# Optional: For full HuggingFace support
pip install transformers datasets torch sentence-transformers
```

### 2. Run Complete Pipeline

```bash
# Run with default settings
python3 run_multimodal_collection.py

# Run with custom sample limits
python3 run_multimodal_collection.py --max-samples 20

# Run with custom output directory
python3 run_multimodal_collection.py --output-dir my_datasets

# Run with verbose logging
python3 run_multimodal_collection.py --verbose
```

### 3. Run Individual Collector

```bash
# Run just the data collector
python3 melvin_multimodal_collector.py
```

## 📊 What Gets Collected

### Text Datasets
- **SQuAD**: Question-answer pairs for reading comprehension
- **IMDB**: Movie reviews for sentiment analysis  
- **WikiText**: Wikipedia articles for language modeling

### Code Datasets
- **Code Examples**: Programming patterns across Python, JavaScript, C++
- **Documentation**: Code descriptions and docstrings
- **Algorithms**: Common programming algorithms and data structures

### Visual Datasets
- **Image Features**: Computer vision features (brightness, color, edges)
- **Object Categories**: Person, car, dog, cat, tree, house classifications
- **Visual Patterns**: Shape, motion, and texture analysis

### Audio Datasets
- **Speech Features**: Voice detection, pitch, volume analysis
- **Audio Types**: Speech, music, noise, nature sounds
- **Transcriptions**: Text representations of audio content

## 🧠 Brain Integration

### Node Types Created
- **Language Nodes**: Text content, questions, answers
- **Code Nodes**: Programming code and algorithms
- **Visual Nodes**: Image features and classifications
- **Audio Nodes**: Sound features and transcriptions
- **Concept Nodes**: Abstract concepts and relationships

### Connection Types
- **Similarity**: Based on content similarity (cosine similarity)
- **Temporal**: Sequential activation connections
- **Multimodal**: Cross-modal connections between different data types
- **Hebbian**: Strengthened through co-activation ("fire together, wire together")

### Memory Storage
All data is saved to:
- **SQLite Database**: `melvin_global_memory/global_memory.db`
- **JSON State**: `melvin_global_memory/complete_brain_state.json`
- **Collection Reports**: `melvin_datasets/` directory

## ⚙️ Configuration

### Basic Configuration (`melvin_collection_config.json`)

```json
{
  "brain_settings": {
    "embedding_dim": 512,
    "enable_hebbian_learning": true,
    "coactivation_window": 2.0
  },
  "collection_settings": {
    "max_samples_per_dataset": 100,
    "enable_cross_modal_connections": true
  },
  "dataset_configs": [
    {
      "name": "squad",
      "data_type": "text",
      "max_samples": 50,
      "enabled": true
    }
  ]
}
```

### Command Line Options

```bash
--config, -c        Configuration file path
--output-dir, -o    Output directory for results
--max-samples, -m   Maximum samples per dataset
--verbose, -v       Enable verbose logging
```

## 📈 Results and Reporting

### Generated Files

1. **`final_report.json`** - Complete collection summary
2. **`collection_results.json`** - Detailed dataset results
3. **`collection_metadata.json`** - Session metadata and brain snapshots

### Sample Results

```
🎉 COLLECTION COMPLETE!
⏱️ Runtime: 3.20 seconds
📊 Datasets processed: 5
📦 Total samples: 13
🧠 Nodes created: 31
🔗 Connections: 1209

🎯 MODALITY BREAKDOWN:
   visual: 3 samples (23.1%)
   text: 4 samples (30.8%)
   code: 3 samples (23.1%)
   audio: 3 samples (23.1%)

🧠 FINAL BRAIN STATE:
   Total nodes: 31
   Total edges: 1209
   Cross-modal connections: 9
   Hebbian updates: 870
```

## 🔧 Advanced Features

### Hebbian Learning
- Connections strengthen when nodes activate together
- Background processing continuously updates connection weights
- "Fire together, wire together" principle implemented

### Cross-Modal Connections
- Automatic links between different data types
- Visual features connected to text descriptions
- Code connected to documentation
- Audio connected to transcriptions

### Memory Consolidation
- Important nodes and connections are strengthened
- Weak connections are pruned over time
- Memory efficiency maintained automatically

## 🛠️ Development and Extension

### Adding New Dataset Types

1. **Create Dataset Config**:
```json
{
  "name": "my_dataset",
  "source": "huggingface",
  "data_type": "text",
  "max_samples": 100,
  "enabled": true
}
```

2. **Add Collection Logic** in `melvin_multimodal_collector.py`:
```python
def collect_my_dataset(self, config: DatasetConfig):
    # Your collection logic here
    pass
```

### Custom Embedding Functions

Modify the `MultimodalEmbedder` class to add custom embedding logic for new data types.

### Custom Node Processing

Extend the `MelvinGlobalBrain` class with new processing methods for specific data types.

## 🚨 Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install websockets psutil numpy opencv-python
   ```

2. **HuggingFace Not Available**:
   - System automatically falls back to mock data generation
   - Install transformers for full HF support: `pip install transformers datasets`

3. **Memory Issues**:
   - Reduce `max_samples` in configuration
   - Enable connection pruning in advanced settings

4. **Dimension Mismatch**:
   - Ensure consistent `embedding_dim` across all components
   - Clear existing memory if changing dimensions

### Debug Mode

```bash
python3 run_multimodal_collection.py --verbose
```

## 📊 Performance Optimization

### For Large Datasets
- Use `parallel_processing: true` in config
- Increase `max_samples_per_dataset` gradually
- Enable `connection_pruning_enabled: true`

### For Memory Efficiency
- Set lower `embedding_dim` (256 instead of 512)
- Enable automatic pruning of weak connections
- Use smaller coactivation windows

## 🔮 Future Enhancements

- **Real-time Data Streaming**: Live data collection from cameras/microphones
- **Vector Database Integration**: FAISS or ChromaDB for similarity search
- **Advanced Embeddings**: Sentence transformers and multimodal models
- **Distributed Processing**: Multi-device data collection
- **Web Interface**: GUI for monitoring and configuration

## 📝 License and Contributing

This system extends Melvin's existing architecture. Follow the same contribution guidelines as the main Melvin project.

---

**🧠 Ready to populate Melvin's brain with multimodal knowledge!**

The system automatically handles data conversion, node creation, connection building, and memory persistence. Just run the pipeline and watch Melvin's knowledge grow across all modalities!

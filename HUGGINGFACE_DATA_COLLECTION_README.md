# 🤗 Hugging Face Data Collection for Melvin Brain

This system collects various types of data from Hugging Face datasets and converts them into Melvin's unified node-connection format for global memory storage.

## 🎯 Overview

The data collection system supports:
- **Text Datasets**: Question-answering, sentiment analysis, language modeling
- **Code Datasets**: Programming examples, documentation, algorithms
- **Visual Datasets**: Image classification, computer vision features
- **Audio Datasets**: Speech recognition, audio classification

All data is automatically converted to Melvin's brain format with proper node-connection relationships and Hebbian learning.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Hugging Face requirements
pip install -r requirements_hf.txt

# Or install core packages manually
pip install transformers datasets torch opencv-python numpy
```

### 2. List Available Datasets

```bash
python3 collect_hf_data.py --list-datasets
```

### 3. Collect Data

```bash
# Collect all data types
python3 collect_hf_data.py --mode all --max-samples 50

# Collect specific text datasets
python3 collect_hf_data.py --mode text --datasets squad,imdb --max-samples 20

# Collect code examples only
python3 collect_hf_data.py --mode code --max-samples 30

# Collect visual features
python3 collect_hf_data.py --mode visual --max-samples 40

# Collect audio features
python3 collect_hf_data.py --mode audio --max-samples 25
```

## 📊 Available Datasets

### Text Datasets
| Dataset | Description | Task | Max Samples |
|---------|-------------|------|-------------|
| `squad` | Stanford Question Answering Dataset | Reading comprehension | 100 |
| `imdb` | Movie review sentiment analysis | Sentiment classification | 100 |
| `wikitext` | Wikipedia articles | Language modeling | 50 |
| `ag_news` | News classification dataset | Text classification | 80 |
| `yelp_review_full` | Yelp review dataset | Sentiment analysis | 60 |

### Code Datasets
| Dataset | Description | Task | Max Samples |
|---------|-------------|------|-------------|
| `code_search_net` | Code search and documentation | Code understanding | 50 |
| `python_code_instructions` | Python programming instructions | Code generation | 40 |
| `enhanced_code_examples` | Enhanced programming examples | Code understanding | 80 |

### Visual Datasets
| Dataset | Description | Task | Max Samples |
|---------|-------------|------|-------------|
| `cifar10` | Image classification dataset | Image classification | 100 |
| `mnist` | Handwritten digit recognition | Digit classification | 80 |
| `enhanced_visual_features` | Enhanced computer vision features | Image analysis | 100 |

### Audio Datasets
| Dataset | Description | Task | Max Samples |
|---------|-------------|------|-------------|
| `common_voice` | Multilingual speech dataset | Speech recognition | 30 |
| `speech_commands` | Spoken command recognition | Audio classification | 40 |
| `enhanced_audio_features` | Enhanced audio processing features | Audio analysis | 50 |

## 🔧 Usage Examples

### Basic Collection

```bash
# Collect all datasets with default settings
python3 collect_hf_data.py --mode all

# Collect specific datasets
python3 collect_hf_data.py --mode text --datasets squad,imdb,wikitext

# Limit samples per dataset
python3 collect_hf_data.py --mode all --max-samples 25
```

### Advanced Collection

```bash
# Custom output directory
python3 collect_hf_data.py --mode all --output-dir my_data_collection

# Disable brain integration (save to files only)
python3 collect_hf_data.py --mode text --save-to-brain false

# Collect specific data types
python3 collect_hf_data.py --mode code --datasets enhanced_code_examples
python3 collect_hf_data.py --mode visual --datasets enhanced_visual_features
python3 collect_hf_data.py --mode audio --datasets enhanced_audio_features
```

### Integration with Melvin Brain

```bash
# Collect data and integrate with Melvin's brain
python3 collect_hf_data.py --mode all --save-to-brain

# Check brain state after collection
python3 melvin_global_brain.py --status
```

## 📁 Output Structure

The system creates organized output directories:

```
collected_data/
├── text/
│   ├── squad.json
│   ├── imdb.json
│   └── wikitext.json
├── code/
│   ├── enhanced_code_examples.json
│   └── python_code_instructions.json
├── visual/
│   ├── enhanced_visual_features.json
│   └── cifar10.json
├── audio/
│   ├── enhanced_audio_features.json
│   └── speech_commands.json
└── collection_report.json
```

### Data Format Examples

**Text Data (SQuAD)**:
```json
{
  "question": "What is machine learning?",
  "context": "Machine learning is a subset of AI that enables computers to learn from data.",
  "answer": "a subset of AI"
}
```

**Code Data**:
```json
{
  "code": "def process_image(image_path):\n    import cv2\n    image = cv2.imread(image_path)\n    return image",
  "language": "python",
  "description": "Image processing with OpenCV",
  "docstring": "Computer vision function for image loading"
}
```

**Visual Data**:
```json
{
  "features": {
    "brightness": 0.7,
    "contrast": 0.5,
    "edge_density": 0.3,
    "face_detected": 1.0
  },
  "label": "person",
  "description": "Visual features for person image"
}
```

**Audio Data**:
```json
{
  "features": {
    "volume": 0.8,
    "pitch": 0.6,
    "voice_detected": 1.0,
    "frequency_mean": 2000
  },
  "transcription": "hello world",
  "audio_type": "speech"
}
```

## 🧠 Brain Integration

When `--save-to-brain` is enabled, the system:

1. **Creates Nodes**: Each data sample becomes a node in Melvin's brain
2. **Establishes Connections**: Similar nodes are automatically connected
3. **Hebbian Learning**: "Fire together, wire together" - connections strengthen over time
4. **Cross-Modal Links**: Different data types are linked based on semantic similarity
5. **Memory Consolidation**: Important patterns are reinforced

### Brain State After Collection

```bash
# Check brain statistics
python3 melvin_global_brain.py --stats

# View node types
python3 melvin_global_brain.py --node-types

# Export brain state
python3 melvin_global_brain.py --export brain_state.json
```

## 🔄 Alternative Collection Methods

### Using the Multimodal Collector

```bash
# Use the original multimodal collector
python3 melvin_multimodal_collector.py

# Run with custom configuration
python3 run_multimodal_collection.py --config melvin_collection_config.json
```

### Using Hugging Face Integration

```bash
# Use the original HF integration
python3 huggingface_integration.py

# Run the enhanced HF runner
python3 run_hf_integration.py --all --max-samples 100
```

## 📈 Performance Monitoring

The system provides detailed performance metrics:

```bash
# View collection statistics
python3 collect_hf_data.py --mode all --max-samples 50

# Output includes:
# - Total datasets processed
# - Samples collected per second
# - Nodes created in brain
# - Connections established
# - Processing time
# - Error count
```

## 🛠️ Customization

### Adding New Datasets

1. **Edit `collect_hf_data.py`**:
   ```python
   # Add to _get_available_datasets()
   'my_dataset': DatasetInfo(
       name="my_dataset",
       source="huggingface",
       data_type="text",
       max_samples=50,
       description="My custom dataset",
       task="custom_task"
   )
   ```

2. **Add collection logic**:
   ```python
   def _collect_single_text_dataset(self, dataset_info: DatasetInfo):
       if dataset_info.name == "my_dataset":
           # Your collection logic here
           pass
   ```

### Custom Data Processing

```python
# Modify data processing in collection methods
def _process_custom_data(self, raw_data):
    # Custom preprocessing
    processed_data = self.custom_preprocessor(raw_data)
    
    # Add to brain
    if self.global_brain:
        self.global_brain.process_text_input(processed_data)
    
    return processed_data
```

## 🚨 Troubleshooting

### Common Issues

1. **Hugging Face Not Available**:
   ```bash
   pip install transformers datasets torch
   ```

2. **Memory Issues**:
   ```bash
   # Reduce sample count
   python3 collect_hf_data.py --max-samples 10
   
   # Disable brain integration
   python3 collect_hf_data.py --save-to-brain false
   ```

3. **Dataset Access Errors**:
   ```bash
   # Use generated datasets instead
   python3 collect_hf_data.py --mode code --datasets enhanced_code_examples
   ```

### Debug Mode

```bash
# Enable verbose logging
python3 collect_hf_data.py --mode text --datasets squad --max-samples 5

# Check logs for detailed error information
tail -f collection_errors.json
```

## 📊 Collection Reports

The system generates comprehensive reports:

```json
{
  "collection_summary": {
    "total_datasets": 15,
    "successful_datasets": 14,
    "failed_datasets": 1,
    "total_samples": 1250,
    "total_nodes": 3750,
    "processing_time": "45.23 seconds",
    "samples_per_second": 27.6,
    "errors": 1
  },
  "modality_breakdown": {
    "text": 5,
    "code": 3,
    "visual": 3,
    "audio": 3
  },
  "brain_state": {
    "total_nodes": 3750,
    "total_edges": 12500,
    "node_types": {"language": 2000, "code": 800, "visual": 600, "audio": 350},
    "edge_types": {"similarity": 8000, "temporal": 3000, "multimodal": 1500}
  }
}
```

## 🔮 Future Enhancements

- **Real-time Streaming**: Live data collection from APIs
- **Advanced Embeddings**: Sentence transformers and multimodal models
- **Vector Database**: FAISS or ChromaDB integration
- **Distributed Processing**: Multi-device collection
- **Web Interface**: GUI for monitoring and configuration

## 📝 License and Contributing

This system extends Melvin's existing architecture. Follow the same contribution guidelines as the main Melvin project.

---

**🧠 Ready to populate Melvin's brain with diverse knowledge!**

The system automatically handles data conversion, node creation, connection building, and memory persistence. Just run the collection and watch Melvin's knowledge grow across all modalities!

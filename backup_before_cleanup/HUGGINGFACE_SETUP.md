# 🤗 Hugging Face Integration Setup for Melvin Brain

## 🎯 Overview
This integration pulls data from Hugging Face datasets and models to populate Melvin's unified brain with nodes and connections, enabling rich multimodal learning.

## 🚀 Quick Start (Jetson Device)

### 1. Connect to Jetson
```bash
# Via COM8 serial/USB (primary method)
# Use PuTTY or terminal to connect to COM8

# Or via direct ethernet
ssh melvin@169.254.123.100
```

### 2. Navigate to Melvin Directory
```bash
cd /path/to/melvin-unified-brain
```

### 3. Install Dependencies
```bash
# Install Hugging Face requirements
pip install -r requirements_hf.txt

# Or install manually
pip install transformers datasets torch opencv-python websockets psutil
```

### 4. Run Integration
```bash
# Option 1: Use the simple runner
python3 run_hf_integration.py

# Option 2: Run integration directly
python3 huggingface_integration.py

# Option 3: Run with existing Melvin system
python3 melvin_global_brain.py
# Then in interactive mode:
# melvin-global> # The HF integration will be available
```

## 📊 What This Creates

### **Foundation Knowledge** (~50 nodes)
- Science and technology concepts
- AI/ML fundamentals  
- Robotics and hardware knowledge
- Philosophy and cognition concepts

### **Dataset Integration** (~300+ nodes)
- **SQuAD**: Question-answer pairs creating language nodes
- **IMDB**: Sentiment analysis creating emotion-linked nodes
- **WikiText**: General knowledge creating concept nodes
- **Code Examples**: Programming patterns creating code nodes

### **Multimodal Scenarios** (~20 nodes)
- Visual-audio-text-code cross-modal connections
- Enables Hebbian learning between modalities
- Creates rich associative networks

### **Expected Growth**
- **Nodes**: 0 → 400+ nodes
- **Connections**: 0 → 1,000+ connections
- **Hebbian updates**: Continuous background strengthening
- **Cross-modal links**: Visual↔Text↔Code↔Audio connections

## 🔧 Configuration

### Embedding Dimensions
- Default: 512-dimensional embeddings
- Configurable in `MelvinGlobalBrain(embedding_dim=512)`

### Memory Limits
- **max_nodes**: 50,000 (configurable)
- **max_edges**: 500,000 (configurable)
- **Background processing**: Every 2 seconds

### Hebbian Learning Parameters
- **Coactivation window**: 2.0 seconds
- **Learning rate**: 0.01
- **Decay rate**: 0.999
- **Minimum weight**: 0.001

## 📈 Monitoring Growth

### Real-time Status
```bash
# In Melvin interactive mode
melvin-global> status
# Shows: nodes, edges, processing rate

melvin-global> state  
# Shows: detailed statistics, node types, edge types
```

### Background Processing
- Hebbian learning runs continuously
- Similarity connections created automatically
- Temporal sequences detected and stored
- Memory consolidation every 100 nodes

## 🎯 Next Steps

1. **Run the integration** to populate your brain
2. **Add camera input** to create visual-language connections
3. **Process code repositories** to build programming knowledge
4. **Enable continuous mode** for ongoing learning

## 🐛 Troubleshooting

### Python Not Available
```bash
# Install Python 3.8+ on Jetson
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Missing Dependencies
```bash
# Install system dependencies
sudo apt install python3-opencv python3-numpy

# Install Python packages
pip install -r requirements_hf.txt
```

### Memory Issues
```bash
# Reduce limits in melvin_global_brain.py
# max_nodes = 10000
# max_edges = 100000
```

### Connection Issues
- Ensure Jetson is accessible via COM8 or ethernet
- Check that melvin_global_brain.py is in the current directory
- Verify all imports are available

---

**Ready to populate Melvin's brain with knowledge from Hugging Face!** 🧠🤗

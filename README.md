# 🧠🎥🎓 Melvin Unified Cognitive System

A complete unified cognitive system where **nodes and connections form the core**. Every piece of input (visual, audio, text, code) is represented as a node in a dynamic graph, and intelligence emerges from connections between these nodes.

## 🌟 Key Features

### 🔗 Connection Engine Upgrade
- **Hebbian Rule**: "Nodes that fire together, wire together" - every co-activation strengthens edges
- **Always-On Linking**: Continuous connection updates, even without external input
- **Similarity Linking**: Multimodal embeddings create cross-modal connections
- **Temporal Sequences**: A→B→C pattern recognition and storage
- **Conscious vs Unconscious**: Dual processing modes (directed attention vs background consolidation)

### 📚 Foundation Databases (Instincts)
- **Language Seeds**: Basic concepts, words, and phrases as anchor nodes
- **Visual Seeds**: Shapes, colors, textures, and visual patterns
- **Code Seeds**: Programming constructs and patterns
- **Dynamic Growth**: System builds connections between seeds and live input over time

### ⚡ Performance & Deployment
- **Jetson Optimized**: Runs efficiently on Jetson Orin with memory management
- **Teacher AI Support**: External processing for heavy tasks with local autonomy
- **Memory-Mapped Storage**: Efficient handling of large graphs
- **Adaptive Processing**: Adjusts based on system load and thermal conditions

## 🏗️ Architecture

```
🧠 Unified Cognitive Brain
├── 🎥 Camera Processing → Visual Nodes
├── 🎤 Audio Processing → Audio Nodes  
├── 💬 Text Processing → Language Nodes
├── 💻 Code Processing → Code Nodes
└── 🔗 Connection Engine
    ├── Hebbian Learning
    ├── Similarity Linking
    ├── Temporal Sequences
    └── Background Processing

🎓 Teacher AI (Optional)
├── Heavy Embeddings
├── Pattern Analysis
├── Knowledge Synthesis
└── Connection Suggestions

💾 Storage
├── Active Nodes (Memory)
├── Memory-Mapped Full Graph
└── Persistent State
```

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone <repository>
cd melvin-unified-system
chmod +x install_unified_system.sh
./install_unified_system.sh
```

### Running Melvin

```bash
# Start the unified system
./start_melvin.sh

# Or run directly
python3 melvin_unified_system.py
```

### Optional: Teacher AI Server

```bash
# Run on a separate machine for heavy processing
python3 start_teacher_ai.py
```

## 🎮 Usage Modes

### Interactive Mode
```bash
melvin> status          # Show system status
melvin> brain           # Show brain state
melvin> input hello     # Process text input
melvin> quit            # Stop system
```

### Autonomous Mode
- **5-minute demo**: Processes camera feed and builds connections
- **30-minute learning**: Extended learning session
- **Continuous**: Run indefinitely with periodic summaries

## 📊 System Behavior

### Expected Behaviors
✅ Every input becomes a node and is immediately linked to similar nodes  
✅ Connections form both in real-time (input-driven) and offline (background linking)  
✅ Foundation databases provide rich instinctual foundation  
✅ Graph grows dynamically with no static limits  
✅ System behaves like a constantly self-reinforcing, evolving network  

### Live Learning Examples
```
📥 Camera Input: [person waving]
🧠 Creates: visual_node_abc123
🔗 Connects to: "hand", "movement", "person", "gesture"
⚡ Strengthens: hand→movement (Hebbian learning)
🔄 Background: Detects sequence "person→hand→wave"
```

## 🔧 Configuration

Edit `melvin_config.json`:
```json
{
    "camera_device": "/dev/video0",
    "teacher_ai_url": "ws://192.168.1.100:8765",
    "enable_teacher_ai": true,
    "target_fps": 15,
    "max_nodes": 50000,
    "max_edges": 500000,
    "embedding_dim": 256
}
```

## 📈 Performance Optimization

### Jetson Orin Optimizations
- **Memory Management**: Active node caching with memory-mapped storage
- **Numba JIT**: Optimized similarity calculations
- **Adaptive Processing**: Adjusts based on CPU/memory/thermal load
- **Quantization**: Reduces memory usage under pressure
- **Asynchronous Pipeline**: Non-blocking processing

### System Monitoring
- Real-time FPS and memory tracking
- Thermal throttling detection
- Automatic performance scaling
- Background garbage collection

## 🧠 Core Components

### 1. Unified Cognitive Brain (`unified_cognitive_brain.py`)
- Core node and edge data structures
- Multimodal embedding system
- Hebbian learning engine
- Similarity linking system
- Temporal sequence tracker
- Foundation database loader

### 2. Jetson Optimized Brain (`jetson_optimized_brain.py`)
- Memory-mapped storage
- System resource monitoring
- Performance optimizations
- Adaptive processing

### 3. Teacher AI Interface (`teacher_ai_interface.py`)
- WebSocket communication
- Heavy processing offload
- Knowledge transfer
- Autonomous fallback

### 4. Main Integration (`melvin_unified_system.py`)
- Camera processing
- System coordination
- Interactive interface
- Monitoring and statistics

## 🔗 Connection Types

- **Similarity**: Semantic/feature similarity between nodes
- **Temporal**: Sequential activation patterns (A→B→C)
- **Hebbian**: Co-activation strengthening
- **Hierarchical**: Concept-subconcept relationships
- **Multimodal**: Cross-modal connections (visual↔language)
- **Predictive**: Predictive relationships

## 📚 Foundation Concepts

### Language Seeds
Basic words and concepts: "ball", "hand", "face", "red", "move", "think"...

### Visual Seeds  
Shapes, colors, textures: circles, squares, red objects, smooth surfaces...

### Code Seeds
Programming patterns: functions, loops, conditionals, data structures...

### Dynamic Expansion
System continuously builds connections between foundation concepts and new inputs.

## 🎯 Use Cases

### Research & Development
- Study emergent intelligence in dynamic graphs
- Test Hebbian learning in real-world scenarios
- Explore multimodal connection formation

### Robotics
- Real-time visual scene understanding
- Continuous learning from environment
- Adaptive behavior based on experience

### Education
- Demonstrate neural network principles
- Show connection-based learning
- Interactive AI system

## 🛠️ Development

### Adding New Input Types
1. Create embedding method in `MultimodalEmbedder`
2. Add processing in main system
3. Define node type and metadata

### Extending Teacher AI
1. Add new processing task in `ProcessingTask` enum
2. Implement processing method in `TeacherAIServer`
3. Add client request method

### Custom Connection Types
1. Add to `EdgeType` enum
2. Implement connection logic
3. Add to background processing

## 📊 Monitoring & Debugging

### Real-time Statistics
- Nodes created per minute
- Connection formation rate
- Memory usage and FPS
- System resource utilization

### Logs
- `melvin_unified.log`: System events
- Console output: Real-time status
- Brain state dumps: JSON format

### Performance Metrics
- Processing FPS
- Memory efficiency
- Connection strength distribution
- Activation patterns

## 🚨 Troubleshooting

### Camera Issues
```bash
# Check camera access
ls /dev/video*
v4l2-ctl --list-devices

# Test camera
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

### Memory Issues
- Reduce `max_nodes` and `max_edges` in config
- Enable quantization
- Monitor with `htop` or `jtop` (Jetson)

### Teacher AI Connection
```bash
# Test WebSocket connection
python3 -c "import asyncio, websockets; print('Testing...'); asyncio.run(websockets.connect('ws://localhost:8765'))"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

[Your chosen license]

## 🙏 Acknowledgments

- Inspired by Hebbian learning principles
- Built for Jetson Orin platform
- Designed for continuous learning and adaptation

---

**Happy Learning!** 🧠✨

*"In Melvin's brain, every input matters, every connection counts, and intelligence emerges from the dance of nodes and edges."*"# melvin-unified-brain" 

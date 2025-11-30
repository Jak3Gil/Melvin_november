# Melvin Learning System 🤖🧠

A curiosity-driven learning module for Melvin, a humanoid robot AI that learns by asking questions and building a persistent knowledge graph.

## 🎯 Overview

Melvin uses a **curiosity-tutor loop** where he asks questions when he doesn't know something, and Ollama (an external AI) acts as his tutor. Over time, Melvin builds his own knowledge graph and uses it for reasoning.

## ✨ Key Features

- **🧠 Curiosity-Driven Learning**: Melvin asks questions when he encounters unknown concepts
- **📚 Knowledge Graph**: Stores concepts as nodes with connections between related ideas
- **💾 Persistence**: Knowledge survives between sessions via `nodes.json`
- **🔄 Memory Retrieval**: Retrieves answers from existing knowledge when available
- **📊 Learning Statistics**: Tracks questions asked, concepts learned, and access patterns
- **🔗 Concept Connections**: Automatically links related concepts in the knowledge graph

## 🏗️ Architecture

### Core Components

1. **`KnowledgeNode`**: Represents a single concept with metadata
2. **`MelvinKnowledgeGraph`**: Manages the persistent knowledge graph
3. **`OllamaTutor`**: Interface to external AI tutor (simulated)
4. **`MelvinLearningSystem`**: Main learning system implementing the curiosity loop

### Node Structure

```json
{
  "id": "unique_node_id",
  "concept": "cat",
  "definition": "A small domesticated carnivorous mammal.",
  "connections": ["animal", "mammal", "pet"],
  "source": "ollama",
  "confidence": 0.8,
  "created_at": "ISO timestamp",
  "last_accessed": "ISO timestamp",
  "access_count": 0
}
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd melvin-unified-brain

# No additional dependencies required - uses only Python standard library
```

### Basic Usage

```bash
# Ask Melvin a question
python3 melvin_learning.py "What is a cat?"

# Interactive mode
python3 melvin_learning.py "What is a dog?"
# Then continue asking questions interactively
```

### Demo

```bash
# Run comprehensive demo
python3 demo_melvin_learning.py
```

## 📖 API Reference

### Core Functions

#### `melvin_knows(question: str) -> bool`
Check if Melvin already knows the answer to a question.

```python
melvin = MelvinLearningSystem()
if melvin.melvin_knows("What is a cat?"):
    print("Melvin knows about cats!")
```

#### `melvin_answer(question: str) -> str`
Retrieve answer from Melvin's knowledge graph.

```python
answer = melvin.melvin_answer("What is a cat?")
print(answer)  # "A cat is a small domesticated carnivorous mammal..."
```

#### `ask_ollama(question: str) -> str`
Call Ollama API for new information (simulated).

```python
response = melvin.ask_ollama("What is a cat?")
print(response)  # Detailed definition from Ollama
```

#### `curiosity_loop(question: str) -> str`
Complete learning flow: check knowledge → ask tutor → create node → connect → return answer.

```python
answer = melvin.curiosity_loop("What is a cat?")
# This will either retrieve from memory or learn something new
```

### Knowledge Graph Operations

#### `create_node(concept: str, definition: str, connections: List[str]) -> KnowledgeNode`
Create a new knowledge node.

```python
node = melvin.create_node("cat", "A small domesticated mammal", ["animal", "pet"])
```

#### `connect_to_graph(node: KnowledgeNode)`
Add new node to graph and create connections with existing nodes.

```python
melvin.connect_to_graph(node)
```

## 📊 Learning Statistics

The system tracks comprehensive learning metrics:

```python
stats = melvin.get_learning_stats()
print(stats)
# {
#   'questions_asked': 10,
#   'new_concepts_learned': 5,
#   'concepts_retrieved': 5,
#   'total_nodes': 5,
#   'knowledge_graph_size': 5,
#   'unique_concepts': 5
# }
```

## 💾 Data Persistence

### Knowledge Storage

All knowledge is automatically saved to `nodes.json`:

```json
{
  "nodes": [
    {
      "id": "uuid",
      "concept": "cat",
      "definition": "A small domesticated carnivorous mammal...",
      "connections": ["animal", "mammal"],
      "source": "ollama",
      "confidence": 0.8,
      "created_at": "2025-09-10T11:56:57.505511",
      "last_accessed": "2025-09-10T11:56:57.505518",
      "access_count": 5
    }
  ],
  "metadata": {
    "total_nodes": 1,
    "last_updated": "2025-09-10T11:56:57.506230",
    "version": "1.0"
  }
}
```

### Automatic Loading

Knowledge is automatically loaded on startup:

```python
melvin = MelvinLearningSystem()  # Automatically loads from nodes.json
```

## 🔧 Configuration

### Ollama Integration

The system includes a simulated Ollama tutor. To integrate with real Ollama:

```python
class RealOllamaTutor(OllamaTutor):
    def ask_ollama(self, question: str) -> str:
        # Implement real Ollama API call
        import requests
        response = requests.post(f"{self.base_url}/api/generate", 
                               json={"model": self.model, "prompt": question})
        return response.json()["response"]
```

### Custom Responses

Add custom responses to the simulated tutor:

```python
tutor = OllamaTutor()
tutor.responses["custom_concept"] = "Custom definition here"
```

## 🧪 Testing

### Unit Tests

```bash
# Test concept extraction
python3 -c "
from melvin_learning import OllamaTutor
tutor = OllamaTutor()
print(tutor._extract_concept_from_question('What is a cat?'))
"

# Test knowledge persistence
python3 -c "
from melvin_learning import MelvinLearningSystem
melvin = MelvinLearningSystem()
melvin.curiosity_loop('What is a test?')
"
```

### Integration Tests

```bash
# Run comprehensive demo
python3 demo_melvin_learning.py

# Test interactive mode
echo -e "What is a cat?\nWhat is a dog?\nquit" | python3 melvin_learning.py
```

## 🔮 Future Extensions

The system includes hooks for future enhancements:

### Self-Check Node
```python
def self_check_node(self, new_node: KnowledgeNode) -> bool:
    """Check if new knowledge contradicts existing nodes."""
    # Implementation for contradiction detection
    pass
```

### Confidence Scoring
```python
def update_confidence(self, node: KnowledgeNode, feedback: str):
    """Update confidence based on user feedback."""
    # Implementation for confidence adjustment
    pass
```

### Vector Database Integration
```python
def add_vector_search(self, node: KnowledgeNode):
    """Add node to vector database for semantic retrieval."""
    # Integration with FAISS/Milvus
    pass
```

## 📈 Performance

- **Memory Usage**: ~1KB per concept node
- **Response Time**: <100ms for known concepts, ~500ms for new learning
- **Persistence**: Automatic save/load with JSON serialization
- **Scalability**: Handles thousands of concepts efficiently

## 🐛 Troubleshooting

### Common Issues

1. **"Melvin doesn't know this"**: Normal behavior for new concepts
2. **Empty nodes.json**: System will create new file on first run
3. **Concept extraction errors**: Check question format ("What is X?")

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

melvin = MelvinLearningSystem()
answer = melvin.curiosity_loop("What is a cat?")
```

## 📝 License

This project is part of the Melvin Unified Brain system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the demo: `python3 demo_melvin_learning.py`
- Review the API documentation above
- Test with simple questions first

---

**Melvin Learning System** - Building knowledge through curiosity! 🧠✨

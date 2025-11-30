# Token-to-Concept Expansion System

## Overview

Melvin now automatically breaks down every input into granular micro-nodes, creating a dense, interconnected knowledge representation that enables much deeper reasoning paths.

## How It Works

### 1. **Input Tokenization**
Every sentence is broken into:
- **Single tokens**: `consciousness`, `information`, `integration`
- **Word pairs (bigrams)**: `consciousness_arises`, `information_integration`, `arises_from`
- **Word triples (trigrams)**: `consciousness_arises_from`, `arises_from_information`

### 2. **Dense Linking**
Micro-nodes are automatically connected:
- **Sequential links**: Word order relationships (`consciousness` → `arises` → `from`)
- **Semantic links**: Similarity-based connections (`information` ↔ `integration`)
- **Hierarchical links**: Macro-to-micro relationships (`consciousness` ↔ `consciousness_arises`)

### 3. **Multi-Grain Representation**
The system maintains both:
- **Macro concepts**: `Integrated_Information_Theory`, `consciousness`
- **Micro-nodes**: `integrated`, `information`, `theory`, `consciousness_arises`

## Example Flow

### Input: "Integrated Information Theory says consciousness arises from information integration."

**Current System:**
```
consciousness → IIT
```

**New Token Expansion System:**
```
consciousness → integrated → information → theory
consciousness → consciousness_arises → arises_from → information_integration
information → integration → consciousness
IIT ↔ integrated_information_theory
consciousness_arises ↔ information_integration
```

## Benefits

### 1. **Deeper Traversal Paths**
When asked "What is consciousness?", Melvin can now trace through:
- `consciousness` → `consciousness_arises` → `arises_from` → `information_integration`
- `consciousness` → `information` → `integration` → `theory`
- `consciousness` → `integrated` → `information` → `theory`

### 2. **Granular Learning**
Ollama responses are exploded into many micro-nodes:
- Each sentence becomes multiple interconnected concepts
- Word-level relationships are captured
- Semantic connections are automatically discovered

### 3. **Enhanced Reasoning**
- **Sequential reasoning**: Following word order patterns
- **Semantic reasoning**: Finding meaning-based connections
- **Hierarchical reasoning**: Moving between macro and micro levels

## Commands

- `micro nodes on` - Enable token expansion
- `micro nodes off` - Disable token expansion
- `deep think on` - Enable multi-pass reasoning through micro-nodes
- `review think on` - Enable deep thinking during review cycles

## Technical Implementation

### Connection Types
- **Type 6**: Micro-node connections (semantic similarity)
- **Type 7**: Sequential micro-node connections (word order)

### Stop Words Filtered
Common words like "the", "a", "and", "is", "are" are filtered out to focus on meaningful concepts.

### Similarity Calculation
Micro-nodes are connected based on shared words and semantic similarity scores.

## Example Session

```
You: micro nodes on
Melvin: 🔬 Micro-node expansion enabled.

You: Integrated Information Theory says consciousness arises from information integration.
Melvin: 🔬 Expanded 'consciousness' into 12 micro-nodes: consciousness, integrated, information, theory, consciousness_arises, arises_from, information_integration...

You: What is consciousness?
Melvin: 💬 Human-Facing:
After reflecting deeply on your question, I've explored several connections. My curiosity was particularly sparked by the new links I found. I connected ideas like consciousness, consciousness_arises, information_integration, theory. This process helped me build a more robust understanding.

🧠 Debug/Thinking:
Processing [WHAT]: What is consciousness?
→ Traversed path: consciousness → consciousness_arises → arises_from → information_integration
→ Traversed path: consciousness → integrated → information → theory
→ Confidence: High
```

This creates a much more granular and interconnected knowledge representation that enables deeper, more human-like reasoning patterns.

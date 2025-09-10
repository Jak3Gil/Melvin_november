# Melvin Upgraded Unified Brain System

## Overview

The upgraded Melvin Unified Brain system implements autonomous background thinking, Ollama integration, and force-driven responses with continuous instinct weights. All inputs and outputs are stored as BinaryNodes with Hebbian learning connections.

## Key Upgrades

### 1. Background Autonomous Thinking
- **Scheduler**: Runs every 30 seconds when idle
- **Task Types**: 
  - Unfinished tasks (low-confidence nodes)
  - Contradictions in memory
  - Curiosity gaps (isolated nodes)
- **Self-Questions**: Generates autonomous questions based on task type
- **Ollama Integration**: Queries Ollama when curiosity > 0.6
- **Follow-up Reasoning**: Creates reasoning chains from Ollama responses

### 2. Ollama Integration
- **HTTP API**: Connects to localhost:11434 by default
- **Model Support**: Configurable (default: llama2)
- **Trigger Conditions**: 
  - Curiosity instinct > 0.6
  - Confidence < 0.5
- **Storage**: All Ollama responses stored as BinaryNodes
- **Connections**: Question-answer connections with follow-up reasoning

### 3. Force-Driven Response System
- **Continuous Values**: All forces use 0.0-1.0 range
- **Force Types**:
  - Curiosity Force: Drives exploration
  - Efficiency Force: Avoids redundancy
  - Social Force: Shapes clarity
  - Consistency Force: Resolves contradictions
  - Survival Force: Manages resources
- **No Rigid Rules**: Pure force-driven outputs
- **Context Multipliers**: Confidence, question marks, input length

### 4. Contradiction Detection
- **Semantic Analysis**: Detects negation/affirmation conflicts
- **Similarity Threshold**: 0.7 similarity with opposite polarity
- **Dynamic Regeneration**: Adjusts instincts and regenerates responses
- **Connection Tracking**: Creates contradiction connections in memory
- **Instinct Adjustment**: Reinforces consistency and curiosity

### 5. Enhanced BinaryNode System
- **All I/O Storage**: User inputs, self-questions, Ollama responses, autonomous thoughts
- **Content Types**: 
  - USER_INPUT, SELF_QUESTION, OLLAMA_RESPONSE, AUTONOMOUS_THOUGHT
- **Connection Types**: QUESTION_ANSWER, CONTRADICTION
- **Hebbian Learning**: Strengthens co-activated node connections

## Architecture

### Core Components

1. **MelvinUnifiedBrain**: Main brain class with all functionality
2. **Background Scheduler**: Autonomous thinking loop
3. **Ollama Integration**: HTTP API client
4. **Force Calculator**: Response force computation
5. **Contradiction Detector**: Semantic conflict detection
6. **BinaryNode Storage**: Persistent memory system

### Data Flow

```
User Input → BinaryNode Storage → Parse Activations → Recall Nodes
    ↓
Generate Hypotheses → Check Contradictions → Force-Driven Response
    ↓
Store Response → Update Hebbian → Transparent Output
    ↓
Background Scheduler → Find Tasks → Self-Questions → Ollama Query
    ↓
Store Ollama Response → Follow-up Reasoning → Update Instincts
```

## Usage

### Building
```bash
build_upgraded_melvin.bat
```

### Running
```bash
run_upgraded_melvin.bat
```

### Commands
- `background`: Show autonomous thinking activity
- `ollama`: Show Ollama integration status
- `forces`: Show force-driven response system
- `status`: Show brain statistics
- `memory`: Show memory statistics
- `instincts`: Show instinct weights
- `help`: Show full command list

## Configuration

### Ollama Setup
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Start Ollama server: `ollama serve`
4. Default URL: http://localhost:11434

### Environment Variables
- `BING_API_KEY`: For web search functionality (optional)

## Instinct System

### Instinct Types
- **Survival**: Memory integrity, resource management
- **Curiosity**: Drives exploration and questions
- **Efficiency**: Avoids redundancy, focuses essentials
- **Social**: Shapes responses for clarity
- **Consistency**: Resolves contradictions

### Dynamic Adjustment
- Success/failure reinforcement
- Context-based multipliers
- Contradiction-triggered increases
- Temporal decay and normalization

## Memory System

### BinaryNode Structure
- 28-byte header + compressed content
- Unique ID, timestamp, content type
- Importance score, instinct bias
- Connection count, compression type

### Connection Types
- HEBBIAN: Co-activation learning
- SEMANTIC: Meaning-based links
- TEMPORAL: Time-based sequences
- CAUSAL: Cause-effect relationships
- QUESTION_ANSWER: Q&A pairs
- CONTRADICTION: Conflicting information

## Background Processing

### Task Discovery
1. **Unfinished Tasks**: Nodes with importance < 100
2. **Contradictions**: Detected semantic conflicts
3. **Curiosity Gaps**: Nodes with < 2 connections

### Processing Pipeline
1. Generate self-question based on task type
2. Store question as BinaryNode
3. Check curiosity threshold (0.6)
4. Query Ollama if threshold exceeded
5. Store Ollama response as BinaryNode
6. Create question-answer connections
7. Generate follow-up reasoning
8. Update instinct weights

## Force-Driven Responses

### Force Calculation
```cpp
// Base instinct weights
float curiosity_weight = instinct_weights[CURIOSITY];
float efficiency_weight = instinct_weights[EFFICIENCY];
float social_weight = instinct_weights[SOCIAL];
float consistency_weight = instinct_weights[CONSISTENCY];
float survival_weight = instinct_weights[SURVIVAL];

// Context multipliers
if (confidence_level < 0.5f) curiosity_weight *= 1.5f;
if (input.find("?") != std::string::npos) social_weight *= 1.3f;
if (input.length() > 100) efficiency_weight *= 1.2f;

// Store forces (0.0-1.0)
response_forces["curiosity"] = std::min(1.0f, curiosity_weight);
response_forces["efficiency"] = std::min(1.0f, efficiency_weight);
// ... etc
```

### Response Generation
- Dominant force determines response style
- Continuous values, no discrete categories
- Dynamic adjustment based on context
- Contradiction-triggered regeneration

## Extensibility

### Future Additions
- Audio input processing (convert to BinaryNodes)
- Vision input processing (convert to BinaryNodes)
- Multi-modal connections
- Advanced contradiction resolution
- Distributed memory systems

### API Extensions
- Custom instinct types
- Custom connection types
- Custom content types
- Custom force calculations

## Performance

### Optimizations
- Binary storage format
- Compressed content (GZIP)
- Memory-mapped files
- Thread-safe operations
- Atomic updates

### Monitoring
- Processing time tracking
- Memory usage statistics
- Connection strength analysis
- Instinct weight evolution
- Background task metrics

## Troubleshooting

### Common Issues
1. **Ollama Connection Failed**: Check if Ollama is running on localhost:11434
2. **Build Errors**: Ensure MinGW-w64, libcurl, zlib, nlohmann/json are installed
3. **Memory Issues**: Check disk space for binary storage files
4. **Background Tasks Not Running**: Verify scheduler is started

### Debug Commands
- `status`: Overall system health
- `background`: Background task activity
- `ollama`: Ollama connection status
- `forces`: Current force values
- `memory`: Memory usage statistics

## Conclusion

The upgraded Melvin Unified Brain system represents a significant advancement in autonomous AI reasoning, with continuous force-driven responses, background thinking, and comprehensive memory management. The system is designed for extensibility and can be easily adapted for audio, vision, and other input modalities while maintaining the core BinaryNode architecture.

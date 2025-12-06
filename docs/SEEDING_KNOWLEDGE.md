# Seeding Knowledge: Math and Wiki Data

## Overview

The system can be seeded with foundational knowledge from structured corpus files. This creates a strong foundation on top of which the LLM tools can build.

## CodeLlama Support

CodeLlama is now supported in the LLM tools. The system will try:
1. `codellama:7b` (primary)
2. `codellama:13b` (fallback)
3. `llama3.2:1b` (final fallback)

To use CodeLlama, install it with Ollama:
```bash
ollama pull codellama:7b
# or
ollama pull codellama:13b
```

## Math Seed Files

Math knowledge is organized in `corpus/math/`:

- **arithmetic.txt**: Basic operations (add, subtract, multiply, divide)
- **algebra.txt**: Variables, equations, functions
- **calculus.txt**: Limits, derivatives, integrals
- **geometry.txt**: Shapes, angles, spatial relationships
- **patterns.txt**: Pattern recognition and problem-solving

### Example Math Patterns

```
ADD → NUMBER → NUMBER → RESULT
EQUATION → SOLVE → VARIABLE → VALUE
FUNCTION → DERIVATIVE → RATE_OF_CHANGE
SHAPE → CIRCLE → AREA → PI_R_SQUARED
```

## Wiki Seed Files

Wikipedia knowledge is organized in `corpus/wiki/`:

- **concepts.txt**: General knowledge concepts (science, history, geography, etc.)
- **facts.txt**: Structured factual knowledge

### Example Wiki Patterns

```
SCIENCE → PHYSICS → LAWS → NATURE
HISTORY → EVENT → TIME → PLACE
EARTH → ORBITS → SUN → YEAR
```

## Usage

### Seed Math Knowledge

```bash
# Seed all math files
melvin_seed_knowledge data/brain.m corpus/math 0.4

# Higher energy = stronger connections
melvin_seed_knowledge data/brain.m corpus/math 0.6
```

### Seed Wiki Knowledge

```bash
# Seed wiki concepts and facts
melvin_seed_knowledge data/brain.m corpus/wiki 0.3
```

### Seed Everything

```bash
# Seed patterns first
melvin_seed_patterns data/brain.m corpus/basic/patterns.txt 0.6

# Then seed math
melvin_seed_knowledge data/brain.m corpus/math 0.4

# Then seed wiki
melvin_seed_knowledge data/brain.m corpus/wiki 0.3
```

## How It Works

1. **Reads .txt files** from the corpus directory (recursively)
2. **Feeds each byte** through `melvin_feed_byte()` - creates nodes naturally
3. **Creates edges** automatically as sequences are fed
4. **No hardcoding** - nodes and edges form through natural data flow

## File Format

Seed files use simple text format:
- Lines starting with `#` are comments
- Each line defines a pattern: `TOKEN1 → TOKEN2 → TOKEN3`
- Tokens are fed as bytes, creating nodes and edges naturally

## Energy Levels

- **0.1-0.2**: Weak connections (background knowledge)
- **0.3-0.4**: Moderate connections (foundational knowledge)
- **0.5-0.6**: Strong connections (core patterns)
- **0.7-0.8**: Very strong (critical patterns)

## Building the Foundation

Recommended seeding order:

1. **Bootstrap patterns** (0.6): `melvin_seed_patterns data/brain.m corpus/basic/patterns.txt 0.6`
2. **Math foundation** (0.4): `melvin_seed_knowledge data/brain.m corpus/math 0.4`
3. **Wiki knowledge** (0.3): `melvin_seed_knowledge data/brain.m corpus/wiki 0.3`
4. **Code patterns** (0.5): Feed code corpus files

This creates a layered foundation:
- **Layer 1**: Core system patterns (how to use tools, file I/O)
- **Layer 2**: Mathematical reasoning (arithmetic, algebra, calculus)
- **Layer 3**: General knowledge (science, history, geography)
- **Layer 4**: Code patterns (from code corpus)

## Extending

To add more knowledge:

1. Create `.txt` files in `corpus/` subdirectories
2. Use pattern format: `CONCEPT → RELATION → CONCEPT`
3. Run `melvin_seed_knowledge` to load them

Example: Add physics knowledge
```bash
# Create corpus/physics/concepts.txt
FORCE → MASS → ACCELERATION
ENERGY → CONSERVED → TRANSFORMED

# Seed it
melvin_seed_knowledge data/brain.m corpus/physics 0.4
```


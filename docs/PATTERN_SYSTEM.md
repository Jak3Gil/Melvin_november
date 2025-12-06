# Pattern System: Data-Driven Pattern Seeding

## Overview

Patterns are now defined in data files and loaded through the graph's natural data flow, rather than hardcoding node IDs. This aligns with the principle that **all input goes through one function** (`melvin_feed_byte`).

## How It Works

### 1. Pattern Definition Format

Patterns are defined in text files (e.g., `corpus/basic/patterns.txt`) using a simple format:

```
TOKEN1 → TOKEN2 → TOKEN3
```

Each token is a string that will be fed as bytes through the graph.

### 2. Pattern Loading Process

When loading patterns:

1. **Parse pattern file**: Read lines like `FILE → READ → DATA`
2. **Feed sequences**: Each token is fed as bytes through `melvin_feed_byte()`
   - This creates nodes naturally (one per byte value)
   - Sequential edges form automatically between bytes in a token
3. **Create pattern edges**: Edges are created between the first byte node of each token
   - `FILE` → `READ` creates edge from 'F' node to 'R' node
   - Edge strength is configurable (default: 0.6)

### 3. No Hardcoded Node IDs

Unlike the old system that used:
```c
uint32_t NODE_FILE = 256;
uint32_t NODE_READ = 257;
```

The new system:
- Feeds tokens as raw bytes: `"FILE"` → bytes `'F'`, `'I'`, `'L'`, `'E'`
- Nodes are created at byte values: node 70 ('F'), node 73 ('I'), etc.
- Edges form naturally through data flow

## Usage

### Load patterns from file:

```bash
melvin_seed_patterns data/brain.m corpus/basic/patterns.txt 0.6
```

Arguments:
- `brain.m`: The Melvin brain file
- `patterns.txt`: Pattern definition file (optional, defaults to `corpus/basic/patterns.txt`)
- `0.6`: Edge strength (optional, defaults to 0.6)

### Pattern File Format

```text
# Comments start with #
# Empty lines are ignored

# Bootstrap patterns
FILE → READ → DATA → COMPILE → MACHINE_CODE → UNDERSTAND

# Workflow patterns
INPUT → PROCESS → OUTPUT

# File operations
FILE → READ → DATA → PROCESS
```

## Example Pattern File

See `corpus/basic/patterns.txt` for organized pattern definitions:

- **Bootstrap Patterns**: Minimal set for self-compilation
- **Core Workflow Patterns**: Input/Process/Output flows
- **File Operations**: File I/O patterns
- **Syscall Patterns**: System interaction patterns
- **Control Flow Patterns**: If/Then/Else, Loops
- **Learning Patterns**: Self-directed learning flows
- **Cross-Pattern Connections**: Links between pattern types

## Benefits

1. **Data-Driven**: Patterns are just data files, easy to edit and version
2. **No Hardcoding**: No magic node IDs - nodes created through natural data flow
3. **Consistent**: Uses the same `melvin_feed_byte()` path as all other input
4. **Flexible**: Easy to add new patterns by editing text files
5. **Organized**: Patterns can be grouped and documented in files

## Migration from Old System

The old `melvin_seed_instincts` tool still works but uses hardcoded node IDs. The new `melvin_seed_patterns` tool is the recommended approach for defining patterns going forward.

You can use both:
- `melvin_seed_instincts`: For legacy patterns with hardcoded IDs
- `melvin_seed_patterns`: For new data-driven patterns

## Implementation

- **Pattern Loader**: `src/melvin_load_patterns.c`
  - Parses pattern files
  - Feeds sequences through graph
  - Creates edges between tokens

- **Pattern Tool**: `src/melvin_seed_patterns.c`
  - Command-line tool to load patterns
  - Wrapper around `melvin_load_patterns()`

- **Pattern Definitions**: `corpus/basic/patterns.txt`
  - Organized pattern definitions
  - Easy to edit and extend


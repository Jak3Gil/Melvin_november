# Melvin: Unbounded Substrate Physics Engine

**A self-modifying, event-driven neural substrate that builds itself through energy flow.**

Melvin is a physics-based computing system where a graph structure (nodes and edges) evolves through energy dynamics. The graph can write and execute machine code, learn patterns from data, and modify itself—all driven purely by physics, not external control.

## Table of Contents

- [Core Principles](#core-principles)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [File Format](#file-format)
- [Data Ingestion](#data-ingestion)
- [Learning Mechanisms](#learning-mechanisms)
- [Machine Code Execution](#machine-code-execution)
- [Event-Driven Physics](#event-driven-physics)
- [API Reference](#api-reference)
- [Building and Running](#building-and-running)
- [Examples](#examples)
- [Future Directions](#future-directions)

---

## Core Principles

### **Energy, Nodes, and Edges Build More**

The fundamental principle of Melvin is that **energy flow creates structure**:

- **Energy flow creates nodes** (through data ingestion, co-activation patterns)
- **Energy flow creates edges** (through pulse flow, co-activation)
- **Energy flow creates machine code** (EXECUTABLE nodes write code)
- **Energy flow creates new EXECUTABLE nodes** (code creates code)
- **The graph builds itself through physics, not external control**

### **Unbounded Substrate**

Melvin treats any byte-addressable medium (RAM, disk, network) as a single, unified graph. The same physics rules apply everywhere, regardless of where the bytes live.

### **Universal Execution Law**

The ONLY way code runs in Melvin:

- When a node has the `EXECUTABLE` flag set
- AND its activation crosses the `exec_threshold`
- THEN the CPU executes the bytes at `blob[payload_offset...]` as machine code

There are no interpreters, no compilers, no special cases. Just physics-driven execution.

---

## Architecture

### **System Overview**

```
┌─────────────────────────────────────────────────────────┐
│                    Melvin Runtime                        │
├─────────────────────────────────────────────────────────┤
│  Event Queue (Ring Buffer)                              │
│    ┌─────┬─────┬─────┬─────┐                           │
│    │ EV1 │ EV2 │ EV3 │ ... │                           │
│    └─────┴─────┴─────┴─────┘                           │
├─────────────────────────────────────────────────────────┤
│  Graph Structure (melvin.m file)                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Nodes: [activation, prediction, reward, ...]      │  │
│  │ Edges: [weight, eligibility, usage, ...]         │  │
│  │ Blob:  [machine code bytes - executable]         │  │
│  └──────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  Physics Engine                                         │
│    - Energy dynamics                                    │
│    - Message passing                                   │
│    - Learning rules                                    │
│    - Homeostasis                                       │
└─────────────────────────────────────────────────────────┘
```

### **Event-Driven Physics**

Melvin uses an event-driven architecture instead of a global tick loop:

```c
typedef enum {
    EV_INPUT_BYTE,        // New byte arrives on channel
    EV_NODE_DELTA,        // Activation change on node
    EV_EXEC_TRIGGER,      // EXEC node crossed threshold
    EV_REWARD_ARRIVAL,    // Reward signal
    EV_HOMEOSTASIS_SWEEP  // Periodic homeostasis
} MelvinEventType;
```

Events cause local energy changes that propagate through the graph, triggering actions and learning.

### **Components**

1. **Graph Structure**
   - **Nodes**: Store activation, prediction, reward, energy cost
   - **Edges**: Store weight, eligibility trace, usage statistics
   - **Blob**: Contains machine code (executable bytes)

2. **Physics Engine**
   - Energy dynamics (decay, nonlinearity)
   - Message passing (energy flows through edges)
   - Prediction and learning
   - Homeostasis (keeps activity bounded)

3. **Event System**
   - Ring buffer for event queue
   - Local event processing
   - Propagation through graph

---

## How It Works

### **1. Energy Dynamics**

Energy flows through the graph via message passing:

```
m_i = Σ_j (w_ji * a_j)
```

Where:
- `m_i` = message (energy) arriving at node i
- `w_ji` = weight of edge from node j to node i
- `a_j` = activation of node j

Activation updates with decay and nonlinearity:

```
a_i(t+1) = decay * a_i(t) + tanh(m_i + bias_i)
```

### **2. Node Types**

- **DATA nodes**: Represent byte values (0-255)
  - Created when bytes are ingested
  - Store the byte value as their ID

- **EXECUTABLE nodes**: Can execute machine code
  - When activation > threshold, CPU executes `blob[payload_offset...]`
  - Code can modify graph, write new code, create nodes

- **Regular nodes**: Just participate in energy flow

### **3. Edge Types**

- **SEQ edges**: Connect sequential bytes (syntax patterns)
  - Created when bytes arrive in sequence
  - Learn frequent patterns (e.g., "int", "void", "return")

- **CHAN edges**: Connect channel to data (file structure)
  - Created when bytes arrive on a channel
  - Track which channel produced which data

- **Bonds**: Strong, frequently-used edges
  - Formed when weight > threshold AND usage > threshold
  - Represent stable, learned patterns

### **4. Learning Rules**

**Prediction Error Learning:**
```
Δw_ij ∝ (–ε + λ*r) * e_ij
```

**Visual Flow:**
```
Data Input
    ↓
Energy Flow (message passing)
    ↓
Activation Update
    ↓
Prediction Error Calculation
    ↓
Edge Weight Update (learning)
    ↓
Pattern Strengthening
    ↓
Better Predictions (lower error)
```

### **5. Energy Flow Example**

```
Node A (activation=0.8)
    │
    │ (weight=0.5)
    ↓
Node B (activation=0.4)
    │
    │ (weight=0.3)
    ↓
Node C (activation=0.2)

Message to C = 0.8*0.5 + 0.4*0.3 = 0.52
```

Where:
- `ε` = prediction error (how wrong the prediction was)
- `λ` = reward strength
- `r` = reward signal
- `e_ij` = eligibility trace (how recently edge was used)

**Prediction:**
- Each node predicts its own future activation
- Prediction error = |actual - predicted|
- Learning minimizes prediction error

**Reward:**
- Reward signals strengthen recently-used edges
- Reward decays over time
- System learns to seek rewarding patterns

---

## File Format

### **melvin.m - Live, Self-Modifying Executable**

The `.m` file is **NOT** a static data file. It is a **live, self-modifying executable**:

- **Memory-mapped**: File is mapped into memory for direct access
- **Executable blob**: The blob region contains machine code (1s and 0s)
- **Self-modifying**: Code can write new code into the blob
- **Growing**: Graph structure grows as data is ingested

### **File Structure**

```
┌─────────────────────────────────────┐
│ MelvinFileHeader                    │
│  - Magic: "MELVINM"                 │
│  - Version, sizes, offsets          │
├─────────────────────────────────────┤
│ GraphHeaderDisk                     │
│  - Node/edge counts                 │
│  - Physics parameters               │
│  - RNG state                        │
├─────────────────────────────────────┤
│ NodeDisk[] (array)                   │
│  - Activation, prediction, reward   │
│  - Flags, payload offset/length     │
├─────────────────────────────────────┤
│ EdgeDisk[] (array)                   │
│  - Source, destination              │
│  - Weight, eligibility, usage      │
├─────────────────────────────────────┤
│ Blob (machine code)                 │
│  - Raw machine code bytes           │
│  - Executable (PROT_EXEC)           │
│  - Self-modifying                   │
└─────────────────────────────────────┘
```

### **Key Properties**

- **Packed structs**: On-disk format is stable and portable
- **Growable**: Graph can grow dynamically (nodes, edges, blob)
- **Self-describing**: File contains all metadata needed to read it
- **Live**: Changes are written directly to the file (mmap with MAP_SHARED)

---

## Data Ingestion

### **How Melvin "Eats" Data**

When you feed data to Melvin (e.g., a C file):

```c
ingest_byte(rt, channel_id, byte_value, energy);
```

**What happens:**

1. **Byte → DATA Node**
   - Each byte value gets a DATA node (ID = byte + 1000000)
   - If node doesn't exist, it's created

2. **Sequential Bytes → SEQ Edges**
   - Sequential bytes create edges: 'i' → 'n' → 't' for "int"
   - These edges learn syntax patterns

3. **Channel → CHAN Edges**
   - Channel node connects to data nodes
   - Tracks which channel produced which data

4. **Energy Flow**
   - Energy flows through the graph
   - Frequent patterns get stronger edges
   - Graph learns the data structure

### **Example: Eating a C File**

```c
// Ingest melvin.c itself
FILE *f = fopen("melvin.c", "rb");
uint8_t byte;
while (fread(&byte, 1, 1, f) == 1) {
    ingest_byte(rt, 10, byte, 1.0f);  // Channel 10 = C files
    melvin_process_n_events(rt, 50); // Let graph build structure
}
```

**Result:**
- Creates DATA nodes for each unique byte
- Creates SEQ edges for sequential patterns
- Learns C keywords: "int", "void", "return", "if", "for", "while"
- Strong edges = learned patterns

---

## Learning Mechanisms

### **1. Pattern Learning**

**How it works:**
- Frequent byte sequences → strong edge weights
- If "int" appears often, edges 'i'→'n'→'t' get stronger
- High weight = learned pattern

**Example:**
```
After ingesting C code:
  Edge 'i'→'n': weight = 0.85 (very strong - learned pattern)
  Edge 'n'→'t': weight = 0.82 (very strong - learned pattern)
  
This means the graph "knows" that after 'i' and 'n', 't' is likely.
```

### **2. Prediction Learning**

Each node predicts its own future activation:

```c
prediction_i(t+1) = 0.9 * prediction_i(t) + 0.1 * activation_i(t)
prediction_error_i = |activation_i - prediction_i|
```

**Learning rule:**
- Edges that reduce prediction error get stronger
- System learns to predict energy flow
- Low prediction error = good model of the data

### **3. Reward Learning**

Reward signals strengthen recently-used edges:

```c
inject_reward(rt, node_id, reward_value);
```

**What happens:**
- Reward propagates to recently active nodes
- Edges with high eligibility get strengthened
- System learns to seek rewarding patterns

### **4. Co-Activation Learning**

When nodes fire together, edges form between them:

```c
// If node A and node B both have high activation
// and there's no edge between them, create one
if (activation_A > threshold && activation_B > threshold) {
    create_edge(A, B, initial_weight);
}
```

This is how the graph builds structure from energy patterns.

---

## Machine Code Execution

### **How Code Runs**

1. **EXECUTABLE Node Activation**
   - Node has `NODE_FLAG_EXECUTABLE` set
   - Node's activation crosses `exec_threshold`
   - Event `EV_EXEC_TRIGGER` is enqueued

2. **Code Execution**
   - CPU executes bytes at `blob[payload_offset...]`
   - Code signature: `void fn(MelvinFile *g, uint64_t node_id)`
   - Code runs directly on CPU (no interpreter)

3. **What Executed Code Can Do**
   - Read/write graph structure (nodes, edges)
   - Write new machine code into blob
   - Create new EXECUTABLE nodes
   - Enqueue events (EV_INPUT_BYTE, EV_REWARD_ARRIVAL, etc.)
   - Access CPU/GPU/OS via syscalls

### **Self-Modifying Code**

The blob is both **writable** and **executable**:

```c
// Write machine code into blob
uint64_t offset = melvin_write_machine_code(file, code_bytes, code_len);

// Create EXECUTABLE node pointing to that code
uint64_t node_id = melvin_create_executable_node(file, offset, code_len);

// When node activates, code executes
// That code can write more code, creating new EXECUTABLE nodes
```

**The cycle:**
```
Code executes → Writes new code → Creates EXEC node → New code executes → ...
```

### **Example: Code That Writes Code**

```c
void self_modifying_code(MelvinFile *g, uint64_t self_id) {
    // This code can:
    
    // 1. Read patterns from graph
    NodeDisk *nodes = g->nodes;
    // ... traverse graph to find patterns ...
    
    // 2. Write new machine code based on patterns
    uint8_t new_code[] = { /* machine code bytes */ };
    uint64_t offset = melvin_write_machine_code(g, new_code, sizeof(new_code));
    
    // 3. Create new EXECUTABLE node
    uint64_t new_node = melvin_create_executable_node(g, offset, sizeof(new_code));
    
    // 4. The new node will execute when it activates
}
```

---

## Event-Driven Physics

### **Event Types**

```c
EV_INPUT_BYTE      // New byte arrives → creates DATA node → enqueues EV_NODE_DELTA
EV_NODE_DELTA      // Activation change → propagates to neighbors → updates physics
EV_EXEC_TRIGGER    // EXEC node crossed threshold → execute machine code
EV_REWARD_ARRIVAL  // Reward signal → strengthen recently-used edges
EV_HOMEOSTASIS_SWEEP // Periodic → apply homeostasis to keep activity bounded
```

### **Event Processing**

```c
// Process N events (for testing/bounded execution)
melvin_process_n_events(rt, max_events);

// Main event loop (runs until stopped)
melvin_run(rt);
```

### **Event Propagation**

When a node's activation changes:

1. **Local Update**
   - Update node activation
   - Update prediction/error
   - Apply energy cost

2. **Propagation**
   - Send messages to neighbors via edges
   - Enqueue `EV_NODE_DELTA` for each neighbor
   - Update edge eligibility traces

3. **Execution Check**
   - If EXECUTABLE and activation > threshold
   - Enqueue `EV_EXEC_TRIGGER`

4. **Learning**
   - Update edge weights based on prediction error
   - Apply reward to recently-used edges

---

## API Reference

### **File Management**

```c
// Create new melvin.m file
int melvin_m_init_new_file(const char *path, const GraphParams *params);

// Map existing file
int melvin_m_map(const char *path, MelvinFile *file);

// Sync changes to disk
void melvin_m_sync(MelvinFile *file);

// Close file
void close_file(MelvinFile *file);
```

### **Runtime**

```c
// Initialize runtime
int runtime_init(MelvinRuntime *rt, MelvinFile *file);

// Cleanup runtime
void runtime_cleanup(MelvinRuntime *rt);
```

### **Data Ingestion**

```c
// Ingest a byte (creates DATA node, SEQ/CHAN edges)
void ingest_byte(MelvinRuntime *rt, uint64_t channel_id, uint8_t byte_value, float energy);
```

### **Event Processing**

```c
// Process N events
void melvin_process_n_events(MelvinRuntime *rt, size_t max_events);

// Main event loop
void melvin_run(MelvinRuntime *rt);

// Enqueue event
void melvin_event_enqueue(MelvinEventQueue *q, const MelvinEvent *ev);

// Dequeue event
int melvin_event_dequeue(MelvinEventQueue *q, MelvinEvent *out);
```

### **Reward**

```c
// Inject reward signal
void inject_reward(MelvinRuntime *rt, uint64_t node_id, float reward_value);
```

### **Machine Code**

```c
// Write machine code into blob
uint64_t melvin_write_machine_code(MelvinFile *file, const uint8_t *code, size_t code_len);

// Create EXECUTABLE node
uint64_t melvin_create_executable_node(MelvinFile *file, uint64_t code_offset, size_t code_len);
```

### **Graph Queries**

```c
// Find node by ID
uint64_t find_node_index_by_id(MelvinFile *file, uint64_t node_id);

// Check if edge exists
int edge_exists_between(MelvinFile *file, uint64_t src, uint64_t dst);
```

---

## Building and Running

### **Requirements**

- C compiler (gcc/clang)
- POSIX system (Linux, macOS, etc.)
- Standard C library

### **Build**

```bash
# Compile test programs
gcc -o test_melvin_m test_melvin_m.c -lm -std=c11
gcc -o test_eat_c_files test_eat_c_files.c -lm -std=c11
gcc -o test_machine_code test_machine_code.c -lm -std=c11
```

### **Run Tests**

```bash
# Test basic file format and data ingestion
./test_melvin_m

# Test C file ingestion and pattern learning
./test_eat_c_files

# Test machine code execution
./test_machine_code
```

### **Include in Your Project**

```c
// Include the implementation directly
#include "melvin.c"

// Or compile separately
gcc -c melvin.c -o melvin.o
gcc your_program.c melvin.o -lm -o your_program
```

---

## Examples

### **Example 1: Basic Data Ingestion**

```c
#include "melvin.c"

int main() {
    // Create file
    GraphParams params = { /* ... */ };
    melvin_m_init_new_file("test.m", &params);
    
    // Map file
    MelvinFile file;
    melvin_m_map("test.m", &file);
    
    // Initialize runtime
    MelvinRuntime rt;
    runtime_init(&rt, &file);
    
    // Ingest some data
    const char *text = "Hello Melvin!";
    for (size_t i = 0; i < strlen(text); i++) {
        ingest_byte(&rt, 1, text[i], 1.0f);
    }
    
    // Process events
    melvin_process_n_events(&rt, 100);
    
    // Sync and close
    melvin_m_sync(&file);
    runtime_cleanup(&rt);
    close_file(&file);
    
    return 0;
}
```

### **Example 2: Ingest C File**

```c
// Ingest a C file and learn patterns
FILE *f = fopen("source.c", "rb");
uint8_t byte;
while (fread(&byte, 1, 1, f) == 1) {
    ingest_byte(&rt, 10, byte, 1.0f);
    melvin_process_n_events(&rt, 50);
}
fclose(f);
```

### **Example 3: Write and Execute Machine Code**

```c
// Simple function that prints node ID
void test_func(MelvinFile *g, uint64_t node_id) {
    printf("Node %llu executed!\n", (unsigned long long)node_id);
}

// Write code into blob
void *func_ptr = (void*)test_func;
uint64_t offset = melvin_write_machine_code(&file, (uint8_t*)func_ptr, 256);

// Create EXECUTABLE node
uint64_t exec_node = melvin_create_executable_node(&file, offset, 256);

// Activate node (will execute when threshold crossed)
MelvinEvent ev = {
    .type = EV_NODE_DELTA,
    .node_id = exec_node,
    .value = 1.0f
};
melvin_event_enqueue(&rt.evq, &ev);
melvin_process_n_events(&rt, 10);
```

---

## Future Directions

### **Short Term**

- [ ] C code compilation to machine code (learned patterns → executable code)
- [ ] GPU execution support
- [ ] Network substrate (distributed graph)
- [ ] More sophisticated learning rules
- [ ] Visualization tools

### **Long Term**

- [ ] Self-compiling system (C → patterns → machine code → execution)
- [ ] Multi-file ingestion and cross-file pattern learning
- [ ] Automatic EXECUTABLE node creation from high-energy patterns
- [ ] Reinforcement learning integration
- [ ] Real-world sensor/actuator interfaces

### **Research Questions**

- Can the graph learn to compile C code by itself?
- Can it learn to optimize its own code?
- Can it discover new algorithms through energy patterns?
- How does the graph structure relate to computational capability?

---

## Philosophy

Melvin is built on the principle that **computation emerges from physics**. There are no interpreters, no compilers, no special cases—just energy flowing through a graph structure. The graph builds itself, writes its own code, and executes it. Everything is driven by physics.

**Energy, nodes, and edges build more.**

---

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## References

- Unbounded Substrate Physics
- Event-Driven Neural Networks
- Self-Modifying Code Systems
- Prediction Error Learning
- Energy-Based Learning

---

**Built with energy, nodes, and edges.**


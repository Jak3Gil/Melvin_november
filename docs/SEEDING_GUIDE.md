# Seeding Guide: Building a Foundation for Melvin

## How Nodes Store Data

### Node Structure
```c
typedef struct {
    float    a;                    // Activation (energy)
    uint8_t  byte;                 // THE DATA: byte value (0-255)
    // ... edges, propensities ...
    uint64_t pattern_data_offset;  // If > 0: points to PatternData (pattern node)
    uint64_t payload_offset;       // If > 0: points to machine code (EXEC node)
} Node;
```

### Key Insight: **Nodes Store Bytes, Not Concepts**

- **Regular nodes**: Store a single byte value (0-255)
- **Words**: Sequences of byte nodes (one node per character)
  - "hello" = 5 nodes: 'h' (104), 'e' (101), 'l' (108), 'l' (108), 'o' (111)
- **Pattern nodes**: Point to PatternData in blob (sequence definition with blanks)
- **EXEC nodes**: Point to machine code in blob (compiled code)

### What Gets Stored Where?

| Data Type | Storage Location | Example |
|-----------|-----------------|---------|
| **Byte value** | `node.byte` | 'h' = 104 |
| **Word sequence** | Multiple nodes + edges | "hello" = 5 nodes connected |
| **Pattern definition** | Blob (PatternData) | Pattern: [blank] + [blank] = [blank] |
| **Machine code** | Blob (code bytes) | Compiled ARM64 instructions |
| **Concept/meaning** | Graph structure (edges) | Connections between nodes |

---

## Seeding Strategy: 1 Million Node Foundation

### Phase 1: Seed Core Data Nodes (256 nodes)
```c
// Every possible byte value gets a node
for (uint8_t b = 0; b < 255; b++) {
    ensure_node(g, b);  // Node ID = byte value
    g->nodes[b].byte = b;  // Store the byte
}
```

### Phase 2: Seed Word Patterns (Sequences)
```c
// Seed common words as sequences
const char *words[] = {"hello", "world", "code", "graph", ...};

for each word {
    uint32_t prev_node = 0;
    for each char in word {
        uint8_t byte = (uint8_t)char;
        uint32_t node_id = byte;  // Node ID = byte value
        
        ensure_node(g, node_id);
        g->nodes[node_id].byte = byte;
        
        // Create sequential edge
        if (prev_node != 0) {
            create_edge(g, prev_node, node_id, 0.5f);
        }
        prev_node = node_id;
    }
}
```

### Phase 3: Seed Concept Nodes (Higher-level)
```c
// Create concept nodes (node IDs 256+)
// These represent concepts, not raw bytes
uint32_t NODE_HELLO_CONCEPT = 256;
uint32_t NODE_WORLD_CONCEPT = 257;
// ...

// Connect concept nodes to their byte sequences
create_edge(g, NODE_HELLO_CONCEPT, 'h', 0.8f);
create_edge(g, NODE_HELLO_CONCEPT, 'e', 0.8f);
// ... connect to all letters in "hello"
```

### Phase 4: Seed Pattern Nodes
```c
// Create patterns from repeated sequences
// Example: "hello world" appears twice → create pattern

PatternData pattern = {
    .magic = PATTERN_MAGIC,
    .element_count = 11,  // "hello world"
    .elements = {
        {'h', 0}, {'e', 0}, {'l', 0}, {'l', 0}, {'o', 0},
        {' ', 0}, {'w', 0}, {'o', 0}, {'r', 0}, {'l', 0}, {'d', 0}
    }
};

// Store in blob, create pattern node pointing to it
uint32_t pattern_node_id = 1000;
g->nodes[pattern_node_id].pattern_data_offset = blob_offset;
```

### Phase 5: Seed EXEC Nodes (Machine Code)
```c
// Compile useful functions to machine code
const char *code = "void useful_function() { ... }";
uint64_t blob_offset, code_size;
sys_compile_c(code, strlen(code), &blob_offset, &code_size);

// Create EXEC node
uint32_t exec_node_id = 2000;
melvin_create_exec_node(g, exec_node_id, blob_offset, 1.0f);
```

---

## Seeding 1 Million Nodes: Practical Approach

### Option 1: Seed from Corpus (Recommended)
```c
// Load large text corpus (Wikipedia, code, etc.)
FILE *corpus = fopen("corpus.txt", "r");
uint64_t nodes_created = 0;
uint32_t prev_node = 0;

while (!feof(corpus) && nodes_created < 1000000) {
    int c = fgetc(corpus);
    if (c == EOF) break;
    
    uint8_t byte = (uint8_t)c;
    uint32_t node_id = byte;  // Reuse byte nodes
    
    ensure_node(g, node_id);
    g->nodes[node_id].byte = byte;
    
    // Create sequential edge
    if (prev_node != 0) {
        find_or_create_edge(g, prev_node, node_id, 0.3f);
    }
    
    prev_node = node_id;
    nodes_created++;
}
```

### Option 2: Seed from Structured Data
```c
// Seed from knowledge base (concepts, relationships)
struct Concept {
    uint32_t node_id;
    const char *name;
    uint32_t *related_concepts;
};

Concept concepts[1000000] = {
    {256, "hello", {257, 258}},  // "hello" relates to "world", "code"
    {257, "world", {256, 259}},
    // ...
};

for each concept {
    // Create concept node
    uint32_t node_id = concept.node_id;
    ensure_node(g, node_id);
    
    // Connect to byte sequence of name
    for each char in concept.name {
        uint8_t byte = (uint8_t)char;
        create_edge(g, node_id, byte, 0.6f);
    }
    
    // Connect to related concepts
    for each related in concept.related_concepts {
        create_edge(g, node_id, related, 0.4f);
    }
}
```

### Option 3: Seed from Code Patterns
```c
// Seed common code patterns as EXEC nodes
const char *common_functions[] = {
    "int add(int a, int b) { return a + b; }",
    "void print(const char *s) { printf(\"%s\", s); }",
    // ... 1000s of useful functions
};

for each function {
    // Compile to machine code
    uint64_t blob_offset, code_size;
    sys_compile_c(function, strlen(function), &blob_offset, &code_size);
    
    // Create EXEC node
    uint32_t exec_node = next_exec_node_id++;
    melvin_create_exec_node(g, exec_node, blob_offset, 1.0f);
    
    // Connect to concept nodes
    create_edge(g, NODE_ADD_CONCEPT, exec_node, 0.7f);
}
```

---

## What Should You Seed?

### Essential Foundation (Start Here)
1. **256 byte nodes** (0-255) - Every possible byte value
2. **Common words** (1000-10000 words) - Frequent sequences
3. **Code patterns** (100-1000 patterns) - Common code structures
4. **EXEC functions** (10-100 functions) - Useful compiled code

### Advanced Foundation (Scale Up)
1. **1M word sequences** - Large vocabulary
2. **100K code patterns** - Programming knowledge
3. **10K EXEC nodes** - Compiled function library
4. **Concept graph** - Semantic relationships

---

## Example: Seeding "Hello World"

```c
// Step 1: Ensure byte nodes exist
for (uint8_t b = 0; b < 255; b++) {
    ensure_node(g, b);
    g->nodes[b].byte = b;
}

// Step 2: Create sequence "hello"
uint8_t hello[] = {'h', 'e', 'l', 'l', 'o'};
uint32_t prev = 0;
for (int i = 0; i < 5; i++) {
    uint32_t node_id = hello[i];
    if (prev != 0) {
        create_edge(g, prev, node_id, 0.5f);
    }
    prev = node_id;
}

// Step 3: Create concept node for "hello"
uint32_t HELLO_CONCEPT = 256;
ensure_node(g, HELLO_CONCEPT);
for (int i = 0; i < 5; i++) {
    create_edge(g, HELLO_CONCEPT, hello[i], 0.8f);
}

// Step 4: If "hello" appears twice, create pattern
// (Pattern system will do this automatically)
```

---

## Key Points

1. **Nodes store bytes, not words**: "hello" = 5 nodes
2. **Words are sequences**: Connected byte nodes
3. **Concepts are connections**: Edges between nodes
4. **Patterns are abstractions**: Stored in blob, pointed to by nodes
5. **EXEC nodes are code**: Machine code in blob, pointed to by nodes

### To Seed 1M Nodes:
- Feed 1M bytes through `melvin_feed_byte()` - creates nodes automatically
- Or use `ensure_node()` + `create_edge()` for structured seeding
- Patterns form automatically when sequences repeat
- EXEC nodes created from compiled code

The system will learn patterns and relationships from the seeded foundation!


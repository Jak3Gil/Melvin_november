# How Text Files Become CPU/GPU Operations

## The Complete Flow

Text seed files don't directly become operations. Instead, they create **structure in the graph** that enables the system to **learn when and how to execute operations**. Here's the complete pipeline:

## Step 1: Text → Bytes → Nodes

When you seed a text file like `corpus/math/arithmetic.txt`:

```
ADD → NUMBER → NUMBER → RESULT
```

Each character becomes a byte, fed through `melvin_feed_byte()`:

```c
// From melvin_feed_byte() in melvin.c
void melvin_feed_byte(Graph *g, uint32_t port_node_id, uint8_t b, float energy) {
    // 1. Find/create node for this byte value
    uint32_t data_id = (uint32_t)b;  // Node ID = byte value
    ensure_node(g, data_id);
    
    // 2. Inject energy (activation)
    g->nodes[port_node_id].a += energy;
    g->nodes[data_id].a += energy;
    
    // 3. Create edges automatically
    create_edge(g, port_node_id, data_id, initial_weight);
    
    // 4. Pattern discovery happens automatically
    pattern_law_apply(g, data_id);
}
```

**Result**: The text "ADD" creates:
- Node 65 ('A'), Node 68 ('D'), Node 68 ('D')
- Edges: port → 'A' → 'D' → 'D'
- Sequential pattern: A-D-D

## Step 2: Patterns Form from Sequences

As sequences repeat, the system discovers patterns:

```c
// From pattern_law_apply() in melvin.c
// When "ADD → NUMBER → NUMBER → RESULT" appears multiple times:
// 1. Pattern is discovered
// 2. Pattern node is created
// 3. Pattern node connects to the sequence nodes
```

**Result**: Pattern nodes like "ADD_OPERATION" form, connected to:
- The byte sequence "ADD"
- Related concepts (NUMBER, RESULT)
- Other patterns (SUBTRACT, MULTIPLY, etc.)

## Step 3: Patterns Trigger Tool Calls

When patterns activate, they can trigger tool gateway nodes (300-699):

```c
// From melvin_tool_layer.c
// Pattern activates → Tool gateway node activates → Tool executes

// Example: Math pattern activates → LLM tool called
if (gateway_base == 500 && syscalls->sys_llm_generate) {
    // Pattern "SOLVE EQUATION" activates node 500
    // Tool generates response
    syscalls->sys_llm_generate(prompt, prompt_len, &response, &response_len);
    
    // Response fed back into graph as bytes
    melvin_feed_byte(g, output_node, response[i], 0.8f);
}
```

**Result**: Text patterns can trigger:
- **LLM tools** (CodeLlama, llama3.2) - generate code/text
- **Vision tools** - process images
- **Audio tools** - speech-to-text, text-to-speech
- **Compilation** - compile C code to machine code

## Step 4: Code Compilation Creates Machine Code

When patterns trigger compilation:

```c
// From host_syscalls.c - sys_compile_c
static int host_compile_c(const uint8_t *c_source, size_t source_len,
                         uint64_t *blob_offset, uint64_t *code_size) {
    // 1. Write C source to temp file
    // 2. Compile with gcc/clang to machine code
    // 3. Read compiled binary
    // 4. Write machine code bytes into blob
    fread(g->blob + offset, 1, bin_size, bin_file);
    
    // 5. Feed machine code bytes into graph as patterns
    melvin_feed_byte(g, target_node, g->blob[offset + i], 0.3f);
    
    *blob_offset = offset;  // Where the code lives in blob
    return 0;
}
```

**Result**: 
- C source code → compiled ARM64 machine code
- Machine code stored in `blob[]` at `blob_offset`
- Machine code bytes also fed into graph as patterns

## Step 5: EXEC Nodes Link to Machine Code

An EXEC node is created that points to the compiled code:

```c
// From melvin.c - melvin_create_exec_node
uint32_t melvin_create_exec_node(Graph *g, uint32_t node_id, 
                                  uint64_t blob_offset, float threshold_ratio) {
    Node *node = &g->nodes[node_id];
    
    // Link node to machine code in blob
    node->payload_offset = blob_offset;  // Points to machine code
    node->exec_threshold_ratio = threshold_ratio;
    
    return node_id;
}
```

**Result**: 
- Node 1000 (example) becomes an EXEC node
- `node->payload_offset` = 256 (where code starts in blob)
- When node activates above threshold → code executes

## Step 6: EXEC Nodes Execute on CPU

When an EXEC node's activation exceeds its threshold:

```c
// From melvin.c - melvin_execute_exec_node
static void melvin_execute_exec_node(Graph *g, uint32_t node_id) {
    Node *node = &g->nodes[node_id];
    
    // 1. Check if activation exceeds threshold
    float threshold = g->avg_activation * node->exec_threshold_ratio;
    if (node->a < threshold) return;  // Not activated enough
    
    // 2. Get function pointer to machine code
    void (*exec_code)(Graph *g, uint32_t) = 
        (void (*)(Graph *g, uint32_t))(g->blob + node->payload_offset);
    
    // 3. EXECUTE THE MACHINE CODE ON CPU
    exec_code(g, node_id);  // ← ACTUAL CPU EXECUTION
}
```

**Result**: 
- CPU jumps to `blob[payload_offset]`
- Executes ARM64 machine code instructions
- Code can call syscalls (GPU, file I/O, etc.)

## Step 7: Executed Code Calls Syscalls

The machine code in the blob can call syscalls:

```c
// Example: Blob code (compiled from C source)
void exec_code(Graph *g, uint32_t self_node_id) {
    MelvinSyscalls *syscalls = g->hdr->syscalls_ptr;
    
    // Call GPU compute
    GPUComputeRequest req = {...};
    syscalls->sys_gpu_compute(&req);  // ← GPU EXECUTION
    
    // Or write file
    syscalls->sys_write_file("/tmp/result.txt", data, len);
    
    // Or compile more code
    syscalls->sys_compile_c(new_source, len, &offset, &size);
}
```

**Result**: 
- CPU executes blob code
- Blob code calls `sys_gpu_compute()` → GPU executes kernel
- Blob code calls `sys_write_file()` → File I/O
- Blob code can compile more code → Creates more EXEC nodes

## The Complete Chain

```
Text File: "ADD → NUMBER → NUMBER → RESULT"
    ↓
Bytes: [65, 68, 68, ...]  (ASCII)
    ↓
Nodes: node[65], node[68], node[68], ...
    ↓
Edges: 65→68→68 (sequential)
    ↓
Pattern: "ADD_OPERATION" pattern node forms
    ↓
Activation: Pattern activates → Tool gateway activates
    ↓
Tool Call: sys_llm_generate("solve: 2+2") → "4"
    ↓
OR: sys_compile_c("int add(int a, int b) { return a+b; }")
    ↓
Machine Code: ARM64 instructions in blob[256...]
    ↓
EXEC Node: node[1000].payload_offset = 256
    ↓
Activation: node[1000].a > threshold
    ↓
CPU Execution: exec_code(g, 1000) → CPU runs ARM64 code
    ↓
Syscall: sys_gpu_compute(&req) → GPU executes kernel
    ↓
Result: GPU computes, writes result, feeds back to graph
```

## Key Insights

1. **Text doesn't directly become code** - it creates structure that enables code generation/execution

2. **Patterns are the bridge** - text patterns connect to:
   - Tool calls (LLM, compile)
   - EXEC nodes (machine code)
   - Other patterns (concepts)

3. **Graph learns associations** - through UEL physics:
   - Which patterns → which tools
   - Which patterns → which EXEC nodes
   - When to execute (activation thresholds)

4. **Self-modification** - executed code can:
   - Compile new code
   - Create new EXEC nodes
   - Modify graph structure
   - Call GPU/CPU operations

5. **No hardcoded logic** - everything emerges from:
   - Energy flow (activation)
   - Edge weights (learned)
   - Pattern formation (automatic)
   - Threshold dynamics (adaptive)

## Example: Math Pattern → GPU Compute

```
1. Seed file: "MULTIPLY → NUMBER → NUMBER → RESULT"
2. Pattern forms: "MULTIPLY_OPERATION" node
3. Pattern activates → LLM tool: "write CUDA kernel for matrix multiply"
4. LLM generates CUDA code
5. sys_compile_c() compiles CUDA → PTX → GPU kernel
6. EXEC node created pointing to kernel launcher code
7. EXEC node activates → CPU executes launcher
8. Launcher calls sys_gpu_compute() → GPU executes kernel
9. Result fed back to graph → reinforces pattern
```

## Summary

**Text files → Structure → Patterns → Tools → Code → Execution**

The text files create the **foundation** (nodes, edges, patterns) that allows the graph to:
- Recognize when to use tools
- Generate/compile code
- Execute operations on CPU/GPU
- Learn from results
- Build more complex behaviors

It's not a direct translation - it's a **learned association** that emerges from the graph dynamics.


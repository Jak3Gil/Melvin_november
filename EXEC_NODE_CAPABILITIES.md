# EXEC Node Capabilities: What Can They Do?

## Current State

### What EXEC Nodes CAN Do:

1. **Execute Pre-Compiled Machine Code**
   - EXEC nodes point to machine code in the blob
   - When activated, they execute that code
   - Code signature: `void exec_code(Graph *g, uint32_t self_node_id)`
   - Code can access the graph structure

2. **Receive Input Values** (NEW)
   - Can receive values from pattern expansion
   - Values stored in blob after code
   - Can execute with those inputs (e.g., EXEC_ADD: input1 + input2)

3. **Return Results** (NEW)
   - Can store results in blob
   - Results converted back to patterns
   - Graph learns result → output mapping

### What EXEC Nodes CANNOT Do (Currently):

1. **Compile Graph Structures Automatically**
   - EXEC nodes don't compile nodes/edges into code
   - They execute pre-compiled code only
   - No automatic graph→code compilation

2. **Generate Code from Patterns**
   - Patterns extract values, but don't generate code
   - No pattern→code compiler

## What EXISTS: sys_compile_c()

There IS a syscall that compiles C code:

```c
int sys_compile_c(const uint8_t *c_source, size_t source_len, uint64_t *blob_offset_out);
```

**What it does:**
- Takes C source code as input
- Compiles it to machine code
- Stores machine code in blob
- Returns blob offset

**What it requires:**
- C source code (as bytes)
- Graph must generate C code first

## The Gap: Graph → Code Compilation

### Current Flow:
1. **Manual**: Write C code → `sys_compile_c()` → Machine code → EXEC node
2. **Pattern**: Pattern extracts values → Pass to EXEC node → Execute

### Missing Flow:
1. **Automatic**: Graph structure → Generate C code → Compile → EXEC node

## What Would Be Needed for Automatic Compilation

### Option 1: Pattern → Code Generation
- Patterns could generate C code from their structure
- Code compiled via `sys_compile_c()`
- Result stored in EXEC node

### Option 2: Graph Structure → Code Generation
- Graph could analyze node/edge structure
- Generate C code that implements that structure
- Compile and execute

### Option 3: Pattern → Direct Compilation
- Patterns could directly compile to machine code
- Skip C intermediate step
- More complex but faster

## Current Capabilities Summary

**EXEC nodes can:**
- ✅ Execute pre-compiled machine code
- ✅ Receive input values from patterns
- ✅ Return results
- ✅ Access graph structure (via Graph *g parameter)

**EXEC nodes cannot:**
- ❌ Automatically compile graph structures
- ❌ Generate code from patterns
- ❌ Compile nodes/edges directly

**System can:**
- ✅ Compile C code via `sys_compile_c()`
- ✅ Store compiled code in blob
- ✅ Create EXEC nodes pointing to compiled code

**System cannot:**
- ❌ Automatically generate C code from graph structures
- ❌ Compile patterns directly to code

## The Answer

**No, EXEC nodes cannot currently compile any node or collection of nodes and edges.**

They can:
- Execute pre-compiled code
- Receive values from patterns
- Access graph structure

But they need:
- Pre-compiled machine code (via `sys_compile_c()` or manual)
- Or a mechanism to generate code from graph structures (not yet implemented)

## What Would Enable This

To allow EXEC nodes to compile graph structures, you'd need:

1. **Graph → Code Generator**: Analyze graph structure, generate C code
2. **Pattern → Code Generator**: Convert patterns to executable code
3. **Automatic Compilation**: Trigger compilation when patterns form

This would be a significant addition, but the infrastructure exists (`sys_compile_c()`), it just needs the code generation layer.


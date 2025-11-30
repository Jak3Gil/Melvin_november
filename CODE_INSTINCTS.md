# Code Instincts Implementation

**Status**: Implemented - Ready for Testing

## Overview

Code instincts implement compilation and execution capabilities as **pure graph structure** (nodes, edges, patterns). The graph learns to use these through energy flow and pattern matching, not hand-coded rules.

## Architecture

### Three Instinct Clusters

#### (A) Port/Layout Instincts

**Purpose**: Define which ports see which bytes and where code/data lives in the blob.

**Nodes Created**:
- `PORT:SRC_IN` (100000) - Source code input port
- `PORT:BIN_IN` (100001) - Binary/machine code input port  
- `PORT:CMD_IN` (100002) - Command input port
- `PORT:OUT_LOG` (100003) - Log/output port
- `REGION:SRC` (100010) - Source code region metadata
- `REGION:BIN` (100011) - Binary region metadata
- `REGION:BLOB` (100012) - Entire blob region
- `BLOCK:ENTRY` (100020) - Code block entry point
- `BLOCK:SIZE` (100021) - Code block size
- `BLOCK:SYMBOL` (100022) - Symbol name

**Edges**: Ports → Regions, Regions → Block metadata

**Graph Structure**: Static nodes with payload labels (just bytes, no special meaning).

---

#### (B) Source↔Binary Mapping Instincts

**Purpose**: Encode "this C snippet corresponds to this machine code" at the byte level.

**Nodes Created**:
- `PATTERN:SRC_BIN` (100050) - Pattern node linking source to binary

**Edges**: SRC_IN → PATTERN → BIN_IN, Regions → PATTERN

**Mechanism**: 
- Creates pattern scaffolding
- Actual pattern slots (SRC:[...], BIN:[...]) are created by pattern induction system when it sees source↔binary pairs
- Graph can strengthen/weaken/clone these patterns

---

#### (C) EXEC Instincts (Compile, Link, Run)

**Purpose**: Provide EXECUTABLE nodes that can call compilers and run code.

**Nodes Created**:
- `EXEC:COMPILE` (100030) - Compiles source → binary (EXECUTABLE)
- `EXEC:LINK` (100031) - Links/loads binary (EXECUTABLE)
- `EXEC:RUN` (100032) - Runs compiled code block (EXECUTABLE)
- `CMD:COMPILE` (100040) - Command to trigger compilation
- `CMD:RUN` (100041) - Command to trigger execution

**Edges**: 
- CMD_COMPILE → EXEC_COMPILE
- CMD_RUN → EXEC_RUN
- SRC_IN → EXEC_COMPILE
- EXEC_COMPILE → BIN_IN, OUT_LOG
- EXEC_LINK → BLOCK_ENTRY
- EXEC_RUN → OUT_LOG

**EXEC Functions** (in `code_exec_helpers.c`):
- `melvin_exec_compile()`: Reads from SRC_IN, calls clang/gcc, writes to BIN_IN
- `melvin_exec_link()`: Links/loads binary into executable memory
- `melvin_exec_run()`: Runs compiled code block

**Key Constraint**: EXEC nodes are graph nodes. When they fire (activation > threshold), they call C functions. But the routing (which EXEC fires, when) is determined by graph energy flow, not external logic.

---

## How Instincts "Teach" the Graph

Instincts are **examples + structures**, not teaching logic. Teaching happens because:

1. **Graph can learn** (proven by Test 1-3: A→B, multi-hop, meta-learning)
2. **Instincts provide high-utility routes** for:
   - Mapping SRC↔BIN
   - Calling EXEC nodes
   - Getting reward when code runs correctly

3. **Episodes carve patterns**:
   - SRC_IN active + CMD_IN="compile" + EXEC_COMPILE fires + BIN_IN gets filled + reward
   - Graph learns: "When SRC and CMD look like this, sending energy through EXEC_COMPILE leads to lower error / more reward"

4. **Later reuse**: With new code, graph can reuse instinct circuits without rediscovering the entire toolchain.

---

## Files

- `code_instincts.c` - Instinct injection (nodes, edges, patterns)
- `code_exec_helpers.c` - EXEC functions (compile, link, run)
- `test_code_instincts.c` - Unit tests (CI-1 through CI-4)

---

## Test Suite (CI-1 through CI-4)

### CI-1: SRC/BIN Port Correctness

**Goal**: Verify port nodes exist and are wired correctly.

**Test**:
1. Create brain with code instincts
2. Verify SRC_IN, BIN_IN, REGION_SRC, REGION_BIN nodes exist
3. Verify EXEC_COMPILE node exists
4. Verify edges: SRC_IN → REGION_SRC, BIN_IN → REGION_BIN

**Success**: All nodes exist, edges are wired.

---

### CI-2: Run-Block Correctness

**Goal**: Verify run-block structure exists.

**Test**:
1. Verify EXEC_RUN node exists
2. Verify BLOCK_ENTRY, BLOCK_SIZE nodes exist
3. Verify EXEC_RUN is wired to BLOCK_ENTRY

**Success**: Run-block structure exists.

**Note**: Full execution test requires compiled code block (future work).

---

### CI-3: Source↔Binary Pattern Matching

**Goal**: Verify pattern scaffolding exists.

**Test**:
1. Verify PATTERN_SRC_BIN node exists
2. Verify pattern is wired to SRC_IN and BIN_IN

**Success**: Pattern scaffolding exists.

**Note**: Full pattern matching requires pattern induction system (future work).

---

### CI-4: "Use Instinct" vs "Ignore Instinct" Learning

**Goal**: Verify instinct wiring exists for learning.

**Test**:
1. Verify CMD_COMPILE → EXEC_COMPILE edge exists
2. Verify CMD_RUN → EXEC_RUN edge exists

**Success**: Instinct wiring exists.

**Note**: Full learning test requires reward-based training (future work).

---

## Running Tests

```bash
gcc -std=c11 -o test_code_instincts test_code_instincts.c -lm
./test_code_instincts
```

---

## Integration with instincts.c

To add code instincts to the main instinct injection:

```c
// In instincts.c, melvin_inject_instincts():
void melvin_inject_instincts(MelvinFile *file) {
    // ... existing injections ...
    melvin_inject_code_instincts(file);  // Add this
}
```

---

## Next Steps

1. **Bind EXEC functions**: Wire EXEC nodes to actual C functions (melvin_exec_compile, etc.)
2. **Full CI-2 test**: Compile actual code block and verify it runs
3. **Full CI-3 test**: Create source↔binary pattern and verify pattern matching
4. **Full CI-4 test**: Train graph with reward, verify it learns to use instincts
5. **Jetson integration**: Wire to real ports, real compilers, real robot

---

## Design Principles

1. **Pure Graph**: All instincts are nodes, edges, patterns - no external logic
2. **Learnable**: Graph can strengthen/weaken/ignore instincts based on experience
3. **Real Tools**: EXEC functions call real compilers (clang/gcc), not simulations
4. **Contract-Compliant**: Only bytes in/out, no direct graph manipulation from outside


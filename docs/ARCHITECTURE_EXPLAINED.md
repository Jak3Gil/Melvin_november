# Melvin Architecture - What Each File Does

## The Core Concept

**You're absolutely right** - conceptually, we should just be "asking melvin.m questions":
- Inject input → melvin.m processes → Read output

But we need C code to actually interact with the `.m` file. Here's what each file does:

## File Breakdown

### 1. `melvin.m` - The Brain State (The Actual Data)
- **What it is**: A binary file containing the persistent graph state
- **Contains**: Nodes, edges, blob (machine code), graph header
- **Analogy**: Like a database file - it's just data on disk
- **Size**: ~1.3MB (the brain state)
- **We don't "read" this directly** - we use APIs to interact with it

### 2. `melvin.c` - The Runtime/Physics Engine (~5600 lines)
- **What it is**: The C code that provides the "operating system" for melvin.m
- **What it does**:
  - Maps `melvin.m` file into memory (`melvin_m_map`)
  - Implements graph physics (energy propagation, learning rules)
  - Provides event loop (`melvin_tick_once`, `melvin_process_n_events`)
  - Executes EXEC nodes (`execute_hot_nodes`)
  - Provides APIs to read/write nodes/edges (`find_node_index_by_id`, etc.)
- **Why tests include it**: Tests need these APIs to:
  - Create/open melvin.m files
  - Inject inputs (write to nodes)
  - Tick the graph (run physics)
  - Read outputs (read from nodes)

### 3. `instincts.c` - Initial Patterns (~1100 lines)
- **What it is**: Code that injects initial graph patterns into a fresh melvin.m
- **What it does**: Creates nodes/edges for:
  - Math operations (MATH:IN_A:I32, EXEC:ADD32, etc.)
  - Channels, body patterns, multi-hop patterns
- **Why tests include it**: To set up the initial graph structure

### 4. `melvin_exec_helpers.c` - EXEC Tool Functions (~200 lines)
- **What it is**: The actual computation functions that EXEC nodes call
- **Contains**: `melvin_exec_add32()`, `melvin_exec_mul32()`, `melvin_exec_select_add_or_mul()`
- **These ARE part of melvin's brain** - they're the tools melvin.m uses
- **Why tests include it**: So the EXEC nodes have code to execute

### 5. `test_1_1_tool_selection.c` - The Test (~500 lines)
- **What it does**:
  1. Creates a `melvin.m` file
  2. Injects instincts (sets up initial patterns)
  3. For each test case:
     - Writes inputs to nodes (opcode, a, b)
     - Ticks the graph (runs physics/EXEC)
     - Reads outputs from nodes
     - Compares to ground truth
- **What it does NOT do**:
  - Does NOT compute the math (that's melvin.m's job)
  - Does NOT select tools (that's melvin.m's job)
  - Does NOT call EXEC functions directly (only via graph event loop)

## The Flow

```
Test Code (C)                    melvin.m (Brain State)
─────────────────                ────────────────────────
1. Create melvin.m file  ──────> Empty file created
2. Inject instincts      ──────> Nodes/edges added
3. Write inputs          ──────> Node states updated
4. Tick graph           ──────> Physics runs, EXEC fires
5. Read outputs          <────── Node states read back
```

## Why So Much C Code?

The test file includes `melvin.c` because:
- **We need the runtime** to interact with melvin.m
- **We need the physics** to actually run the graph
- **We need the APIs** to read/write nodes

But conceptually, you're right - we're just:
1. Asking melvin.m: "What is 1 + 2?"
2. Melvin.m processes (via graph + EXEC)
3. Reading the answer back

## The Key Insight

**melvin.m is the program** - it's the persistent brain state that does the work.

**melvin.c is the runtime** - it's the "operating system" that lets us interact with melvin.m.

**The test is the environment** - it just provides inputs and checks outputs.

## What We're Actually Testing

We're testing that **melvin.m can**:
- Store inputs in nodes
- Route signals through the graph
- Select which EXEC tool to use
- Execute the tool
- Store results in nodes

All of this happens **inside melvin.m** - the C code is just the interface.


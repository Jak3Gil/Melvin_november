# What The Test Actually Does (Simplified)

## The Core Test Logic (What Matters)

Here's what the test actually does, stripped of all the boilerplate:

```c
// 1. Create melvin.m brain file
melvin_m_init_new_file("test_1_1.m", &params);
melvin_m_map("test_1_1.m", &file);

// 2. Set up initial patterns (one-time setup)
melvin_inject_instincts(&file);  // Creates MATH nodes, EXEC nodes, etc.

// 3. For each test case - THIS IS THE ACTUAL TEST:
for (each test case) {
    // A. Write inputs to melvin.m
    write_to_node(&file, "MATH:IN_A:I32", a);      // "Here's the first number"
    write_to_node(&file, "MATH:IN_B:I32", b);      // "Here's the second number"
    write_to_node(&file, "TOOL:OPCODE:I32", op);   // "0 = add, 1 = multiply"
    
    // B. Ask melvin.m to process
    melvin_tick_once(&file);  // "Go think about this"
    melvin_tick_once(&file);  // (run physics, execute tools)
    melvin_tick_once(&file);
    
    // C. Read the answer from melvin.m
    int32_t result = read_from_node(&file, "MATH:OUT:I32");
    
    // D. Check if melvin.m got it right
    assert(result == expected);
}
```

## That's It!

The test is just:
1. **Write inputs** → melvin.m
2. **Tick the graph** → melvin.m processes
3. **Read outputs** → melvin.m's answer
4. **Check correctness** → did melvin.m get it right?

## Why All The C Code?

The `#include "melvin.c"` is needed because:

- `write_to_node()` is defined in melvin.c
- `read_from_node()` is defined in melvin.c  
- `melvin_tick_once()` is defined in melvin.c
- `melvin_m_map()` is defined in melvin.c

**We're not running melvin.c's logic** - we're using melvin.c's **APIs** to talk to melvin.m.

## The Analogy

Think of it like a database:

- **melvin.m** = The database file (the actual data)
- **melvin.c** = The database library (provides `db_query()`, `db_write()`, etc.)
- **The test** = A script that uses the library to query the database

We're not running the database engine's code - we're using its APIs to interact with the data.

## What's Actually In melvin.m?

The `.m` file contains:
- **Nodes**: Data structures with state, flags, payloads
- **Edges**: Connections between nodes with weights
- **Blob**: Machine code for EXEC nodes
- **Header**: Metadata about the graph

When we "tick" the graph, melvin.c reads this data, runs the physics, updates it, and writes it back.

## The Key Point

**melvin.m is the program** - it's the persistent brain that does computation.

**melvin.c is the runtime** - it's the interface we use to interact with melvin.m.

**The test is the question** - it just asks melvin.m questions and checks the answers.


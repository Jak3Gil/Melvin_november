# General Mechanism Proposal: Let Graph Learn, Don't Hardcode

## The Problem

**We want the graph to learn number parsing, but we don't want to:**
- Add special cases that restrict emergence
- Miss other important patterns
- Make the graph dependent on us

## Current Pattern System (What Graph Can Do Alone)

### What Patterns Can Discover:
1. **Repeated Sequences**: Graph automatically finds patterns when sequences repeat
2. **Pattern Generalization**: Patterns with blanks can match variations
3. **Pattern Expansion**: When patterns activate, they expand to underlying nodes
4. **Nested Patterns**: Patterns can reference other patterns
5. **Edge Strengthening**: Frequently used connections get stronger

### What Patterns CANNOT Do (Current Limitation):
- Extract values from sequences (bytes → integers)
- Pass values to EXEC nodes
- Get results from EXEC nodes

## Proposed Solution: General Value Extraction Mechanism

### Instead of Special-Casing Numbers:
**Provide a general mechanism that lets the graph learn value extraction for ANY pattern type.**

### The Mechanism:

```c
/* General pattern value extraction - not number-specific */
typedef struct {
    uint32_t pattern_node_id;  /* Pattern that extracts this value */
    uint32_t value_type;       /* Graph learns: 0=number, 1=string, 2=concept, etc. */
    uint64_t value_data;       /* The actual value (interpreted by type) */
} PatternValue;
```

### How It Works:

1. **Pattern Expansion Enhancement**: When a pattern expands, it can extract a "value"
   - Not hardcoded for numbers
   - Works for any pattern that matches a sequence
   - Graph learns which patterns extract which values

2. **Value Storage**: Extracted values can be stored in pattern nodes
   - Pattern node can have a "value" field
   - Graph learns to associate patterns with values

3. **Value Routing**: Values can be passed to EXEC nodes
   - General mechanism (not number-specific)
   - Graph learns which values go to which EXEC nodes

### What Graph Learns (Through Examples):

**We feed examples, graph learns mappings:**

```
Example 1: "100" → integer 100
Example 2: "200" → integer 200
Example 3: "hello" → string "hello"
Example 4: "red" → color concept
```

**Graph discovers:**
- Which patterns extract which values
- Which values route to which EXEC nodes
- When to trigger computation

### What We Provide (General Mechanisms):

1. **Pattern Value Extraction**: Patterns can extract values (general, not number-specific)
2. **EXEC Node I/O**: EXEC nodes can receive values and return results (general)
3. **Pattern→EXEC Bridge**: Patterns can trigger EXEC nodes with values (general)

### What Graph Provides (Learned Behavior):

1. **Specific Mappings**: "100" → 100 (learned from examples)
2. **Routing Decisions**: Which values go to which EXEC nodes (learned from patterns)
3. **Novel Discoveries**: New value extractions we didn't anticipate (emergent)

## Example: How Graph Would Learn Number Parsing

### Step 1: Feed Examples
```
"100" appears in context where integer 100 is used
"200" appears in context where integer 200 is used
"100+200=300" shows computation
```

### Step 2: Graph Discovers Patterns
- Pattern: [digit][digit][digit] → extracts to value
- Pattern: [value] + [value] = [value] → computation pattern

### Step 3: Graph Learns Mappings
- "100" pattern → extracts integer 100
- "200" pattern → extracts integer 200
- Addition pattern → routes to EXEC_ADD

### Step 4: Graph Generalizes
- Any [digit]+ pattern → extracts integer
- Any addition pattern → routes to EXEC_ADD
- Discovers new patterns we didn't teach

## Benefits of General Mechanism:

1. **Emergent**: Graph can discover novel value extractions
2. **General**: Works for numbers, strings, concepts, etc.
3. **Autonomous**: Graph learns specifics, we provide general mechanism
4. **Extensible**: Graph can learn new value types we didn't anticipate

## What We DON'T Hardcode:

❌ Number parsing logic
❌ String parsing logic  
❌ Type conversion rules
❌ Routing decisions

## What We DO Provide:

✅ General value extraction mechanism
✅ General EXEC I/O mechanism
✅ General pattern→EXEC bridge

## The Key Insight:

**We're not teaching the graph WHAT to learn (numbers), we're providing HOW to learn (general mechanisms).**

The graph:
- Learns number parsing through examples
- Learns string parsing through examples
- Learns concept mapping through examples
- Discovers novel patterns we didn't anticipate

We provide:
- Tools for learning (general mechanisms)
- Graph uses tools to learn specifics

## Balance: Scaffolding vs. Emergence

**Too much scaffolding**: Graph becomes dependent, can't discover
**Too little scaffolding**: Graph can't do anything, fails

**Sweet spot**: General mechanisms + examples → graph learns specifics

## Conclusion

**Don't add special cases. Add general mechanisms that let the graph learn autonomously.**

The graph should discover:
- Number parsing (from examples)
- String parsing (from examples)
- Concept mapping (from examples)
- Novel patterns (emergent)

We should provide:
- General value extraction
- General EXEC I/O
- General pattern→EXEC bridge

The graph fills in:
- Specific mappings
- Routing decisions
- Novel discoveries


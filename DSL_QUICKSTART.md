# Graph DSL Quick Start

## What is the DSL?

The Graph DSL (Domain-Specific Language) lets you **program the graph as code/data** instead of hardcoding intelligence in C. This keeps C as pure hardware - all smarts live in the graph.

## Basic Usage

```bash
# Build the DSL terminal
make dsl

# Run it
./melvin_dsl
```

## Example Session

```graph
> pattern "ab" { atom 0: byte 'a'; atom 1: byte 'b'; }
Created pattern "ab" (id: 9223372036854775808, 2 atoms)

> feed "ababab"
Fed 6 bytes as DATA nodes

> show patterns
=== Patterns ===
Pattern id=9223372036854775808, q=0.5000

> learn episodes=10
Running 10 learning episodes with 1 patterns...
  Episode 0: error=0.0000
  ...
  Episode 9: error=0.0000

> show patterns
=== Patterns ===
Pattern id=9223372036854775808, q=0.9463

> show bindings
=== Pattern bindings ===
PATTERN id=9223372036854775808 q=0.946
  -> DATA id=0 byte=a (0x61) w=1.000
  -> DATA id=1 byte=b (0x62) w=1.000
  -> DATA id=2 byte=a (0x61) w=1.000
  ...

> quit
```

## Commands

### Define Patterns
```graph
pattern "name" { atom 0: byte 'x'; atom 1: byte 'y'; }
```
- Creates a pattern with explicit atoms
- Each atom specifies: `atom DELTA: byte 'VALUE'`
- Pattern is added to graph immediately

### Feed Data
```graph
feed "your data string"
```
- Adds data as DATA nodes
- Creates sequence edges automatically
- Graph is ready for pattern matching

### Run Learning
```graph
learn [episodes=N] [threshold=X]
```
- Runs self-consistency episodes
- Updates pattern qualities based on reconstruction error
- Default: 10 episodes, threshold 0.8

### Show Graph State
```graph
show patterns    # List all patterns and their qualities
show stats       # Graph statistics (nodes, edges, etc.)
show bindings    # Pattern-to-data bindings
```

## Workflow

1. **Define patterns** (explicitly or via induction - coming soon)
2. **Feed data** to the graph
3. **Run learning** to update pattern qualities
4. **Inspect results** with `show` commands
5. **Iterate** - modify patterns, feed new data, learn again

## Benefits

✅ **No C recompilation** - change graph logic without rebuilding  
✅ **Graph as data** - store graph programs as files  
✅ **Intelligence in graph** - all smarts encoded as patterns  
✅ **Rapid experimentation** - test different pattern sets quickly  

## Next Steps

The DSL will be extended to support:
- Pattern induction from data (`pattern from_data { scan: bigrams }`)
- Learning rule definitions
- Output generation rules
- Graph program files (`.gdsl` files)

This moves ALL intelligence from C to the graph, keeping C as pure hardware.


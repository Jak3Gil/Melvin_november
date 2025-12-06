# .m File Format: Self-Contained Brain

## Yes, Everything is in the .m File!

The `.m` file is a **self-contained binary format** that stores the entire graph state.

---

## File Layout

```
┌─────────────────────────────────────────────────────────┐
│ Header (4096 bytes)                                      │
│ - Magic: "MLVN"                                          │
│ - Version, flags                                         │
│ - Offsets to all regions (64-bit, TB-scale ready)       │
│ - Counts: node_count, edge_count, blob_size, etc.        │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Nodes Array (hot region)                                 │
│ - All Node structures (64 bytes each)                    │
│ - Activation, edges, propensities, pattern data, etc.    │
│ - Grows as graph learns                                  │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Edges Array (hot region)                                 │
│ - All Edge structures (20 bytes each)                   │
│ - Source, destination, weight, linked lists               │
│ - Grows as graph learns                                  │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Blob (hot region)                                         │
│ - Machine code (EXEC nodes)                               │
│ - Pattern data structures                                 │
│ - Pattern instances                                       │
│ - Learned values                                          │
│ - Self-modifiable code                                    │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Cold Data (optional, read-only)                          │
│ - Corpus data                                             │
│ - Training examples                                       │
│ - Reference material                                      │
│ - Graph can copy from cold→blob (self-directed learning) │
└─────────────────────────────────────────────────────────┘
```

---

## What's Stored

### ✅ In the .m File:

1. **All Nodes** (hot region)
   - Activation values
   - Edge pointers (first_in, first_out)
   - Propensities (input/output/memory)
   - Pattern data offsets
   - EXEC node payload offsets
   - Everything needed to reconstruct the graph

2. **All Edges** (hot region)
   - Source/destination nodes
   - Weights
   - Linked list pointers (next_in, next_out)

3. **Blob** (hot region)
   - Machine code (EXEC nodes)
   - Pattern data structures
   - Pattern instances
   - Learned values
   - Self-modifiable code

4. **Cold Data** (optional)
   - Read-only corpus
   - Training examples
   - Reference material

5. **Header
   - Magic number ("MLVN")
   - Version
   - All offsets and counts
   - Entry points

### ❌ NOT in the .m File:

- Runtime arrays (last_activation, prop_queue, etc.)
  - These are allocated in RAM on boot
  - Not persisted (recalculated from nodes on boot)

- Sequence buffers
  - Temporary tracking buffers
  - Not persisted

---

## Persistence

### What Persists:
✅ **Graph structure** (nodes, edges)  
✅ **Node state** (activation, propensities)  
✅ **Edge weights**  
✅ **Pattern data** (in blob)  
✅ **Machine code** (in blob)  
✅ **Learned values** (in blob)  

### What Doesn't Persist:
❌ **Runtime tracking arrays** (recalculated on boot)  
❌ **Propagation queue** (recreated on boot)  
❌ **Sequence buffers** (temporary)  

---

## File Size Calculation

For a 1TB graph:
- **Nodes**: ~16 billion × 64 bytes = ~1TB
- **Edges**: ~50 billion × 20 bytes = ~1TB
- **Blob**: Variable (patterns, code)
- **Cold data**: Optional (corpus)

**Total: ~1-2TB** (depending on blob/cold data size)

---

## Self-Contained

The `.m` file is **completely self-contained**:
- ✅ No external dependencies
- ✅ No separate index files
- ✅ No database
- ✅ Just one binary file

**You can copy the .m file and it contains everything!**

---

## Memory Mapping

When you `mmap()` the .m file:
- **Nodes array** → `g->nodes` (direct pointer)
- **Edges array** → `g->edges` (direct pointer)
- **Blob** → `g->blob` (direct pointer)
- **Cold data** → `g->cold_data` (direct pointer)

**All data is accessed directly from the file** - no copying into RAM (except pages loaded on-demand).

---

## Conclusion

**Yes, everything is in the .m file!**

The entire graph state is persisted in a single self-contained binary file. When you boot, you just `mmap()` it and access data directly. No loading, no copying - just virtual memory mapping.


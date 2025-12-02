# Edge Capacity Explanation

**Question**: "What do you mean by edge capacity?"

---

## 📊 Edge Capacity = Pre-allocated Space

**"5000 edge capacity"** means:
- The brain file **pre-allocates space** for 5000 edges when first created
- This is **initial space**, not a hard limit
- Edges can **grow dynamically** beyond 5000

---

## 🔄 How It Works

### Initial Creation
When you create a brain with `melvin_open(path, 1000, 5000, 65536)`:
- **1000 nodes**: Pre-allocated space for 1000 nodes
- **5000 edges**: Pre-allocated space for 5000 edges  
- **65536 bytes**: Blob size (64KB for EXEC code)

The file is created with space for 5000 edges, but starts with **0 edges** (empty).

### Dynamic Growth
When you create edges and exceed 5000:

```c
// From create_edge() in melvin.c
if (g->edge_count >= max_edges) {
    /* Need to grow file to accommodate more edges */
    /* Adaptive growth: double edges (or at least 10k, whichever is larger) */
    uint64_t new_edge_count = g->edge_count * 2;
    if (new_edge_count < g->edge_count + 10000) {
        new_edge_count = g->edge_count + 10000;  /* Minimum growth */
    }
    // File is extended automatically
}
```

**The file automatically grows** when you exceed the initial capacity:
- Doubles the edge space, OR
- Adds at least 10,000 more edges (whichever is larger)

---

## 📈 Example Growth

**Initial State**:
- Nodes: 1000 (pre-allocated)
- Edges: 0 (using 0 of 5000 capacity)
- File size: ~244KB

**After Creating 5,000 Edges**:
- Nodes: 1000
- Edges: 5,000 (using all 5000 capacity)
- File size: ~244KB

**After Creating 5,001 Edges**:
- Nodes: 1000
- Edges: 5,001
- File size: **~488KB** (automatically doubled to ~10,000 edge capacity)

**After Creating 10,001 Edges**:
- Nodes: 1000
- Edges: 10,001
- File size: **~976KB** (automatically doubled to ~20,000 edge capacity)

---

## ✅ Why Pre-allocate?

1. **Performance**: Pre-allocating avoids frequent file growth operations
2. **Efficiency**: Fewer `ftruncate()` and `mremap()` calls
3. **Predictability**: Initial file size is known upfront

---

## 🎯 Summary

- **"5000 edge capacity"** = Initial pre-allocated space for 5000 edges
- **Not a limit**: Edges can grow beyond 5000
- **Auto-growth**: File automatically expands when capacity is exceeded
- **Current state**: Brain starts with 0 edges, can grow to millions

**Think of it like a parking lot**: You build space for 5000 cars initially, but you can expand the lot when needed. The 5000 is just the starting size, not a maximum.


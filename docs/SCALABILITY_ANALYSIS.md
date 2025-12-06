# Scalability Analysis: 1TB Graph Boot Time

## Current Bottlenecks

### Problem: Linear Memory Allocation
For a 1TB graph with ~16 billion nodes (assuming 64 bytes/node):
- **6 arrays × 4 bytes × 16B nodes = 384 GB RAM** just for tracking arrays!
- **prop_queue arrays = 16 GB** more
- **Total: ~400 GB RAM** just to boot

This is **not scalable**.

---

## Current Boot Operations (Time Analysis)

### Fast Operations (O(1) or O(log n)):
1. **mmap()** - ~0.01s (lazy mapping, doesn't load all data)
2. **Read header** - ~0.001s (fixed size, ~100 bytes)
3. **Sample nodes/edges** - ~0.1s (we limit to 1000 samples)

### Slow Operations (O(n)):
1. **Array allocation** - **O(n)** where n = node_count
   - `calloc(node_count, sizeof(float))` × 6 arrays
   - For 16B nodes: **~400 GB allocation = minutes to hours**

2. **prop_queue allocation** - **O(n)**
   - `calloc(node_count, sizeof(uint32_t))`
   - For 16B nodes: **~64 GB allocation**

3. **Array initialization loop** - **O(n)** (but we limit to 256)
   - Currently only 256 nodes, so fast

---

## Solution: Truly Lazy Allocation

### Option 1: Sparse Arrays (Hash Tables)
- Only allocate memory for nodes that are actually used
- Use hash table: `node_id → value`
- Boot time: **O(1)** - no allocation on startup
- Runtime: **O(1)** average lookup

### Option 2: Fixed-Size Circular Buffers
- Only track last N nodes (e.g., 1M most recent)
- Boot time: **O(1)** - fixed allocation
- Runtime: **O(1)** lookup for recent nodes

### Option 3: On-Demand Allocation
- Allocate arrays only when first accessed
- Use `realloc()` to grow as needed
- Boot time: **O(1)** - no allocation
- Runtime: **O(1)** after first access

---

## Recommended: Hybrid Approach

1. **Small graphs (< 1M nodes)**: Current approach (fast, simple)
2. **Large graphs (> 1M nodes)**: Sparse hash tables
3. **prop_queue**: Fixed-size circular buffer (1M entries max)

### Implementation:
```c
if (g->node_count < 1000000) {
    // Fast path: allocate arrays
    g->last_activation = calloc(g->node_count, sizeof(float));
} else {
    // Sparse path: hash table
    g->sparse_last_activation = hash_table_create();
}
```

---

## Expected Boot Times

### Current (with optimizations):
- **Small graph (1K nodes)**: ~0.1s ✅
- **Medium graph (1M nodes)**: ~1s ✅
- **Large graph (1B nodes)**: **~1000s (16 minutes)** ❌
- **Huge graph (16B nodes)**: **~hours** ❌

### With Sparse Arrays:
- **Small graph (1K nodes)**: ~0.1s ✅
- **Medium graph (1M nodes)**: ~1s ✅
- **Large graph (1B nodes)**: **~1s** ✅
- **Huge graph (16B nodes)**: **~1s** ✅

---

## Memory Usage Comparison

### Current (Dense Arrays):
- 1K nodes: 24 KB
- 1M nodes: 24 MB
- 1B nodes: 24 GB
- 16B nodes: **384 GB** ❌

### Sparse Arrays (Hash Table):
- 1K nodes: 24 KB
- 1M nodes: 24 MB
- 1B nodes: 24 MB (only active nodes)
- 16B nodes: **24 MB** (only active nodes) ✅

---

## Next Steps

1. Implement sparse array option for large graphs
2. Make prop_queue fixed-size (1M max)
3. Add detection: `if (node_count > threshold) use_sparse()`


# Why Node/Edge Limits Changed to Disk-Based

## ❌ The Problem

**Before:** Hardcoded limits were set at initialization:
- 10 million nodes (fixed)
- 100 million edges (fixed)
- ~3.6 GB total file size

**Problem:** These limits are **arbitrary** and **tiny** compared to available disk space!

For example, on a system with 691 GB available:
- Old limit: 3.6 GB (uses 0.5% of disk!)
- New limit: ~622 GB (uses 90% of disk)

## ✅ The Solution

**Now:** Limits are **calculated from available disk space**:

1. **Check available disk space** (where `melvin.m` will be created)
2. **Reserve 10%** for other uses (safety margin)
3. **Use 90%** for the brain file
4. **Allocate:**
   - 30% to nodes (30% of usable space ÷ 40 bytes = node capacity)
   - 70% to edges (70% of usable space ÷ 32 bytes = edge capacity)

## 📊 Example Calculation

On a system with **691 GB available**:

```
Available disk: 691 GB
Reserve 10%:    69 GB (safety margin)
Usable space:   622 GB (90%)

Allocation:
  Nodes: 30% = 186 GB ÷ 40 bytes = ~5 billion nodes
  Edges: 70% = 436 GB ÷ 32 bytes = ~14 billion edges

Total file: ~622 GB
```

Compare to **old hardcoded limits**:
```
Nodes: 10 million * 40 bytes = 400 MB
Edges: 100 million * 32 bytes = 3.2 GB
Total: ~3.6 GB
```

**New limit is 170x larger!**

## 🎯 Why This Makes Sense

1. **Memory-mapped files** can use disk space directly
   - No need to load everything into RAM
   - OS handles paging automatically
   - Only accessed portions are in RAM

2. **The real limit IS disk space**
   - No reason to artificially limit it
   - Let Melvin use all available space

3. **Dynamic sizing**
   - Works on systems with any disk size
   - Small disk = smaller brain file
   - Large disk = larger brain file

## 📋 How It Works

### Initialization (`init_melvin_jetson.sh`):

```bash
# Get available disk space
AVAILABLE_SPACE=$(df -k . | tail -1 | awk '{print $4}')

# Reserve 10% for safety
USABLE_SPACE=$((AVAILABLE_SPACE * 90 / 100))

# Calculate node/edge capacities
NODE_CAP=$((USABLE_SPACE * 30 / 100 / 40))  # 30% ÷ 40 bytes
EDGE_CAP=$((USABLE_SPACE * 70 / 100 / 32))  # 70% ÷ 32 bytes
```

### Runtime Checks (`melvin.c`):

```c
// alloc_node checks against capacity
if (g->header->num_nodes >= g->header->node_cap) return 0;

// add_edge checks against capacity  
if (g->header->num_edges >= g->header->edge_cap) return;
```

**These checks still work** - they just check against a much larger, disk-based limit now!

## 🚀 Impact

### Before:
- **Maximum:** 10M nodes, 100M edges
- **File size:** ~3.6 GB
- **Wasted space:** 688 GB unused on 691 GB disk

### After:
- **Maximum:** ~5 billion nodes, ~14 billion edges (on 691 GB disk)
- **File size:** ~622 GB (uses 90% of disk)
- **Wasted space:** Only 10% reserved for safety

## 💡 Key Insight

**The node/edge limit is now the disk limit, not an arbitrary number!**

- Small disk (10 GB) → Smaller brain (~9 GB)
- Large disk (1 TB) → Larger brain (~900 GB)
- Infinite disk → Infinite growth potential!

## ⚠️ Note

The file is **pre-allocated** at initialization (sparse file or full allocation depending on OS). This means:

1. **Initialization takes time** (allocating a 622 GB file)
2. **File exists at full size** (even if empty)
3. **Future growth:** Would require file extension (not yet implemented)

For future work, we could implement **dynamic file extension** so the file grows as needed instead of pre-allocating everything.

## ✅ Summary

**Before:** Arbitrary hardcoded limits (10M nodes, 100M edges)  
**After:** Disk-based limits (90% of available space)  
**Result:** Melvin can now use nearly all available disk space!  
**The limit is now the disk, not an arbitrary number.**


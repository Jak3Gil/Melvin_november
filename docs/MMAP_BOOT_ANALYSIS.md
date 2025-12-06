# mmap() Boot Time Analysis: 1TB File

## How mmap() Actually Works

### mmap() is Lazy/Virtual
- **Does NOT read the entire file into RAM**
- **Only creates virtual address space mappings**
- **Pages are loaded on-demand** when first accessed (page fault)

### What Happens During Boot

1. **mmap() call** - ~0.01s
   - Creates virtual address mappings
   - Does NOT read file data
   - Just tells OS "this file maps to this virtual address range"
   - Time is **constant** regardless of file size

2. **Read header** - ~0.001s
   - `read()` or direct memory access to first 4KB
   - OS loads first page (4KB) on first access
   - **Constant time** (4KB read)

3. **Sample nodes/edges** - ~0.1s
   - Access 1000 nodes/edges
   - OS loads ~64KB-128KB of pages on-demand
   - **Constant time** (fixed number of samples)

4. **File system overhead** - Variable
   - **This is the real question!**
   - Does file system metadata scale with file size?

---

## File System Overhead

### ext4/XFS (Linux):
- **Metadata is separate** from data
- File size stored in inode (fixed size)
- **mmap() time: constant** (doesn't depend on file size)
- **First page access: constant** (4KB read)

### NTFS (Windows):
- Similar - metadata separate
- **mmap() time: constant**

### HFS+/APFS (macOS):
- Similar behavior
- **mmap() time: constant**

### Potential Issues:

1. **Fragmentation**:
   - If 1TB file is fragmented, first access might be slower
   - But still only loads pages we access (4KB-128KB)

2. **File system journal**:
   - Journal replay on mount (if filesystem wasn't cleanly unmounted)
   - **This is filesystem-level, not file-size dependent**

3. **Directory lookup**:
   - Finding the file in directory
   - **Constant time** (hash table lookup)

---

## Actual Boot Time Breakdown (1TB File)

### Fast Operations (Constant):
1. **mmap()** - ~0.01s ✅ (virtual mapping only)
2. **Read header** - ~0.001s ✅ (4KB page)
3. **Sample nodes** - ~0.1s ✅ (1000 samples = ~64KB)
4. **Allocate arrays** - ~0.01s ✅ (24 MB fixed)

### Potential Variable Operations:
1. **File system metadata read** - ~0.01-0.1s
   - Inode lookup
   - Extent tree traversal (if file is fragmented)
   - **Usually constant, but can vary with fragmentation**

2. **First page fault** - ~0.001s
   - OS loads first 4KB page
   - **Constant** (always 4KB)

---

## Real-World Test

To verify, we'd need to:
```bash
# Create 1TB sparse file (doesn't actually allocate 1TB)
truncate -s 1T test_1tb.m

# Time mmap
time ./test_mmap_boot test_1tb.m

# Should show ~0.1-0.2s regardless of file size
```

---

## Conclusion

**Boot time is approximately constant** because:

1. ✅ **mmap() is lazy** - doesn't read file
2. ✅ **Only access what we need** - header + 1000 samples
3. ✅ **OS pages on-demand** - only loads accessed pages
4. ⚠️ **File system overhead** - may vary slightly with fragmentation, but usually negligible

**Expected boot time: 0.1-0.3 seconds** (not exactly constant, but close)

The key insight: **We don't read the 1TB file - we just map it and access tiny portions on-demand.**


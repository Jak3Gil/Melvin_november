# What is a Blob?

## Definition

A **blob** is a continuous byte region in the `melvin.m` file that contains **raw machine code** (CPU instructions).

Think of it as:
- A **heap for machine code**
- A **continuous byte array** where EXEC nodes store their code
- **One blob region per .m file** (not multiple blobs)
- **Multiple EXEC nodes** can point to **different offsets** in the same blob

## File Structure

```
melvin.m file:
  [Header] → [Graph] → [Nodes] → [Edges] → [BLOB] ← Machine code here
                                                      ↑
                                              EXEC nodes point here
```

## How It Works

1. **Machine code is written to blob**: `melvin_write_machine_code()` appends code to the blob
2. **EXEC nodes point to offsets**: Each EXEC node has `payload_offset` and `payload_len`
3. **CPU executes from blob**: When EXEC node fires, CPU jumps to `blob[payload_offset]` and runs it
4. **Blob is RWX**: Read-Write-Execute protection allows code to run

## Example

```c
// Write machine code to blob
uint64_t offset = melvin_write_machine_code(&file, code_bytes, code_len);
// offset = 0 (first chunk)

// Create EXEC node pointing to that code
uint64_t exec_id = melvin_create_executable_node(&file, offset, code_len);
// EXEC node now points to blob[0]

// When EXEC fires, CPU runs:
void *code_ptr = blob + exec_id->payload_offset;
// CPU jumps to code_ptr and executes
```

## Multiple Code Chunks

**Question**: Can we have multiple blobs?

**Answer**: There is **ONE blob region**, but **multiple EXEC nodes** can point to **different offsets**:

```
Blob (single continuous region):
  [0x0000] Code chunk 1 (EXEC node 1) - 8 bytes
  [0x0008] Code chunk 2 (EXEC node 2) - 8 bytes
  [0x0010] Code chunk 3 (EXEC node 3) - 8 bytes
  [0x0018] Code chunk 4 (EXEC node 4) - 16 bytes
  ...
  [0xFFFFF] (capacity: 1 MB)
```

Each EXEC node has:
- `payload_offset`: Where its code starts in the blob
- `payload_len`: How many bytes its code uses

## How We Know It's Real

### Test Results

✅ **Bytes are stored**: Machine code bytes are written to the file
✅ **EXEC nodes point to bytes**: `payload_offset` correctly references blob
✅ **Memory is executable**: Blob region is marked RWX (Read-Write-Execute)
✅ **CPU can run it**: When EXEC fires, CPU jumps to blob[offset]
✅ **Code persists**: All code chunks survive save/reload

### Verification

```c
// 1. Write code
uint64_t offset = melvin_write_machine_code(&file, code, len);

// 2. Verify bytes are in blob
for (int i = 0; i < len; i++) {
    assert(file.blob[offset + i] == code[i]);  // ✓ Matches
}

// 3. Create EXEC node
uint64_t exec_id = melvin_create_executable_node(&file, offset, len);

// 4. Verify EXEC points to blob
NodeDisk *exec = find_node(exec_id);
assert(exec->payload_offset == offset);  // ✓ Points to blob
assert(exec->payload_len == len);        // ✓ Correct length

// 5. Verify memory protection
mprotect(blob, size, PROT_READ | PROT_WRITE | PROT_EXEC);  // ✓ RWX set

// 6. CPU can execute
void *code_ptr = blob + exec->payload_offset;
// CPU jumps here and runs the code
```

## Capacity

- **Current capacity**: 1,048,576 bytes (1 MB)
- **Can grow**: File can be extended if needed
- **Usage**: Code chunks are appended sequentially
- **No fragmentation**: Simple append-only allocation

## Key Points

1. **One blob per file**: Single continuous region
2. **Multiple EXEC nodes**: Each points to different offset
3. **Sequential allocation**: Code chunks appended one after another
4. **RWX protection**: Memory is executable
5. **Persistent**: All code survives save/reload
6. **Real machine code**: Raw CPU instructions, not interpreted

## Test

Run `test_blob_explained.c` to see:
- Blob structure and layout
- Machine code bytes in hex
- EXEC nodes pointing to different offsets
- Memory protection verification
- Persistence across save/reload

## Conclusion

The blob is **real, executable machine code** stored in the file. Multiple EXEC nodes can use the same blob by pointing to different offsets, like pointers into a heap.


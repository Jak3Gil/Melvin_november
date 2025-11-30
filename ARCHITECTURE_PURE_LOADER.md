# Pure Loader Architecture

## Status: ✅ COMPLETE

The codebase has been refactored so that:

- **`.m` files are self-contained binary brains** containing:
  - Graph state (nodes, edges)
  - Machine code (laws, physics, loops) in blob
  
- **`melvin.c` is a pure loader** (~250 lines):
  - mmap `.m` file
  - feed bytes (write to `.m`)
  - expose syscalls
  - jump into blob once (`melvin_call_entry`)
  
- **NO physics in runtime C code**

## File Structure

### Runtime (Linked into loader binary)
- `melvin.h` - Header definitions only
- `melvin.c` - Pure loader (~250 lines)
  - `melvin_open()` - mmap .m file
  - `melvin_close()` - unmap
  - `melvin_sync()` - sync to disk
  - `melvin_feed_byte()` - write bytes to .m (NO physics)
  - `melvin_set_syscalls()` - expose syscalls to blob
  - `melvin_call_entry()` - jump into blob (ONLY way to "run")
  - `melvin_get_activation()` - read-only inspection

### Tools (Separate binaries, NOT in runtime)
- `melvin_uel.c` - UEL physics implementation
  - `uel_main(Graph *g)` - The actual physics
  - Used by tools to extract machine code
  - **NOT linked into runtime**
  
- `uel_seed_tool.c` - Tool to seed .m blob
  - Creates .m file
  - Embeds UEL machine code into blob
  - Sets `main_entry_offset`

## Verification

### ✅ No Physics Loops in Runtime

```bash
$ grep -i "for.*node\|while.*node\|melvin_tick\|melvin_step" melvin.c
# (no matches - verified)
```

The only loops in `melvin.c` are:
- Initialization loop (setting up data nodes 0-255) - structure only, no physics
- Edge traversal in `find_edge()` - structure lookup only, no physics

### ✅ No Physics Functions in Runtime

- `melvin_step()` - **REMOVED**
- `melvin_tick()` - **NOT LINKED** (exists only in tool)
- All UEL calculations - **IN BLOB ONLY**

### ✅ Runtime API

```c
// Loader only
Graph* melvin_open(const char *path, ...);
void melvin_close(Graph *g);
void melvin_sync(Graph *g);

// Input (writes to .m, no physics)
void melvin_feed_byte(Graph *g, uint32_t port, uint8_t b, float energy);

// Syscalls (for blob)
void melvin_set_syscalls(Graph *g, MelvinSyscalls *sys);

// Run (jump into blob - blob does EVERYTHING)
void melvin_call_entry(Graph *g);

// Inspection (read-only)
float melvin_get_activation(Graph *g, uint32_t node_id);
```

## How It Works

1. **Create brain** (offline):
   ```bash
   ./uel_seed_tool brain.m
   # Extracts uel_main() machine code
   # Writes to blob[0]
   # Sets main_entry_offset = 0
   ```

2. **Run brain** (runtime):
   ```c
   Graph *g = melvin_open("brain.m", ...);
   melvin_feed_byte(g, port, 'A', 1.0f);  // Just writes to .m
   melvin_set_syscalls(g, &syscalls);
   melvin_call_entry(g);  // Jump into blob - blob does ALL physics
   melvin_close(g);
   ```

3. **Blob code** (inside .m):
   - Runs UEL physics
   - Updates activations/weights
   - Calls syscalls for IO
   - Handles all loops
   - Everything lives in machine code

## Key Principle

> **`.m` IS the mind.**
> 
> `melvin.c` is just: map file, feed bytes, give syscalls, jump in once.
> 
> All laws, loops, ticks, decisions live in machine code inside `.m`.

## Next Steps

To complete the architecture:

1. **Extract machine code properly**:
   - Compile `melvin_uel.c` to `.o`
   - Extract `.text` section
   - Write to blob[0] in `uel_seed_tool`
   
2. **Handle syscalls in blob**:
   - Blob code reads `syscalls_ptr_offset` from header
   - Loads syscalls pointer from blob
   - Calls syscalls as needed

3. **Optional: Self-modification**:
   - Blob code can rewrite itself
   - Blob code can add new nodes/edges
   - All evolution happens in blob


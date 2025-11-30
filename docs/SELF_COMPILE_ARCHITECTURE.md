# Self-Compilation Architecture

## Status: ✅ IMPLEMENTED

The system now supports self-compilation where `.m` can compile C code and store it back in the blob, while keeping `melvin.c` as a pure loader.

## Key Principle

> **Everything is bytes + energy**
> 
> C files, compiled binaries, vision frames, motor CAN frames, text - all are just bytes that enter the same graph and become energy/routes by the same laws.

## Architecture

### Runtime (Pure Loader)
- `melvin.c` - Still pure loader (~270 lines)
  - NO physics
  - NO understanding of C/code
  - Just: mmap, feed bytes, syscalls, jump into blob

### Host Syscalls
- `host_syscalls.c` - Host-side implementations
  - `sys_write_text` - stdout
  - `sys_write_file` - write file
  - `sys_read_file` - read file
  - `sys_run_cc` - compile C (calls clang)
  - `sys_send_motor_frame` - motor control

### Compiler Agent (Blob Machine Code)
- `mc_compile_agent.c` - Compiler agent (TOOL, not runtime)
  - `mc_compile_c()` - Compiles C source, stores in blob
  - `ingest_bytes_as_energy()` - Ingest bytes as graph energy
  - `mc_find_free_blob_region()` - Find free blob space

## How It Works

### 1. Feed Source as Bytes

```c
// Host feeds C source as plain bytes - no special handling
const char *source = "int add(int a, int b) { return a + b; }";
for (size_t i = 0; i < strlen(source); i++) {
    melvin_feed_byte(g, port, source[i], energy);
}
```

The graph sees these as:
- DATA nodes for each byte
- SEQ edges between consecutive bytes
- Activation energy flowing through the pattern

### 2. Blob Decides to Compile

The blob machine code (not C runtime) detects:
- Pattern of C source bytes
- Decides: "this reduces chaos, I should compile this"
- Calls `mc_compile_c()` via function pointer

### 3. Compilation Happens in Blob

```c
// Inside blob machine code (mc_compile_c):
void mc_compile_c(Graph *g, const char *src_path, ...) {
    MelvinSyscalls *sys = melvin_get_syscalls_from_blob(g);
    
    // 1. Call syscall to compile
    sys->sys_run_cc(src_path, out_path);
    
    // 2. Read compiled binary
    sys->sys_read_file(out_path, &bin, &bin_len);
    
    // 3. Copy into blob
    memcpy(g->blob + dest_offset, bin, bin_len);
    
    // 4. Ingest compiled bytes as energy (same as source!)
    ingest_bytes_as_energy(g, bin, bin_len, port);
}
```

### 4. Unified Ingestion

Both source and compiled bytes go through the same path:
- Create/update DATA nodes
- Create SEQ edges
- Flow activation energy
- Form patterns

The graph learns relationships between:
- Source text patterns
- Compiled machine code patterns
- Their correlations

## File Structure

```
melvin.c              - Pure loader (NO physics, NO compilation logic)
melvin.h              - Header with extended syscalls
host_syscalls.c       - Host syscall implementations
mc_compile_agent.c    - Compiler agent (TOOL - embedded in blob)
test_self_compile.c   - Test demonstrating self-compilation
```

## Verification

### ✅ No Physics in Runtime
```bash
$ grep -i "melvin_tick\|melvin_step\|uel_main" melvin.c
# (no matches - verified)
```

### ✅ No Compilation Logic in Runtime
```bash
$ grep -i "compile\|clang\|gcc" melvin.c
# (no matches - verified)
```

### ✅ All Compilation in Blob
- `mc_compile_agent.c` is a TOOL file
- Compiled and embedded into blob
- Runtime never links it directly

## Usage

### 1. Initialize Host Syscalls
```c
MelvinSyscalls syscalls;
melvin_init_host_syscalls(&syscalls);
melvin_set_syscalls(g, &syscalls);
```

### 2. Feed Source as Bytes
```c
// Write source to file
FILE *f = fopen("source.c", "w");
fputs(c_source, f);
fclose(f);

// Feed bytes into graph
for (size_t i = 0; i < strlen(c_source); i++) {
    melvin_feed_byte(g, port, c_source[i], energy);
}
```

### 3. Blob Handles Everything
```c
melvin_call_entry(g);  // Blob decides what to do
```

The blob code:
- Detects C source pattern
- Calls `mc_compile_c()` 
- Stores compiled code in blob
- Ingests both source and compiled bytes as energy

## Key Insight

The loader (`melvin.c`) has **zero knowledge** of:
- What C is
- What compilation means
- What files contain
- What bytes represent

It only:
- Maps `.m` file
- Feeds bytes (writes to `.m`)
- Exposes syscalls (generic functions)
- Jumps into blob

All understanding, decisions, and logic live in the blob machine code.

## Next Steps

1. **Embed compiler agent**: Use `uel_seed_tool` to embed `mc_compile_agent` machine code into blob
2. **Pattern detection**: Blob code learns to detect "C source" patterns
3. **Self-modification**: Blob code can rewrite itself using compiled code
4. **Multi-language**: Same mechanism works for ASM, Python, etc. - all just bytes


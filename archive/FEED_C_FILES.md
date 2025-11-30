# Feeding C Files to Melvin on Jetson

**Yes!** You can keep giving Melvin C files even when it's running on the Jetson.

## How It Works

Melvin can ingest C files in several ways:

### Method 1: Via stdin (Real-time)

While Melvin is running on the Jetson, you can send C files directly:

```bash
# On your Mac/PC
cat my_file.c | ssh melvin@169.254.123.100 "cd ~/melvin_system && ./melvin"

# Or pipe a file
ssh melvin@169.254.123.100 "cd ~/melvin_system && cat - | ./melvin" < my_file.c

# Or send interactively
ssh melvin@169.254.123.100
cd ~/melvin_system
./melvin &
echo "void my_function() { ... }" | ./melvin
```

**How it works:**
- Melvin reads from stdin (non-blocking)
- `mc_chat_in` or `mc_stdio_in` processes the input
- Creates word/byte nodes for the C code
- Graph learns the structure

### Method 2: Copy to Jetson File System

Copy C files to the Jetson and Melvin will find them:

```bash
# Copy file to Jetson
scp my_file.c melvin@169.254.123.100:~/melvin_system/

# Melvin will auto-detect and parse (if mc_parse is active)
# Or create a "data" or "corpus" directory
ssh melvin@169.254.123.100 "mkdir -p ~/melvin_system/data"
scp my_file.c melvin@169.254.123.100:~/melvin_system/data/
```

**How it works:**
- `mc_fs_read` or `mc_parse_c_file` scans for C files
- Parses them and stores structure in graph
- Creates nodes/edges for functions, calls, etc.

### Method 3: Via GitHub URL (Auto-clone)

Send GitHub URLs and Melvin will clone and learn:

```bash
# On Jetson
echo "https://github.com/user/repo.git" | ./melvin

# Or add to github_urls.txt
echo "https://github.com/user/repo.git" >> github_urls.txt
```

**How it works:**
- `mc_git_auto_learn` detects GitHub URLs
- Clones repository to `ingested_repos/`
- Parses all C files automatically
- Stores everything in `melvin.m`

## What Gets Stored

When Melvin receives a C file, it:

1. **Parses the structure:**
   - Functions → Function nodes
   - Function calls → Call edges
   - Parameters → Parameter nodes
   - Variables → Variable nodes
   - Types → Type nodes

2. **Stores in graph:**
   - All structure stored as nodes/edges
   - Connected by relationships
   - Activation-based learning
   - Pattern formation

3. **Everything in melvin.m:**
   - No source files needed
   - All knowledge in graph
   - Permanently stored

## Requirements

For Melvin to parse C files, you need **one of these**:

### Option A: Plugin File (Current)

The `mc_parse` plugin needs to be compiled on Jetson:

```bash
# Build parse plugin on Jetson
cd ~/melvin_system
mkdir -p plugins
gcc -shared -fPIC -O3 -march=armv8-a -I. -o plugins/mc_parse.so plugins/mc_parse.c -lm -ldl
```

**Then register in melvin.c:**
```c
MCFn mc_parse_c_file = load_plugin_function("mc_parse", "mc_parse_c_file");
register_mc("parse_c", mc_parse_c_file);
```

### Option B: Graph-Native (Future)

Eventually, Melvin could learn to parse C files purely from the graph:
- Parsing patterns stored in `melvin.m`
- Graph itself does the parsing
- No plugin files needed!

## Example: Feeding C Files to Jetson

### Setup (One-time)

```bash
# 1. Build parse plugin on Jetson
ssh melvin@169.254.123.100
cd ~/melvin_system
mkdir -p plugins data

# 2. Copy plugin source (or it's already in melvin.m from initial deployment)
scp plugins/mc_parse.c melvin@169.254.123.100:~/melvin_system/plugins/

# 3. Build plugin
gcc -shared -fPIC -O3 -march=armv8-a -I. -o plugins/mc_parse.so plugins/mc_parse.c -lm -ldl
```

### Feed C Files Continuously

**Method 1: Copy files**
```bash
# Copy new C file
scp my_new_file.c melvin@169.254.123.100:~/melvin_system/data/

# Melvin running on Jetson will detect and parse it
```

**Method 2: Send via stdin**
```bash
# While Melvin is running on Jetson
ssh melvin@169.254.123.100
cd ~/melvin_system

# In another terminal, send file
cat my_file.c | ssh melvin@169.254.123.100 "cd ~/melvin_system && nc localhost 9999"
# (Requires mc_chat_in to be listening on port)

# Or simpler - just cat to running process
echo "void new_function() { return 42; }" | ./melvin
```

**Method 3: GitHub repos**
```bash
# Clone and learn entire repos
echo "https://github.com/user/cool-project.git" | ./melvin
```

## What Happens

1. **Melvin receives C file** (via stdin, filesystem, or git)
2. **Plugin parses it** (`mc_parse_c_file`)
3. **Structure extracted:**
   - Functions, calls, parameters, types
4. **Stored in graph:**
   - New nodes created
   - New edges created
   - Patterns formed
5. **Everything in melvin.m:**
   - Source file not needed
   - Knowledge is graph-native
   - Permanently stored

## Continuous Learning

**You can keep feeding Melvin C files forever:**

```bash
# Continuous feed loop
while true; do
    scp new_file*.c melvin@169.254.123.100:~/melvin_system/data/
    sleep 60  # Wait for Melvin to process
done
```

Melvin will:
- ✅ Detect new files
- ✅ Parse them
- ✅ Store in graph
- ✅ Learn patterns
- ✅ Grow knowledge

## Key Points

1. **Everything in graph:** Once parsed, the source C files aren't needed
2. **Continuous ingestion:** Can keep feeding files while running
3. **Multiple methods:** stdin, filesystem, GitHub URLs all work
4. **Graph-native:** All knowledge stored as nodes/edges in `melvin.m`
5. **Permanent storage:** Everything saved in `melvin.m`

## Future: Pure Graph Parsing

Eventually, Melvin could learn to parse C files **purely from the graph**:
- Parsing patterns learned and stored
- Graph itself does parsing
- No plugin files needed
- Truly graph-native!

**For now, you need the `mc_parse.so` plugin, but the knowledge is all in the graph!**



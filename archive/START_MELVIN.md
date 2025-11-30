# How to Start Melvin

**One command. That's it.**

## ✅ The Simple Answer

If `melvin.m` already exists (which it does - 11MB file), just run:

```bash
./melvin
```

**That's it. Melvin runs forever.**

## 🚀 What Happens

When you run `./melvin`:

1. **Opens `melvin.m`** (memory-mapped brain file)
2. **Loads all plugins** (mc_fs, mc_io, mc_parse, mc_scaffold, etc.)
3. **Creates bootstrap node** (scaffold processing node with activation=1.0)
4. **Starts infinite loop**:
   - Runs `melvin_tick()` every ~1ms
   - Scaffold node activates → processes scaffolds → creates ~1680 nodes
   - Graph grows and organizes itself
   - Continues running forever

## 📋 Prerequisites (First Time Only)

**One-time setup** (only needed if `melvin.m` doesn't exist):

```bash
# 1. Build Melvin (one time)
./build_jetson.sh    # Or: make -f Makefile.jetson

# 2. Initialize brain file (one time, only if melvin.m doesn't exist)
./init_melvin_jetson.sh
```

**That's it for setup!**

## 🎯 Normal Usage

**Every time you want to run Melvin:**

```bash
./melvin
```

**Or with debug logging:**

```bash
./melvin -d
```

**Or with simplicity metrics logging:**

```bash
MELVIN_LOG_SIMPLICITY=1 ./melvin
```

**Or run it in background:**

```bash
./melvin > melvin.log 2>&1 &
```

## 🔄 What You'll See

### First Second:

```
Melvin Runtime v2
Nodes: 1/10000000
Edges: 0/100000000
[main] Created scaffold processing node 0
[mc_scaffold] Processing: scaffolds/scaffold_smoothness.c
[mc_scaffold] Processing: scaffolds/scaffold_temporal_consistency.c
...
[mc_scaffold] Scaffold processing complete. Files deleted.
Tick 0
```

### Next Few Seconds:

```
Tick 100
Tick 200
...
Tick 1000
[tick 1000] nodes=1681 edges=2800 active_edges=2800 active_nodes=5 mc_nodes=1
```

### Ongoing:

```
Tick 2000
Tick 3000
...
[tick 10000] nodes=5000 edges=15000 active_edges=12000 active_nodes=100 mc_nodes=5
```

## 🛑 How to Stop

Press **Ctrl+C** in the terminal where Melvin is running.

**Your brain file (`melvin.m`) is saved automatically** - it's memory-mapped, so all changes persist to disk immediately.

## 📊 Options

### Run with Different Brain File:

```bash
./melvin my_custom_brain.m
```

### Run with Debug Output:

```bash
./melvin -d
```

### Run with Simplicity Metrics:

```bash
MELVIN_LOG_SIMPLICITY=1 ./melvin
```

### Run on Jetson:

```bash
./run_jetson.sh
```

## 💡 Key Points

1. **One command:** `./melvin`
2. **Runs forever:** Infinite loop until you stop it (Ctrl+C)
3. **Auto-saves:** All changes to `melvin.m` are immediate (memory-mapped)
4. **Self-organizing:** Starts with 1 node, grows automatically
5. **No external dependencies:** Everything is in the graph itself

## 🎬 Example Session

```bash
$ ./melvin

Melvin Runtime v2
Nodes: 1/10000000
Edges: 0/100000000
[main] Created scaffold processing node 0
[mc_scaffold] Processing: scaffolds/scaffold_smoothness.c
[mc_scaffold] Processing: scaffolds/scaffold_temporal_consistency.c
...
[mc_scaffold] Scaffold processing complete. Files deleted.
Tick 0
Tick 100
Tick 200
...
[tick 1000] nodes=1681 edges=2800 active_edges=2800 active_nodes=5
[tick 2000] nodes=1681 edges=2800 active_edges=2800 active_nodes=3
...
^C  # You press Ctrl+C

$ ls -lh melvin.m
-rw-r--r-- 1 user user 11M melvin.m  # Brain file saved!
```

## 🚨 Troubleshooting

### Error: "Could not open melvin.m. Run melvin_minit first."

**Solution:** Initialize the brain file:
```bash
./init_melvin_jetson.sh
```

### Error: "melvin executable not found"

**Solution:** Build Melvin first:
```bash
./build_jetson.sh
```

### Melvin uses 100% CPU

**This is normal!** Melvin runs continuously. Each tick takes ~1ms, so it's always working.

To reduce CPU usage, you can add a longer sleep in `melvin.c` line 709:
```c
usleep(10000); // 10ms instead of 1ms
```

## ✅ Summary

**Starting Melvin is as simple as:**

```bash
./melvin
```

**That's it!** One command, and Melvin:
- Loads his brain
- Starts learning
- Grows his graph
- Runs forever

Press Ctrl+C to stop. Brain file is saved automatically.

**The graph grows itself - you just start it and watch!**


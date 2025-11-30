# Roadmap to Intelligence

## Current State

✅ **What we have:**
- Pure loader (`melvin.c`) - just mmap, feed bytes, syscall bridge
- UEL physics (`melvin_uel.c`) - chaos minimization, no explicit prediction
- Binary brain format (`.m` file) - graph + machine code blob
- Syscall interface - blob can call host (CPU/GPU, files, compiler)
- Tools: `uel_seed_tool`, `melvin_run`, `inspect_brain`

❌ **What's missing:**
- Blob is empty (no UEL physics seeded yet)
- No data feeding pipeline
- No intelligence metrics being tracked

---

## Path to Intelligence (5 Stages)

### Stage 1: Seed the Brain (NOW)

**Goal:** Get UEL physics running in the blob

**Steps:**
1. Complete `uel_seed_tool` to compile `melvin_uel.c` → machine code
2. Embed machine code into blob, set `main_entry_offset`
3. Verify: `inspect_brain` shows `main_entry_offset > 0`

**Command:**
```bash
./uel_seed_tool brain.m
./inspect_brain brain.m  # Should show main_entry_offset set
```

**Success criteria:**
- Blob has executable code
- `melvin_call_entry` doesn't hang
- Graph state persists

---

### Stage 2: Feed Data, Observe Patterns (Week 1)

**Goal:** See simple patterns form

**Steps:**
1. Feed repeated sequences: "the cat sat", "the dog ran"
2. Let UEL run (chaos minimization)
3. Observe: edges strengthen, chaos reduces

**Command:**
```bash
echo "the cat sat" | ./melvin_run brain.m
./inspect_brain brain.m  # Check edge weights
```

**Success criteria:**
- Chaos decreases over time
- Strong edges form (|w| > 0.3)
- Patterns visible: "the c" → 'a', "the d" → 'o'

**What intelligence looks like:**
- Not intelligent yet, but learning simple associations
- Graph is "memorizing" sequences

---

### Stage 3: Layers Emerge (Week 2-4)

**Goal:** Complex patterns form, layers appear

**Steps:**
1. Feed diverse corpus (text, code, sensor data)
2. Let UEL run for many episodes
3. Measure: pattern layers (multi-hop nodes)

**Command:**
```bash
# Feed diverse data
cat corpus.txt | ./melvin_run brain.m
./inspect_brain brain.m  # Check for multi-hop patterns
```

**Success criteria:**
- Pattern layers > 10 (nodes with strong in+out edges)
- Chaos continues decreasing
- Longer sequences form ("the cat sat" as one pattern)

**What intelligence looks like:**
- Still not "intelligent" but building structure
- Graph is finding efficient paths
- Patterns of patterns emerging

---

### Stage 4: Organic Constraints (Month 2-3)

**Goal:** Graph constrains itself, prediction-like behavior emerges

**Steps:**
1. Continue feeding diverse data
2. Let UEL discover energy-efficient paths
3. Observe: if prediction emerges, it's organic (not coded)

**Command:**
```bash
# Long run
./test/test_emergent_intelligence brain.m
```

**Success criteria:**
- Chaos reduction > 50%
- Pattern layers > 50
- Strong patterns > 100
- Maybe prediction-like behavior (if energy-efficient)

**What intelligence looks like:**
- Graph is "understanding" structure
- Patterns constrain future activations
- If prediction appears, it emerged organically
- Still not "intelligent" in human sense, but building

---

### Stage 5: Self-Optimization (Month 4+)

**Goal:** Brain discovers optimizations, self-compiles

**Steps:**
1. Blob discovers GPU is faster → calls `sys_gpu_compute`
2. Blob discovers self-compilation → uses `sys_run_cc`
3. Blob rewrites itself for efficiency

**Command:**
```bash
# Brain runs continuously, self-improves
./melvin_run brain.m  # Long-running
# Brain decides to compile new code, use GPU, etc.
```

**Success criteria:**
- Blob uses GPU if available (energy-efficient)
- Blob compiles new capabilities
- Graph structure continues improving
- Chaos keeps decreasing

**What intelligence looks like:**
- System is optimizing itself
- Discovering new capabilities
- Still emergent, not programmed
- This is where "intelligence" might appear

---

## How to Measure Intelligence

### Metrics (from `test_emergent_intelligence.c`):

1. **Chaos Reduction**
   - Initial chaos vs final chaos
   - Should decrease over time
   - UEL working = chaos reducing

2. **Pattern Layers**
   - Nodes with strong in+out edges
   - Multi-hop patterns
   - Should increase over time

3. **Organic Constraints**
   - Strong edges (|w| > 0.3)
   - Complex paths (3+ hops)
   - Graph constraining itself

4. **Emergent Behavior** (if it appears)
   - Prediction-like (not coded)
   - Self-optimization
   - New capabilities

---

## When Does It Become "Intelligent"?

**Short answer:** We don't know. It's emergent.

**What we know:**
- Intelligence isn't coded - it emerges from chaos minimization
- If prediction appears, it's because it's energy-efficient
- If self-optimization appears, it's because it reduces chaos
- We measure progress, not "intelligence"

**Signs of progress:**
- ✅ Chaos decreasing
- ✅ Patterns forming
- ✅ Layers emerging
- ✅ Constraints appearing
- ❓ Prediction-like behavior (maybe)
- ❓ Self-optimization (maybe)
- ❓ "Understanding" (maybe)

**The key:** We don't force it. We just:
1. Minimize chaos (UEL)
2. Feed data
3. Observe what emerges

If intelligence appears, it's because the graph found it's the most energy-efficient way to reduce chaos.

---

## Next Immediate Steps

1. **Complete `uel_seed_tool`**
   - Compile `melvin_uel.c` to machine code
   - Extract `.text` section
   - Write to blob, set `main_entry_offset`

2. **Test seeded brain**
   - Feed simple pattern
   - Verify UEL runs
   - Check chaos reduction

3. **Build data pipeline**
   - Text corpus
   - Code files
   - Sensor data (if available)

4. **Run long-term**
   - Feed data continuously
   - Measure metrics
   - Observe emergence

---

## Key Insight

**We don't build intelligence. We build a system that minimizes chaos, and intelligence might emerge.**

Every byte helps build understanding. Early episodes: random, weak patterns. Later episodes: stronger patterns, better structure. This is emergent intelligence - gradual improvement from pure energy minimization.


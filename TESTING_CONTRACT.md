# MELVIN TESTING CONTRACT
## The Rules That Keep Tests Honest

**PURPOSE:** These rules ensure we test Melvin's **graph intelligence**, not our hand-coded hacks.

---

## THE 5 CORE RULES (NON-NEGOTIABLE)

### Rule 1: **melvin.m is the brain**

- Nodes, edges, patterns, stats, weights, activations — **everything** lives in `melvin.m`
- The file is memory-mapped (mmap) for direct access
- The graph structure **is** the intelligence
- No external state that affects behavior

**Enforcement:** Any code that stores task-specific logic outside `melvin.m` violates this rule.

---

### Rule 2: **External interface is minimal**

**Allowed operations from outside:**

1. **Map `melvin.m`** — Open and memory-map the file
2. **Stream bytes in** — Send `(channel_id, byte_value)` pairs via `ingest_byte()`
3. **Tick physics** — Call `melvin_process_n_events()` to advance simulation
4. **Read bytes out** — Read output channels (OUT_TEXT, OUT_MOTOR, etc.)
5. **Read stats** — Read-only access to summary statistics (prediction error, reward totals)

**NOT allowed:**

- Direct node/edge manipulation (except for initial setup of universal substrate)
- Direct weight modifications
- Direct pattern creation (patterns must emerge from physics)
- Any task-specific logic in test harness

**Enforcement:** Test harness can only call:
- `melvin_m_map()`, `runtime_init()`, `ingest_byte()`, `melvin_process_n_events()`
- Read-only accessors for output channels and stats
- File I/O (copy snapshots, save/load `.m` files)

---

### Rule 3: **No hidden resets**

- **DO NOT** zero nodes/edges/patterns between tests
- **DO NOT** re-initialize weights unless explicitly creating a new brain
- Every test **starts from the brain state produced by all previous tests**
- Tests accumulate knowledge — Test N sees all learning from Tests 1..N-1

**Exception:** When explicitly testing "fresh brain" baseline:
- Keep a snapshot: `melvin_clean.m`
- Copy `melvin_clean.m` → `melvin.m` to restore baseline
- This is file I/O, not internal reset

**Enforcement:** Any code that calls `memset()` on graph data between tests violates this rule.

---

### Rule 4: **All learning is internal**

- No external code nudging weights
- No task-specific logic in C like `if (byte == 'A') increase_weight(...)`
- The only learning signals from outside must be **encoded as bytes**

**Learning signals via bytes:**

- Reward: Send bytes on `CH_REWARD` channel (e.g., `b'R+'`, `b'R-'`)
- Episode markers: Send special bytes (e.g., `b'\n'` for end-of-episode)
- Task context: Encode as bytes on appropriate channels

**The graph must interpret these bytes through its own learned patterns.**

**Enforcement:** Search codebase for:
- `edge->weight =` (should only appear in learning rules, not test code)
- `node->state =` (should only appear in physics updates, not test code)
- Task-specific `if` statements that modify graph state

---

### Rule 5: **melvin.m is append/evolve-only**

- We don't rebuild the graph from C on each run
- The graph structure evolves continuously through:
  - Data ingestion (creates DATA nodes, SEQ edges)
  - Energy flow (creates co-activation edges)
  - Pattern induction (creates PATTERN nodes)
  - Learning (updates weights)

**Allowed rebuilding:**

- Only for the **universal substrate** (data structures, physics engine, IO)
- Not for learned structure (patterns, weights, topology)

**Exception:** Initial instinct injection (one-time setup of universal primitives like EXEC_TEMPLATE, CODE_WRITE nodes)

**Enforcement:** Any code that rebuilds learned patterns or weights from scratch violates this rule.

---

## WHAT THIS MEANS FOR TESTS

### Test Harness (Outside) Responsibilities:

1. **Open brain:**
   ```c
   MelvinFile file;
   melvin_m_map("melvin.m", &file);
   MelvinRuntime rt;
   runtime_init(&rt, &file);
   ```

2. **Feed inputs (bytes only):**
   ```c
   ingest_byte(&rt, CH_TEXT, 'H', 1.0f);
   ingest_byte(&rt, CH_TEXT, 'e', 1.0f);
   ingest_byte(&rt, CH_TEXT, 'l', 1.0f);
   // ... etc
   ```

3. **Tick physics:**
   ```c
   melvin_process_n_events(&rt, 1000);  // Process 1000 events
   ```

4. **Read outputs (bytes or stats only):**
   ```c
   // Read from OUT_TEXT channel (if pattern exists)
   uint64_t out_node = find_node_by_label(&file, "OUT_TEXT");
   float activation = file.nodes[out_node].state;
   // OR read output bytes that graph chose to emit
   ```

5. **Send rewards (bytes only):**
   ```c
   if (output_correct) {
       ingest_byte(&rt, CH_REWARD, 'R', 1.0f);  // Reward byte
   } else {
       ingest_byte(&rt, CH_REWARD, 'P', 1.0f);  // Punishment byte
   }
   ```

6. **Grade externally:**
   ```c
   // Compute metrics OUTSIDE the graph
   float accuracy = correct_count / total_count;
   printf("Accuracy: %.2f%%\n", accuracy * 100.0f);
   // Give feedback back as bytes if needed
   ```

### Melvin (Inside) Responsibilities:

1. **Ingest bytes** → Create DATA nodes, SEQ edges, CHAN edges
2. **Update activations** → Energy flows through edges
3. **Learn patterns** → Edge weights update via free-energy minimization
4. **Form abstractions** → PATTERN nodes emerge when they reduce FE
5. **Generate outputs** → Patterns activate output channels
6. **Evolve structure** → Graph grows, prunes, adapts

**Everything happens through physics — no external logic.**

---

## TEST SEQUENCE EXAMPLE

```
TEST 1: A→B Pattern
--------------------
1. Open melvin.m (or create new)
2. Feed: "A" "B" "A" "B" ... (repeated pattern)
3. Tick: 1000 events
4. Test: Feed "A", tick 100 events, check if "B" is predicted
5. Grade: Compare predicted vs actual (EXTERNAL to graph)
6. Reward: Send reward bytes if correct
7. Close: Keep melvin.m for next test

TEST 2: Arithmetic (uses TEST 1's learned structure)
----------------------------------------------------
1. Open same melvin.m (NOT reset)
2. Feed: "3+3=6" "2+2=4" ... (arithmetic examples)
3. Tick: 5000 events
4. Test: Feed "5+5=?", check output
5. Grade: Compare output vs expected (EXTERNAL)
6. Reward: Send reward bytes
7. Close: Keep melvin.m

TEST 3: ... (all previous learning available)
```

---

## CONTRACT VIOLATION DETECTION

### Automated Checks (grep patterns):

```bash
# Find weight manipulation outside learning rules
grep -r "->weight\s*=" --include="*.c" --exclude="melvin.c" | grep -v "learning\|update_edge"

# Find node state manipulation in tests
grep -r "node.*->state\s*=" --include="test_*.c"

# Find task-specific logic
grep -r "if.*byte.*==" --include="*.c" | grep -v "ingest_byte\|diagnostic"

# Find pattern creation in tests
grep -r "create_pattern\|create_edge.*pattern" --include="test_*.c"
```

### Manual Review Checklist:

- [ ] Does test reset graph state? → **VIOLATION**
- [ ] Does test modify weights directly? → **VIOLATION**
- [ ] Does test have task-specific logic? → **VIOLATION**
- [ ] Does test only use bytes in/out? → **GOOD**
- [ ] Does test reuse melvin.m from previous tests? → **GOOD**

---

## ENFORCEMENT IN CODE

Add to test harness template:

```c
// ========================================================================
// CONTRACT ENFORCEMENT: This test only uses bytes in/out
// ========================================================================
// 
// ALLOWED:
//   - ingest_byte(rt, channel, byte, energy)
//   - melvin_process_n_events(rt, N)
//   - Read output channels / stats
//   - File I/O (open, copy, save melvin.m)
// 
// FORBIDDEN:
//   - Direct node/edge manipulation
//   - Direct weight modifications
//   - Task-specific C logic
//   - Graph state resets
// ========================================================================
```

---

## THE GOAL

**When a test passes, we know:**

- The graph (melvin.m) learned something
- Learning happened through physics (energy, patterns, free-energy minimization)
- No cheating (no hand-coded logic made it work)
- The intelligence is **in the graph**, not in our code

**This is how we prove Melvin's capabilities are real.**

---

## EXCEPTIONS (Must be documented)

If you MUST violate a rule for a specific test:

1. Document the exception clearly
2. Explain why it's necessary
3. Mark it as "NOT testing graph intelligence"
4. Create a separate test that follows the contract

Example:
```c
// EXCEPTION: This test violates Rule 4 to verify EXEC mechanism works
// This is NOT testing graph intelligence, only testing the execution subsystem
// A proper test would encode this as bytes and let the graph learn it
```

---

**Last Updated:** [Date]
**Status:** ACTIVE - All tests must follow this contract


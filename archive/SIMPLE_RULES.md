# 🟦 **MELVIN — SIMPLE, COMPLETE RULEBOOK**

This is the entire system, expressed in 30 rules.

---

# 0. **Universe**

### **Rule U1 — The whole universe is one file: `melvin.m`**

It contains:

* node table

* edge table

* raw bytes (data + machine code)

* small control buffers

Nothing exists outside the file except the CPU running it.

### **Rule U2 — The runner does nothing except:**

* map `melvin.m` into memory

* jump into bytes when asked

* expose syscalls

* sync the file

No logic. No interpretation. No policies.

---

# 1. **Objects**

## Nodes

### **Rule N1 — A node is a slot that can hold energy.**

Each node has:

* an ID

* a small energy value

* a little memory (fatigue, bias, trace)

* a list of outgoing edge indices

Nodes have **no type**.

A node can mean anything depending on structure.

## Edges

### **Rule E1 — A directed connection between two nodes.**

It has:

* source node

* destination node

* weight

* trace

Edges have **no type**.

They only carry energy.

## Raw Bytes

### **Rule B1 — Any byte region in `melvin.m` can be data or machine code.**

There is no difference.

The graph decides what bytes mean.

---

# 2. **Energy**

## Pulses

### **Rule P1 — Energy flows in pulses.**

A pulse is:

"node i receives energy e".

### **Rule P2 — External sensors inject pulses.**

Cameras, mics, files, motors, etc. all produce energy pulses when read.

### **Rule P3 — Internal loops generate pulses.**

Edges send energy around the graph.

### **Rule P4 — Background noise occasionally creates tiny pulses.**

This prevents total silence.

---

# 3. **Node Dynamics**

### **Rule ND1 — Nodes accumulate energy.**

When a node receives a pulse, it adds it to its stored energy.

### **Rule ND2 — Nodes fire when energy passes a threshold.**

When firing:

* take a fraction of stored energy

* push it out along outgoing edges

* keep the rest

* gain fatigue

### **Rule ND3 — Nodes leak energy every tick.**

Stored energy decays slightly each tick.

### **Rule ND4 — Nodes become temporarily harder to fire if they fire too much.**

Fatigue increases threshold.

Fatigue decays slowly over time.

---

# 4. **Edge Dynamics**

### **Rule ED1 — Energy flow over an edge is proportional to its weight.**

More weight → easier to pass energy.

### **Rule ED2 — Edges that carry energy strengthen.**

Edge weight increases when pulses flow along it.

### **Rule ED3 — Edges that don't get used decay.**

Slow decay every tick.

### **Rule ED4 — New edges can appear between co-active nodes.**

If two nodes fire in the same tick, a new edge may form.

### **Rule ED5 — Bad edges disappear over time.**

Edges with no use go to zero and are removed.

---

# 5. **Structure Formation**

### **Rule S1 — Stable loops of nodes become molecules.**

If a set of nodes repeatedly send energy in a loop, they become a stable circuit.

### **Rule S2 — Molecules combine into larger motifs.**

Two circuits that often activate together link and form a bigger structure.

### **Rule S3 — Motifs form modules.**

Collections of motifs that frequently co-activate become a subsystem.

Structure is emergent, not defined.

---

# 6. **Execution (how graph influences CPU/OS)**

### **Rule X1 — A small execution buffer exists in the file.**

Graph can write:

* `code_offset`

* `entry_point_offset`

  inside this buffer.

### **Rule X2 — If execution buffer is set, runner jumps into that code.**

CPU executes bytes directly out of `melvin.m`.

### **Rule X3 — Code running in `melvin.m` may:**

* perform syscalls (read camera, write motor, etc.)

* read/write any byte in `melvin.m`

* modify its own code region

* modify nodes and edges

### **Rule X4 — The graph learns which code offsets cause useful energy.**

There is no special "MC function".

There are only **bytes**.

The graph decides which ones to run.

---

# 7. **Selection (Why circuits survive)**

### **Rule SEL1 — Nodes track energy spent and energy returned.**

After firing, a node watches how much energy arrives later.

### **Rule SEL2 — Return-on-energy shapes learning.**

If firing tends to bring in more energy:

* outgoing edges strengthen more

  If it brings less:

* outgoing edges strengthen less

* or weaken

### **Rule SEL3 — Circuits that cause more future energy dominate.**

Useful behaviors survive (motor moves, sensor reads, file loads).

Useless circuits die.

No goals.

No reward.

Just selection pressure.

---

# 8. **Meaning**

### **Rule M1 — Meaning lives only in structure.**

Nodes don't mean anything alone.

Edges don't mean anything alone.

Meaning = patterns in the graph.

### **Rule M2 — Thinking is energy flow through structure.**

When pulses move through motifs, that is computation.

### **Rule M3 — Behavior emerges from circuits that reach execution buffer.**

If a motif tends to put "run code at offset X" into the buffer, that code runs.

If running that code tends to bring in new energy → motif strengthens.

---

# 9. **No Forbidden Magic**

### **Rule F1 — No node types, no edge types, no symbolic labels.**

Everything is untyped.

### **Rule F2 — No special MC functions.**

Only code blocks inside the file.

The graph chooses which blocks to run.

### **Rule F3 — The runner cannot interpret graph state.**

It only:

* watches one buffer

* executes code

* exposes syscalls

All intelligence is inside `melvin.m`.

---

# 🟧 **FINAL SUMMARY (one sentence)**

**Nodes store energy, edges move energy, energy reshapes the structure, structure triggers code execution, code execution changes the world, the world sends new energy back, and circuits that increase future energy survive.**

Everything else—memory, perception, concept, planning, motor control, "intelligence"—emerges automatically.

---


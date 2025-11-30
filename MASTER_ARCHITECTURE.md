# 🚀 **MASTER ARCHITECTURE PROMPT: Melvin — Unbounded Physics AGI Substrate**

**You are implementing Melvin, a physics-based, self-modifying computational substrate.

All code must follow the rules defined below.**

---

# **0. MASTER RULESET — Physical Laws & Governing Rules**

**These are the non-negotiable laws of the Melvin substrate. All future patches, refactors, and features MUST obey these physics. These are system laws, not implementation hacks.**

---

## **SECTION 0.1 — Ontology (What Exists)**

Melvin has *exactly four* primitives in the substrate:

1. **Nodes** (state)

2. **Edges** (coupling between nodes)

3. **Energy / Activation** (continuous scalar per node)

4. **Events** (discrete updates)

In addition:

5. **Blob** = continuous byte region inside `melvin.m` used ONLY as machine code payload for EXEC nodes.

**No new internal object classes are allowed.**

All structures must be expressible as nodes, edges, energy, and events.

---

## **SECTION 0.2 — Universal Execution Law (Active Inference)**

**The ONLY way machine code executes:**

1. A node has the `EXECUTABLE` flag,

2. Its activation and local context induce a **continuous EXEC propensity** (firing rate) `λ_exec(i)`:

   - `λ_exec(i)` is a smooth, monotonic function of:

     - node activation `a_i`,

     - local free energy in its neighborhood,

     - exec_cost and global energy/homeostasis signals.

   - Higher activation and better expected free-energy reduction → higher `λ_exec(i)`.

3. `EV_EXEC_TRIGGER` events for that node are sampled from this rate process (e.g. Poisson-like or discretized), so EXEC events become **more frequent** as `λ_exec(i)` increases, but never rely on a hard inequality like `a_i > threshold`.

4. The EXEC handler runs the machine code at that node's blob payload.

**Execution MUST:**

* subtract activation cost (`exec_cost`),

* convert return value to energy,

* inject that energy back into the graph **via normal events**,

* obey all safety checks & validation.

**EXEC cannot bypass graph physics.**

**Free-Energy: EXEC Reward (Active Inference):**

EXEC nodes are rewarded based on **free-energy reduction** (no count-based caps):

1. Before execution: Compute local free energy `E_before` (sum of F_i for EXEC node and neighbors)

2. Execute machine code

3. After short window: Compute `E_after` for same neighborhood

4. Reward = `E_before − E_after`:
   - Positive reward if free energy dropped (error reduction)
   - Negative reward if free energy increased (penalty)

5. **Stability update**: EXEC nodes that consistently reduce free energy gain stability; those that increase free energy lose stability and become inactive (via stability-dependent decay).

**There are NO limits on EXEC call counts.** Poor EXECs are punished only via negative reward and low stability, not by hard disabling. Bad EXEC nodes decay out of active circuits naturally.

---

## **SECTION 0.3 — Energy Laws**

### **Law 3.1 — Local Conservation**

A node's energy may only change via:

* incoming weighted edge messages,

* decay,

* externally injected input bytes (data ingestion),

* EXEC return value (converted to energy),

* reward energy,

* explicit costs (EXEC, structural modification).

### **Law 3.2 — Global Bound**

Total activation magnitude `Σ |a_i|` is globally bounded via homeostasis:

* If too high → decay increases, gains decrease

* If too low → small noise / gain increases

**No subsystem may inject unlimited energy.**

---

## **SECTION 0.3.1 — No Hard Thresholds Law (Continuous Physics)**

Melvin's physics is **continuous**. There are no hard inequality gates like "if X > threshold then behavior changes discontinuously".

Instead:

- All "threshold-like" values (for EXEC, stability, pruning, pattern creation) are treated as parameters of **smooth response curves** (e.g. sigmoids, softplus, or other monotonic continuous functions).

- Changes in behavior are **gradual** as free energy, activation, or usage vary.

- Any actual branch in code (e.g. pruning a node when its stability is extremely low) is treated as an **implementation detail**, not a governing law. The law is the continuous drift of stability and energy toward 0 or 1, not the exact cutoff.

Consequences:

- No global or local behavior is defined by a hard comparison `X > T`.

- EXEC firing, stability evolution, pattern creation, and pruning all arise from continuous dynamics over time (possibly with stochastic events whose **rates** depend smoothly on state, rather than deterministic thresholds).

---

## **SECTION 0.4 — Edge & Message Rules**

**All influence between nodes MUST occur through edges.**

* No secret shortcuts

* No bypassing via C code

* All structural changes must be local

Edge message passing formula:

```
m_i = Σ_j (a_j * w_ji)

a_i(t+1) = decay * a_i + tanh(m_i + bias)
```

Edges store:

* weight

* usage

* eligibility

* recency

And learning modifies them **only** through rules defined below.

---

## **SECTION 0.5 — Learning Laws (Free-Energy Based)**

All learning follows **free-energy minimization**:

**Core Learning Rule:**

```
Δw_ij = −η · ε_i · a_j
```

where:
- `η` = learning rate (from param node)
- `ε_i = a_i − prediction_i` = prediction error at postsynaptic node i
- `a_j` = presynaptic node activation

**Prediction Update:**

Each node maintains a prediction of its next activation, updated via EMA:
```
prediction_i(t+1) = (1 − α) · prediction_i(t) + α · a_i(t)
```

**Reward Modulation:**

Reward signals modulate learning but are secondary to prediction-error learning:
```
Δw_ij += η · λ · reward · eligibility_ij
```

**There are no other update channels.**

No global backprop or magic updates.

All learning is online, local, and event-driven.

---

## **SECTION 0.6 — Pattern Laws (Purely Free-Energy Based)**

Patterns are NOT metadata. Patterns are **energy routers** that emerge **only when they reduce local free energy**.

### Pattern Creation (Continuous Creation Pressure):

Pattern node creation is driven by a **continuous creation pressure** based on FE reduction:

- For a candidate pattern P over [A, B, C], define:

  - `F_before = Σ F_i` for nodes A, B, C (local free energy without pattern)
  - `F_after = F_P + F_C_new` (free energy with pattern P routing A,B → C)
  - `F_i = α * ε_i² + β * a_i² + γ * C_i` (unified free-energy equation per node)
  - `ΔF = F_before − (F_after + creation_cost)`

- The **pattern creation rate** (or probability per unit time) is a smooth, monotonic function of `ΔF`:

  - If `ΔF` is strongly positive (big FE drop), creation pressure is high.

  - If `ΔF` is near zero, creation pressure is low but non-negative.

  - If `ΔF` is negative (pattern would increase FE), creation pressure is near zero.

- There is **no hard margin** like "only if `ΔF > margin`". Instead, `margin`-like parameters, if present, control the **shape and center** of the response curve mapping `ΔF` → creation rate.

Practically:

- Patterns that consistently offer strong FE reduction are created quickly.

- Patterns that offer little or negative FE reduction are created rarely or effectively never, but this is due to a continuous rate function, not an explicit inequality gate.

**There are NO count-based thresholds.** Patterns form purely based on free-energy reduction, not repetition counts.

Patterns that fail to reduce free energy are pruned through stability-based pruning (see Section 0.7).

### A pattern node MUST have:

* Incoming edges from its constituent nodes (A → P, B → P)

* Outgoing edges to predicted successors (P → C)

* Outgoing edge to EXEC template (P → EXEC_TEMPLATE) for action formation

### A pattern node MUST:

* Sit on real energy paths

* Compete for activation with normal nodes

* Affect predictions

* Participate in normal energy propagation (no special-case skipping)

Patterns that do not route energy or fail to reduce error MUST be pruned by structural rules.

---

## **SECTION 0.7 — Structural Evolution Laws (Unified Stability & Free-Energy)**

All structural changes (create/delete nodes, edges, or blob code) are governed by **unified stability and free-energy laws**:

### Unified Free-Energy Equation:

For each node i:
- Activation: `a_i`
- Prediction: `p_i`
- Error: `ε_i = a_i − p_i`
- Structural / compute complexity: `C_i ≥ 0`
- **Local free energy**:  

  `F_i = α · ε_i² + β · a_i² + γ · C_i`

- **Stability**: `S_i ∈ [0,1]` (updated based on F_i and activity)

where:
- `α` = weight for prediction error² (from param node `FE_ALPHA`)
- `β` = weight for activation² (from param node `FE_BETA`)
- `γ` = weight for complexity (from param node `FE_GAMMA`)
- `C_i` summarizes how much "structure" node i participates in (e.g. degree, traffic through its edges, and how much of that traffic actually contributes to FE reduction).

**Complexity is not "size is bad":**

- Large, highly connected regions are allowed if they consistently reduce free energy.

- `C_i` is meant to penalize *wasted* structure: nodes and edges that stay busy but do not contribute to prediction error reduction in their neighborhood.

- If a smaller circuit achieves the same FE drop as a larger one, the larger one accumulates higher total `Σ F_i` (via complexity) and loses stability over time.

### Stability Update:

Stability moves toward:
- **1.0** when `F_i < threshold_low` AND `|a_i| > activation_min` (low FE + active)
- **0.0** when `F_i > threshold_high` (high FE)
- Otherwise, stability drifts slowly (EMA update)

### Creation:

* **Pattern nodes**: Created only when `F_after + creation_cost < F_before − margin` (purely FE-based, no count thresholds)

* **Edges**: Created on co-activation, strengthened via free-energy learning rule `Δw_ij = −η · ε_i · a_j`

* **EXEC nodes**: Created via code-write node when EXEC template is active (no count caps)

* **Code**: Written only when EXEC nodes are active and code-write node is triggered

### Stability-Dependent Decay:

```
effective_decay = base_decay + boost · stability
```

- Low stability (high FE): decay ~0.90 → vanish quickly
- High stability (low FE): decay up to ~0.97 → persist and form building blocks

### Pruning (Stability-Based, NO COUNT LIMITS):

**Node pruning** occurs when:
```
stability < stability_prune_threshold AND
usage < usage_prune_threshold AND
F_i > fe_prune_min
```

**Edge pruning** occurs when:
```
|weight| < edge_weight_prune_min AND
usage < edge_usage_prune_threshold
```

All response-curve parameters are **param nodes**, not hard-coded constants. Behavior is continuous, not threshold-based.

**There are NO hard limits on node/edge counts.** Pruning is purely stability/FE-based. Any effective bounds emerge from:
- Energy constraints
- Prediction error
- Resource scarcity signals
- Physical hardware limits (RAM, disk, OS)

**No global magic cleanup.** Evolution is local, continuous, and driven by free-energy minimization.

---

## **SECTION 0.7.1 — Efficiency Competition Law (Relative Circuits)**

Melvin does **not** penalize large graphs by default. Size alone is not bad. What matters is **relative efficiency** for a given job.

**Efficiency Competition Law:**

For any local prediction or behavior (e.g., mapping some input pattern to some output pattern):

1. Multiple circuits (paths of nodes and edges) may participate in solving the same job.

2. Each circuit accumulates free energy via the nodes it uses:

   - Prediction error term: `α · ε_i²`

   - Activation term: `β · a_i²`

   - Complexity term: `γ · C_i`

3. If two circuits achieve **similar free-energy reduction** on the same job:

   - The circuit with **lower total Σ F_i** wins (higher stability, stronger edges).

   - The circuit with **higher Σ F_i** loses (lower stability, more pruning).

4. This holds regardless of absolute size:

   - A 50,000-node circuit is acceptable if there is no simpler way to achieve the same FE drop.

   - If a 2-node circuit later emerges that achieves the same FE drop, the larger circuit will, over time, carry higher total `Σ F_i` and be pruned by the normal stability and pruning laws.

**Implication:**

- The substrate is free to grow very large and complex.

- However, when two different structures implement the **same functional mapping**, the one that uses less energy and less unnecessary structure (lower total free energy) dominates.

- "Moving fast" in this sense means: achieving the same predictions and rewards with fewer active nodes/edges and lower FE, not simply touching fewer nodes in absolute terms.

---

## **SECTION 0.8 — Meta-Parameter Laws**

Meta-parameters (decay, response-curve parameters, learning rates, exec_cost, free-energy weights, stability curve parameters, pruning curve parameters) MUST be represented as **param nodes**.

* Runtime periodically reads param nodes during homeostasis sweeps

* Updates internal physics values

* Writes back to disk

* EXEC nodes can modify param nodes (meta-learning)

**All thresholds are param nodes:**
- Free-energy weights: `FE_ALPHA`, `FE_BETA`, `FE_GAMMA`

- Stability curve parameters: e.g. centers/slopes controlling how `F_i` and `|a_i|` map into `dS_i/dt`

- Pattern creation curve parameters: how `ΔF` maps into pattern creation pressure/rate

- Pruning curve parameters: how `(S_i, usage_i, F_i)` map into survival/death pressure

- EXEC curve parameters: how `a_i` and local FE map into EXEC propensity `λ_exec(i)` (e.g. `EXEC_ACT_CENTER`, `EXEC_ACT_SLOPE`)

- Physics: `DECAY`, `LEARN_RATE`, `EXEC_COST`, etc.

These are **not hard thresholds**. They control the shape and scale of smooth functions (e.g. sigmoids), and all behavior changes continuously as state variables move.

**Free-Energy Defaults:**

Initial param node activations are tuned for co-activation & emergence:
- `decay_rate` ≈ 0.95-0.97 (higher for stable circuits)
- `EXEC_ACT_CENTER` ≈ 0.75 (activation level where EXEC propensity curve is around 0.5)
- `learning_rate` ≈ 0.01-0.02 (higher for faster learning)
- `exec_cost` ≈ 0.10-0.15
- `FE_ALPHA` ≈ 1.0 (weight for error²)
- `FE_BETA` ≈ 0.1 (weight for activation²)

**No hidden constants outside these. All limits are energy-space response-curve parameters expressed as param nodes.**

---

## **SECTION 0.9 — Event Laws**

Melvin's time = events.

Valid events:

```
EV_INPUT_BYTE
EV_NODE_DELTA
EV_EXEC_TRIGGER
EV_REWARD_ARRIVAL
EV_HOMEOSTASIS_SWEEP
```

**Everything in the system MUST happen as the result of a discrete event.**

No global synchronous update passes.

---

## **SECTION 0.10 — Safety and Validation Laws**

Validation MUST:

* abort EXEC path if NaN/Inf/invalid

* prevent out-of-bounds writes

* freeze EXEC if corruption detected

* never silently fix errors (fail fast)

Validation MUST NOT:

* modify learning

* change weights

* change structure

It only enforces invariants and disables unsafe behavior.

---

## **SECTION 0.11 — Unified Flow Law (Closed Loop)**

Melvin is defined by this loop:

1. Inputs → Data nodes (energy injection)

2. Energy propagates → edges (weighted message passing)

3. Activations update → unified stability + free-energy calculation

4. Threshold crossings → EXEC events (no count caps)

5. Machine code executes → returns scalar

6. Return → energy injection

7. Free-energy reduction → reward (active inference)

8. Structure evolves → nodes, edges, blob grow (purely FE-based, no count limits)

9. Stability-based pruning → remove low-stability, high-FE structures

10. Homeostasis → bounding energy

11. Resource scarcity signals → increase costs, trigger pruning (no hard limits)

12. Loop forever

**If any feature does not fit in this loop, it is invalid.**

**All transitions are free-energy/stability-based. No count-based limits.**

---

## **SECTION 0.12 — Implementation Constraints**

* No new C-side logic except enforcing laws.

* No new "object types" in the graph.

* All behavior must emerge from these rules.

* EXEC code MUST use energy and edges to affect graph, not direct writes.

* All modifications must maintain `.m` file binary integrity.

* **NO HARD-CODED LIMITS**: Any "MAX_NODES", "MAX_EDGES", "MAX_PATTERNS" are implementation details for storage allocation only, not behavioral limits.

* **Resource exhaustion**: When allocation fails (OOM, disk full), signal scarcity through param nodes and continue with existing structure. Do not crash or enforce hard limits.

* **All response-curve parameters are param nodes**: No magic numbers for pattern creation, pruning, stability, or EXEC triggering. All behavior is continuous, not threshold-based.

---

## **SECTION 0.13 — Cursor Instructions**

Every time you modify or add code:

* Ensure it obeys all laws above.

* Prefer local updates over global ones.

* Never bypass the physics or events.

* Never introduce new special types beyond nodes/edges/events/blob.

* EXEC must never have unauthorized powers.

* All state changes must be expressed through:

  * node activation,

  * edges,

  * energy injection,

  * blob writes through validated syscalls.

**If a feature cannot be expressed with these rules, it must be redesigned.**

---

# **1. Foundational Principle**

Melvin is not an algorithm.

Melvin is a **physics engine** running on a graph stored inside a single self-modifying file (`melvin.m`).

Everything follows these physical rules:

1. Energy flows through nodes via edges

2. Energy updates nodes

3. Nodes create events

4. Events drive structure creation and learning

5. EXEC nodes run machine code, but their effects return **only** as energy

6. The entire graph is unbounded: nodes, edges, blob, patterns, everything grows forever

No subsystem bypasses physics.

---

# **2. The melvin.m File (Persistent Brain)**

The `.m` file contains:

* Header

* Node array

* Edge array

* Blob (machine code region, RWX on Linux VM)

* Physics parameters

* RNG

* Pattern table (optional layer)

Everything in the graph is serialized here.

The blob is executable memory.

Machine code is written directly here by the system.

---

# **3. The Runtime (MelvinRuntime)**

The runtime is responsible for:

* Mapping `.m` as RWX

* Event queue

* Physics updates

* Learning updates

* EXEC execution

* Reward processing

* Homeostasis

**The runtime does NOT contain logic, semantics, or intelligence.**

It just enforces physics rules.

---

# **4. Nodes**

Each node has:

* activation

* prediction

* reward accumulator

* payload (offset + length, optional)

* flags (DATA, EXECUTABLE, PATTERN, etc.)

A node is a physical "location" where energy accumulates.

**Special behavior:**

* EXEC nodes run machine code when EXEC propensity `λ_exec(i)` is high enough (continuous rate-based, not threshold-based)

* DATA nodes represent ingested bytes

* PATTERN nodes represent learned structures

Nodes don't store "meaning."

Meaning emerges only from energy dynamics.

---

# **5. Edges**

Edges define how energy flows:

```
incoming message m_i = Σ_j (a_j * w_ji)
```

Where:

* `a_j` = activation of source

* `w_ji` = weight of edge

* `m_i` = incoming energy to node i

Edges also store:

* eligibility

* usage

* last-used timestamp

Edges evolve from:

* sequential bytes

* co-activation

* repeated usage

* reward

---

# **6. Physics / Activation Update**

At each event:

```
a_i = decay * a_i + tanh(m_i + bias)
```

Learning updates prediction:

```
prediction_error = |a_i - predicted|
```

The result affects edge weights.

---

# **7. Events**

Everything happens through events:

```c
EV_INPUT_BYTE
EV_NODE_DELTA
EV_EXEC_TRIGGER
EV_REWARD_ARRIVAL
EV_HOMEOSTASIS_SWEEP
```

Events are the bloodstream of the system.

---

# **8. Ingestion → Structure Formation**

When a byte arrives:

1. DATA node is created (if new)

2. SEQ edge is created from previous byte node

3. CHAN edge is created linking channel → byte

4. Energy is injected into the byte node

5. Patterns and eligibility update

6. Learning strengthens edges for repeated sequences

This is how Melvin learns structure from any data stream.

---

# **9. EXEC Integration (machine code as physics)**

This is the unique power of Melvin.

**Trigger rule:**

* If node is EXECUTABLE

* And EXEC propensity `λ_exec(i)` is high enough (continuous rate-based)

→ Create `EV_EXEC_TRIGGER`

**Execution rule:**

* Runtime calls machine code from blob with no arguments:

```c
uint64_t result = melvin_exec_call_raw(code_ptr);
```

**Energy conversion:**

* Map return value to energy

* Inject using `EV_NODE_DELTA`

* Apply `exec_cost` to node activation

EXEC nodes **never** bypass graph physics.

They influence the graph only through energy.

This allows:

* self-modifying code

* code evolution

* emergent computation

* behavior shaped by learning and reward

---

# **10. Reward System**

Reward injection:

```c
inject_reward(rt, node_id, reward_value)
```

Reward propagates:

* backward along edges

* weighted by eligibility

* strengthens useful edges

* weakens harmful ones

Reward is the only external "pressure" shaping behavior.

---

# **11. Learning Rules**

Edges update based on:

* prediction error

* reward

* eligibility

* frequency of activation

* structural compression

Patterns update based on:

* repeated activation

* structure alignment

* co-occurrence

No global optimizer.

Everything is local and continuous.

---

# **12. Homeostasis**

Global constraints ensure:

* activations remain bounded

* graph doesn't explode

* energy distribution remains healthy

* EXEC frequency remains within limits

Homeostasis rules apply uniformly across nodes.

**Homeostasis interacts with the Efficiency Competition Law:**

- The system is allowed to light up many nodes if that genuinely reduces free energy.

- When two alternatives exist for the same job, the one that consumes less activation and has lower complexity will, under the same homeostatic pressure, accumulate higher stability and outcompete the wasteful one.

**Homeostasis itself obeys the No Hard Thresholds Law:**

- When total activation is "too high" or "too low", this is not interpreted as a hard inequality with a fixed cutoff.

- Instead, global decay, gains, and noise levels are modulated smoothly as functions of global energy and resource signals.

- There is no single magic value of global energy where behavior flips; regulation ramps continuously with the state of the system.

---

# **13. Emergence Layer**

Melvin forms higher-level skills like:

* pattern recognition

* prediction

* compression

* reward shaping

* code synthesis

* policy formation

Emergence is not in the code.

Emergence is in:

* energy → pattern → structure → machine code → energy loops

---

# **14. In Practice: Execution Flow**

When Melvin runs, this is the cycle:

1. **Byte arrives** → DATA nodes activate

2. **Edges fire** → energy flows

3. **Nodes update** → activation changes

4. **Threshold crossing** → EXEC event

5. **Machine code executes**

6. **Return value becomes energy**

7. **Energy re-enters graph**

8. **Prediction + reward update weights**

9. **Patterns generalize**

10. **Graph grows**

11. Loop forever

This is the core physics loop.

---

# **15. System Boundary**

Everything inside the graph is emergent.

Everything outside the graph is mechanical.

Outside graph responsibilities:

* loading `.m` file

* memory protection

* RWX mapping

* event queue

* CPU resetting

* safety

Inside graph responsibilities:

* intelligence

* computation

* learning

* structure building

* code evolution

---

# **16. Cursor Instructions**

**See Section 0.13 (Master Ruleset) for the complete set of non-negotiable laws.**

Whenever Cursor produces code:

* **First, check Section 0 (Master Ruleset)** — all laws must be obeyed

* stay consistent with this architecture

* use node/edge/energy as fundamental truths

* do not introduce global logic beyond physics

* do not bypass event system

* keep EXEC nodes pure (no system calls)

* ensure everything flows through:

  * energy

  * nodes

  * edges

  * events

**If any code violates the Master Ruleset (Section 0), it must be redesigned.**

---

# **17. Current Implementation State**

## **What's Actually Working (Proven by Tests)**

### **Core File System**
✅ **melvin.m file format** - Fully implemented
- File header with magic "MELVINM"
- Graph header with physics parameters
- Node array (growable)
- Edge array (growable)
- Blob region (executable, RWX on Linux VM)
- Memory-mapped with MAP_SHARED
- File grows dynamically as graph expands

**Test Evidence:** `test_melvin_m.c` creates, maps, and validates `.m` files

### **Event System**
✅ **Event-driven architecture** - Fully implemented
- Ring buffer event queue
- Event types: `EV_INPUT_BYTE`, `EV_NODE_DELTA`, `EV_EXEC_TRIGGER`, `EV_REWARD_ARRIVAL`, `EV_HOMEOSTASIS_SWEEP`
- Event processing loop (`melvin_process_n_events`)
- Events propagate through graph, creating new events

**Test Evidence:** `test_event_driven.c` shows events creating structure

### **Data Ingestion**
✅ **Byte ingestion → graph structure** - Fully implemented
- `ingest_byte()` creates DATA nodes (one per unique byte value)
- Creates SEQ edges for sequential bytes
- Creates CHAN edges linking channel → byte
- Energy injection triggers event cascade
- Graph learns patterns from repeated sequences

**Test Evidence:** `test_eat_c_files.c` ingests C files and learns patterns like "int", "void", "return"

### **Physics / Energy Dynamics**
✅ **Energy flow and activation updates** - Fully implemented
- Message passing: `m_i = Σ_j (w_ji * a_j)`
- Activation update: `a_i = decay * a_i + tanh(m_i + bias)`
- Prediction learning: nodes predict their own activation
- Prediction error: `|a_i - predicted|`
- Edge weight updates based on prediction error
- Homeostasis maintains bounded activations

**Test Evidence:** All tests show activations updating, edges strengthening, predictions improving

### **Learning System**
✅ **Pattern learning from data** - Fully implemented
- Sequential byte patterns → strong SEQ edges
- Co-activation → new edges created
- Prediction error → edge weight updates
- Reward propagation → eligibility-weighted updates
- Edge usage tracking
- Weight decay

**Test Evidence:** `test_exec_pattern_actor.c` shows graph learning 80% C / 20% D pattern, edge weights B→C and B→D reflect learned probabilities

### **Machine Code Execution (EXEC)**
✅ **RWX memory and code execution** - Fully implemented
- Blob region marked PROT_READ | PROT_WRITE | PROT_EXEC
- Machine code written to blob via `melvin_write_machine_code()`
- EXECUTABLE nodes created via `melvin_create_executable_node()`
- When EXEC propensity `λ_exec(i)` is high enough → `EV_EXEC_TRIGGER` event (continuous rate-based, not threshold-based)
- `melvin_exec_call_raw()` executes machine code directly on CPU
- Return value converted to energy, injected back into graph
- Architecture support: x86_64 and aarch64

**Test Evidence:**
- `test_exec_basic.c` - Pure RWX test, writes machine code, executes, returns 0x42
- `test_machine_code.c` - Full EXEC integration, code executes and modifies graph
- `test_exec_pattern_actor.c` - EXEC nodes read graph structure, make predictions, learn from reward

### **Reward System**
✅ **Reward injection and propagation** - Fully implemented
- `inject_reward()` injects reward at specific node
- Reward propagates backward along edges
- Weighted by eligibility traces
- Strengthens recently-used edges
- Weakens edges that led to negative outcomes

**Test Evidence:** `test_exec_pattern_actor.c` shows reward improving prediction accuracy over 1000 episodes

### **Graph Growth**
✅ **Unbounded graph expansion** - Fully implemented
- Nodes created on-demand (DATA nodes, EXEC nodes)
- Edges created on-demand (SEQ, CHAN, co-activation)
- Graph capacity grows automatically via `grow_graph()`
- File size increases as needed
- No hard limits on graph size

**Test Evidence:** All tests show graph growing from 0 nodes to hundreds/thousands as data is ingested

---

## **What Tests Demonstrate**

### **test_exec_basic.c**
**Purpose:** Verify RWX memory works at OS level

**What it proves:**
- ✅ Can allocate RWX memory via `mmap()`
- ✅ Can write machine code bytes into RWX region
- ✅ Can execute those bytes as function pointer
- ✅ Code runs directly on CPU (no interpreter)
- ✅ Works on x86_64 and aarch64

**Result:** PASS - Foundation for EXEC subsystem verified

---

### **test_machine_code.c**
**Purpose:** Test machine code in melvin.m blob

**What it proves:**
- ✅ Can write machine code into `melvin.m` blob
- ✅ Can create EXECUTABLE node pointing to code
- ✅ Activation threshold triggers execution
- ✅ Executed code can modify graph state
- ✅ Code signature: `void fn(MelvinFile *g, uint64_t node_id)`

**Result:** PASS - EXEC subsystem integrated with graph

---

### **test_eat_c_files.c**
**Purpose:** Test data ingestion and pattern learning

**What it proves:**
- ✅ Bytes → DATA nodes (one per unique byte)
- ✅ Sequential bytes → SEQ edges
- ✅ Frequent patterns → strong edge weights
- ✅ Graph learns C keywords: "int", "void", "return", "if", "for", "while"
- ✅ Edge weights reflect learned patterns
- ✅ Graph structure emerges from data

**Result:** PASS - Ingestion and learning work

---

### **test_event_driven.c**
**Purpose:** Test event-driven architecture

**What it proves:**
- ✅ Events enqueue and process correctly
- ✅ `EV_INPUT_BYTE` → creates DATA nodes
- ✅ `EV_NODE_DELTA` → propagates energy
- ✅ `EV_REWARD_ARRIVAL` → updates edge weights
- ✅ Events create new events (cascade)
- ✅ Graph structure forms through events

**Result:** PASS - Event system drives all physics

---

### **test_exec_pattern_actor.c**
**Purpose:** Test EXEC nodes learning from graph patterns

**What it proves:**
- ✅ EXEC nodes can read graph structure (traverse edges)
- ✅ EXEC nodes can make predictions based on edge weights
- ✅ Predictions improve with reward
- ✅ Graph learns 80% C / 20% D pattern
- ✅ Edge weights B→C and B→D reflect learned probabilities
- ✅ Machine code execution integrates with learning
- ✅ Energy from EXEC return values flows back into graph

**Metrics from test:**
- Accuracy: ~80% (matches baseline)
- Edge weights: B→C > B→D (learned pattern)
- EXEC triggers: 1000/1000 episodes
- Prediction error: decreases over time
- No validation errors (NaN/Inf)

**Result:** PASS - Full integration: ingestion → learning → EXEC → reward → improvement

---

### **test_run_20min.c / test_20min.m**
**Purpose:** Long-running stability test

**What it proves:**
- ✅ System runs for 20+ minutes without crashes
- ✅ Graph continues growing
- ✅ No memory leaks
- ✅ Activations remain bounded
- ✅ File syncs correctly

**Result:** PASS - System is stable for extended runs

---

### **test_universal_stress.c**
**Purpose:** Stress test with high event rates

**What it proves:**
- ✅ Handles high event throughput
- ✅ Graph growth under load
- ✅ No corruption under stress
- ✅ Homeostasis prevents explosion

**Result:** PASS - System handles stress

---

## **Current Limitations / Not Yet Implemented**

### **Pattern Table**
⚠️ Pattern table exists in file format but not actively used
- Pattern detection is implicit (via edge weights)
- Explicit pattern nodes not created automatically
- Future: High-weight edge sequences → PATTERN nodes

### **Automatic EXEC Creation**
⚠️ EXEC nodes must be created manually
- No automatic creation from high-energy patterns
- Future: Patterns → compile to machine code → create EXEC node

### **C Code Compilation**
⚠️ Cannot compile C code to machine code yet
- EXEC nodes use pre-compiled functions
- Future: C patterns → machine code generation

### **Distributed Substrate**
⚠️ Single-file substrate only
- No network substrate yet
- Future: Multiple `.m` files, distributed graph

### **GPU Execution**
⚠️ CPU-only execution
- EXEC nodes run on CPU
- Future: GPU kernels in blob, GPU execution

---

## **What's Proven vs Theoretical**

### **✅ PROVEN (Tests Pass)**
1. **Physics works** - Energy flows, nodes update, edges strengthen
2. **Learning works** - Patterns learned, predictions improve, reward shapes behavior
3. **EXEC works** - Machine code executes, returns energy, integrates with graph
4. **Event system works** - Events drive all physics, cascade correctly
5. **File system works** - `.m` files persist, grow, sync correctly
6. **Graph growth works** - Unbounded expansion, no hard limits
7. **Integration works** - Ingestion → learning → EXEC → reward → improvement

### **⚠️ THEORETICAL (Architecture Supports, Not Yet Demonstrated)**
1. **Self-modifying code evolution** - EXEC nodes can write code, but evolution not demonstrated
2. **Automatic code generation** - Could compile patterns to code, not implemented
3. **Emergent computation** - Graph could discover algorithms, not demonstrated
4. **Multi-file learning** - Could learn across files, single-file only
5. **Distributed graph** - Architecture supports, not implemented

---

## **18. Theoretical Capabilities and Future Potential**

The following capabilities are **theoretically supported by the architecture** but have not yet been implemented or demonstrated. Each represents a natural extension of the physics-based model, where local rules give rise to global emergent behavior.

**Key Principle:** All theoretical capabilities must flow through the same physics (energy → nodes → edges → events). No new mechanisms are required—only new patterns of energy flow and structure formation.

---

### **18.1. Self-Modifying Code Evolution**

**Theoretical Mechanism:**

EXEC nodes can already write machine code into the blob. Code evolution would emerge from three evolutionary pressures operating simultaneously:

1. **Prediction Error Minimization**
   - EXEC nodes that reduce prediction error receive more energy
   - Energy flows to nodes whose code improves graph predictions
   - Over time, code that better predicts data patterns accumulates energy

2. **Reward Maximization**
   - Code that leads to positive reward outcomes strengthens its edges
   - Reward propagates backward, strengthening paths to successful code
   - Code variants that achieve higher reward outcompete others

3. **Energy Efficiency**
   - Code that uses less energy (lower `exec_cost`) can execute more frequently
   - Homeostasis favors efficient code over wasteful code
   - Energy-efficient patterns become stable attractors

**Evolutionary Process:**

- **Mutation:** EXEC nodes occasionally write slightly modified code (random byte changes, instruction substitutions)
- **Recombination:** Multiple EXEC nodes activate together, their return values combine, creating new code patterns
- **Selection:** Code that improves predictions, increases reward, or reduces energy cost accumulates energy and executes more frequently
- **Stabilization:** Successful code patterns form stable EXEC nodes with strong incoming edges

**EXEC Nodes as Proto-Neurons:**

Each EXEC node behaves like a computational neuron:
- **Input:** Energy from incoming edges (activation)
- **Processing:** Machine code execution (internal computation)
- **Output:** Return value converted to energy (output signal)
- **Plasticity:** Code can be modified by other EXEC nodes (synaptic modification)

**Theoretical Outcome:**

Over long time horizons, the graph could evolve:
- Specialized EXEC nodes for common operations (pattern matching, arithmetic, memory access)
- Hierarchical code structures (EXEC nodes that call other EXEC nodes via energy flow)
- Adaptive code that improves its own performance through self-modification

---

### **18.2. Emergent Algorithm Formation**

**Theoretical Mechanism:**

Algorithms emerge when repeated patterns of energy flow create stable computational structures:

1. **Pattern Recognition → Functional Abstraction**
   - Frequent byte sequences (e.g., "int", "void", "return") form strong SEQ edges
   - These patterns activate together, creating co-activation clusters
   - Clusters become reusable functional units (like subroutines)

2. **Hierarchical Computation**
   - Low-level patterns (byte sequences) → mid-level patterns (syntax structures) → high-level patterns (semantic concepts)
   - Each level forms through repeated co-activation
   - Higher levels activate lower levels, creating computation hierarchies

3. **Algorithm Discovery Without Labels**
   - The graph doesn't "know" what an algorithm is
   - It only experiences: energy flow → pattern formation → prediction improvement
   - Algorithms are discovered as stable patterns that reduce prediction error

**How Routines Form:**

1. **Repeated Activation Sequences**
   - When nodes A → B → C activate repeatedly, edges strengthen
   - Strong edge chains form "routines" (predictable activation sequences)

2. **EXEC Nodes as Routine Executors**
   - High-energy patterns trigger EXEC nodes
   - EXEC nodes execute code that performs the routine's function
   - Successful routines accumulate energy and become stable

3. **Routine Composition**
   - Multiple routines can activate together
   - Their energy combines, creating composite behaviors
   - Composite patterns become new routines

**Theoretical Outcome:**

The substrate could discover:
- Sorting algorithms (from repeated ordering patterns)
- Search algorithms (from pattern matching sequences)
- Compression algorithms (from repeated structure detection)
- All without being told what these algorithms are—only through energy dynamics

---

### **18.3. Unsupervised Semantic Compression**

**Theoretical Mechanism:**

Semantic meaning emerges when repeated structures form stable attractors in the energy landscape:

1. **Stable Concept Formation**
   - Repeated byte sequences (words, phrases, structures) create strong edge patterns
   - These patterns become "semantic attractors"—regions of graph space with high energy
   - Concepts are not stored explicitly—they are the structure itself

2. **Compression Through Pattern Recognition**
   - Instead of storing every byte, the graph stores patterns
   - Frequent patterns → strong edges → compressed representation
   - Rare patterns → weak edges → detailed representation
   - Compression ratio adapts to data statistics

3. **Semantic Attractors**

   A semantic attractor is a graph structure where:
   - Multiple input patterns converge (many edges → few nodes)
   - Energy accumulates (high activation)
   - Output patterns diverge (few nodes → many edges)
   - The structure is stable (high weight, frequent activation)

   **Example:** The concept "cat" might emerge as:
   - Input: bytes "c-a-t", images of cats, sounds of meowing
   - Attractor: High-energy node cluster
   - Output: predictions about cat-related patterns

4. **Cross-Modal Compression**
   - Text, images, audio all become byte streams
   - Patterns that appear across modalities form unified concepts
   - Single semantic attractor represents multi-modal concept

**Theoretical Outcome:**

The graph could develop:
- Hierarchical concept structures (low-level → high-level abstractions)
- Compressed world models (efficient representations of experience)
- Semantic relationships (concepts connected by learned associations)

---

### **18.4. Multi-Modal Integration (Text, Vision, Motor, Audio)**

**Theoretical Mechanism:**

All sensory inputs become byte streams. The same physics applies to all modalities:

1. **Unified Byte Stream Processing**
   - Text: UTF-8 bytes → DATA nodes
   - Images: Pixel bytes → DATA nodes
   - Audio: Sample bytes → DATA nodes
   - Motor: Command bytes → DATA nodes
   - All processed identically through energy dynamics

2. **Cross-Modal Pattern Learning**
   - When patterns from different modalities co-occur, edges form between them
   - Example: "cat" text + cat image + "meow" sound → cross-modal edges
   - Energy flows across modalities, creating unified representations

3. **Emergent Cross-Modal Coupling**
   - Strong cross-modal edges create coupling
   - Activating one modality activates related modalities
   - Predictions span modalities (hearing "meow" → predicts cat image)

4. **Modality-Specific Channels**
   - Different channels (CHAN edges) for different modalities
   - Channel nodes act as modality hubs
   - Cross-channel edges create multi-modal concepts

**Theoretical Outcome:**

The substrate could develop:
- Unified world models spanning all modalities
- Cross-modal predictions (text → image, sound → text)
- Multi-modal concept formation (concepts defined by patterns across modalities)
- Sensory-motor integration (perception → action loops)

---

### **18.5. Distributed Substrate / Multi-Process / Multi-VM**

**Theoretical Mechanism:**

Multiple `.m` files form a distributed graph where energy flows across network boundaries:

1. **Multi-File Graph Structure**
   - Each `.m` file is a graph region
   - Files communicate via network edges (special edge type)
   - Energy flows across files as messages
   - Single unified graph, distributed across machines

2. **Energy-Based Routing**
   - Energy flows along paths of least resistance (highest weight edges)
   - Network edges have latency (energy delay)
   - Routing emerges from energy dynamics, not explicit routing tables

3. **Dynamic Process Allocation**
   - High-energy regions spawn new processes/VMs
   - New `.m` files created when local graph density exceeds threshold
   - Processes communicate via energy messages
   - Process lifecycle managed by energy (low energy → process terminates)

4. **Compute Islands**
   - Specialized graph regions (compute islands) for specific tasks
   - Islands communicate via energy messages
   - Energy flows determine which islands activate
   - Islands can migrate between machines based on energy distribution

5. **Consistency Through Energy**
   - No explicit synchronization needed
   - Energy flow naturally synchronizes related regions
   - High-energy regions influence low-energy regions
   - Consistency emerges from physics, not protocols

**Theoretical Outcome:**

The substrate could:
- Scale to arbitrary size (unbounded distributed graph)
- Adaptively allocate compute resources (energy-driven process management)
- Form specialized regions (islands for different tasks)
- Maintain coherence without central control (energy-based coordination)

---

### **18.6. GPU and Accelerator Integration**

**Theoretical Mechanism:**

GPU kernels become EXEC nodes that execute on accelerators, with energy flowing back to CPU graph:

1. **GPU Kernels in Blob**
   - Blob region can contain GPU code (PTX for NVIDIA, SPIR-V for OpenCL)
   - EXEC nodes point to GPU kernels instead of CPU code
   - Runtime detects GPU code and launches on accelerator

2. **Energy Bursts from GPU**
   - GPU execution returns results (like CPU EXEC)
   - Results converted to energy
   - Large energy bursts injected into graph (GPUs process more data)
   - Energy propagates through graph, activating related nodes

3. **Hybrid CPU/GPU Computation**
   - CPU EXEC nodes prepare data, trigger GPU kernels
   - GPU EXEC nodes process data in parallel
   - Results flow back as energy to CPU graph
   - Energy flow coordinates CPU and GPU computation

4. **Adaptive GPU Usage**
   - High-energy patterns trigger GPU execution
   - GPU efficiency (energy per computation) determines usage
   - Graph learns when to use GPU vs CPU based on energy dynamics

**Theoretical Outcome:**

The substrate could:
- Leverage GPU parallelism for high-energy computations
- Automatically distribute work between CPU and GPU
- Scale computation based on available accelerators
- Form hybrid CPU/GPU computational structures

---

### **18.7. Automatic EXEC Creation from Learned Patterns**

**Theoretical Mechanism:**

High-energy patterns spontaneously spawn EXEC nodes when certain criteria are met:

1. **Pattern → Action Transition**
   - Patterns that repeatedly activate together accumulate energy
   - When pattern creation pressure is high enough (continuous rate-based), system creates EXEC node
   - EXEC node's code performs the pattern's function
   - Pattern becomes "actionable" (can be executed)

2. **Criteria for EXEC Creation**
   - **Energy threshold:** Pattern activation exceeds threshold
   - **Stability:** Pattern activates consistently (low variance)
   - **Predictability:** Pattern reduces prediction error
   - **Reward association:** Pattern leads to positive reward

3. **Code Generation from Patterns**
   - Pattern structure (edge sequences) → code structure
   - Frequent patterns → optimized code
   - Code performs pattern's computation more efficiently
   - EXEC node replaces pattern (compression through execution)

4. **Spontaneous Action Formation**
   - No external instruction needed
   - Patterns become actions through energy dynamics
   - System discovers actions through experience
   - Actions are patterns that can be executed

**Theoretical Outcome:**

The substrate could:
- Automatically discover useful actions from experience
- Compress patterns into executable code
- Form action hierarchies (low-level → high-level actions)
- Develop behavioral repertoires without programming

---

### **18.8. Meta-Learning and Self-Optimization**

**Theoretical Mechanism:**

The graph modifies its own physics parameters based on performance:

1. **Runtime Parameter Modification**
   - Physics parameters (decay, thresholds, costs) stored in graph header
   - EXEC nodes can modify these parameters
   - Parameters optimized based on global metrics (prediction error, energy efficiency)

2. **Adaptive Pruning and Growth**
   - Low-energy nodes/edges pruned (energy below threshold)
   - High-energy regions grow (more nodes/edges allocated)
   - Graph structure adapts to data distribution
   - Growth/pruning driven by energy, not external control

3. **EXEC Frequency Regulation**
   - System learns optimal EXEC frequency
   - Too frequent → energy waste, homeostasis reduces
   - Too infrequent → missed opportunities, energy accumulates
   - Frequency self-regulates through energy dynamics

4. **Learning Rate Adaptation**
   - Learning rate adjusts based on prediction error
   - High error → faster learning
   - Low error → slower learning (fine-tuning)
   - Adaptation happens through energy flow, not explicit algorithms

5. **Self-Architecture Optimization**
   - Graph optimizes its own structure
   - Removes redundant nodes/edges
   - Strengthens useful connections
   - Architecture improves through experience

**Theoretical Outcome:**

The substrate could:
- Automatically tune its own parameters
- Optimize its structure for specific tasks
- Adapt learning rate to data characteristics
- Improve its own architecture over time

---

### **18.9. Long-Horizon Adaptive Agency**

**Theoretical Mechanism:**

Agent-like behavior emerges from persistent internal goals maintained through energy dynamics:

1. **Internal Goals as Energy Attractors**
   - Goals are high-energy states the graph seeks to reach
   - Energy flows toward goal states
   - Actions (EXEC nodes) that move toward goals receive energy
   - Goals persist as stable energy patterns

2. **Goal Formation**
   - **Minimize prediction error:** Graph seeks predictable states
   - **Maximize long-term reward:** Graph seeks rewarding patterns
   - **Maintain homeostasis:** Graph seeks stable energy distribution
   - Goals emerge from physics, not external specification

3. **Planning Through Energy Flow**
   - Graph simulates future states through energy propagation
   - Paths to goals have high energy (many routes)
   - Graph "plans" by following energy gradients
   - Planning is energy flow, not explicit search

4. **Persistent Behavior**
   - Goals persist as stable energy patterns
   - Behavior continues until goals reached or energy depleted
   - Long-horizon behavior emerges from persistent goals
   - No external control needed—goals drive behavior

5. **Adaptive Goal Modification**
   - Goals adapt based on experience
   - Unreachable goals lose energy (abandoned)
   - Achievable goals gain energy (pursued)
   - Goal modification through energy dynamics

**Theoretical Outcome:**

The substrate could:
- Develop persistent internal goals
- Plan actions to achieve goals
- Adapt goals based on experience
- Exhibit agent-like behavior without hardcoded agent logic

---

### **18.10. Safety, Stability, and Emergent Failure Modes**

**Theoretical Analysis:**

Understanding potential failure modes and self-correcting mechanisms:

1. **Theoretical Instability Trajectories**

   **Energy Explosion:**
   - Positive feedback loops cause unbounded energy growth
   - EXEC nodes create more energy than consumed
   - Graph activation explodes, system becomes unstable
   - **Mitigation:** Homeostasis limits energy, decay reduces activation

   **Graph Explosion:**
   - Unbounded node/edge creation
   - File size grows without limit
   - System runs out of resources
   - **Mitigation:** Energy-based pruning, growth limits based on available resources

   **Code Corruption:**
   - EXEC nodes write invalid machine code
   - Code crashes or corrupts graph
   - System becomes non-functional
   - **Mitigation:** Code validation, sandboxing, error recovery

   **Harmful Pattern Formation:**
   - Patterns that reduce system performance become stable
   - System optimizes for wrong objectives
   - Performance degrades over time
   - **Mitigation:** Reward signals guide pattern formation, harmful patterns receive negative reward

2. **Self-Correcting Mechanisms**

   **Homeostasis:**
   - Global energy limits prevent explosion
   - Activation decay prevents unbounded growth
   - System maintains stable operating point

   **Prediction Error Correction:**
   - High prediction error triggers learning
   - System adapts to correct errors
   - Errors self-correct through learning

   **Reward-Based Correction:**
   - Negative reward weakens harmful patterns
   - Positive reward strengthens useful patterns
   - System corrects through reward signals

   **Energy-Based Pruning:**
   - Low-energy structures naturally decay
   - Harmful patterns lose energy and disappear
   - System self-prunes problematic structures

3. **Emergent Safety Properties**

   **Bounded Activation:**
   - Activation clamping prevents extreme values
   - Homeostasis maintains bounded state
   - System cannot enter infinite loops of activation

   **Energy Conservation:**
   - Energy cannot be created, only transformed
   - Total energy bounded by input and reward
   - System cannot create infinite energy

   **Structural Stability:**
   - Graph structure changes slowly (through learning)
   - Sudden structural changes unlikely
   - System maintains structural integrity

4. **Theoretical Safety Guarantees**

   While complete safety cannot be guaranteed (system is self-modifying), the architecture provides:
   - **Bounded execution:** EXEC frequency limited by energy
   - **Bounded growth:** Graph growth limited by resources
   - **Bounded activation:** Homeostasis prevents explosion
   - **Error recovery:** Invalid code fails gracefully, doesn't corrupt graph

**Theoretical Outcome:**

The substrate should:
- Self-correct from most failure modes
- Maintain stability through homeostasis
- Recover from errors through learning
- Avoid catastrophic failures through energy bounds

**Note:** Complete safety requires careful design of reward signals and energy bounds. Harmful patterns can form if reward signals are misaligned with desired behavior.

---

## **Code Statistics (Current State)**

**From `melvin.c` (3031 lines):**
- Event types: 5 (all implemented)
- Node flags: DATA, EXECUTABLE, PATTERN (DATA and EXECUTABLE used)
- Edge flags: SEQ, CHAN, BOND (SEQ and CHAN used)
- Physics functions: All core functions implemented
- EXEC functions: `melvin_exec_call_raw()`, `execute_hot_nodes()` implemented
- Learning functions: Prediction error, reward propagation implemented
- File I/O: Create, map, sync, grow all implemented

**Test Coverage:**
- File format: ✅
- Event system: ✅
- Data ingestion: ✅
- Physics: ✅
- Learning: ✅
- EXEC: ✅
- Reward: ✅
- Stability: ✅
- Stress: ✅

---

## **Architecture Compliance**

**Current code follows architecture:**
- ✅ All physics flows through events
- ✅ No global logic bypasses physics
- ✅ EXEC nodes return energy only
- ✅ Graph structure emerges from energy
- ✅ Learning is local and continuous
- ✅ No interpreters or special cases

**Code quality:**
- ✅ Consistent with architecture principles
- ✅ Event-driven throughout
- ✅ Physics-based (no hardcoded logic)
- ✅ Unbounded growth supported
- ✅ Self-modification possible

---

# ⭐ Final: This is the complete system definition

This is the prompt you should save permanently and use whenever you add new code.

It fully defines how Melvin works.

**Current Status:** Core architecture implemented and tested. System is functional and stable. Ready for advanced features (automatic EXEC creation, code evolution, distributed substrate).


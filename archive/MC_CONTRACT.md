# Melvin MC + Emergence Invariants

**Version:** 1.0  
**Status:** Core System Contract  
**Purpose:** Defines the boundary between Machine Code (MC) and the emergent graph intelligence

---

## Philosophy

Melvin's intelligence emerges from graph structure and pulse dynamics. Machine Code (MC) functions provide I/O and utilities only. **MC is the body; the graph is the brain.**

All meaning, patterns, concepts, and behaviors emerge from topology, weights, and pulse flow—not from MC logic, type tags, or external metadata.

---

## A. MC Boundary Rules (MC is body, graph is brain)

### Rule M1 — MC functions are I/O only

**MC functions MAY:**

* Read sensors (camera, mic, encoders, files, network)
* Write actuators (motors, text, audio, packets, files)
* Perform local utilities (hash, compress, random, math)

**MC functions MUST NOT:**

* Contain task-specific policies ("if obstacle, turn left")
* Contain high-level logic or planning
* Implement search, planning, or learning algorithms that bypass the graph
* Directly create or delete large subgraphs except when invoked through graph-driven codegen (see Rule G1-G3)

**Interpretation:** MC is a syscall layer, not a cognitive layer.

---

### Rule M2 — MC functions are fixed across runs

* The set of MC functions and their signatures are defined at compile-time
* Their **semantics do not change** at runtime:
  * Same inputs → same outputs (plus environment noise)
* The graph can change **when** and **how often** they're used, but **not what they do**

**Interpretation:** MC gives a stable world interface. Learning happens in the graph, not by editing MC semantics.

---

### Rule M3 — MC functions have no types in the graph

* Nodes and edges never carry a "this is an MC node" tag in the physics layer
* Binding from node to MC function is kept in a **separate binding table**:
  ```
  MCBinding: node_id -> mc_block_id
  ```
* The physics engine doesn't care what that binding means; it just knows:
  * "if node X passes activation threshold and is in MCBinding → call mc_block"

**Interpretation:** The graph "discovers" which nodes matter for I/O; physics doesn't special-case them.

---

## B. Activation Sources (graph doesn't start/stop at MC)

### Rule A1 — Sensors inject pulses, not types

* All external inputs (camera, mic, text, encoders, files, etc.) are converted into **plain pulses at existing or newly created nodes**
* Sensors do **not** create special node kinds; they just:
  * Create new nodes when needed
  * Emit pulses at those node IDs

**Interpretation:** The graph sees all input as "more energy," not tagged symbols.

---

### Rule A2 — Internal noise is mandatory

* Every tick, a small number of nodes are chosen stochastically to receive tiny pulses, **even if no sensors fire**
* Noise must be:
  * Low amplitude
  * Widely spread over time
* Purpose:
  * Seed new patterns
  * Keep dormant areas occasionally probed
  * Avoid dead, frozen attractors

**Interpretation:** The graph can start activity from nothing; MC I/O is not required to "boot thought."

---

### Rule A3 — Reward is just another pulse source

* Reward (positive/negative) is represented as:
  * Pulses into certain nodes (e.g., "value subgraphs")
  * Or a small global modulation of local learning rates
* Reward is **never** a direct instruction ("do X, don't do Y")
* Reward is **never** used to hard-code policies; it only biases plasticity

**Interpretation:** Reward shapes which circuits survive, not what they do explicitly.

---

## C. Internal Dynamics Rules (self-sustaining emergence)

### Rule D1 — Graph can run with zero MC calls

The physics engine must be able to:

* Propagate pulses
* Strengthen/decay edges
* Create/delete nodes/edges
* Form attractors and motifs

**Even if no MC function is ever called.**

There is always some combination of:

* Noise
* Old traces
* Residual pulses

That keeps internal dynamics alive (within energy limits).

**Interpretation:** MC is not the start or end of activity; it's just one influence.

---

### Rule D2 — Edge formation/decay are purely local

* New edges form when pulses co-occur on two nodes (i, j), regardless of MC usage
* Edges decay based on:
  * Lack of usage
  * Age
  * Local energy constraints
* No global controller or MC function decides where edges should form

**Interpretation:** Structure is grown by traffic, not by MC logic.

---

### Rule D3 — Attractors and motifs must be able to form solely from pulse statistics

* Stable subgraphs (loops, oscillators, assemblies) must emerge from:
  * Repeated pulse paths
  * Strengthening of frequently used edges
  * Decay of unused edges
* No MC function is allowed to "stamp" or template these motifs in one shot
* MC may assist with **low-level utilities** (e.g., random numbers, hashing) but not shape motifs directly

**Interpretation:** Internal building blocks are truly emergent, not hand-constructed by code.

---

## D. MC Usage and Chaining (emergent behavior)

### Rule P1 — MC invocation is gated by emergent subgraphs

* The runtime only calls an MC function if:
  * A node `n` exceeded some activation measure (e.g. pulses per tick)
  * AND `n` has an entry in `MCBinding`
* There is no hardcoded "call mc_X every tick" outside of debugging
* The graph must learn to generate the conditions for MC firing

**Interpretation:** The graph decides *when* to use MC, based on learned circuits.

---

### Rule P2 — Conflicting MC outputs are resolved by activation competition

* If multiple MC-bound nodes want to fire on the same tick:
  * Their activation levels are compared
  * A **winner-takes-most** or softmax policy chooses which MC calls execute
* This selection is **not** hard-coded by semantics ("motor always wins over text"); it's based on activation and, optionally, learned inhibitory circuits

**Interpretation:** "Highest activation wins" is the arbitration mechanism, not semantic priority.

---

### Rule P3 — MC cannot directly wire the graph

**MC functions MUST NOT:**

* Create or delete arbitrary edges/nodes with global knowledge
* Reset large subgraphs based on pre-programmed heuristics
* Implement external graph compression/simplification logic

**The only allowed MC-induced wiring changes are:**

* Local, triggered by pulses at specific nodes
* Or via a separate, graph-driven "codegen" function (see Rule G1-G3)

**Interpretation:** MC is not allowed to be a god-mode graph editor. It can propose local changes, but physics rules still govern.

---

### Rule P4 — MC output is always fed back as input pulses

* Any MC action that changes the world (move motor, write text, etc.) should, when possible, create **new sensory pulses later**:
  * Robot moves → encoders + vision change → new pulses
  * Text is emitted → environment responds → new text/audio pulses
* The graph must be able to "see" the consequences of using an MC function as new input, closing the loop

**Interpretation:** Outputs become future inputs, which is necessary for learning causal structure.

---

## E. Code Generation Rules (graph creating new machine blocks)

This is the high-power part: letting the graph grow its own code while keeping MC from taking over cognition.

### Rule G1 — New machine code is generated by a dedicated MC "compiler" block

* You can have **one or a small set** of special MC functions that:
  * Take some bytes from the graph (or external source)
  * Compile or assemble them into machine code
  * Append new CodeBlockHeaders + bytes into the `.m` file's code region
* These "compiler MC functions":
  * Do not decide *what* to compile
  * Only implement the technical work of turning bytes into executable code

**Interpretation:** The graph chooses content; MC "compiler" just turns it into machine code.

---

### Rule G2 — The graph chooses what to compile and when

* The decision to invoke the codegen MC block must:
  * Come from normal pulse dynamics and emergent motifs
  * Be mediated by nodes that have learned to chain "concept → codegen"
* There is no hardcoded schedule like "compile every N ticks"
* The graph must learn situations where new code helps

**Interpretation:** Code generation itself is an emergent skill.

---

### Rule G3 — New code blocks are treated as additional MC primitives, not cognitive types

* Once a new code block is created:
  * It receives a new `code_block_id`
  * Optional new `MCBinding` entries can be created linking some nodes → this block
* The physics layer:
  * Treats it exactly like any other MC block
  * Has no notion that "this is emergent code vs base code"

**Interpretation:** Emergently-generated code is just new body tricks; intelligence stays in the graph.

---

### Rule G4 — Graph must still be able to function with zero generated MC code

* Even if codegen MC functions are never called, the system must:
  * Form internal motifs
  * Build concepts
  * Control fixed MC I/O
* Codegen is an **expansion mechanism**, not a dependency

**Interpretation:** Emergence is not contingent on codegen; it just makes the system more powerful.

---

## F. Hard Invariants (if broken, emergence dies)

To keep the whole thing honest:

### Invariant 1: No cognition in MC logic

MC never encodes policies like "if X then Y"; all such behavior must come from the graph's structure.

**Violation Detection:** Any MC function that contains conditionals based on semantic interpretation (not pure I/O or utility).

---

### Invariant 2: No semantic tags in nodes/edges

No enums, roles, types, or labels at the physics layer.

**Violation Detection:** Any field in `NodeDisk` or `EdgeDisk` that encodes semantic meaning rather than pure physics state.

---

### Invariant 3: Graph activity can start and continue without MC

Noise + old traces + attractors are enough.

**Violation Detection:** System freezes or becomes static when all MC functions are disabled.

---

### Invariant 4: MC is always optional

Removing MC should leave a running, internally dynamic graph (just blind & mute).

**Violation Detection:** System crashes or requires MC initialization to function.

---

### Invariant 5: All "meaning" is encoded in graph topology + weights + dynamics

Not in MC, not in type tags, not in external metadata.

**Violation Detection:** Behavior that cannot be explained by looking at graph structure, weights, and pulse flow alone.

---

## Implementation Checklist

When implementing or reviewing code, verify:

- [ ] MC functions only perform I/O and utilities
- [ ] MC function semantics never change at runtime
- [ ] MC binding is in separate table, not encoded in node/edge types
- [ ] Sensors inject plain pulses, not typed nodes
- [ ] Internal noise is injected every tick
- [ ] Reward is just pulse modulation, not instructions
- [ ] Graph can run with zero MC calls
- [ ] Edge formation/decay are purely local
- [ ] Attractors form from pulse statistics alone
- [ ] MC invocation gated by activation thresholds
- [ ] MC conflicts resolved by activation competition
- [ ] MC cannot directly wire graph (except local/codegen)
- [ ] MC outputs feed back as input pulses
- [ ] Codegen MC functions are pure compilers
- [ ] Graph chooses when/what to compile
- [ ] Generated code treated like base MC primitives
- [ ] System works with zero generated code

---

## Testing Protocols

### Test 1: MC-Free Operation
Disable all MC functions. System should:
- Continue propagating pulses
- Form and decay edges
- Generate internal activity via noise
- Build stable patterns

### Test 2: MC Semantics Stability
Run same graph state with same MC functions. Verify identical outputs (modulo noise).

### Test 3: Activation-Based Invocation
Manually manipulate node activations. Verify MC functions only fire when activation thresholds exceeded.

### Test 4: No Semantic Tags
Dump graph structure. Verify no fields encode semantic meaning beyond physics state.

### Test 5: Feedback Loop Closure
Enable MC output. Verify outputs generate new input pulses within reasonable time.

---

## Version History

- **v1.0** (Initial): Core MC boundary and emergence invariants established

---

## Notes for Implementers

This contract is the foundation of Melvin's emergent intelligence. Violating these rules will prevent true emergence and trap the system in pre-programmed behavior.

When in doubt, ask: **"Does this decision move intelligence into the graph or keep it in MC?"**

Always err on the side of putting intelligence in the graph.


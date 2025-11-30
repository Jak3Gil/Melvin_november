# Instincts.m v0.1 Training Plan

## Goal

Build `instincts.m v0.1` - a competent OS + tool user that can:
- Ingest bytes
- Read C files
- Compile & run code (in sandbox)
- Interpret compile/test outcomes as reward signals
- Use basic safe motor control in simple body/world

**No empathy yet, no personality yet.** Just infrastructure.

---

## What instincts.m v0.1 Must Contain

### 1. Stable DATA + CHAN Structure

All relevant channels used:
- `CHAN_CODE_RAW` (100) - Raw C source code bytes
- `CHAN_COMPILE_LOG` (101) - Compiler output
- `CHAN_TEST_IO` (102) - Test execution output
- `CHAN_MOTOR` (200) - Motor commands
- `CHAN_PROPRIO` (201) - Proprioceptive feedback
- `CHAN_SENSOR` (202) - Sensor data

Healthy SEQ and CHAN edges (not just sparse noise).

### 2. Strong Patterns For

- **Function-like C sequences**: Repeated structural motifs (e.g., `int main() { return 0; }`)
- **Compile success/fail patterns**: Code → log → reward patterns
- **Safe motion patterns**: Motor behaviors that don't hit limits or high penalty

### 3. Reasonable FE Landscape

Clear low-FE islands around:
- Frequent C structure
- Frequently-successful motor behaviors

---

## Training Phases

### Phase 1: Pure Bytes + C Literacy (No Motors)

**Environment:**
- Feed mix of:
  - Simple C files
  - Compiler logs
  - Test outputs (text)
- Reward:
  - Small +R on successful compile
  - +R proportional to tests passed

**Loop:**
1. Stream `code.c` bytes → `CHAN_CODE_RAW`
2. EXEC node compiles it → `CHAN_COMPILE_LOG` bytes
3. EXEC node runs tests → `CHAN_TEST_IO` bytes
4. Inject reward: `R+` for success, `R-` for failure

**Train until:**
- Clear patterns around code/log/test channels
- FE_ema in that subgraph decreasing and stabilizing

**Snapshot:** `melvin_instincts_c_io_v0.m`

---

### Phase 2: Sim Body Survival (Motors + Sensors, No Personality)

**Environment:**
- Simple simulated body:
  - Few joints
  - Gravity
  - Collisions
- Reward:
  - +R for staying upright / within safe joint ranges
  - -R for collisions / joint limit violations

**Loop:**
1. Env sends PROPRIO + SENSOR bytes
2. Graph updates, emits MOTOR bytes
3. Env applies action, computes reward
4. Feed back reward

**Train until:**
- Subgraph that reliably avoids catastrophic motions
- FE_ema and activation stats stable in motor/sensor channels

**Snapshot:** `melvin_instincts_body_v0.m`

---

### Phase 3: Combine C + Body (Infrastructure Brain)

**Alternate or interleave:**
- C-IO episodes (Phase 1 style)
- Body survival episodes (Phase 2 style)

Same graph learns:
- Code
- Basic motor control
- Unified reward notions

**Train to budget** (e.g. total event count) and snapshot as:

> `melvin_instincts_v0.1.m`

This is your **base survival/infrastructure brain.**

---

## Metrics Per Phase

For each training phase, track:

- **FE_ema distributions**: Mean, variance, histogram
- **Pattern counts**: In relevant channels
- **Compile/test success rates**: (Phase 1)
- **Survival metrics**: In sim (Phase 2)
- **Edge statistics**: SEQ, CHAN, bond counts
- **Node statistics**: Total nodes, active nodes

All logged to `training_metrics.csv`.

---

## Checkpoint Schedule

Save `.m` snapshots at fixed intervals:
- Every `CHECKPOINT_INTERVAL_EVENTS` (default: 100,000 events)
- Label clearly: `phase1_step_100k.m`, `phase2_step_500k.m`, etc.

Checkpoints saved in `checkpoints/` directory.

---

## Rollback Strategy

If a later phase goes weird (FE spikes, behavior collapses):
1. Identify last "good" snapshot (healthy metrics)
2. Roll back to that checkpoint
3. Adjust curriculum/parameters, not physics
4. Resume training

---

## Running Training

### Local
```bash
# Run all phases
./instincts_training

# Or specify C files directory
./instincts_training -c /path/to/c/files

# Or start from existing file
./instincts_training -f existing_instincts.m
```

### On Jetson
```bash
# Copy to Jetson
scp instincts_training.c melvin.c melvin.h melvin_diagnostics.* melvin@169.254.123.100:/home/melvin/

# SSH and compile
ssh melvin@169.254.123.100
cd /home/melvin
gcc -o instincts_training instincts_training.c -lm -std=c11 -Wall -O2

# Run training (long-running, consider screen/tmux)
./instincts_training
```

---

## Success Criteria for v0.1

Before moving to personality/shaping, verify:

1. **Numerically healthy:**
   - FE_ema stable, not exploding
   - No NaN/Inf
   - Graph structure growing but bounded

2. **Competent with code:**
   - Patterns form around C structure
   - Compile/test success rates > baseline

3. **Competent with body:**
   - Survival rate > random baseline
   - Motor patterns avoid catastrophic failures

4. **Unified:**
   - Both C and body subgraphs coexist
   - No interference/collapse when alternating

---

## After v0.1

Once `instincts_v0.1.m` is stable and competent:

**Then** fork it and add personality/shaping:
- Social channels
- Empathy/humor reward shaping
- Identity-like goals
- etc.

Right now, focus on **getting v0.1 instincts.m to exist and be stable**.


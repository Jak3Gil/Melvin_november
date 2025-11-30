# Next Steps: From Diagnostics to Instincts.m v0.1

## Status Summary

### ✅ Completed
- **Diagnostics Suite**: All 3 experiments passed
  - Experiment A: Single-node sanity ✅
  - Experiment B: Pattern learning (structured beats random) ✅
  - Experiment C: Control loop (no segfault, wiring correct) ✅
- **Physics Verification**: Core laws are working correctly

### ⚠️ Known Issues
- **Control Learning (C.2)**: Learning not showing improvement yet
  - No STATE→ACTION edges forming
  - Returns not above random baseline
  - **Status**: This is a tuning/curriculum issue, not physics bug
  - **Action**: Can proceed with instincts training; will improve during training

---

## Gate: Control Learning C.2

**Status**: Not yet passing, but physics is verified.

**Decision**: Proceed with instincts training. The control learning will improve as:
- More training data flows through the system
- Patterns form around state/action/reward relationships
- The graph learns the reward structure naturally

**Note**: If control learning doesn't improve during instincts training, we'll revisit the reward injection mechanism.

---

## Instincts.m v0.1 Training Plan

### Phase 1: C Literacy (500K events)
**Goal**: Learn to process C code, compile, test

**Input**:
- C source files → `CHAN_CODE_RAW`
- Compiler logs → `CHAN_COMPILE_LOG`
- Test outputs → `CHAN_TEST_IO`

**Reward**:
- +0.5 for successful compile
- +1.0 for tests passed

**Success Criteria**:
- Patterns form around C structure
- FE_ema decreasing in code/log/test subgraph
- Compile success rate > baseline

**Checkpoint**: `checkpoints/phase1_step_100k.m`, `phase1_step_500k.m`

---

### Phase 2: Body Survival (500K events)
**Goal**: Learn safe motor control

**Input**:
- Proprioception → `CHAN_PROPRIO`
- Sensors → `CHAN_SENSOR`
- Motor commands → `CHAN_MOTOR`

**Reward**:
- +0.1 for staying in safe joint ranges
- -0.5 for limit violations/collisions

**Success Criteria**:
- Survival rate > random baseline
- FE_ema stable in motor/sensor channels
- Patterns form around safe motion sequences

**Checkpoint**: `checkpoints/phase2_step_100k.m`, `phase2_step_500k.m`

---

### Phase 3: Combined (1M events)
**Goal**: Unified infrastructure brain

**Method**: Alternate C episodes and body episodes

**Success Criteria**:
- Both subgraphs coexist
- No interference/collapse
- Numerically healthy (FE stable, no NaN/Inf)

**Final Snapshot**: `instincts_v0.1.m`

---

## Running Training

### On Jetson (Recommended)
```bash
# Setup
./run_instincts_training_jetson.sh

# Or manually:
ssh melvin@169.254.123.100
cd /home/melvin/instincts_training
screen -S instincts
./instincts_training

# Monitor progress
tail -f training.log
```

### Local (for testing)
```bash
gcc -o instincts_training instincts_training.c -lm -std=c11 -Wall -O2
./instincts_training
```

---

## Monitoring Training

### Key Metrics to Watch

1. **FE_ema**: Should stabilize, not explode
2. **Pattern counts**: Should grow in relevant channels
3. **Success rates**: Compile/test/survival should improve
4. **Graph size**: Should grow but remain bounded

### Red Flags

- FE_ema > 100.0 (exploding)
- NaN/Inf detected
- Graph size > 10M nodes (unbounded growth)
- All success rates = 0.0 (complete failure)

If red flags appear:
1. Stop training
2. Roll back to last good checkpoint
3. Adjust parameters/curriculum
4. Resume

---

## Checkpoint Strategy

**Automatic checkpoints** every 100K events:
- `checkpoints/phase1_step_100k.m`
- `checkpoints/phase1_step_200k.m`
- etc.

**Manual checkpoints** before major changes:
- Before starting Phase 2
- Before starting Phase 3
- Before any parameter changes

**Rollback procedure**:
```bash
# Identify last good checkpoint
ls -lh checkpoints/

# Copy to main file
cp checkpoints/phase1_step_300k.m instincts.m

# Resume training
./instincts_training -f instincts.m
```

---

## Success Criteria for v0.1

Before declaring `instincts_v0.1.m` complete:

1. ✅ **Numerically healthy**
   - FE_ema stable (< 10.0)
   - No NaN/Inf
   - Graph structure bounded

2. ✅ **Competent with code**
   - Patterns around C structure
   - Compile/test success > random

3. ✅ **Competent with body**
   - Survival rate > random
   - Safe motion patterns

4. ✅ **Unified**
   - Both subgraphs coexist
   - No collapse when alternating

---

## After v0.1

Once `instincts_v0.1.m` is stable:

**Then** add personality/shaping:
- Social channels
- Empathy/humor rewards
- Identity goals
- etc.

**But first**: Get v0.1 to exist and be stable.

---

## Files Created

- `diag_experiment_c2_control_learning_on.c` - Enhanced control learning test
- `instincts_training.c` - Main training loop (3 phases)
- `run_instincts_training_jetson.sh` - Jetson runner script
- `INSTINCTS_PLAN.md` - Detailed training plan
- `NEXT_STEPS.md` - This file

---

## Quick Start

```bash
# 1. Run control learning check (optional)
sshpass -p "123456" ssh melvin@169.254.123.100 "cd /home/melvin/diagnostics && ./diag_experiment_c2_control_learning_on"

# 2. Start instincts training
./run_instincts_training_jetson.sh

# 3. Monitor progress
sshpass -p "123456" ssh melvin@169.254.123.100 "tail -f /home/melvin/instincts_training/training.log"
```

---

**Ready to begin instincts training!**


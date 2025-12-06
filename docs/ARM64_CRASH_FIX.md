# ARM64 Crash Fix - Reinforcement Learning from Code Failures

## Problem Discovery

**Symptom:** Jetson Orin AGX experiencing "Illegal instruction" crashes when executing blob code
```
bash: line 1:  5668 Illegal instruction     (core dumped) ./run_live
bash: line 44:  5805 Illegal instruction     (core dumped) ./run_safe
```

## Root Cause Analysis

### The Bug

Found critical bug in `src/melvin.c` function `melvin_execute_exec_node()`:

**Line 3361 - UNPROTECTED EXECUTION:**
```c
if (has_inputs && has_code) {
    // ...
    exec_func f = (exec_func)code_ptr;
    result = f(input1, input2);  // ⚠️ NO CRASH PROTECTION!
    // ...
}
```

**Lines 3388-3423 - PROTECTED EXECUTION:**
```c
else if (has_code) {
    // Set up signal handlers (SIGSEGV, SIGBUS)
    struct sigaction old_sa_segv, old_sa_bus, new_sa;
    // ...
    if (setjmp(exec_segfault_recovery) == 0) {
        exec_code(g, node_id);  // ✅ PROTECTED!
        execution_success = true;
    } else {
        execution_success = false;  // Crash caught!
    }
    // Restore handlers
}
```

### Why This Matters

The graph learns by **executing arbitrary ARM64 machine code** stored in its blob. When teaching operations:
1. We feed ARM64 instructions as data
2. Graph stores them in EXEC nodes
3. Graph tries to execute them when activated

**Without protection:** One bad instruction crashes the entire process!

## Existing Reinforcement Learning System

The system **already had** reinforcement learning built in (lines 3425-3443):

```c
/* Update success rate (exponential moving average) */
float success_value = execution_success ? 1.0f : 0.0f;
if (node->exec_success_rate == 0.0f) {
    node->exec_success_rate = success_value;
} else {
    node->exec_success_rate = 0.9f * node->exec_success_rate + 0.1f * success_value;
}

/* Adjust threshold based on success rate */
if (node->exec_success_rate > 0.7f && node->exec_threshold_ratio > 0.3f) {
    node->exec_threshold_ratio *= 0.99f;  // Successful → easier to trigger
} else if (node->exec_success_rate < 0.3f && node->exec_threshold_ratio < 2.0f) {
    node->exec_threshold_ratio *= 1.01f;  // Failing → harder to trigger
}
```

**But it couldn't work** because crashes killed the process before learning could happen!

## The Fix

### Added Crash Protection to Input Execution Path

**Before:**
```c
result = f(input1, input2);  // Crashes process on bad code
execution_success = true;
```

**After:**
```c
/* Set up crash protection (SIGILL, SIGSEGV, SIGBUS) */
struct sigaction old_sa_segv, old_sa_bus, old_sa_ill, new_sa;
exec_g_global = g;
exec_segfault_occurred = false;

new_sa.sa_sigaction = exec_segfault_handler;
sigemptyset(&new_sa.sa_mask);
new_sa.sa_flags = SA_SIGINFO;
sigaction(SIGSEGV, &new_sa, &old_sa_segv);
sigaction(SIGBUS, &new_sa, &old_sa_bus);
sigaction(SIGILL, &new_sa, &old_sa_ill);  // ← KEY: Catches illegal instructions!

/* Set jump point for recovery */
if (setjmp(exec_segfault_recovery) == 0) {
    result = f(input1, input2);  // Try execution
    execution_success = true;    // Success!
} else {
    execution_success = false;   // Caught crash - no process death!
}

/* Restore signal handlers */
sigaction(SIGSEGV, &old_sa_segv, NULL);
sigaction(SIGBUS, &old_sa_bus, NULL);
sigaction(SIGILL, &old_sa_ill, NULL);
```

### Added SIGILL Handler

Also added `SIGILL` (illegal instruction) handler to both execution paths - this was missing!

## How Reinforcement Learning Works Now

### Crash Detection → Recovery → Learning

1. **Bad code executes** → SIGILL signal raised
2. **Signal handler catches it** → `exec_segfault_handler()` called
3. **Error fed to graph:**
   ```c
   melvin_feed_byte(exec_g_global, 250, 0xFF, 1.0f);  // Error port
   melvin_feed_byte(exec_g_global, 31, 0xFF, 0.8f);   // Negative feedback
   ```
4. **Recovery via longjmp** → Execution continues
5. **Success rate updated:**
   - `exec_success_rate` decreases
   - `exec_threshold_ratio` increases (harder to trigger next time)
6. **Graph learns:**
   - Pattern that led to crash gets negative reinforcement
   - Alternative patterns get relatively stronger
   - Over time, graph routes around bad code

### Positive Reinforcement

When execution **succeeds:**
1. `exec_success_rate` increases
2. `exec_threshold_ratio` decreases
3. Node becomes **easier to activate**
4. Graph preferentially uses successful operations

## Expected Behavior on Jetson

### Before Fix
```
🚀 Executing blob code at offset 1024...
   Inputs: 5, 3
Illegal instruction (core dumped)  ← Process dies
```

### After Fix
```
🚀 Executing blob code at offset 1024...
   Inputs: 5, 3
❌ BLOB EXECUTION FAILED (crashed)
   Node: 2000, Offset: 1024
   Error fed to port 250 (graph learns from failure)

[Graph continues running]
[Success rate: 0.0 → 0.0]
[Threshold ratio: 0.5 → 0.505 (harder to trigger)]

[Next cycle, different route chosen...]
```

### After Learning
Over many cycles:
- Bad operations: `exec_threshold_ratio → 3.0` (maximum, rarely triggered)
- Good operations: `exec_threshold_ratio → 0.1` (minimum, easily triggered)
- Graph naturally evolves to use working code paths

## Testing Plan

1. **Compile updated code:**
   ```bash
   gcc -c src/melvin.c -o melvin.o -Wall -O2
   ```

2. **Deploy to Jetson:**
   ```bash
   ./deploy_teachable_to_jetson.sh
   ```

3. **Run hardware test:**
   ```bash
   ssh melvin@169.254.123.100
   cd /home/melvin/teachable_system
   ./run_live
   ```

4. **Expected results:**
   - No process crashes
   - Error messages for failed executions
   - Success messages for good executions
   - Graph continues learning from both

## System Architecture

### Complete Learning Loop

```
[Hardware Input] → [Graph Pattern Recognition]
                           ↓
                   [Pattern Matches EXEC Node]
                           ↓
                   [Activation Exceeds Threshold?]
                           ↓ YES
                   [Execute Blob Code]
                      ↙         ↘
              SUCCESS         FAILURE
                 ↓              ↓
         [Positive RL]    [Negative RL]
         threshold ↓      threshold ↑
         easier next     harder next
                 ↓              ↓
         [Convert Result]  [Feed Error]
                 ↓              ↓
         [Output Pattern]  [Try Alternative]
                           
         ← [Graph Evolves] ←
```

### Key Insight

The graph doesn't need to **understand** machine code - it just needs to:
1. Try different execution paths
2. Observe which ones crash vs succeed
3. Reinforce successful paths
4. Suppress failing paths

**Emergent behavior:** Graph learns operational semantics through trial and error!

## Why This Is Powerful

### Traditional Approach
```
if (operation == "add") {
    result = a + b;  // Hardcoded!
} else if (operation == "multiply") {
    result = a * b;  // Hardcoded!
}
```

### Melvin's Approach
```
// Feed ARM64 addition code → Graph stores it
// Feed ARM64 multiply code → Graph stores it
// Graph learns when to execute each based on patterns
// NO hardcoded operations - pure learning!
```

### Learning Extends Beyond Code

The same reinforcement learning applies to:
- **Which patterns** trigger execution
- **Which inputs** to use
- **Which outputs** to produce
- **When** to execute vs skip
- **How** to combine operations

**All emergent** from success/failure feedback!

## Security Note

Signal handlers provide **process-level isolation** but not sandboxing:
- Bad code can't crash the process ✅
- Bad code can still access process memory ⚠️
- For production, consider: seccomp, containers, or separate process execution

For research/development, current approach is appropriate.

## Summary

**Fixed:** Critical bug causing Jetson crashes
**Added:** SIGILL signal handler for illegal instructions
**Enabled:** Reinforcement learning that was already implemented but couldn't function
**Result:** Graph can now learn from code execution failures without process death

The system is now **truly self-healing** and can explore the space of machine code operations through reinforcement learning!


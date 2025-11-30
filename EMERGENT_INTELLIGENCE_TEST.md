# Testing Emergent Intelligence

## The Goal

Test that the system is **building understanding**, not just memorizing.

## What We're Measuring

### 1. Pattern Formation
- Do edges strengthen over time?
- Do common sequences (like "the cat") form stronger connections?
- Does coherence increase?

### 2. Prediction Ability
- Given context "the c", can it predict 'a' (to complete "the cat")?
- Does prediction accuracy improve over episodes?
- Can it complete words/sequences it's seen?

### 3. Generalization
- If it learns "the cat sat" and "the dog ran"
- Can it predict "the cat ran" (combining patterns)?
- Does it understand word boundaries?

## The Test

```bash
./test_emergent_intelligence
```

**What it does:**
1. Feeds corpus: "the cat sat", "the dog ran", etc.
2. Tests prediction after each phase
3. Measures coherence (how connected the graph is)
4. Shows gradual improvement

## What Success Looks Like

### Early Episodes (1-20)
- Random activations
- Weak edges (all ~0.1)
- Low coherence (~0.01)
- Prediction score: ~0.0 (can't predict)

### Middle Episodes (40-60)
- Some patterns forming
- Some edges stronger (0.2-0.3)
- Coherence increasing (~0.1)
- Prediction score: ~0.1 (weak but improving)

### Later Episodes (80-100)
- Strong patterns visible
- Many edges > 0.3
- High coherence (~0.3+)
- Prediction score: ~0.2+ (can predict some patterns)

## Key Metrics

1. **Coherence**: How connected is the graph? (0.0 = random, 1.0 = fully connected)
2. **Strong patterns**: How many edges have |w| > 0.3?
3. **Prediction score**: Can it predict next byte from context?

## The Point

**Intelligence emerges gradually:**
- Episode 1: Random noise
- Episode 10: Weak patterns
- Episode 50: Stronger patterns
- Episode 100: Can predict some sequences

**Each byte helps:**
- "the cat" appears → edges form
- "the cat sat" appears → edges strengthen
- "the cat" appears again → prediction improves

This is **emergent intelligence** - understanding builds from many small interactions.

## Real Test = Long Running

For real intelligence testing:
```bash
# Feed large corpus over days/weeks
cat large_corpus.txt | ./melvin_run brain.m

# Test periodically
./test_emergent_intelligence

# Measure improvement over time
```

The system should show:
- Gradual improvement (not instant)
- Pattern formation
- Better predictions over time
- Understanding building from bytes


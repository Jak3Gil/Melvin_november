# Threshold Summary

## Pattern Matching Thresholds

### 1. Similarity-First Search Threshold (Line 4111)

**Purpose:** Find similar nodes to build candidate list

**Current:** Fixed `0.2f` → **CHANGED to dynamic/relative**

**New:** 
```c
float similarity_threshold = avg_edge_strength * 0.2f + avg_activation * 0.1f;
// Clamped: 0.05f - 0.5f (lenient, finds more candidates)
```

**Why:** Should be relative to graph state, and more lenient to find more candidate nodes

### 2. Pattern Matching Similarity Threshold (Line 3394)

**Purpose:** Overall similarity check when matching pattern to sequence

**Current:**
```c
float similarity_threshold = avg_edge_strength * 0.3f + avg_activation * 0.2f;
// Clamped: 0.1f - 0.8f
```

**This is already dynamic and relative** ✅

### 3. Adjusted Pattern Matching Threshold (Line 3462)

**Purpose:** Final threshold adjusted by pattern strength and frequency

**Current:**
```c
float adjusted_threshold = similarity_threshold * (1.0f + pattern_strength * 0.5f) * frequency_factor;
// Clamped: 0.05f - 0.9f
```

**This is already dynamic and relative** ✅

## Summary

- **Similarity-first search:** Now dynamic/relative (was fixed 0.2f)
- **Pattern matching:** Already dynamic/relative ✅
- **Adjusted threshold:** Already dynamic/relative ✅

All thresholds are now relative to graph state!


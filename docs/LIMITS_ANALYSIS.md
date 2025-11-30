# Limits and Thresholds Analysis

## Architecture: EVENT-DRIVEN (NOT Tick-Based)

The `.m` file runs **continuously via events**, not ticks:
- **Event-driven**: When `melvin_feed_byte()` is called, nodes are added to a propagation queue
- **Continuous processing**: `melvin_call_entry()` processes the queue until empty
- **No global ticks**: Actions are triggered by energy injection, not time-based cycles
- **Queue-based**: Uses a lock-free circular buffer for event propagation

## Found Limits and Thresholds

### 1. Initial Scaffolding Limits
**Location**: `create_initial_edge_suggestions()`
- `max_initial_edges = 1000` - Limits initial scaffolding edges
- **Impact**: Only creates 1000 initial edges, even if graph could use more
- **Can remove**: Yes - let graph create as many initial edges as needed

### 2. Worker Thread Cap
**Location**: `uel_propagate_from()` line 1617
- `if (num_workers > 8) num_workers = 8;` - Caps worker threads at 8
- **Impact**: Limits parallelism on systems with >8 cores
- **Can remove**: Yes - use all available cores, or let graph adapt

### 3. Edge Growth Increment
**Location**: `create_edge()` line 1058
- `uint64_t new_edge_count = g->edge_count + 10000;` - Adds 10k edges at a time
- **Impact**: May grow more than needed, or not enough
- **Can replace**: Yes - grow by percentage (e.g., 2x) or based on demand

### 4. Safety Limit in find_edge()
**Location**: `find_edge()` line 938
- `uint32_t max_iterations = (uint32_t)(g->edge_count + 1);` - Prevents infinite loops
- **Impact**: Safety check for corrupted edge lists
- **Keep**: Yes - this is a safety check for corrupted data structures

### 5. Minimum Thresholds
**Locations**: Various places
- `if (g->avg_activation < 0.01f) g->avg_activation = 0.1f;` - Minimum activation
- `if (g->avg_chaos < 0.01f) g->avg_chaos = 0.1f;` - Minimum chaos
- `if (g->avg_edge_strength < 0.01f) g->avg_edge_strength = 0.1f;` - Minimum edge strength
- **Impact**: Prevents division by zero, but may mask very low activity
- **Can replace**: Yes - use epsilon values or adaptive minimums

### 6. Queue Size Minimum
**Location**: `melvin_open_with_cold()` line 820
- `g->prop_queue_size = (g->node_count > 0) ? g->node_count : 256;` - Minimum 256
- **Impact**: Small graphs get 256-size queue (may be wasteful)
- **Can replace**: Yes - start smaller, grow dynamically

### 7. Initial Edge Creation Limits
**Location**: `create_initial_edge_suggestions()` lines 248-410
- Multiple loops with `edges_created < max_initial_edges` checks
- Limits to 10 input ports, 10 memory ports, 10 output ports
- **Impact**: Only creates edges for first 10 of each port type
- **Can remove**: Yes - create edges for all available ports

### 8. Sampling Limit for Chaos Calculation
**Location**: Initialization code line 561
- `for (uint64_t i = 0; i < g->node_count && i < 100; i++)` - Only samples first 100 nodes
- **Impact**: Chaos estimate may be inaccurate for large graphs
- **Can replace**: Yes - sample more nodes or use statistical sampling

### 9. ⚠️ TICK-LIKE LIMIT in uel_main() (CRITICAL)
**Location**: `uel_main()` line 1829
- `uint32_t max_iterations = 1000;` - Limits processing to 1000 nodes per call
- **Impact**: **This is a TICK LIMIT!** If queue has >1000 nodes, some won't be processed
- **Problem**: Actions ARE limited by this - it's essentially a tick-based limit
- **Must remove**: Yes - process until queue is empty, not limited by iteration count

## Recommendations

### Remove/Replace These:
1. ✅ **Remove `max_initial_edges` limit** - Let graph create all needed initial edges
2. ✅ **Remove worker thread cap** - Use all available cores
3. ✅ **Replace fixed 10k edge growth** - Use percentage-based growth (2x)
4. ✅ **Remove port count limits** - Create edges for all ports, not just first 10
5. ✅ **Replace minimum thresholds** - Use adaptive minimums based on graph state
6. ✅ **Replace queue size minimum** - Start smaller, grow dynamically
7. ✅ **Replace sampling limit** - Sample more nodes or use statistical methods
8. ⚠️ **CRITICAL: Remove `max_iterations` in uel_main()** - Process until queue empty, not limited by count

### Keep These:
1. ✅ **Keep `max_iterations` in find_edge()** - Safety check for corrupted data
2. ✅ **Keep basic safety checks** - Null pointers, bounds checking, etc.

## Action Items

1. Remove artificial limits on initial scaffolding
2. Remove worker thread cap
3. Replace fixed growth increments with adaptive growth
4. Remove port count limits in edge creation
5. Replace minimum thresholds with adaptive values
6. Make queue size fully dynamic


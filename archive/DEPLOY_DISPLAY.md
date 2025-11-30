# Deploy Display Plugin to Jetson

## What This Does

The `mc_display` plugin enables Melvin to continuously display the graph visualization on the Jetson's display port (HDMI/eDP).

## Features

- **Framebuffer Support**: Uses `/dev/fb0` for direct display output
- **Force-Directed Layout**: Automatically positions nodes using physics simulation
- **Real-Time Rendering**: Updates display every 10 ticks for performance
- **Console Fallback**: Falls back to console output if framebuffer unavailable
- **Performance Optimized**: Limits display to 1000 nodes / 5000 edges for smooth rendering

## Deployment

### 1. Transfer Files to Jetson

```bash
# Deploy the display plugin
scp plugins/mc_display.c melvin@169.254.123.100:~/melvin_system/

# Deploy scaffold (optional - patterns will be in graph)
scp scaffolds/scaffold_display_continuous.c melvin@169.254.123.100:~/melvin_system/scaffolds/
```

### 2. Build Plugin on Jetson

```bash
ssh melvin@169.254.123.100
cd ~/melvin_system

# Build plugin as shared library
gcc -shared -fPIC -O3 -march=armv8-a -I. -o plugins/mc_display.so plugins/mc_display.c -lm -ldl

# Verify it compiled
ls -lh plugins/mc_display.so
```

### 3. Register in melvin.c (if needed)

The plugin will auto-load when Melvin tries to use it. Or manually add to `melvin.c`:

```c
MCFn mc_display_graph = load_plugin_function("mc_display", "mc_display_graph");
MCFn mc_display_init = load_plugin_function("mc_display", "mc_display_init");
register_mc("display_graph", mc_display_graph);
register_mc("display_init", mc_display_init);
```

### 4. Create Display Node in Graph

The scaffold will create display nodes automatically, or create manually:

```c
uint64_t display_node = alloc_node(&g);
g.nodes[display_node].kind = NODE_KIND_CONTROL;
g.nodes[display_node].mc_id = <display_init_id>;
g.nodes[display_node].bias = 5.0f;
g.nodes[display_node].a = 1.0f;
```

### 5. Run Melvin

```bash
./melvin
```

The display will automatically:
- Initialize framebuffer
- Start continuous rendering
- Update every 10 ticks
- Show nodes as colored circles
- Show edges as connecting lines

## Display Output

### Framebuffer Mode (HDMI/eDP)
- Full screen graph visualization
- Nodes as colored circles (activation-based color)
- Edges as lines (weight-based intensity)
- Force-directed layout

### Console Mode (Fallback)
- Text-based stats output
- Node counts and activations
- Top active nodes list
- Updates in real-time

## Troubleshooting

### No Display Output

1. **Check framebuffer access:**
   ```bash
   ls -l /dev/fb0
   sudo chmod 666 /dev/fb0  # If needed
   ```

2. **Check display connection:**
   ```bash
   xrandr  # List displays
   ```

3. **Try console mode:**
   - Plugin automatically falls back to console if framebuffer unavailable

### Performance Issues

- **Too many nodes:** Plugin auto-limits to 1000 nodes for display
- **Too slow:** Rendering happens every 10 ticks (adjustable in code)
- **Layout jittery:** Increase iteration count in `layout_nodes()`

## Integration with Graph

The display system is graph-native:
- Display patterns stored in `melvin.m`
- Display nodes activated by graph
- Continuous rendering driven by graph activity
- All display logic learned through patterns

**Everything in the graph - no hardcoded display logic!**


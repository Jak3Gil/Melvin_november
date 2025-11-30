# Enable Display Visualization on Jetson

## Status

✅ Display plugin deployed: `plugins/mc_display.so`
✅ Scaffold deployed: `scaffolds/scaffold_display_continuous.c`

## Next Steps to Enable Display

### Option 1: Quick Test (Current Setup)

The display plugin is built but needs to be registered. Currently Melvin will try to auto-load it.

### Option 2: Add Display Registration to melvin.c

Edit `melvin.c` on Jetson and add after line 667:

```c
// Load display plugin
MCFn mc_display_graph = load_plugin_function("mc_display", "mc_display_graph");
MCFn mc_display_init = load_plugin_function("mc_display", "mc_display_init");

// Register display functions
if (mc_display_graph) register_mc("display_graph", mc_display_graph);
if (mc_display_init) register_mc("display_init", mc_display_init);
```

Then rebuild melvin:
```bash
gcc -std=c11 -O3 -march=armv8-a -o melvin melvin.c -lm -ldl -lpthread
```

### Option 3: Let Graph Handle It (Recommended)

The scaffold will teach the graph to use display automatically:

1. Melvin processes scaffolds (on next start or if not done yet)
2. Display patterns stored in graph
3. Graph activates display nodes automatically
4. Visualization starts!

## What the Display Shows

- **Nodes**: Colored circles (activation-based color intensity)
- **Edges**: Lines connecting nodes (weight-based thickness)
- **Layout**: Force-directed physics simulation
- **Updates**: Every 10 ticks for performance

## Display Modes

### Framebuffer Mode (HDMI/eDP)
- Direct hardware access via `/dev/fb0`
- Full screen visualization
- Best performance

### Console Mode (Fallback)
- Text-based stats
- Node/edge counts
- Top active nodes
- Works if framebuffer unavailable

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

3. **Check if running:**
   ```bash
   ps aux | grep melvin
   tail -f ~/melvin_system/melvin.log | grep display
   ```

### Performance Issues

- Display limits to 1000 nodes / 5000 edges for performance
- Updates every 10 ticks (adjustable in `mc_display.c`)
- Console mode is always available as fallback

## Commands

**Check Display Status:**
```bash
ssh melvin@169.254.123.100 "tail -f ~/melvin_system/melvin.log | grep display"
```

**View on Jetson Display:**
The visualization will appear on the Jetson's HDMI/eDP output automatically when active.

**Force Console Mode:**
If framebuffer doesn't work, the plugin automatically falls back to console output.



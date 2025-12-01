# Starting Melvin and Talking to It

## Quick Start

### 1. Start Melvin
```bash
cd ~/melvin
./tools/melvin_service.sh start
```

### 2. Check Status
```bash
./tools/melvin_service.sh status
```

Should show: `Melvin is RUNNING`

### 3. Monitor Activity
```bash
# Watch logs in real-time
./tools/melvin_service.sh logs

# Or tail the log file
tail -f ~/melvin/melvin.log
```

### 4. Talk to Melvin
- **Speak into the USB microphone**
- You should hear echo immediately (direct passthrough)
- After 5-10 minutes, graph learns → you'll hear processed responses

## What to Watch For

### Graph Activity Indicators:
```
[0] 🟢 ACTIVE | Nodes: 1000 | Edges: 2551 | Chaos: 0.123 | Activation: 0.234
      Audio: 1024 read, 512 written
      💬 Conversation: STT=0.234 LLM=0.456 TTS=0.123
```

**Look for:**
- 🟢 **ACTIVE** status (graph is processing)
- **Nodes/Edges growing** (graph is learning)
- **Audio bytes increasing** (mic/speaker working)
- **Conversation indicators** (STT/LLM/TTS activation)

### Tool Usage:
- **STT** activation = Graph is hearing you
- **LLM** activation = Graph is thinking/responding
- **TTS** activation = Graph is speaking

### Pattern Creation:
- **Edges increasing** = Graph is creating connections
- **Nodes growing** = Graph is learning new patterns
- **Chaos changing** = Graph is processing/learning

## Troubleshooting

### Not Hearing Anything:
```bash
# Check audio devices
aplay -l
arecord -l

# Test direct playback
aplay /usr/share/sounds/alsa/Front_Left.wav
```

### Graph Not Processing:
```bash
# Check if running
./tools/melvin_service.sh status

# Check logs
./tools/melvin_service.sh logs

# Restart if needed
./tools/melvin_service.sh restart
```

### No Tool Activity:
- Wait 5-10 minutes for graph to learn
- Speak louder
- Check logs for tool calls

## Expected Timeline

**0-2 minutes:**
- Graph initializes
- Echo works (direct mic → speaker)
- Status shows 🟢 ACTIVE

**2-5 minutes:**
- Graph starts learning
- Nodes/edges growing
- Tool gateways activating

**5-10 minutes:**
- Conversation path strengthens
- Graph can route: Mic → STT → LLM → TTS → Speaker
- You'll hear processed responses (not just echo)

**10+ minutes:**
- Graph continues learning
- Responses improve
- Patterns build up

## Stop Melvin

```bash
./tools/melvin_service.sh stop
```

## Summary

1. ✅ Start: `./tools/melvin_service.sh start`
2. ✅ Monitor: `./tools/melvin_service.sh logs`
3. ✅ Talk: Speak into mic
4. ✅ Wait: 5-10 minutes for learning
5. ✅ Listen: Hear responses!

**Melvin is running and ready to learn!** 🎉


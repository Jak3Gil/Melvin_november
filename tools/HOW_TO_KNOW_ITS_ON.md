# How to Know Melvin is On and Working

## Quick Status Check

### 1. **Service Status**
```bash
./tools/melvin_service.sh status
```
Shows if Melvin is running, PID, and uptime.

### 2. **Console Output**
When running, you'll see:
```
[0] 🟢 ACTIVE | Nodes: 1000 | Edges: 2547 | Chaos: 0.123456 | Activation: 0.234567
      Audio: 1024 read, 512 written | Video: 0 read, 0 written
      💬 Conversation: STT=0.234 LLM=0.456 TTS=0.123
```

**Status Indicators:**
- 🟢 **ACTIVE** = Graph is processing (chaos > 0.01 or activation > 0.01)
- ⚪ **IDLE** = Graph is in low-activity state (self-regulating)

**Conversation Indicators:**
- **STT** = Speech-to-text activation (hearing you)
- **LLM** = Language model activation (thinking/responding)
- **TTS** = Text-to-speech activation (speaking)

### 3. **Audio Feedback**
- **Echo Test**: Speak into mic → should hear echo immediately (bypasses graph)
- **Graph Response**: After graph learns, you'll hear processed responses
- **No Audio** = Check mic/speaker connections

### 4. **Dashboard**
```bash
python3 tools/melvin_dashboard_app.py
```
Open: http://169.254.123.100:8080

**Visual Indicators:**
- Real-time node/edge counts (should be growing)
- Chaos/activation graphs (should show activity)
- 3D visualization (nodes/edges moving)
- Output log (shows tool calls, responses)

### 5. **Logs**
```bash
./tools/melvin_service.sh logs
```
Shows:
- Startup messages
- Tool calls (STT, LLM, TTS)
- Errors (if any)
- Status updates

## Conversation Readiness

### Out of the Box:
✅ **Hardware connected** - Mic, speaker, cameras detected
✅ **Tools installed** - Ollama, Whisper, Piper, Vision working
✅ **Conversation path** - Mic → STT → LLM → TTS → Speaker
✅ **Patterns seeded** - Initial instinct patterns created

### What to Expect:
1. **First few minutes**: Graph is learning, strengthening connections
2. **Echo works immediately**: Direct mic → speaker passthrough
3. **Conversation starts**: After graph learns the path (5-10 minutes)
4. **Responses improve**: As graph builds more patterns

### How to Test Conversation:
1. **Start Melvin:**
   ```bash
   ./tools/melvin_service.sh start
   ```

2. **Check status:**
   ```bash
   ./tools/melvin_service.sh status
   ```

3. **Watch console/logs:**
   ```bash
   ./tools/melvin_service.sh logs
   ```
   Look for:
   - 🟢 ACTIVE status
   - Audio bytes being read/written
   - Conversation indicators (STT/LLM/TTS)

4. **Speak into mic:**
   - You should hear echo immediately
   - After learning, you'll hear responses

5. **Monitor dashboard:**
   - Watch node/edge counts grow
   - See chaos/activation change
   - Check output log for tool calls

## Troubleshooting

### Not Hearing Anything:
- Check mic/speaker connections
- Verify audio device: `aplay -l` and `arecord -l`
- Check volume: `alsamixer`

### No Conversation Indicators:
- Graph may be in low-activity state (normal)
- Speak louder or wait for graph to process
- Check logs for errors

### Graph Not Processing:
- Check if service is running: `./tools/melvin_service.sh status`
- Check logs: `./tools/melvin_service.sh logs`
- Restart: `./tools/melvin_service.sh restart`

## Summary

**You'll know it's on when:**
1. ✅ Service status shows "RUNNING"
2. ✅ Console shows 🟢 ACTIVE or audio bytes increasing
3. ✅ You hear echo from mic → speaker
4. ✅ Dashboard shows growing nodes/edges
5. ✅ Logs show tool calls and processing

**Conversation is ready when:**
- All tools working (tested)
- Conversation path exists (verified)
- Graph is processing (🟢 ACTIVE)
- Audio I/O working (bytes read/written)

**It's working!** 🎉


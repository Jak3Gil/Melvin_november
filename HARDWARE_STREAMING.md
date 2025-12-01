# Hardware Streaming Guide: USB Cameras, Mics, and Speakers

## ✅ Yes, Melvin Supports Continuous Hardware Streaming!

Melvin has **built-in support** for:
- **USB Cameras** (via V4L2)
- **USB Microphones** (via ALSA)
- **USB Speakers** (via ALSA)

All with **continuous streaming** - no stopping, runs 24/7.

---

## 🎥 How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│  Hardware Threads (Continuous I/O)      │
├─────────────────────────────────────────┤
│  Mic Thread → Port 0 → Graph           │
│  Camera Thread → Port 10 → Graph       │
│  Graph → Port 100 → Speaker Thread     │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Main Loop (Continuous Processing)      │
├─────────────────────────────────────────┤
│  melvin_call_entry() → UEL Physics     │
│  melvin_tool_layer_invoke() → Tools    │
└─────────────────────────────────────────┘
```

### Key Components

1. **Hardware Threads**: Read/write hardware continuously
2. **Main Loop**: Processes graph continuously
3. **Tool Layer**: Invokes tools when graph activates gateways
4. **Event-Driven**: Hardware feeds bytes → Graph processes → Tools invoked

---

## 🎤 Audio Streaming (Mic + Speaker)

### How It Works

**Microphone (Input)**:
```c
// Audio reader thread (runs continuously)
while (running) {
    // Read audio samples from USB mic (ALSA)
    snd_pcm_readi(capture_handle, buffer, frames);
    
    // Feed each byte to graph via port 0
    for (each byte in buffer) {
        melvin_feed_byte(g, 0, byte, 0.3f);  // Port 0 = mic input
    }
    
    // Also feed to working memory (200-209) and STT gateway (300)
    melvin_feed_byte(g, 300, buffer[0], 0.4f);  // Triggers STT tool
}
```

**Speaker (Output)**:
```c
// Audio writer thread (runs continuously)
while (running) {
    // Monitor output port 100 for audio data
    if (graph->nodes[100].a > threshold) {
        // Collect audio bytes from graph
        // Write to USB speaker (ALSA)
        snd_pcm_writei(playback_handle, audio_data, frames);
    }
}
```

### Port Mapping

- **Port 0**: Microphone input (raw audio bytes)
- **Port 100**: Speaker output (audio bytes to play)
- **Port 200-209**: Working memory (audio processing)
- **Port 300-309**: STT gateway (speech-to-text tool)
- **Port 600-609**: TTS gateway (text-to-speech tool)

### Continuous Flow

```
Mic → Port 0 → Graph → STT Tool → Text → LLM → TTS Tool → Port 100 → Speaker
```

---

## 📷 Video Streaming (Cameras)

### How It Works

**Camera (Input)**:
```c
// Video reader thread (runs continuously)
while (running) {
    for (each camera) {
        // Read frame from USB camera (V4L2)
        int frame_size = read_camera_frame(camera, frame_buffer);
        
        // Feed frame bytes to graph via port 10 + camera_index
        uint32_t port = 10 + cam_idx;  // Port 10 = camera 0, 11 = camera 1
        
        // Feed in chunks (256 bytes at a time)
        for (each chunk) {
            for (each byte) {
                melvin_feed_byte(g, port, byte, 0.15f);
            }
        }
        
        // Also feed to working memory (201-210) and Vision gateway (400)
        melvin_feed_byte(g, 400, frame_buffer[0], 0.4f);  // Triggers Vision tool
    }
}
```

### Port Mapping

- **Port 10**: Camera 0 input (raw frame bytes)
- **Port 11**: Camera 1 input (raw frame bytes)
- **Port 201-210**: Working memory (video processing)
- **Port 400-409**: Vision gateway (image recognition tool)

### Continuous Flow

```
Camera → Port 10 → Graph → Vision Tool → Labels → Graph → Patterns
```

---

## 🚀 Running with Hardware

### Basic Usage

```bash
# Run with hardware (auto-detects devices)
./melvin_hardware_runner brain.m

# Specify devices explicitly
./melvin_hardware_runner brain.m \
    "default" \          # Audio capture (mic)
    "default" \          # Audio playback (speaker)
    "/dev/video0" \      # Camera 0
    "/dev/video2"        # Camera 1 (optional)
```

### What Happens

1. **Hardware Initialization**:
   - Opens USB mic (ALSA)
   - Opens USB speaker (ALSA)
   - Opens USB cameras (V4L2)
   - Starts reader/writer threads

2. **Continuous Streaming**:
   - Mic thread reads audio → feeds to graph
   - Camera threads read frames → feed to graph
   - Main loop processes graph continuously
   - Tool layer invokes tools when gateways activate
   - Speaker thread writes audio from graph

3. **Learning**:
   - Graph learns patterns from audio/video
   - Tools process data (STT, Vision, TTS)
   - Graph learns to route audio/video through tools
   - Patterns form around repeated sequences

---

## 🔧 Technical Details

### Audio (ALSA)

**Format**:
- **Sample Rate**: 16kHz (configurable)
- **Channels**: Stereo (2 channels)
- **Format**: 16-bit PCM
- **Buffer**: 1024 samples (~64ms)

**Devices**:
- **Capture**: `default` or specific ALSA device (e.g., `hw:0,0`)
- **Playback**: `default` or specific ALSA device

**Threads**:
- **Reader**: Continuously reads from mic, feeds to graph
- **Writer**: Continuously monitors port 100, writes to speaker

### Video (V4L2)

**Format**:
- **Resolution**: 640x480 (configurable)
- **FPS**: 30 (configurable)
- **Format**: MJPEG (preferred) or YUYV (fallback)
- **Buffers**: 4 frame buffers (V4L2 MMAP)

**Devices**:
- **Camera 0**: `/dev/video0` (default)
- **Camera 1**: `/dev/video2` or `/dev/video1` (optional)

**Threads**:
- **Reader**: Continuously reads frames, feeds to graph in chunks

---

## 📊 Continuous Processing

### Main Loop

```c
while (running) {
    // 1. Process graph (UEL physics)
    melvin_call_entry(g);
    
    // 2. Invoke tools (when gateways activate)
    melvin_tool_layer_invoke(g);
    
    // 3. Sync to disk periodically
    if (time_to_sync) {
        melvin_sync(g);
    }
}
```

### Event Flow

```
Hardware Event → Feed Byte → Queue Node → Process → Tools → Output
     ↓              ↓            ↓           ↓        ↓        ↓
  Mic reads    melvin_feed  prop_queue  UEL physics  STT    Speaker
  Camera       _byte()      _add()      processes    Vision  writes
  captures     (port 0/10)  (node_id)   (energy)     TTS    (port 100)
```

---

## 🎯 What the System Learns

### From Audio

- **Speech patterns**: Common words, phrases
- **Audio features**: Frequency patterns, rhythms
- **Conversation flow**: Question → Answer patterns
- **Tool usage**: When to use STT, when to use TTS

### From Video

- **Visual patterns**: Objects, scenes, movements
- **Spatial relationships**: Object positions, layouts
- **Temporal patterns**: Motion, sequences
- **Tool usage**: When to use Vision tool

### From Combined

- **Audio-visual correlation**: Sounds with visual events
- **Multimodal patterns**: Speech + gestures, actions
- **Context understanding**: Situational awareness

---

## ⚙️ Configuration

### Audio Settings

```c
#define AUDIO_SAMPLE_RATE 16000  // 16kHz
#define AUDIO_CHANNELS 2          // Stereo
#define AUDIO_BUFFER_SIZE 1024    // Samples
```

### Video Settings

```c
#define VIDEO_WIDTH 640
#define VIDEO_HEIGHT 480
#define VIDEO_FPS 30
```

### Port Assignments

```c
#define AUDIO_PORT_INPUT 0        // Mic input
#define AUDIO_PORT_OUTPUT 100     // Speaker output
#define VIDEO_PORT_INPUT 10       // Camera input
```

---

## 🔍 Monitoring

### Status Output

```
[100] Nodes: 10000 | Edges: 50000 | Chaos: 0.123 | Activation: 0.456
Audio: 1234567 bytes read, 987654 bytes written
Video: 500 frames read
```

### What to Watch

- **Audio bytes read/written**: Confirms hardware is working
- **Video frames read**: Confirms cameras are capturing
- **Graph activity**: Nodes/edges growing, chaos reducing
- **Tool invocations**: STT, Vision, TTS being called

---

## 🐛 Troubleshooting

### No Audio

```bash
# Check ALSA devices
arecord -l  # List capture devices
aplay -l    # List playback devices

# Test manually
arecord -f cd test.wav  # Record
aplay test.wav          # Play
```

### No Video

```bash
# Check V4L2 devices
v4l2-ctl --list-devices

# Test camera
v4l2-ctl --device=/dev/video0 --stream-mmap --stream-count=1
```

### Thread Issues

- Hardware threads run independently
- Main loop processes graph continuously
- No blocking - all I/O is non-blocking
- Graceful error handling - continues on device errors

---

## 📝 Summary

**✅ Melvin supports continuous hardware streaming**

- **USB Cameras**: V4L2, continuous frame capture
- **USB Microphones**: ALSA, continuous audio capture
- **USB Speakers**: ALSA, continuous audio playback
- **Streaming**: 24/7, no stopping
- **Learning**: Graph learns patterns from hardware input
- **Tools**: STT, Vision, TTS invoked automatically

**Just run `melvin_hardware_runner` and it streams continuously!**


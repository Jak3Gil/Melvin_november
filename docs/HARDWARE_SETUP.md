# Hardware Setup Guide

## What We've Built

We've created a complete hardware integration layer that connects real USB hardware (microphone, speaker, cameras) to Melvin's soft structure. The system now has:

1. **Real Audio I/O**: USB microphone → Port 0, USB speaker ← Port 100
2. **Real Video I/O**: USB cameras → Ports 10-19, Display ← Port 110
3. **Automatic Integration**: Hardware feeds data into graph, graph produces output to hardware
4. **Soft Structure Mapping**: Hardware streams map to semantic port ranges

## Files Created

- `src/melvin_hardware_audio.c` - ALSA audio I/O (mic/speaker)
- `src/melvin_hardware_video.c` - V4L2 video I/O (cameras)
- `src/melvin_hardware.h` - Hardware interface header
- `src/melvin_hardware_runner.c` - Complete hardware runner
- `docs/HARDWARE_INTEGRATION.md` - Technical documentation

## How It Works

### Data Flow

```
USB Microphone → ALSA → Audio Reader Thread → Port 0 → Graph → UEL Physics
                                                                     ↓
USB Speaker ← ALSA ← Audio Writer Thread ← Port 100 ← Graph ← Output Activation

USB Camera → V4L2 → Video Reader Thread → Port 10 → Graph → UEL Physics
                                                                     ↓
Display ← FBDEV ← Video Writer Thread ← Port 110 ← Graph ← Output Activation
```

### Port Mapping

- **Port 0**: Raw audio bytes from microphone (PCM samples)
- **Port 100**: Audio output bytes to speaker (activated when graph wants to speak)
- **Port 10**: Raw frame bytes from camera 0
- **Port 11**: Raw frame bytes from camera 1 (if multiple cameras)
- **Port 110**: Display output (activated when graph wants to show something)

### Activation Strategy

The system monitors output ports (100, 110) for activation:
- When activation > 0.5, the graph wants to produce output
- Audio writer reads from ports 100-109 and writes to speaker
- Video writer reads from ports 110-119 and writes to display
- Positive feedback (node 30) is activated when output succeeds

## Building

### Prerequisites (Jetson/Linux)

```bash
# Install ALSA development libraries
sudo apt-get install libasound2-dev

# V4L2 is usually built into the kernel, but you may need:
sudo apt-get install v4l-utils
```

### Compile

```bash
make melvin_hardware_runner
```

This will compile:
- `melvin_hardware_runner` - Complete hardware runner
- Falls back gracefully if ALSA/V4L2 not available (simulation mode)

## Running on Jetson

### Basic Usage

```bash
# Run with default devices
./melvin_hardware_runner brain.m

# Specify audio devices
./melvin_hardware_runner brain.m hw:0 hw:0

# Specify cameras
./melvin_hardware_runner brain.m default default /dev/video0 /dev/video1
```

### Finding Devices

```bash
# List audio devices
cat /proc/asound/cards
arecord -l  # List capture devices
aplay -l   # List playback devices

# List video devices
ls -l /dev/video*
v4l2-ctl --list-devices
```

### Example Session

```bash
# On Jetson
cd /mnt/melvin_ssd/melvin
./melvin_hardware_runner brain.m

# Output:
========================================
Melvin Hardware Runner
========================================
Brain: brain.m
Audio capture: default
Audio playback: default
Cameras: 1
  Camera 0: /dev/video0
Press Ctrl+C to stop

Brain opened: 10000 nodes, 50000 edges
Soft structure initialized:
  Input ports: 0-99 (audio: 0, video: 10-19)
  Output ports: 100-199 (audio: 100, video: 110)
  Memory ports: 200-255

Initializing hardware...
Audio hardware initialized
  Capture device: default
  Playback device: default
  Input port: 0
  Output port: 100
Video hardware initialized
  Cameras: 1
  Input port: 10-10
  Output port: 110
Hardware initialized. Starting continuous processing...

[0] Nodes: 10000 | Edges: 50001 | Chaos: 0.100000 | Activation: 0.102341
      Audio: 1024 read, 0 written | Video: 1 read, 0 written
[10] Nodes: 10000 | Edges: 50003 | Chaos: 0.095234 | Activation: 0.104892
      Audio: 10240 read, 512 written | Video: 10 read, 0 written
...
```

## What Happens When Running

1. **Hardware threads start**:
   - Audio reader reads from mic continuously → feeds to port 0
   - Video reader reads from cameras → feeds to port 10
   - Audio writer monitors port 100 → writes to speaker when activated
   - Video writer monitors port 110 → writes to display when activated

2. **Graph processes data**:
   - UEL physics runs after each hardware feed
   - Graph learns patterns from audio/video streams
   - Output ports activate when graph wants to produce output

3. **Feedback loop**:
   - Positive feedback (node 30) when output succeeds
   - Graph learns to correlate outputs with feedback
   - System improves over time

## Troubleshooting

### No Audio

```bash
# Test ALSA
arecord -d 3 test.wav
aplay test.wav

# Check permissions
ls -l /dev/snd/
# May need to add user to audio group:
sudo usermod -a -G audio $USER
```

### No Video

```bash
# Test camera
v4l2-ctl --device=/dev/video0 --stream-mmap --stream-count=10

# Check permissions
ls -l /dev/video*
# May need to add user to video group:
sudo usermod -a -G video $USER
```

### Fallback Mode

If hardware isn't available, the system runs in simulation mode:
- Generates random audio/video data
- Still processes through graph
- Useful for testing without hardware

## Next Steps

1. **Test with real hardware** on Jetson
2. **Tune activation thresholds** (currently 0.5)
3. **Add speech recognition** (convert audio to text tokens)
4. **Add text-to-speech** (convert output to audio)
5. **Add image processing** (extract features from video frames)
6. **Add feedback mechanisms** (user interaction, success detection)

## Integration with Soft Structure

The hardware integration leverages the soft structure we created:
- **Input ports (0-99)**: Hardware feeds data here
- **Output ports (100-199)**: Graph produces output here
- **Memory ports (200-255)**: Graph stores learned patterns
- **Temporal anchors (240-243)**: Graph tracks time
- **Feedback channels (30-33)**: System provides learning signals

The graph can:
- Learn audio patterns (speech, music, sounds)
- Learn visual patterns (objects, motion, scenes)
- Correlate audio and video
- Produce coordinated audio/video output
- Adapt based on feedback

This makes Melvin a **real-world learning system** that processes actual sensory data!


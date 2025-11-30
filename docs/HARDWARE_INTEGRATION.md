# Hardware Integration with Soft Structure

## Overview

This document describes how to connect real hardware (USB mic, speaker, cameras) to Melvin's soft structure, enabling the system to process real-world sensory data and produce real outputs.

## Port Mapping Strategy

The soft structure defines semantic port ranges. We map hardware streams to these ports:

### Audio Input (Microphone) → Input Ports
- **Ports 0-9**: Text input ports (primary)
  - **Port 0**: Raw audio bytes (PCM samples)
  - **Port 1**: Audio features (frequency bands, volume)
  - **Port 2**: Speech detection (voice activity)
  - **Port 3-9**: Reserved for processed audio features

### Visual Input (Cameras) → Input Ports  
- **Ports 10-19**: Sensor data ports
  - **Port 10**: Camera 0 - raw frame bytes
  - **Port 11**: Camera 0 - processed features (edges, motion)
  - **Port 12**: Camera 1 - raw frame bytes (if multiple cameras)
  - **Port 13**: Camera 1 - processed features
  - **Port 14-19**: Reserved for additional camera streams/features

### Audio Output (Speaker) ← Output Ports
- **Ports 100-109**: Text output ports (primary)
  - **Port 100**: Audio output bytes (PCM samples)
  - **Port 101**: Speech synthesis control
  - **Port 102-109**: Reserved for audio output channels

### Visual Output (Display/Actuators) ← Output Ports
- **Ports 110-119**: Action output ports
  - **Port 110**: Display frame output
  - **Port 111**: Motor/actuator control
  - **Port 112-119**: Reserved for additional actuators

## Hardware Access Layer

### Linux/Jetson Hardware APIs

**Audio (ALSA/PulseAudio)**:
- Microphone: Read PCM samples via ALSA/PulseAudio
- Speaker: Write PCM samples via ALSA/PulseAudio
- Format: 16-bit PCM, 16kHz sample rate (configurable)

**Video (V4L2)**:
- Cameras: Read frames via Video4Linux2 (V4L2)
- Format: MJPEG or raw YUV (convert to bytes)
- Resolution: 320x240 or 640x480 (configurable)

### Integration Points

1. **Hardware Reader Thread**: Continuously reads from mic/cameras
2. **Graph Feeder**: Feeds hardware data into graph via `melvin_feed_byte()`
3. **Graph Monitor**: Monitors output ports for activation
4. **Hardware Writer Thread**: Writes activated output to speaker/display

## Data Flow

```
USB Microphone → [ALSA/Pulse] → Audio Reader → Port 0 (raw audio bytes)
                                                      ↓
                                              Graph Processing
                                                      ↓
USB Speaker ← [ALSA/Pulse] ← Audio Writer ← Port 100 (audio output)

USB Camera → [V4L2] → Frame Reader → Port 10 (raw frame bytes)
                                          ↓
                                    Graph Processing
                                          ↓
Display ← [FBDEV/DRM] ← Frame Writer ← Port 110 (display output)
```

## Implementation Strategy

### Phase 1: Audio I/O
1. Add ALSA/PulseAudio syscalls to `MelvinSyscalls`
2. Create `melvin_hardware_audio.c` for mic/speaker access
3. Create continuous reader/writer threads
4. Feed audio bytes to port 0, read from port 100

### Phase 2: Video I/O
1. Add V4L2 syscalls to `MelvinSyscalls`
2. Create `melvin_hardware_video.c` for camera access
3. Convert frames to byte streams
4. Feed frame bytes to port 10, read from port 110

### Phase 3: Integration
1. Create `melvin_hardware_runner.c` that combines all I/O
2. Continuous loop: read hardware → feed graph → process → write hardware
3. Monitor output port activations to trigger hardware writes

## Port Activation Strategy

### Reading Output Ports

The graph activates output ports (100-199) when it wants to produce output. We need to:

1. **Monitor output ports** periodically (every 100ms or so)
2. **Detect activation** (activation > threshold, e.g., 0.5)
3. **Read activated nodes** and convert to hardware output
4. **Provide feedback** (activate feedback nodes based on success)

### Example: Audio Output

```c
// Monitor port 100 (audio output)
float activation = melvin_get_activation(g, 100);
if (activation > 0.5f) {
    // Graph wants to produce audio
    // Read surrounding nodes (101-109) for audio data
    uint8_t audio_bytes[1024];
    int count = 0;
    for (int i = 100; i < 110 && count < 1024; i++) {
        float a = melvin_get_activation(g, i);
        if (a > 0.3f) {
            audio_bytes[count++] = (uint8_t)(a * 255.0f);  // Convert activation to byte
        }
    }
    // Write to speaker
    write_audio_output(audio_bytes, count);
    
    // Provide feedback (positive if audio was produced)
    melvin_feed_byte(g, 30, 1, 0.5f);  // Positive feedback node
}
```

## Feedback Mechanism

### Audio Feedback
- **Positive feedback (node 30)**: Activate when audio output is clear/understood
- **Negative feedback (node 31)**: Activate when audio output is unclear/noisy
- **Uncertainty (node 32)**: Activate when output is ambiguous

### Visual Feedback
- **Positive feedback**: Activate when camera detects expected objects/patterns
- **Negative feedback**: Activate when camera detects unexpected/confusing scenes

## Continuous Operation

The hardware runner should:
1. **Read hardware continuously** (mic, cameras) in separate threads
2. **Feed data to graph** via appropriate input ports
3. **Trigger UEL propagation** after each feed
4. **Monitor output ports** for activation
5. **Write to hardware** when output ports activate
6. **Provide feedback** based on hardware state/success

## Jetson-Specific Considerations

- **USB devices**: May need udev rules for consistent device paths
- **ALSA devices**: Check `/proc/asound/cards` for available audio devices
- **V4L2 devices**: Check `/dev/video*` for available cameras
- **Performance**: Use separate threads for I/O to avoid blocking graph processing
- **Power**: Jetson may throttle - monitor temperature/performance

## Next Steps

1. Implement `melvin_hardware_audio.c` with ALSA/PulseAudio access
2. Implement `melvin_hardware_video.c` with V4L2 access
3. Create `melvin_hardware_runner.c` that integrates everything
4. Test with real USB mic/speaker/camera on Jetson
5. Tune port mappings and activation thresholds


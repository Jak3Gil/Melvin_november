# Hardware Runner Design - Teachable System

**Goal**: Runner that feeds hardware data to brain, brain learns and responds autonomously

---

## 🎯 THE ARCHITECTURE

```
┌─────────────────────────────────────────────────┐
│ USB HARDWARE                                     │
│ Mic (card 0) → Cameras (video0-3)               │
└────────────┬────────────────────────────────────┘
             │ Raw bytes
             ↓
┌─────────────────────────────────────────────────┐
│ CAPTURE THREADS                                  │
│ - audio_capture_thread() → Port 0               │
│ - camera1_thread() → Port 10                     │
│ - camera2_thread() → Port 11                     │
└────────────┬────────────────────────────────────┘
             │ Continuous stream
             ↓
┌─────────────────────────────────────────────────┐
│ BRAIN (brain.m on 4TB SSD)                       │
│                                                   │
│ Pattern Discovery (auto):                        │
│   Level 1: Raw audio/video patterns              │
│   Level 2: Feature patterns (composed)           │
│   Level 3: Semantic patterns (composed)          │
│   Level 4: Behavior patterns (composed)          │
│                                                   │
│ EXEC Nodes (taught):                             │
│   Node 2000: Audio playback (ARM64)              │
│   Node 2001: GPIO control (ARM64)                │
│   Node 2002: Servo control (ARM64)               │
│                                                   │
│ Routing (learned):                               │
│   Patterns → EXEC (edges strengthen with use)    │
└────────────┬────────────────────────────────────┘
             │ EXEC outputs
             ↓
┌─────────────────────────────────────────────────┐
│ OUTPUT DISPATCHER                                 │
│ - Check EXEC output buffer                       │
│ - Execute hardware commands                      │
└────────────┬────────────────────────────────────┘
             │ Commands
             ↓
┌─────────────────────────────────────────────────┐
│ USB HARDWARE OUTPUTS                              │
│ Speaker, LEDs, Motors                             │
└───────────────────────────────────────────────────┘
```

**Everything learned, nothing hardcoded!**

---

## 💡 **HOW TO ADD NEW HARDWARE**

### **Example: Add USB Temperature Sensor**

**Step 1: Connect physically**
```bash
# Plug in USB thermometer
# Check it appears:
lsusb | grep -i temp
ls /dev/ttyUSB*  # If serial device
```

**Step 2: Feed pattern to brain** (NO code changes!)
```bash
# Teach brain about the sensor by feeding pattern
for i in {1..10}; do
    echo "TEMPERATURE_SENSOR_PORT_30" | \
        ./tools/feed_pattern /mnt/melvin_ssd/brains/main_brain.m 30
done

# Pattern created! Brain knows about port 30 now.
```

**Step 3: Start feeding data**
```bash
# Read sensor and feed to Melvin
while true; do
    temp=$(cat /dev/ttyUSB0)  # Read temp
    echo "TEMP_$temp" | \
        ./tools/feed_pattern /mnt/melvin_ssd/brains/main_brain.m 30
    sleep 1
done
```

**Step 4: Brain learns automatically**
```
Brain discovers:
- Temperature varies with time
- High temp co-activates with [some pattern]
- Learns: high_temp → activate_fan EXEC
- All through co-activation!

NO code changes needed!
```

---

## 🔧 **HARDWARE RUNNER CODE**

### **Minimal Implementation** (`melvin_hardware_runner.c`):

```c
/* Teachable Hardware Runner - Feeds hardware data, brain learns */

#include "src/melvin.h"
#include <pthread.h>
#include <alsa/asoundlib.h>  /* For USB audio */
#include <linux/videodev2.h>  /* For USB cameras */

#define BRAIN_PATH "/mnt/melvin_ssd/brains/main_brain.m"

/* Audio capture thread */
void* audio_thread(void* arg) {
    Graph *brain = (Graph*)arg;
    
    /* Open USB mic (card 0, device 0) */
    snd_pcm_t *capture_handle;
    snd_pcm_open(&capture_handle, "hw:0,0", SND_PCM_STREAM_CAPTURE, 0);
    
    /* Set format: 16kHz mono */
    /* ... ALSA setup ... */
    
    /* Continuous capture */
    uint8_t buffer[1024];
    while (1) {
        /* Read audio samples */
        snd_pcm_readi(capture_handle, buffer, 512);
        
        /* Feed to Melvin port 0 */
        for (int i = 0; i < 512; i++) {
            melvin_feed_byte(brain, 0, buffer[i], 1.0f);
        }
        
        /* Let brain process every N bytes */
        if (bytes_fed % 100 == 0) {
            melvin_call_entry(brain);
        }
    }
}

/* Camera capture thread */
void* camera_thread(void* arg) {
    Graph *brain = (Graph*)arg;
    
    /* Open camera */
    int fd = open("/dev/video0", O_RDWR);
    
    /* Set format: 640x480 MJPEG */
    /* ... V4L2 setup ... */
    
    /* Continuous capture */
    while (1) {
        /* Capture frame */
        uint8_t frame[307200];  /* 640x480 */
        read(fd, frame, sizeof(frame));
        
        /* Feed to Melvin port 10 (sample every Nth byte) */
        for (int i = 0; i < sizeof(frame); i += 100) {
            melvin_feed_byte(brain, 10, frame[i], 0.8f);
        }
        
        /* Process */
        melvin_call_entry(brain);
    }
}

int main() {
    /* Open brain on SSD */
    Graph *brain = melvin_open(BRAIN_PATH, 10000, 50000, 131072);
    
    if (!brain) {
        fprintf(stderr, "❌ Failed to open brain\n");
        return 1;
    }
    
    printf("✅ Brain opened from SSD\n");
    printf("   File: %s\n", BRAIN_PATH);
    printf("   Nodes: %llu\n", brain->node_count);
    printf("   Starting hardware learning...\n\n");
    
    /* Start hardware threads */
    pthread_t audio_tid, camera_tid;
    pthread_create(&audio_tid, NULL, audio_thread, brain);
    pthread_create(&camera_tid, NULL, camera_thread, brain);
    
    /* Main loop: just run UEL and check outputs */
    while (1) {
        /* UEL propagation (brain learns patterns) */
        melvin_call_entry(brain);
        
        /* Check if EXEC nodes want to output */
        /* (would check output buffer or specific nodes) */
        
        /* Autosave every 60 seconds */
        static time_t last_save = 0;
        if (time(NULL) - last_save > 60) {
            melvin_close(brain);
            brain = melvin_open(BRAIN_PATH, 10000, 50000, 131072);
            last_save = time(NULL);
        }
        
        usleep(10000);  /* 10ms */
    }
    
    return 0;
}
```

**Brain learns from continuous hardware stream!**

---

## 🎓 **TEACHABILITY EXAMPLES**

### **Example 1: Add Light Sensor**

```bash
# 1. Connect sensor to GPIO pin 12

# 2. Feed pattern (teach brain about sensor)
for i in {1..10}; do
    echo "LIGHT_SENSOR_GPIO_12" | feed_pattern brain.m 40
done

# 3. Start feeding readings
while true; do
    light=$(gpio_read 12)
    echo "LIGHT_$light" | feed_pattern brain.m 40
done

# Brain learns:
# - Light varies with time
# - Dark → turn_on_light EXEC
# - Bright → turn_off_light EXEC
# All autonomous!
```

---

### **Example 2: Add Motor Controller**

```bash
# 1. Connect motor to PWM pin

# 2. Teach brain the control operation
./tools/teach_operation brain.m motor_control_arm64_code.bin "MOTOR_CONTROL"

# 3. Feed pattern
echo "MOTOR_PWM_PIN_5" | feed_pattern brain.m 50

# 4. Brain learns:
# - When to activate motor (through pattern co-activation)
# - How much power (through EXEC parameters)
# - All from experience!
```

---

## 🚀 **NEXT STEPS**

### **1. Create Simple Hardware Runner** (2 hours)
```c
// Just audio for now
// Proves the concept
// Can add camera/motor later
```

### **2. Test on Jetson** (1 hour)
```bash
# Run with USB mic/speaker
# Feed audio data
# Brain discovers patterns
# Verify learning works
```

### **3. Add Camera Integration** (2 hours)
```c
// Feed video frames
// Brain learns visual patterns
// Composes with audio
```

### **4. Full Multimodal** (1 week)
```c
// All hardware integrated
// Hierarchical composition
// Emergent behaviors!
```

---

## 🎯 **SUMMARY**

**Hardware Status**:
- ✅ 4TB SSD ready (3.7TB free!)
- ✅ USB audio working (mic + speaker)
- ✅ USB cameras ready (2+ devices)
- ✅ All drivers installed

**Software Status**:
- ✅ Teachable brain created (1.9MB)
- ✅ Tools built and working
- ✅ 148 patterns pre-seeded
- ✅ 100 bootstrap edges
- ✅ 5 EXEC operations taught

**Ready to Run**:
- Brain on 4TB SSD ✅
- Hardware connected ✅
- Drivers configured ✅
- System dedicated to Melvin ✅

**Adding new hardware**: Just feed patterns! NO code changes! ✨

**Want me to create the hardware runner?** 🤖



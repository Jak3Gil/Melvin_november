# Jetson Hardware Status & Configuration

**Date**: December 2, 2025  
**System**: Jetson AGX Orin (Dedicated Melvin System)

---

## 💾 **STORAGE STATUS**

### **Main Storage** (eMMC):
```
/dev/mmcblk0p1:  57GB total
                 32GB used
                 23GB free (59% used)
```

### **4TB NVMe SSD** ✅:
```
/dev/nvme0n1p2:  3.7TB total
Mounted at:      /mnt/melvin_ssd
Used:            238MB (0%)
Available:       3.7TB FREE! 🎉

Perfect for:
- Massive brain files (can grow to TB scale!)
- Pattern storage
- Blob code library
- Training data corpus
```

**Recommendation**: Store all brain files on SSD:
```bash
/mnt/melvin_ssd/brains/
├── hardware_brain.m (starting ~2MB, can grow to GB/TB!)
├── learned_patterns/
└── blob_library/
```

---

## 🎤 **AUDIO HARDWARE**

### **USB Audio Device** ✅:
```
Card 0: AB13X USB Audio
  Capture:  ✅ (Microphone)
  Playback: ✅ (Speaker)
  
ALSA Device:
  Record:  hw:0,0 (card 0, device 0)
  Play:    hw:0,0 (same device)
```

**Status**: Mic + Speaker on SAME USB device - perfect for full-duplex!

**Test Commands**:
```bash
# Test microphone
arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 test.wav

# Test speaker  
aplay -D hw:0,0 test.wav
```

---

## 📷 **CAMERA HARDWARE**

### **USB Cameras** ✅:
```
/dev/video0  ✅ Camera 1
/dev/video1  ✅ Camera 2
/dev/video2  (Additional)
/dev/video3  (Additional)
```

**Status**: At least 2 cameras available!

**Test Commands**:
```bash
# Test camera 1
v4l2-ctl --device=/dev/video0 --list-formats

# Capture frame
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 frame.jpg
```

---

## 🔌 **USB DEVICES**

```
USB Hubs:     2x (providing ports for expansion)
USB Audio:    AB13X (mic + speaker)
USB Serial:   HL-340 adapter
USB Bluetooth: Realtek radio
```

**Status**: Well-equipped for I/O!

---

## 🎯 **DEDICATED MELVIN SYSTEM**

### **Current Use**:
- Jetson is 100% dedicated to Melvin
- No other services needed
- Can optimize everything for Melvin

### **Recommendations**:

**1. Use 4TB SSD for Brain Storage**:
```bash
# Move brain files to SSD
mkdir -p /mnt/melvin_ssd/brains
ln -s /mnt/melvin_ssd/brains /home/melvin/brains

# Brain files can grow HUGE!
# Pattern library: Up to 1TB+
# Blob code library: Up to 100GB+
# Training corpus: Up to 1TB+
```

**2. Disable Unnecessary Services**:
```bash
# Free up RAM and CPU for Melvin
sudo systemctl disable bluetooth
sudo systemctl disable cups  
sudo systemctl disable avahi-daemon
# Keep only: SSH, networking, USB drivers
```

**3. Set USB Power Management**:
```bash
# Prevent USB devices from suspending
sudo sh -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend'
```

---

## 🔧 **DRIVER CONFIGURATION**

### **Audio Drivers** ✅:

```bash
# ALSA Configuration for USB Audio
cat > /home/melvin/.asoundrc << EOF
pcm.!default {
    type hw
    card 0
    device 0
}

ctl.!default {
    type hw
    card 0
}
EOF

# Test
arecord -D default -d 3 -f S16_LE -r 16000 test.wav
aplay -D default test.wav
```

---

### **Video Drivers** ✅:

```bash
# V4L2 should be built-in
# Check camera capabilities:
v4l2-ctl --device=/dev/video0 --all

# Set camera format:
v4l2-ctl --device=/dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat=MJPG
```

---

## 🎯 **PORT MAPPING FOR HARDWARE**

### **Melvin Port Assignments**:

```
INPUT PORTS:
  Port 0:  USB Mic (card 0, device 0, capture)
  Port 10: USB Camera 1 (/dev/video0)
  Port 11: USB Camera 2 (/dev/video1)
  Port 12: USB Camera 3 (/dev/video2) [if connected]
  Port 13: USB Camera 4 (/dev/video3) [if connected]
  
SEMANTIC PORTS (AI outputs):
  Port 100: Whisper transcription text
  Port 110: MobileNet vision labels
  Port 120: Ollama reasoning/context
  
OUTPUT PORTS:
  Port 200: USB Speaker (card 0, device 0, playback)
  Port 210: GPIO outputs (LEDs, etc.)
  Port 220: PWM outputs (servos, motors)
```

---

## 🚀 **ADDING NEW HARDWARE** (By Feeding Patterns!)

### **Example: Add a new USB sensor**

**Step 1: Connect Hardware**
```bash
# Plug in new USB device
# Check it appears:
lsusb  # See new device
```

**Step 2: Feed Pattern to Brain** (NO CODE CHANGES!)
```bash
# Just feed the pattern!
echo "NEW_SENSOR_PORT_30" | ./feed_pattern brain.m 30

# Repeat 10 times to create strong pattern
for i in {1..10}; do
    echo "USB_THERMOMETER_PORT_30" | ./feed_pattern brain.m 30
done
```

**Step 3: Start Feeding Data**
```bash
# Read from sensor and feed to port 30
while true; do
    temp=$(read_thermometer)
    echo "$temp" | ./feed_pattern brain.m 30
done
```

**Step 4: Brain Learns Autonomously!**
```
Brain discovers:
- Temperature patterns
- Co-activation with other sensors
- When temperature matters
- What to do about it (through EXEC routing)

NO hardcoding needed!
```

---

## 🎯 **SYSTEM CONFIGURATION**

### **1. Brain Storage on SSD**:

```bash
# Create brain directory on SSD
ssh melvin@jetson
sudo mkdir -p /mnt/melvin_ssd/brains
sudo chown melvin:melvin /mnt/melvin_ssd/brains

# Create brain there
cd /home/melvin/teachable_system
./create_teachable_hardware_brain.sh /mnt/melvin_ssd/brains/main_brain.m

# Symlink for easy access
ln -s /mnt/melvin_ssd/brains /home/melvin/brains
```

---

### **2. Hardware Runner Configuration**:

```bash
# Create config file
cat > /home/melvin/hardware_config.conf << EOF
# Melvin Hardware Configuration

[brain]
path = /mnt/melvin_ssd/brains/main_brain.m
autosave_interval = 60  # seconds

[audio]
capture_device = hw:0,0
playback_device = hw:0,0
sample_rate = 16000
channels = 1

[video]
camera1 = /dev/video0
camera2 = /dev/video1
resolution = 640x480
fps = 30

[ports]
audio_in = 0
camera1_in = 10
camera2_in = 11
whisper_out = 100
vision_out = 110
speaker_out = 200
EOF
```

---

### **3. Auto-Start on Boot**:

```bash
# Create systemd service
sudo cat > /etc/systemd/system/melvin-hardware.service << 'EOF'
[Unit]
Description=Melvin Hardware Learning System
After=network.target

[Service]
Type=simple
User=melvin
WorkingDirectory=/home/melvin/teachable_system
ExecStart=/home/melvin/teachable_system/melvin_hardware_runner /mnt/melvin_ssd/brains/main_brain.m
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable
sudo systemctl enable melvin-hardware
sudo systemctl start melvin-hardware
```

---

## 📊 **CURRENT STATUS**

| Component | Status | Device | Notes |
|-----------|--------|--------|-------|
| **4TB SSD** | ✅ Ready | /mnt/melvin_ssd | 3.7TB free! |
| **USB Audio** | ✅ Ready | Card 0, hw:0,0 | Mic + Speaker |
| **USB Cameras** | ✅ Ready | /dev/video0-3 | 2-4 cameras |
| **Drivers** | ✅ Installed | ALSA, V4L2 | Built-in |
| **Teachable Brain** | ✅ Created | 1.9MB | On Jetson |
| **Tools** | ✅ Built | All 3 tools | Working |

---

## 🎉 **READY TO RUN!**

**Everything is in place**:
- ✅ 4TB SSD for massive brain growth
- ✅ USB audio working (mic + speaker)
- ✅ USB cameras ready (2+ cameras)
- ✅ Teachable brain created
- ✅ Tools built and working
- ✅ No hardcoding - pure learning!

**Next**: Create the hardware runner and start learning!



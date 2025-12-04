#!/bin/bash
# Fix Audio Routing - Force USB Speaker as Default

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  AUDIO ROUTING FIX                                    ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

echo "1. CURRENT AUDIO SETUP"
echo "════════════════════════════════════════════════════════"
cat /proc/asound/cards
echo ""

echo "2. CHECKING DEFAULT DEVICE"
echo "════════════════════════════════════════════════════════"
cat ~/.asoundrc 2>/dev/null || echo "No ~/.asoundrc file"
cat /etc/asound.conf 2>/dev/null || echo "No /etc/asound.conf file"
echo ""

echo "3. SETTING USB SPEAKER (card 0) AS DEFAULT"
echo "════════════════════════════════════════════════════════"
cat > ~/.asoundrc << 'EOF'
defaults.pcm.card 0
defaults.pcm.device 0
defaults.ctl.card 0
EOF

echo "✅ Created ~/.asoundrc to force USB as default"
cat ~/.asoundrc
echo ""

echo "4. CHECKING PULSEAUDIO (if running)"
echo "════════════════════════════════════════════════════════"
if pgrep pulseaudio >/dev/null; then
    echo "PulseAudio is running"
    
    # List current sinks
    pactl list sinks short
    echo ""
    
    # Find USB sink
    USB_SINK=$(pactl list sinks short | grep -i "usb\|AB13X" | awk '{print $1}')
    if [ -n "$USB_SINK" ]; then
        echo "Setting USB sink as default: $USB_SINK"
        pactl set-default-sink $USB_SINK
        echo "✅ USB speaker set as PulseAudio default"
    else
        echo "⚠️  Could not find USB sink in PulseAudio"
    fi
else
    echo "PulseAudio not running (using ALSA directly - that's fine)"
fi
echo ""

echo "5. MAXIMIZING USB SPEAKER VOLUME"
echo "════════════════════════════════════════════════════════"
amixer -c 0 scontrols
echo ""
amixer -c 0 set Master 100% unmute 2>&1 | tail -2
amixer -c 0 set PCM 100% unmute 2>&1 | tail -2
amixer -c 0 set Speaker 100% unmute 2>&1 | tail -2
amixer -c 0 set Headphone 100% unmute 2>&1 | tail -2
echo ""

echo "6. TESTING USB SPEAKER"
echo "════════════════════════════════════════════════════════"
echo "🔊 Playing test tone NOW (you should hear it!)..."
echo ""

# Try multiple methods
echo "Method 1: Using default device (should be USB now)"
timeout 3 speaker-test -t sine -f 800 -c 2 -l 1 2>&1 | head -10 &
PID1=$!
sleep 3
kill $PID1 2>/dev/null
wait $PID1 2>/dev/null
echo ""

echo "Method 2: Explicit USB device"
timeout 3 speaker-test -D plughw:0,0 -t sine -f 1000 -c 2 -l 1 2>&1 | head -10 &
PID2=$!
sleep 3
kill $PID2 2>/dev/null
wait $PID2 2>/dev/null
echo ""

echo "Method 3: Playing recorded audio"
if [ -f /tmp/hw_test_mic.wav ]; then
    echo "Playing your microphone recording..."
    aplay /tmp/hw_test_mic.wav 2>&1 | head -5
else
    # Create a beep
    sox -n /tmp/test_beep.wav synth 1 sine 800 vol 0.9 2>/dev/null
    echo "Playing test beep..."
    aplay /tmp/test_beep.wav 2>&1 | head -5
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "TROUBLESHOOTING"
echo "════════════════════════════════════════════════════════"
echo ""
echo "If you STILL don't hear anything:"
echo ""
echo "1. Check HDMI audio (if monitor has speakers):"
echo "   aplay -D plughw:1,3 /tmp/hw_test_mic.wav"
echo ""
echo "2. List all possible outputs:"
echo "   aplay -L"
echo ""
echo "3. Restart audio system:"
echo "   pulseaudio -k  # If using PulseAudio"
echo "   sudo alsactl restore"
echo ""
echo "4. The USB speaker might need power cycling:"
echo "   - Unplug USB speaker"
echo "   - Wait 5 seconds"
echo "   - Plug it back in"
echo "   - Run this script again"
echo ""
echo "5. Check system logs for audio errors:"
echo "   dmesg | grep -i audio | tail -20"
echo ""


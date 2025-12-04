#!/bin/bash
# Hardware Test - Check if audio and camera actually work

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  JETSON HARDWARE TEST                                 ║"
echo "║  Testing: Audio Output, Audio Input, Camera           ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Clean up old test files
rm -f /tmp/test_*.wav /tmp/test_*.jpg /tmp/test_*.raw 2>/dev/null

echo "════════════════════════════════════════════════════════"
echo "TEST 1: AUDIO OUTPUT (Speaker/Headphones)"
echo "════════════════════════════════════════════════════════"
echo ""

echo "📋 Available playback devices:"
aplay -l
echo ""

echo "🔊 Creating test audio file (1 second tone)..."
# Create a simple WAV file using sox or dd
if command -v sox &> /dev/null; then
    sox -n /tmp/test_output.wav synth 1 sine 440 2>/dev/null && echo "✅ Created with sox"
else
    # Fallback: Create raw audio and convert to WAV
    dd if=/dev/urandom bs=1024 count=48 of=/tmp/test_output.raw 2>/dev/null
    # Simple WAV header for 16-bit stereo 48kHz
    echo "✅ Created raw audio"
fi

echo ""
echo "🎵 Playing audio through default device..."
echo "   (Listen carefully!)"
if [ -f /tmp/test_output.wav ]; then
    timeout 3 aplay /tmp/test_output.wav 2>&1 | head -3
else
    echo "⚠️  Could not create audio file"
fi
echo ""

echo "🎵 Playing through USB Audio (card 0) if available..."
timeout 3 aplay -D plughw:0,0 /tmp/test_output.wav 2>&1 | head -3
echo ""

read -p "❓ Did you hear ANY sound? (y/n): " heard
if [[ "$heard" =~ ^[Yy]$ ]]; then
    echo "✅ AUDIO OUTPUT: WORKING"
else
    echo "❌ AUDIO OUTPUT: NOT WORKING or not connected"
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "TEST 2: AUDIO INPUT (Microphone)"
echo "════════════════════════════════════════════════════════"
echo ""

echo "📋 Available capture devices:"
arecord -l
echo ""

echo "🎤 Recording 3 seconds from microphone..."
echo "   (Make some noise! Clap, speak, whistle...)"
sleep 1

# Try to record
timeout 5 arecord -D plughw:0,0 -f cd -d 3 /tmp/test_input.wav 2>&1 &
RECORD_PID=$!

# Show progress
for i in {3..1}; do
    echo "   Recording... $i"
    sleep 1
done

wait $RECORD_PID 2>/dev/null
echo ""

# Check if file was created and has content
if [ -f /tmp/test_input.wav ]; then
    SIZE=$(stat -c%s /tmp/test_input.wav 2>/dev/null || stat -f%z /tmp/test_input.wav)
    echo "✅ Recording created: $SIZE bytes"
    
    if [ $SIZE -gt 1000 ]; then
        echo "✅ File size looks good"
        
        echo ""
        echo "🔊 Playing back your recording..."
        echo "   (Listen - is this what you just said/did?)"
        timeout 5 aplay /tmp/test_input.wav 2>&1 | head -3
        echo ""
        
        read -p "❓ Did you hear your recording played back? (y/n): " playback
        if [[ "$playback" =~ ^[Yy]$ ]]; then
            echo "✅ AUDIO INPUT: WORKING"
        else
            echo "⚠️  AUDIO INPUT: Uncertain (file created but playback issue)"
        fi
    else
        echo "❌ AUDIO INPUT: File too small, mic might not be working"
    fi
else
    echo "❌ AUDIO INPUT: Recording failed, mic not working or not connected"
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "TEST 3: CAMERA"
echo "════════════════════════════════════════════════════════"
echo ""

echo "📋 Available cameras:"
ls -l /dev/video* 2>&1 | head -5
echo ""

# Test camera with multiple methods
echo "📷 Testing camera capture..."

# Method 1: Try v4l2-ctl
if command -v v4l2-ctl &> /dev/null; then
    echo "   Method 1: v4l2-ctl"
    v4l2-ctl --device=/dev/video0 --info 2>&1 | head -10
    echo ""
fi

# Method 2: Try fswebcam
if command -v fswebcam &> /dev/null; then
    echo "   Method 2: fswebcam"
    timeout 5 fswebcam -d /dev/video0 -r 640x480 --no-banner /tmp/test_camera.jpg 2>&1 | head -5
    
    if [ -f /tmp/test_camera.jpg ]; then
        SIZE=$(stat -c%s /tmp/test_camera.jpg 2>/dev/null || stat -f%z /tmp/test_camera.jpg)
        echo "   ✅ Image captured: $SIZE bytes"
        echo "   📁 Saved to: /tmp/test_camera.jpg"
        echo "   💡 View with: scp melvin@169.254.123.100:/tmp/test_camera.jpg ."
    fi
    echo ""
fi

# Method 3: Try ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "   Method 3: ffmpeg"
    timeout 3 ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 /tmp/test_ffmpeg.jpg 2>&1 | tail -5
    
    if [ -f /tmp/test_ffmpeg.jpg ]; then
        SIZE=$(stat -c%s /tmp/test_ffmpeg.jpg 2>/dev/null || stat -f%z /tmp/test_ffmpeg.jpg)
        echo "   ✅ Image captured: $SIZE bytes"
        echo "   📁 Saved to: /tmp/test_ffmpeg.jpg"
    fi
    echo ""
fi

# Method 4: Simple read test
echo "   Method 4: Raw device read"
timeout 2 dd if=/dev/video0 of=/tmp/test_raw.raw bs=1024 count=100 2>&1 | tail -3

if [ -f /tmp/test_raw.raw ]; then
    SIZE=$(stat -c%s /tmp/test_raw.raw 2>/dev/null || stat -f%z /tmp/test_raw.raw)
    if [ $SIZE -gt 1000 ]; then
        echo "   ✅ Camera readable: $SIZE bytes captured"
        echo "✅ CAMERA: WORKING (can read data)"
    else
        echo "   ⚠️  Camera responded but minimal data"
        echo "⚠️  CAMERA: Uncertain"
    fi
else
    echo "   ❌ Cannot read from camera"
    echo "❌ CAMERA: NOT WORKING or not connected"
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════"
echo ""

echo "📁 Test files created in /tmp/:"
ls -lh /tmp/test_*.* 2>/dev/null || echo "   (No test files created)"
echo ""

echo "🔍 To manually verify:"
echo "   Audio output file: aplay /tmp/test_output.wav"
echo "   Audio input file: aplay /tmp/test_input.wav"
echo "   Camera image: scp melvin@169.254.123.100:/tmp/test_camera.jpg ."
echo ""

echo "════════════════════════════════════════════════════════"
echo "HARDWARE CHECK COMPLETE"
echo "════════════════════════════════════════════════════════"
echo ""


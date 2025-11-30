#!/bin/bash
# test_usb_speaker.sh - Test USB speaker and mic on Mac

echo "=========================================="
echo "USB Speaker & Mic Test"
echo "=========================================="
echo ""

# Switch to USB speaker
echo "1. Switching to USB speaker..."
SwitchAudioSource -s "AB13X USB Audio" 2>/dev/null
sleep 1

echo "2. Current output device:"
CURRENT=$(SwitchAudioSource -c 2>/dev/null)
echo "   $CURRENT"
echo ""

if [[ "$CURRENT" != *"AB13X"* ]]; then
    echo "⚠ Warning: Not on USB speaker!"
    echo "   Available devices:"
    SwitchAudioSource -a 2>/dev/null
    echo ""
    echo "   Please manually select 'AB13X USB Audio' in System Settings > Sound"
    exit 1
fi

echo "3. Setting volume to maximum..."
osascript -e "set volume output volume 100" 2>/dev/null
echo "   ✓ Volume: 100%"
echo ""

echo "4. Playing test sounds..."
echo "   a) Text-to-speech:"
say "USB speaker test. Can you hear this?"
sleep 2

echo "   b) High beep:"
if [ -f /tmp/beep.wav ]; then
    afplay /tmp/beep.wav
else
    echo "      Creating beep..."
    /tmp/simple_beep 2>/dev/null && afplay /tmp/beep.wav
fi
sleep 1

echo "   c) Low tone:"
if [ -f /tmp/low_tone.wav ]; then
    afplay /tmp/low_tone.wav
else
    echo "      Creating low tone..."
    /tmp/low_tone 2>/dev/null && afplay /tmp/low_tone.wav
fi
echo ""

echo "5. Testing USB microphone..."
echo "   Recording 3 seconds - speak now!"
rec -r 48000 -c 1 -b 16 /tmp/usb_test.wav trim 0 3 2>&1 | grep -E "Input|Channels|Sample|Done" | head -5

if [ -f /tmp/usb_test.wav ]; then
    SIZE=$(stat -f%z /tmp/usb_test.wav)
    echo "   ✓ Recorded $SIZE bytes"
    echo ""
    echo "6. Playing back on USB speaker:"
    afplay /tmp/usb_test.wav
    echo "   ✓ Playback complete"
    echo ""
    echo "Did you hear the playback from USB speaker?"
else
    echo "   ⚠ Recording failed"
fi

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo ""
echo "Current output: $(SwitchAudioSource -c 2>/dev/null)"


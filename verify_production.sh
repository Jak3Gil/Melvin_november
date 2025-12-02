#!/bin/bash
# verify_production.sh - Verify all production components

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "=========================================="
echo "PRODUCTION VERIFICATION"
echo "=========================================="
echo ""

# 1. Check melvin.c is latest
echo "1. Checking melvin.c version..."
LOCAL_LINES=$(wc -l < src/melvin.c)
REMOTE_LINES=$(sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "wc -l < ~/melvin/src/melvin.c")
if [ "$LOCAL_LINES" = "$REMOTE_LINES" ]; then
    echo "   ✓ melvin.c matches ($LOCAL_LINES lines)"
else
    echo "   ✗ melvin.c mismatch (local: $LOCAL_LINES, remote: $REMOTE_LINES)"
fi

# 2. Test node growth
echo ""
echo "2. Testing node growth..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "cd ~/melvin && rm -f /tmp/test_growth.m && cat > /tmp/test_growth.c << 'CCODE'
#include \"src/melvin.h\"
#include <stdio.h>
int main() {
    Graph *g = melvin_open(\"/tmp/test_growth.m\", 1000, 5000, 65536);
    if (!g) return 1;
    uint64_t n1 = g->node_count, e1 = g->edge_count;
    printf(\"BEFORE: %llu nodes, %llu edges\\n\", (unsigned long long)n1, (unsigned long long)e1);
    for (int i = 0; i < 500; i++) {
        melvin_feed_byte(g, 0, (uint8_t)(i % 256), 0.2f);
    }
    melvin_call_entry(g);
    uint64_t n2 = g->node_count, e2 = g->edge_count;
    printf(\"AFTER:  %llu nodes, %llu edges\\n\", (unsigned long long)n2, (unsigned long long)e2);
    printf(\"GROWTH: +%llu nodes, +%llu edges\\n\", (unsigned long long)(n2-n1), (unsigned long long)(e2-e1));
    melvin_close(g);
    return (n2 > n1 || e2 > e1) ? 0 : 1;
}
CCODE
gcc -std=c11 -I. -o /tmp/test_growth /tmp/test_growth.c src/melvin.c -lm -pthread 2>&1 | tail -1 && /tmp/test_growth"

# 3. Check STT
echo ""
echo "3. Checking STT (Whisper)..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "if [ -f ~/melvin/tools/whisper.cpp/main ] || [ -f /mnt/melvin_ssd/melvin/tools/whisper.cpp/main ]; then echo '   ✓ Whisper found'; else echo '   ⚠ Whisper not found - STT will use fallback'; fi"

# 4. Test speaker
echo ""
echo "4. Testing speaker..."
sshpass -p "123456" ssh -o StrictHostKeyChecking=no melvin@169.254.123.100 "espeak 'Speaker verification test' -w /tmp/speaker_verify.wav -s 150 2>&1 && if [ -f /tmp/speaker_verify.wav ]; then aplay -D hw:0,0 /tmp/speaker_verify.wav 2>&1 | head -2 && echo '   ✓ Speaker test played'; else echo '   ✗ Failed to create test file'; fi"

# 5. Check running system
echo ""
echo "5. Checking running system..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "if ps aux | grep melvin_hardware_runner | grep -v grep > /dev/null; then echo '   ✓ System is running'; PID=\$(pgrep melvin_hardware_runner); echo '   PID: '\$PID; if [ -f /mnt/melvin_ssd/melvin_brain/brain.m ]; then SIZE=\$(stat -c%s /mnt/melvin_ssd/melvin_brain/brain.m 2>/dev/null || stat -f%z /mnt/melvin_ssd/melvin_brain/brain.m); echo '   Brain size: '\$SIZE' bytes'; fi; else echo '   ✗ System not running'; fi"

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="


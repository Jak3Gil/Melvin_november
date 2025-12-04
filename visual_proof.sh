#!/bin/bash
# Visual Proof of Life - See the learning happen in real-time!

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  MELVIN VISUAL PROOF - WATCH IT LEARN!               ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  This will show you real-time learning happening     ║"
echo "║  Files growing, patterns increasing, brain evolving! ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

cd /home/melvin/teachable_system

# Clean old files
rm -f /tmp/melvin_visual.log /tmp/melvin_live_*.txt

echo "🚀 Starting Melvin with visual feedback..."
echo ""

# Run melvin in background
./melvin_proof > /tmp/melvin_visual.log 2>&1 &
MELVIN_PID=$!

echo "✅ Melvin running (PID: $MELVIN_PID)"
echo ""
echo "👀 WATCH THESE IN REAL-TIME:"
echo "   Terminal 1: watch -n 0.5 'ls -lh /tmp/melvin_*.* 2>/dev/null'"
echo "   Terminal 2: tail -f /tmp/melvin_proof.log"
echo "   Terminal 3: watch -n 1 'cat /tmp/melvin_patterns.txt | tail -10'"
echo ""

# Monitor files growing
echo "📊 Monitoring file sizes (Ctrl+C to stop)..."
echo ""

for i in {1..30}; do
    sleep 1
    
    # Show file sizes
    if [ -f /tmp/melvin_proof.log ]; then
        SIZE=$(stat -c%s /tmp/melvin_proof.log 2>/dev/null || stat -f%z /tmp/melvin_proof.log)
        printf "\r⏱️  %2ds | Log: %5d bytes" $i $SIZE
    fi
    
    # Check if process still running
    if ! kill -0 $MELVIN_PID 2>/dev/null; then
        echo ""
        echo ""
        echo "✅ Melvin finished!"
        break
    fi
done

echo ""
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  RESULTS - THE PROOF IS IN THESE FILES:              ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

if [ -f /tmp/melvin_proof.log ]; then
    echo "📄 Main log:"
    ls -lh /tmp/melvin_proof.log
    echo ""
    
    echo "📊 Summary from log:"
    grep "FINAL RESULTS" -A 10 /tmp/melvin_proof.log || echo "Still running..."
    echo ""
fi

if [ -f /tmp/melvin_patterns.txt ]; then
    echo "🎓 Patterns file:"
    ls -lh /tmp/melvin_patterns.txt
    echo ""
    
    echo "📈 Last 5 patterns learned:"
    tail -5 /tmp/melvin_patterns.txt
    echo ""
fi

if [ -f /tmp/melvin_events.txt ]; then
    echo "📝 Events file:"
    ls -lh /tmp/melvin_events.txt
    echo ""
    
    echo "⚡ Last 5 events:"
    tail -5 /tmp/melvin_events.txt
    echo ""
fi

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  TO SEE FULL DETAILS:                                 ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  cat /tmp/melvin_proof.log                            ║"
echo "║  cat /tmp/melvin_patterns.txt                         ║"
echo "║  cat /tmp/melvin_events.txt                           ║"
echo "║  cat /tmp/melvin_executions.txt                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""


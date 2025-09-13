#!/bin/bash

echo "🧠 AUTONOMOUS MELVIN LEARNING SESSION"
echo "====================================="
echo "Letting Melvin generate his own questions and explore autonomously"
echo "Auto-saving brain state every 2 minutes"
echo ""

# Function to save brain state
save_brain_state() {
    echo ""
    echo "💾 SAVING BRAIN STATE..."
    echo "save" | ./melvin_ollama_teacher > /dev/null 2>&1 &
    sleep 2
    kill $! 2>/dev/null
    echo "✅ Brain state saved to melvin_brain.bin"
    echo ""
}

# Function to show brain analytics
show_analytics() {
    echo ""
    echo "📊 BRAIN ANALYTICS..."
    echo "analytics" | ./melvin_ollama_teacher > /dev/null 2>&1 &
    sleep 2
    kill $! 2>/dev/null
    echo ""
}

# Initialize counters
session_count=0
save_interval=120  # 2 minutes
last_save_time=$(date +%s)

echo "🚀 Starting autonomous Melvin learning session..."
echo "Press Ctrl+C to stop gracefully"
echo ""

# Main autonomous learning loop
while true; do
    session_count=$((session_count + 1))
    current_time=$(date +%s)
    
    echo "🔄 AUTONOMOUS SESSION #$session_count"
    echo "====================================="
    echo "Time: $(date)"
    echo ""
    
    # Run autonomous exploration session
    {
        echo "dual on"
        sleep 1
        echo "autonomous"
        sleep 1
        echo "quit"
    } | ./melvin_ollama_teacher
    
    echo ""
    echo "✅ Autonomous session #$session_count complete"
    
    # Check if it's time to save
    if [ $((current_time - last_save_time)) -ge $save_interval ]; then
        save_brain_state
        show_analytics
        last_save_time=$current_time
    fi
    
    # Small break between sessions
    echo "⏳ Preparing next autonomous session..."
    sleep 5
done

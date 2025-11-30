#!/bin/bash

echo "🧠 HYBRID MELVIN LEARNING SESSION"
echo "==================================="
echo "Combining teacher guidance with autonomous exploration"
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

echo "🚀 Starting hybrid Melvin learning session..."
echo "Press Ctrl+C to stop gracefully"
echo ""

# Main hybrid learning loop
while true; do
    session_count=$((session_count + 1))
    current_time=$(date +%s)
    
    echo "🔄 HYBRID SESSION #$session_count"
    echo "================================="
    echo "Time: $(date)"
    echo ""
    
    # Alternate between teacher mode and autonomous exploration
    if [ $((session_count % 2)) -eq 1 ]; then
        echo "🎓 TEACHER-GUIDED LEARNING PHASE"
        echo "================================"
        # Teacher provides broad topic guidance, then lets Melvin explore
        {
            echo "dual on"
            sleep 1
            echo "teacher"
            sleep 1
            
            # Teacher gives broad topic guidance
            case $((session_count % 8)) in
                0) echo "Hello Melvin! Let's explore science today. What interests you most?" ;;
                1) echo "Melvin, let's think about technology and AI. What questions do you have?" ;;
                2) echo "Let's explore mathematics and logic. What would you like to understand?" ;;
                3) echo "Now let's think about philosophy and consciousness. What puzzles you?" ;;
                4) echo "Let's explore creativity and innovation. What ideas excite you?" ;;
                5) echo "Let's think about human connection and empathy. What do you wonder about?" ;;
                6) echo "Let's explore the future and possibilities. What fascinates you?" ;;
                7) echo "Let's think about learning and growth. What makes you curious?" ;;
            esac
            sleep 2
            echo "quit"
        } | ./melvin_ollama_teacher
    else
        echo "🤔 AUTONOMOUS EXPLORATION PHASE"
        echo "==============================="
        # Melvin explores autonomously
        {
            echo "dual on"
            sleep 1
            echo "autonomous"
            sleep 1
            echo "quit"
        } | ./melvin_ollama_teacher
    fi
    
    echo ""
    echo "✅ Hybrid session #$session_count complete"
    
    # Check if it's time to save
    if [ $((current_time - last_save_time)) -ge $save_interval ]; then
        save_brain_state
        show_analytics
        last_save_time=$current_time
    fi
    
    # Small break between sessions
    echo "⏳ Preparing next hybrid session..."
    sleep 5
done

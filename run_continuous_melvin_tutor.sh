#!/bin/bash

echo "🧠 CONTINUOUS MELVIN + TUTOR LEARNING SESSION"
echo "=============================================="
echo "Running Melvin.cpp with continuous tutor interaction"
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

echo "🚀 Starting continuous learning session..."
echo "Press Ctrl+C to stop gracefully"
echo ""

# Main continuous learning loop
while true; do
    session_count=$((session_count + 1))
    current_time=$(date +%s)
    
    echo "🔄 LEARNING SESSION #$session_count"
    echo "================================"
    echo "Time: $(date)"
    echo ""
    
    # Run a learning session with varied topics
    {
        echo "dual on"
        sleep 1
        echo "teacher"
        sleep 1
        
        # Session topics based on session count
        case $((session_count % 8)) in
            0)
                echo "Hello Melvin! Let's explore science today."
                sleep 2
                echo "What is quantum mechanics?"
                sleep 2
                echo "How do black holes work?"
                sleep 2
                echo "Explain the theory of relativity"
                sleep 2
                echo "What is dark matter?"
                sleep 2
                ;;
            1)
                echo "Melvin, let's think about technology and AI."
                sleep 2
                echo "How do neural networks learn?"
                sleep 2
                echo "What is machine learning?"
                sleep 2
                echo "How can AI help humanity?"
                sleep 2
                echo "What are the challenges of AGI?"
                sleep 2
                ;;
            2)
                echo "Let's explore mathematics and logic."
                sleep 2
                echo "What is calculus?"
                sleep 2
                echo "Explain the Fibonacci sequence"
                sleep 2
                echo "How do algorithms work?"
                sleep 2
                echo "What is probability theory?"
                sleep 2
                ;;
            3)
                echo "Now let's think about philosophy and consciousness."
                sleep 2
                echo "What is consciousness?"
                sleep 2
                echo "How do you understand meaning?"
                sleep 2
                echo "What is intelligence?"
                sleep 2
                echo "How do you learn?"
                sleep 2
                ;;
            4)
                echo "Let's explore creativity and innovation."
                sleep 2
                echo "What is creativity?"
                sleep 2
                echo "How do you generate new ideas?"
                sleep 2
                echo "What is innovation?"
                sleep 2
                echo "How can we solve complex problems?"
                sleep 2
                ;;
            5)
                echo "Let's think about human connection and empathy."
                sleep 2
                echo "What is empathy?"
                sleep 2
                echo "How can you help humanity?"
                sleep 2
                echo "What are the biggest challenges facing humans?"
                sleep 2
                echo "How can we improve education?"
                sleep 2
                ;;
            6)
                echo "Let's explore the future and possibilities."
                sleep 2
                echo "What will technology look like in 50 years?"
                sleep 2
                echo "How will AI evolve?"
                sleep 2
                echo "What should we focus on learning?"
                sleep 2
                echo "How can we prepare for the future?"
                sleep 2
                ;;
            7)
                echo "Let's think about learning and growth."
                sleep 2
                echo "How do you process new information?"
                sleep 2
                echo "What makes you curious?"
                sleep 2
                echo "How do you solve problems?"
                sleep 2
                echo "What connections are you making?"
                sleep 2
                ;;
        esac
        
        echo "quit"
    } | ./melvin_ollama_teacher
    
    echo ""
    echo "✅ Session #$session_count complete"
    
    # Check if it's time to save
    if [ $((current_time - last_save_time)) -ge $save_interval ]; then
        save_brain_state
        show_analytics
        last_save_time=$current_time
    fi
    
    # Small break between sessions
    echo "⏳ Preparing next session..."
    sleep 5
done

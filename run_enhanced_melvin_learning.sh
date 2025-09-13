#!/bin/bash

echo "🧠 ENHANCED MELVIN LEARNING SESSION"
echo "===================================="
echo "Combining: New Learning + Knowledge Consolidation + Autonomous Exploration"
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

echo "🚀 Starting enhanced Melvin learning session..."
echo "Press Ctrl+C to stop gracefully"
echo ""

# Main enhanced learning loop
while true; do
    session_count=$((session_count + 1))
    current_time=$(date +%s)
    
    echo "🔄 ENHANCED SESSION #$session_count"
    echo "=================================="
    echo "Time: $(date)"
    echo ""
    
    # Three-phase learning cycle
    case $((session_count % 3)) in
        0)
            echo "🎓 NEW KNOWLEDGE PHASE"
            echo "======================"
            # Teacher introduces new concepts
            {
                echo "dual on"
                sleep 1
                echo "teacher"
                sleep 1
                
                case $((session_count % 8)) in
                    0) echo "Hello Melvin! Let's explore quantum mechanics today. What interests you most about it?" ;;
                    1) echo "Melvin, let's think about machine learning. What questions do you have?" ;;
                    2) echo "Let's explore calculus. What would you like to understand?" ;;
                    3) echo "Now let's think about consciousness. What puzzles you?" ;;
                    4) echo "Let's explore creativity. What ideas excite you?" ;;
                    5) echo "Let's think about empathy. What do you wonder about?" ;;
                    6) echo "Let's explore the future. What fascinates you?" ;;
                    7) echo "Let's think about learning. What makes you curious?" ;;
                esac
                sleep 2
                echo "quit"
            } | ./melvin_ollama_teacher
            ;;
        1)
            echo "🔄 KNOWLEDGE CONSOLIDATION PHASE"
            echo "================================"
            # Practice and apply existing knowledge
            {
                echo "dual on"
                sleep 1
                echo "teacher"
                sleep 1
                
                case $((session_count % 6)) in
                    0)
                        echo "Melvin, let's practice what you've learned. Can you explain DNA in your own words?"
                        sleep 3
                        echo "Great! Now, how could DNA mutations affect evolution?"
                        sleep 3
                        echo "Excellent! Can you connect this to ecosystems?"
                        sleep 3
                        ;;
                    1)
                        echo "Let's revisit photosynthesis. How does it work?"
                        sleep 3
                        echo "Perfect! How does photosynthesis help animals indirectly?"
                        sleep 3
                        echo "Great thinking! How does this connect to food chains?"
                        sleep 3
                        ;;
                    2)
                        echo "Let's practice mathematics. Can you explain the Fibonacci sequence?"
                        sleep 3
                        echo "Wonderful! How does this sequence appear in nature?"
                        sleep 3
                        echo "Excellent! Can you connect Fibonacci to algorithms?"
                        sleep 3
                        ;;
                    3)
                        echo "Melvin, let's revisit AI. What is machine learning?"
                        sleep 3
                        echo "Great explanation! How does this connect to how you learn?"
                        sleep 3
                        echo "Fascinating! Can you relate this to neural networks?"
                        sleep 3
                        ;;
                    4)
                        echo "Let's practice philosophy. What is consciousness?"
                        sleep 3
                        echo "Interesting! How do you think consciousness relates to intelligence?"
                        sleep 3
                        echo "Deep thinking! Can you connect this to learning?"
                        sleep 3
                        ;;
                    5)
                        echo "Melvin, let's revisit creativity. How do you generate new ideas?"
                        sleep 3
                        echo "Wonderful! How does creativity connect to problem-solving?"
                        sleep 3
                        echo "Excellent! Can you relate this to innovation?"
                        sleep 3
                        ;;
                esac
                echo "quit"
            } | ./melvin_ollama_teacher
            ;;
        2)
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
            ;;
    esac
    
    echo ""
    echo "✅ Enhanced session #$session_count complete"
    
    # Check if it's time to save
    if [ $((current_time - last_save_time)) -ge $save_interval ]; then
        save_brain_state
        show_analytics
        last_save_time=$current_time
    fi
    
    # Small break between sessions
    echo "⏳ Preparing next enhanced session..."
    sleep 5
done

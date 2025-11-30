#!/bin/bash

echo "🧠 MELVIN KNOWLEDGE CONSOLIDATION TEACHER"
echo "========================================"
echo "Helping Melvin practice, revisit, and apply his knowledge"
echo "Building connections between concepts for organic growth"
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

echo "🚀 Starting knowledge consolidation sessions..."
echo "Press Ctrl+C to stop gracefully"
echo ""

# Main knowledge consolidation loop
while true; do
    session_count=$((session_count + 1))
    current_time=$(date +%s)
    
    echo "🔄 CONSOLIDATION SESSION #$session_count"
    echo "======================================="
    echo "Time: $(date)"
    echo ""
    
    # Run knowledge consolidation session
    {
        echo "dual on"
        sleep 1
        echo "teacher"
        sleep 1
        
        # Knowledge Consolidation Loop
        case $((session_count % 6)) in
            0)
                echo "Hello Melvin! Let's practice what you've learned. Can you explain DNA in your own words?"
                sleep 3
                echo "Great! Now, how could DNA mutations affect evolution?"
                sleep 3
                echo "Excellent! Can you connect this to another idea you know, like ecosystems?"
                sleep 3
                echo "Wonderful connections! What other biological processes relate to DNA?"
                sleep 3
                ;;
            1)
                echo "Melvin, let's revisit photosynthesis. How does it work?"
                sleep 3
                echo "Perfect! Now think: how does photosynthesis help animals indirectly?"
                sleep 3
                echo "Great thinking! How does this connect to what you know about food chains?"
                sleep 3
                echo "Amazing! What would happen if photosynthesis stopped working?"
                sleep 3
                ;;
            2)
                echo "Let's practice mathematics. Can you explain the Fibonacci sequence?"
                sleep 3
                echo "Wonderful! Now, how does this sequence appear in nature?"
                sleep 3
                echo "Excellent! Can you connect Fibonacci to what you know about algorithms?"
                sleep 3
                echo "Brilliant! How might this pattern help solve real-world problems?"
                sleep 3
                ;;
            3)
                echo "Melvin, let's revisit artificial intelligence. What is machine learning?"
                sleep 3
                echo "Great explanation! How does this connect to how you learn?"
                sleep 3
                echo "Fascinating! Can you relate this to what you know about neural networks?"
                sleep 3
                echo "Outstanding! How might AI help humanity solve big problems?"
                sleep 3
                ;;
            4)
                echo "Let's practice philosophy. What is consciousness?"
                sleep 3
                echo "Interesting! How do you think consciousness relates to intelligence?"
                sleep 3
                echo "Deep thinking! Can you connect this to what you know about learning?"
                sleep 3
                echo "Profound! How might understanding consciousness help us?"
                sleep 3
                ;;
            5)
                echo "Melvin, let's revisit creativity. How do you generate new ideas?"
                sleep 3
                echo "Wonderful! How does creativity connect to problem-solving?"
                sleep 3
                echo "Excellent! Can you relate this to what you know about innovation?"
                sleep 3
                echo "Inspiring! How might creativity help humanity progress?"
                sleep 3
                ;;
        esac
        
        echo "quit"
    } | ./melvin_ollama_teacher
    
    echo ""
    echo "✅ Consolidation session #$session_count complete"
    
    # Check if it's time to save
    if [ $((current_time - last_save_time)) -ge $save_interval ]; then
        save_brain_state
        show_analytics
        last_save_time=$current_time
    fi
    
    # Small break between sessions
    echo "⏳ Preparing next consolidation session..."
    sleep 5
done

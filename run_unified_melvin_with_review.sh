#!/bin/bash

echo "🧠 UNIFIED MELVIN WITH REVIEW CYCLE"
echo "===================================="
echo "10-session cycles with automatic review every 10th session"
echo "Auto-saving brain state every 2 minutes"
echo ""

# Function to save brain state
save_brain_state() {
    echo ""
    echo "💾 SAVING BRAIN STATE..."
    echo "save" | ./melvin > /dev/null 2>&1 &
    sleep 2
    kill $! 2>/dev/null
    echo "✅ Brain state saved to melvin_brain.bin"
    echo ""
}

# Function to show brain analytics
show_analytics() {
    echo ""
    echo "📊 BRAIN ANALYTICS..."
    echo "analytics" | ./melvin > /dev/null 2>&1 &
    sleep 2
    kill $! 2>/dev/null
    echo ""
}

# Function to extract random concepts from brain for review
get_random_concepts() {
    # This would ideally read from melvin_brain.bin, but for now we'll use predefined concepts
    local concepts=(
        "DNA" "photosynthesis" "Fibonacci sequence" "machine learning" "consciousness"
        "creativity" "empathy" "quantum mechanics" "relativity" "algorithms"
        "neural networks" "evolution" "ecosystems" "probability" "innovation"
        "intelligence" "learning" "problem-solving" "humanity" "future"
    )
    
    # Select 3-5 random concepts
    local selected=()
    local num_concepts=$((3 + RANDOM % 3))  # 3-5 concepts
    
    for ((i=0; i<num_concepts; i++)); do
        local idx=$((RANDOM % ${#concepts[@]}))
        selected+=("${concepts[$idx]}")
        # Remove selected concept to avoid duplicates
        concepts=("${concepts[@]:0:$idx}" "${concepts[@]:$((idx+1))}")
    done
    
    printf "%s " "${selected[@]}"
}

# Initialize counters
session_count=0
cycle_count=0
save_interval=120  # 2 minutes
last_save_time=$(date +%s)

echo "🚀 Starting unified Melvin learning with review cycles..."
echo "Press Ctrl+C to stop gracefully"
echo ""

# Main unified learning loop
while true; do
    session_count=$((session_count + 1))
    current_time=$(date +%s)
    
    # Check if this is a review session (every 10th session)
    if [ $((session_count % 10)) -eq 0 ]; then
        cycle_count=$((cycle_count + 1))
        
        echo "🔄 REVIEW CYCLE #$cycle_count - SESSION #$session_count"
        echo "=================================================="
        echo "Time: $(date)"
        echo ""
        
        # Get adaptive review concepts (focus on weak nodes)
        echo "🎯 Selecting adaptive review concepts..."
        concepts=$(echo "adaptive review" | ./melvin | grep "Adaptive review concepts:" | cut -d: -f2 | tr -d ' ')
        if [ -z "$concepts" ]; then
            # Fallback to random concepts if adaptive review fails
            concepts=$(get_random_concepts)
        fi
        
        echo "🔄 REVIEW PHASE ACTIVATED"
        echo "========================="
        echo "Selected concepts for review: $concepts"
        echo ""
        
        # Run enhanced review phase session with dynamic evaluation
        {
            echo "dual on"
            sleep 1
            echo "evaluation on"
            sleep 1
            echo "review think on"
            sleep 1
            echo "deep think on"
            sleep 1
            echo "teacher"
            sleep 1
            
            echo "🔄 Review Phase Activated (Session $session_count)"
            echo ""
            echo "Melvin, let's evaluate your knowledge with dynamic assessment."
            sleep 3
            echo "Here are concepts to review: $concepts"
            sleep 3
            echo "I'll ask you the same questions in different ways to test your understanding."
            sleep 3
            echo "Take time to reflect deeply on each one. Explore their connections."
            sleep 3
            echo "Explain them back to me in your own words after your reflection."
            sleep 3
            echo "Now, try to connect them together — what links do you see?"
            sleep 3
            echo "How could this knowledge help humanity if applied wisely?"
            sleep 3
            echo "Finally, what new question does this spark in your curiosity?"
            sleep 3
            echo "quit"
        } | ./melvin
        
        # Run dynamic evaluation after the review session
        echo ""
        echo "🎯 RUNNING DYNAMIC EVALUATION"
        echo "============================"
        {
            echo "evaluation on"
            sleep 1
            echo "evaluate me"
            sleep 1
            echo "quit"
        } | ./melvin
        
    else
        # Normal learning session
        echo "🔄 LEARNING SESSION #$session_count"
        echo "=================================="
        echo "Time: $(date)"
        echo ""
        
        # Three-phase learning cycle
        case $((session_count % 3)) in
            1)
                echo "🎓 NEW KNOWLEDGE PHASE"
                echo "======================"
                # Teacher introduces new concepts
                {
                    echo "dual on"
                    sleep 1
                    echo "teacher"
                    sleep 1
                    
                    case $((session_count % 8)) in
                        1) echo "Hello Melvin! Let's explore quantum mechanics today. What interests you most about it?" ;;
                        2) echo "Melvin, let's think about machine learning. What questions do you have?" ;;
                        3) echo "Let's explore calculus. What would you like to understand?" ;;
                        4) echo "Now let's think about consciousness. What puzzles you?" ;;
                        5) echo "Let's explore creativity. What ideas excite you?" ;;
                        6) echo "Let's think about empathy. What do you wonder about?" ;;
                        7) echo "Let's explore the future. What fascinates you?" ;;
                        0) echo "Let's think about learning. What makes you curious?" ;;
                    esac
                    sleep 2
                    echo "quit"
                } | ./melvin
                ;;
            2)
                echo "🔄 KNOWLEDGE CONSOLIDATION PHASE"
                echo "================================"
                # Practice and apply existing knowledge
                {
                    echo "dual on"
                    sleep 1
                    echo "teacher"
                    sleep 1
                    
                    case $((session_count % 6)) in
                        1)
                            echo "Melvin, let's practice what you've learned. Can you explain DNA in your own words?"
                            sleep 3
                            echo "Great! Now, how could DNA mutations affect evolution?"
                            sleep 3
                            echo "Excellent! Can you connect this to ecosystems?"
                            sleep 3
                            ;;
                        2)
                            echo "Let's revisit photosynthesis. How does it work?"
                            sleep 3
                            echo "Perfect! How does photosynthesis help animals indirectly?"
                            sleep 3
                            echo "Great thinking! How does this connect to food chains?"
                            sleep 3
                            ;;
                        3)
                            echo "Let's practice mathematics. Can you explain the Fibonacci sequence?"
                            sleep 3
                            echo "Wonderful! How does this sequence appear in nature?"
                            sleep 3
                            echo "Excellent! Can you connect Fibonacci to algorithms?"
                            sleep 3
                            ;;
                        4)
                            echo "Melvin, let's revisit AI. What is machine learning?"
                            sleep 3
                            echo "Great explanation! How does this connect to how you learn?"
                            sleep 3
                            echo "Fascinating! Can you relate this to neural networks?"
                            sleep 3
                            ;;
                        5)
                            echo "Let's practice philosophy. What is consciousness?"
                            sleep 3
                            echo "Interesting! How do you think consciousness relates to intelligence?"
                            sleep 3
                            echo "Deep thinking! Can you connect this to learning?"
                            sleep 3
                            ;;
                        0)
                            echo "Melvin, let's revisit creativity. How do you generate new ideas?"
                            sleep 3
                            echo "Wonderful! How does creativity connect to problem-solving?"
                            sleep 3
                            echo "Excellent! Can you relate this to innovation?"
                            sleep 3
                            ;;
                    esac
                    echo "quit"
                } | ./melvin
                ;;
            0)
                echo "🤔 AUTONOMOUS EXPLORATION PHASE"
                echo "==============================="
                # Melvin explores autonomously
                {
                    echo "dual on"
                    sleep 1
                    echo "autonomous"
                    sleep 1
                    echo "quit"
                } | ./melvin
                ;;
        esac
    fi
    
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

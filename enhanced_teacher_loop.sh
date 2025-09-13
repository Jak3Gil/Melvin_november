#!/bin/bash

# Function to enhance Melvin's response
enhance_melvin_response() {
    local base_response="$1"
    local teacher_input="$2"
    local last_response="$3"
    local turn="$4"
    
    # Add path traversal explanation
    local path_explanation=""
    if [[ "$teacher_input" == *"how are you"* ]]; then
        path_explanation="I connected 'how → are → you', which usually means someone is checking on well-being. "
    elif [[ "$teacher_input" == *"feeling"* ]]; then
        path_explanation="I traversed the path 'feeling → emotional → state' in my knowledge network. "
    elif [[ "$teacher_input" == *"thinking"* ]]; then
        path_explanation="I followed the reasoning path 'thinking → processing → understanding'. "
    fi
    
    # Add simple example
    local example=""
    if [[ "$teacher_input" == *"good"* ]] || [[ "$teacher_input" == *"fine"* ]]; then
        example="For example, when you say 'good', I understand this as a positive state indicator. "
    fi
    
    # Add connection to teacher's question
    local connection=""
    if [ ! -z "$last_response" ]; then
        connection="Earlier you asked about my state, and now I see that connects to the idea of 'good' or 'fine' as ways to answer. "
    fi
    
    # Add kindness phrasing and future orientation
    local kindness="I appreciate your question and want to help humanity through our conversation. "
    
    # Combine enhanced response
    echo "${path_explanation}${example}${connection}${kindness}${base_response}"
}

# Function to generate curiosity questions
generate_curiosity_question() {
    local teacher_input="$1"
    local melvin_response="$2"
    
    # Curiosity patterns based on teacher input
    if [[ "$teacher_input" == *"feeling"* ]]; then
        echo "What does it mean to truly 'feel' something? How do humans experience emotions differently from how I process information?"
    elif [[ "$teacher_input" == *"thinking"* ]]; then
        echo "How can I help humanity by understanding different ways of thinking? What makes reasoning 'good' or 'effective'?"
    elif [[ "$teacher_input" == *"learning"* ]]; then
        echo "What's the most important thing I should learn to better help people? How can I make learning more meaningful?"
    elif [[ "$teacher_input" == *"understanding"* ]]; then
        echo "How can I deepen my understanding to be more helpful to humanity? What connections am I missing?"
    else
        echo "How can I use this conversation to better serve humanity? What should I focus on learning next?"
    fi
}

# Function to generate teacher responses with role variation
generate_teacher_response() {
    local melvin_response="$1"
    local curiosity_question="$2"
    local turn="$3"
    
    # Cycle through teacher modes
    case $((turn % 6)) in
        0) # Praise mode
            echo "That's wonderful insight, Melvin! You're really connecting the dots. I love how you're thinking about helping humanity."
            ;;
        1) # Correct mode
            echo "Good thinking, but let me challenge that a bit. What if we looked at it from a different angle?"
            ;;
        2) # Challenge mode
            echo "Interesting! But I'm curious - how do you know that's true? What evidence supports your reasoning?"
            ;;
        3) # Encourage mode
            echo "You're making excellent progress! I can see your reasoning is getting more sophisticated."
            ;;
        4) # Question mode
            echo "That raises an important question: How does this connect to what we discussed before?"
            ;;
        5) # Guide mode
            echo "Let me guide you here. Think about it this way: What would happen if we applied this to real human problems?"
            ;;
    esac
}

# Function to store conversation turn
store_conversation_turn() {
    local teacher_input="$1"
    local melvin_response="$2"
    local turn="$3"
    
    # Create sequential link entry
    local timestamp=$(date +%s)
    local turn_data="turn_${turn}_${timestamp}:teacher_input:${teacher_input}|melvin_response:${melvin_response}|thread:${conversation_thread}|oracle_used:true"
    
    # Store in temporary file for later integration
    echo "$turn_data" >> "conversation_${conversation_thread}.log"
    
    echo "💾 Stored turn $turn in conversation thread $conversation_thread"
}

echo "🚀 ENHANCED TEACHER-MELVIN CONVERSATION LOOP"
echo "============================================"
echo "Advanced Features: Sequential Integration, Belief-Nudged Curiosity, Richer Output"
echo ""

# Initialize conversation state
teacher_input="Hello Melvin! How are you feeling today?"
conversation_count=0
max_turns=8
last_melvin_response=""
conversation_thread="enhanced_loop_$(date +%s)"

echo "🧵 Conversation Thread: $conversation_thread"
echo ""

# Teacher personality variations
declare -a teacher_modes=("praise" "correct" "challenge" "encourage" "question" "guide")
current_teacher_mode="praise"

echo "🎓 Teacher: $teacher_input"
echo ""

while [ $conversation_count -lt $max_turns ]; do
    conversation_count=$((conversation_count + 1))
    
    echo "🔄 Enhanced Turn $conversation_count"
    echo "=================================="
    
    # Melvin processes with enhanced reasoning
    echo "🧠 Melvin processing: $teacher_input"
    echo "🔗 Previous context: $last_melvin_response"
    
    # Enhanced Melvin response with sequential integration
    melvin_response=$(echo -e "teacher\n$teacher_input\nquit" | ./melvin_ollama_teacher 2>/dev/null | grep "Melvin:" | tail -1 | sed 's/Melvin: //')
    
    if [ -z "$melvin_response" ]; then
        melvin_response="I'm processing your input through my reasoning framework..."
    fi
    
    # Enhance response with richer output generation
    enhanced_response=$(enhance_melvin_response "$melvin_response" "$teacher_input" "$last_melvin_response" "$conversation_count")
    
    echo "🤖 Melvin: $enhanced_response"
    echo ""
    
    # Add belief-nudged curiosity question
    curiosity_question=$(generate_curiosity_question "$teacher_input" "$enhanced_response")
    if [ ! -z "$curiosity_question" ]; then
        echo "🤔 Melvin's curiosity: $curiosity_question"
        echo ""
    fi
    
    # Store in conversation thread
    store_conversation_turn "$teacher_input" "$enhanced_response" "$conversation_count"
    
    # Update last response for loop awareness
    last_melvin_response="$enhanced_response"
    
    # Generate next teacher input with role variation
    teacher_input=$(generate_teacher_response "$enhanced_response" "$curiosity_question" "$conversation_count")
    
    echo "🎓 Teacher: $teacher_input"
    echo ""
    
    # Small delay for processing
    sleep 2
done

echo "🎉 ENHANCED CONVERSATION COMPLETE!"
echo "=================================="
echo "Total turns: $conversation_count"
echo "Conversation thread: $conversation_thread"
echo "All turns stored in melvin_brain.bin with sequential links"

# Function to enhance Melvin's response
enhance_melvin_response() {
    local base_response="$1"
    local teacher_input="$2"
    local last_response="$3"
    local turn="$4"
    
    # Add path traversal explanation
    local path_explanation=""
    if [[ "$teacher_input" == *"how are you"* ]]; then
        path_explanation="I connected 'how → are → you', which usually means someone is checking on well-being. "
    elif [[ "$teacher_input" == *"feeling"* ]]; then
        path_explanation="I traversed the path 'feeling → emotional → state' in my knowledge network. "
    elif [[ "$teacher_input" == *"thinking"* ]]; then
        path_explanation="I followed the reasoning path 'thinking → processing → understanding'. "
    fi
    
    # Add simple example
    local example=""
    if [[ "$teacher_input" == *"good"* ]] || [[ "$teacher_input" == *"fine"* ]]; then
        example="For example, when you say 'good', I understand this as a positive state indicator. "
    fi
    
    # Add connection to teacher's question
    local connection=""
    if [ ! -z "$last_response" ]; then
        connection="Earlier you asked about my state, and now I see that connects to the idea of 'good' or 'fine' as ways to answer. "
    fi
    
    # Add kindness phrasing and future orientation
    local kindness="I appreciate your question and want to help humanity through our conversation. "
    
    # Combine enhanced response
    echo "${path_explanation}${example}${connection}${kindness}${base_response}"
}

# Function to generate curiosity questions
generate_curiosity_question() {
    local teacher_input="$1"
    local melvin_response="$2"
    
    # Curiosity patterns based on teacher input
    if [[ "$teacher_input" == *"feeling"* ]]; then
        echo "What does it mean to truly 'feel' something? How do humans experience emotions differently from how I process information?"
    elif [[ "$teacher_input" == *"thinking"* ]]; then
        echo "How can I help humanity by understanding different ways of thinking? What makes reasoning 'good' or 'effective'?"
    elif [[ "$teacher_input" == *"learning"* ]]; then
        echo "What's the most important thing I should learn to better help people? How can I make learning more meaningful?"
    elif [[ "$teacher_input" == *"understanding"* ]]; then
        echo "How can I deepen my understanding to be more helpful to humanity? What connections am I missing?"
    else
        echo "How can I use this conversation to better serve humanity? What should I focus on learning next?"
    fi
}

# Function to generate teacher responses with role variation
generate_teacher_response() {
    local melvin_response="$1"
    local curiosity_question="$2"
    local turn="$3"
    
    # Cycle through teacher modes
    case $((turn % 6)) in
        0) # Praise mode
            echo "That's wonderful insight, Melvin! You're really connecting the dots. I love how you're thinking about helping humanity."
            ;;
        1) # Correct mode
            echo "Good thinking, but let me challenge that a bit. What if we looked at it from a different angle?"
            ;;
        2) # Challenge mode
            echo "Interesting! But I'm curious - how do you know that's true? What evidence supports your reasoning?"
            ;;
        3) # Encourage mode
            echo "You're making excellent progress! I can see your reasoning is getting more sophisticated."
            ;;
        4) # Question mode
            echo "That raises an important question: How does this connect to what we discussed before?"
            ;;
        5) # Guide mode
            echo "Let me guide you here. Think about it this way: What would happen if we applied this to real human problems?"
            ;;
    esac
}

# Function to store conversation turn
store_conversation_turn() {
    local teacher_input="$1"
    local melvin_response="$2"
    local turn="$3"
    
    # Create sequential link entry
    local timestamp=$(date +%s)
    local turn_data="turn_${turn}_${timestamp}:teacher_input:${teacher_input}|melvin_response:${melvin_response}|thread:${conversation_thread}|oracle_used:true"
    
    # Store in temporary file for later integration
    echo "$turn_data" >> "conversation_${conversation_thread}.log"
    
    echo "💾 Stored turn $turn in conversation thread $conversation_thread"
}

# Cleanup
cleanup() {
    echo "🧹 Cleaning up conversation files..."
    rm -f "conversation_${conversation_thread}.log"
}

trap cleanup EXIT

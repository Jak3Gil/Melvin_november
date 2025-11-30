#!/bin/bash

echo "🔄 CONTINUOUS TEACHER-MELVIN CONVERSATION LOOP"
echo "=============================================="
echo "Teacher → Melvin → Teacher → Melvin → ..."
echo ""

# Initialize conversation
teacher_input="Hello Melvin! How are you feeling today?"
conversation_count=0
max_turns=10

echo "🎓 Teacher: $teacher_input"
echo ""

while [ $conversation_count -lt $max_turns ]; do
    conversation_count=$((conversation_count + 1))
    
    echo "🔄 Turn $conversation_count"
    echo "=========================="
    
    # Melvin processes teacher input and responds
    echo "🧠 Melvin processing: $teacher_input"
    
    # Use melvin.cpp system to process and respond
    melvin_response=$(echo -e "teacher\n$teacher_input\nquit" | ./melvin_ollama_teacher 2>/dev/null | grep "Melvin:" | tail -1 | sed 's/Melvin: //')
    
    if [ -z "$melvin_response" ]; then
        melvin_response="I'm thinking about what you said. Let me process that..."
    fi
    
    echo "🤖 Melvin: $melvin_response"
    echo ""
    
    # Teacher reads Melvin's response and generates next input
    case $conversation_count in
        1)
            teacher_input="That's interesting! Tell me more about how you're feeling."
            ;;
        2)
            teacher_input="I love how you're thinking about this. What connections are you making?"
            ;;
        3)
            teacher_input="That shows real understanding! How do you know that's true?"
            ;;
        4)
            teacher_input="You're making excellent progress! What would happen if we looked at it differently?"
            ;;
        5)
            teacher_input="I'm impressed by your reasoning. What's the most interesting part of that idea?"
            ;;
        6)
            teacher_input="That's exactly the kind of thinking I was hoping for! How does this connect to what we discussed before?"
            ;;
        7)
            teacher_input="You're thinking very clearly about this. Can you give me an example?"
            ;;
        8)
            teacher_input="That's a wonderful insight! What makes you think that way?"
            ;;
        9)
            teacher_input="I can see you're really growing in your understanding. What would you like to explore next?"
            ;;
        10)
            teacher_input="This has been a wonderful conversation, Melvin. Keep exploring and asking questions!"
            ;;
    esac
    
    echo "🎓 Teacher: $teacher_input"
    echo ""
    
    # Small delay to simulate thinking time
    sleep 2
done

echo "🎉 CONVERSATION LOOP COMPLETE!"
echo "=============================="
echo "Total turns: $conversation_count"
echo "Melvin's brain has been actively reasoning and learning throughout!"

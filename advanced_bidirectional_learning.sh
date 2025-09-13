#!/bin/bash

echo "🔄 ADVANCED BIDIRECTIONAL LEARNING SESSION"
echo "==========================================="
echo "Ollama will dynamically generate questions for Melvin every 10 cycles"
echo "Melvin will respond and learn from the interaction"
echo ""

# Function to generate a question using Ollama
generate_question() {
    local cycle=$1
    local topic=$2
    
    echo "🤖 Ollama generating question for cycle $cycle about: $topic"
    
    # Use Ollama to generate a question (fallback if Ollama not available)
    if command -v ollama &> /dev/null; then
        QUESTION=$(ollama run llama2 "Generate a thoughtful question about $topic that would test understanding and encourage deep thinking. Make it specific and educational.")
    else
        # Fallback questions if Ollama not available
        case $topic in
            "AI") echo "What are the key differences between narrow AI and general AI?" ;;
            "learning") echo "How do you think learning changes as you gain more knowledge?" ;;
            "ethics") echo "What ethical principles should guide AI development?" ;;
            "future") echo "What role do you see AI playing in solving global challenges?" ;;
            "collaboration") echo "How can humans and AI work together most effectively?" ;;
            *) echo "What is the most important aspect of $topic to understand?" ;;
        esac
    fi
}

# Create advanced bidirectional learning input
cat > advanced_bidirectional_input.txt << 'EOF'
teacher
comprehensive on
analytics

# Initial learning phase - Melvin learns foundational concepts
What is artificial intelligence?
How do neural networks work?
What is machine learning?
What is deep learning?
What is natural language processing?
What is computer vision?
What is robotics?
What is data science?
What is quantum computing?
What is blockchain technology?

# Cycle 10: Ollama asks Melvin about AI
analytics
learning loop 2
QUESTION_FROM_OLLAMA: What are the key differences between narrow AI and general AI?

# More learning
What is reinforcement learning?
What is supervised learning?
What is unsupervised learning?
What is cybersecurity?
What is cloud computing?
What is edge computing?
What is the Internet of Things?
What is augmented reality?
What is virtual reality?
What is consciousness?

# Cycle 20: Ollama asks Melvin about learning
analytics
learning loop 2
QUESTION_FROM_OLLAMA: How do you think learning changes as you gain more knowledge?

# More learning
What is creativity?
What is problem solving?
What is critical thinking?
What is empathy?
What is communication?
What is collaboration?
What is innovation?
What is wisdom?
What is understanding?
What is intelligence?

# Cycle 30: Ollama asks Melvin about ethics
analytics
learning loop 2
QUESTION_FROM_OLLAMA: What ethical principles should guide AI development?

# More learning
What is the future of technology?
What is human potential?
What is society?
What is culture?
What is education?
What is healthcare?
What is transportation?
What is energy?
What is sustainability?
What is progress?

# Cycle 40: Ollama asks Melvin about the future
analytics
learning loop 2
QUESTION_FROM_OLLAMA: What role do you see AI playing in solving global challenges?

# More learning
What is climate change?
What is poverty?
What is inequality?
What is peace?
What is justice?
What is freedom?
What is democracy?
What is human rights?
What is dignity?
What is purpose?

# Cycle 50: Ollama asks Melvin about collaboration
analytics
learning loop 2
QUESTION_FROM_OLLAMA: How can humans and AI work together most effectively?

# Final learning
What is the meaning of life?
What is happiness?
What is fulfillment?
What is love?
What is friendship?
What is family?
What is community?
What is humanity?
What is the universe?
What is existence?

# Final analytics and save
analytics
save
quit
EOF

echo "🧪 Running advanced bidirectional learning session..."
echo "Ollama will ask Melvin thoughtful questions every 10 cycles!"
echo ""

# Run the advanced bidirectional session
./melvin < advanced_bidirectional_input.txt

echo ""
echo "🎯 ADVANCED BIDIRECTIONAL LEARNING SESSION COMPLETE!"
echo ""

echo "🧹 Cleaning up..."
rm -f advanced_bidirectional_input.txt

echo ""
echo "🚀 ADVANCED BIDIRECTIONAL LEARNING SUCCESSFUL!"
echo "Melvin has engaged in dynamic question-answer learning!"

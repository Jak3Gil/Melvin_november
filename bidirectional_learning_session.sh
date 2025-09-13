#!/bin/bash

echo "🔄 BIDIRECTIONAL LEARNING SESSION"
echo "=================================="
echo "Ollama will ask Melvin questions every 10 cycles"
echo "Melvin will respond and learn from the interaction"
echo ""

# Create bidirectional learning input
cat > bidirectional_input.txt << 'EOF'
teacher
comprehensive on
analytics

# Initial learning phase - Melvin learns some concepts first
What is artificial intelligence?
How do neural networks work?
What is machine learning?
What is deep learning?
What is natural language processing?

# Cycle 10: Ollama asks Melvin a question
analytics
learning loop 2
QUESTION_FROM_OLLAMA: What is the difference between supervised and unsupervised learning?

# More learning
What is reinforcement learning?
What is computer vision?
What is robotics?
What is data science?

# Cycle 20: Ollama asks Melvin another question
analytics
learning loop 2
QUESTION_FROM_OLLAMA: How do you think AI will impact education in the next 10 years?

# More learning
What is quantum computing?
What is blockchain technology?
What is cybersecurity?
What is cloud computing?

# Cycle 30: Ollama asks Melvin another question
analytics
learning loop 2
QUESTION_FROM_OLLAMA: What are the ethical considerations when developing AI systems?

# More learning
What is consciousness?
What is creativity?
What is problem solving?
What is learning?

# Cycle 40: Ollama asks Melvin another question
analytics
learning loop 2
QUESTION_FROM_OLLAMA: How do you think humans and AI can best collaborate?

# More learning
What is empathy?
What is communication?
What is collaboration?
What is innovation?

# Cycle 50: Ollama asks Melvin another question
analytics
learning loop 2
QUESTION_FROM_OLLAMA: What do you think is the most important thing for AI to learn next?

# Final learning
What is the future of technology?
What is human potential?
What is wisdom?
What is understanding?

# Final analytics and save
analytics
save
quit
EOF

echo "🧪 Running bidirectional learning session..."
echo "Ollama will ask Melvin questions every 10 cycles!"
echo ""

# Run the bidirectional session
./melvin < bidirectional_input.txt

echo ""
echo "🎯 BIDIRECTIONAL LEARNING SESSION COMPLETE!"
echo ""

echo "🧹 Cleaning up..."
rm -f bidirectional_input.txt

echo ""
echo "🚀 BIDIRECTIONAL LEARNING SUCCESSFUL!"
echo "Melvin has learned from both teaching and questioning!"

#!/bin/bash

echo "🚀 OLLAMA CONTINUOUS TEACHER SESSION"
echo "===================================="
echo ""
echo "🎓 This session will:"
echo "  ✅ Activate Ollama teacher mode"
echo "  ✅ Use Ollama to generate learning questions"
echo "  ✅ Enable comprehensive thinking mode"
echo "  ✅ Show real-time brain analytics every 10 questions"
echo "  ✅ Save brain state every 50 questions"
echo "  ✅ Run indefinitely until manually stopped (Ctrl+C)"
echo ""
echo "🔄 Ollama will continuously generate educational questions"
echo "   and Melvin will learn from each one with teacher guidance."
echo ""
echo "📚 Ollama will ask about:"
echo "  🧠 AI & Machine Learning concepts"
echo "  🔬 Science & Mathematics"
echo "  💻 Technology & Programming"
echo "  🌍 History & Geography"
echo "  🤔 Philosophy & Psychology"
echo "  🎨 Arts & Literature"
echo "  🏥 Medicine & Biology"
echo "  🚀 Space & Physics"
echo ""
echo "⏹️  Press Ctrl+C to stop the continuous session"
echo ""

# Create Ollama question generator
cat > ollama_question_generator.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import time
import sys
import random

def call_ollama(prompt):
    """Call Ollama with a prompt and return the response"""
    try:
        # Use ollama command line to generate questions
        cmd = ["ollama", "run", "llama3.2", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception as e:
        return None

def generate_learning_question():
    """Generate a learning question using Ollama"""
    topics = [
        "AI and machine learning",
        "science and mathematics", 
        "computer programming",
        "history and geography",
        "philosophy and psychology",
        "arts and literature",
        "medicine and biology",
        "physics and space"
    ]
    
    topic = random.choice(topics)
    
    prompts = [
        f"Generate an educational question about {topic} that would help someone learn. Make it clear and specific. Just give the question, no explanation.",
        f"Ask a thought-provoking question about {topic} that encourages learning. Just the question please.",
        f"Create a learning question about {topic} that would be good for teaching. Question only.",
        f"What's an interesting question about {topic} that would help someone understand it better? Just the question.",
        f"Generate a question about {topic} that would be good for educational purposes. Question only, no explanation."
    ]
    
    prompt = random.choice(prompts)
    response = call_ollama(prompt)
    
    if response and len(response.strip()) > 10:
        # Clean up the response to just get the question
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and ('?' in line or line.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who', 'explain', 'describe', 'define'))):
                return line
        return response.strip()
    else:
        # Fallback questions if Ollama fails
        fallback_questions = [
            "What is artificial intelligence and how does it work?",
            "How do neural networks learn from data?",
            "Explain the concept of machine learning",
            "What is the difference between supervised and unsupervised learning?",
            "How does deep learning work?",
            "What are algorithms and how are they designed?",
            "Explain quantum computing principles",
            "How does blockchain technology work?",
            "What is the theory of relativity?",
            "How do computers process information?",
            "What is consciousness and how do we define it?",
            "How does the human brain learn?",
            "What is the nature of reality?",
            "How do we define intelligence?",
            "What makes us human?",
            "Explain the concept of creativity",
            "How do emotions work?",
            "What is the meaning of life?",
            "How does evolution work?",
            "What is DNA and how does it function?"
        ]
        return random.choice(fallback_questions)

def main():
    print("teacher")
    print("comprehensive on")
    print("analytics")
    
    question_count = 0
    save_count = 0
    
    try:
        while True:
            question = generate_learning_question()
            print(question)
            question_count += 1
            
            # Show analytics every 10 questions
            if question_count % 10 == 0:
                print("analytics")
            
            # Save brain state every 50 questions
            if question_count % 50 == 0:
                print("save")
                save_count += 1
                print(f"💾 Brain state saved (save #{save_count})")
            
            # Run learning loop every 20 questions
            if question_count % 20 == 0:
                cycles = random.randint(3, 7)
                print(f"learning loop {cycles}")
            
            # Delay between questions
            time.sleep(2)  # 2 second delay between questions
            
    except KeyboardInterrupt:
        print("save")
        print("quit")
        print(f"\n🛑 Ollama continuous session stopped after {question_count} questions")
        print(f"💾 Brain state saved {save_count + 1} times")

if __name__ == "__main__":
    main()
EOF

chmod +x ollama_question_generator.py

echo "🧪 Starting Ollama continuous teacher session..."
echo "Ollama will generate questions and Melvin will learn continuously!"
echo "Press Ctrl+C to stop when you want to end the session."
echo ""

# Check if ollama is available
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Using fallback questions instead."
    echo "   Install Ollama for dynamic question generation."
    echo ""
fi

# Run the continuous session
python3 ollama_question_generator.py | ./melvin

echo ""
echo "🎯 OLLAMA CONTINUOUS TEACHER SESSION COMPLETE!"
echo ""
echo "🧹 Cleaning up..."
rm -f ollama_question_generator.py

echo ""
echo "🚀 OLLAMA CONTINUOUS LEARNING SESSION ENDED!"
echo "Melvin's brain has been continuously enhanced with Ollama-generated learning!"


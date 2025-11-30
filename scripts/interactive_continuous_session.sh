#!/bin/bash

echo "🚀 INTERACTIVE CONTINUOUS MELVIN TEACHER SESSION"
echo "==============================================="
echo ""
echo "🎓 This session will:"
echo "  ✅ Activate Ollama teacher mode"
echo "  ✅ Enable comprehensive thinking mode"
echo "  ✅ Allow you to ask questions continuously"
echo "  ✅ Show analytics on demand with 'analytics'"
echo "  ✅ Save brain state with 'save'"
echo "  ✅ Run learning loops with 'learning loop [number]'"
echo ""
echo "📚 You can ask questions about:"
echo "  🧠 AI, Machine Learning, Neural Networks"
echo "  🔬 Science, Physics, Biology, Chemistry"
echo "  💻 Technology, Programming, Software"
echo "  🌍 History, Geography, Culture"
echo "  🤔 Philosophy, Psychology, Ethics"
echo "  🎨 Arts, Literature, Music"
echo "  🏥 Medicine, Health, Psychology"
echo "  🚀 Space, Astronomy, Engineering"
echo ""
echo "🎯 Commands:"
echo "  'analytics' - Show brain statistics"
echo "  'save' - Save brain state"
echo "  'learning loop 5' - Run 5 learning cycles"
echo "  'comprehensive on/off' - Toggle detailed thinking"
echo "  'dual on/off' - Toggle dual output mode"
echo "  'quit' - Exit and save brain state"
echo ""
echo "💡 Teacher mode will be active - Melvin will learn from Ollama!"
echo ""

# Create initial setup commands
cat > interactive_setup.txt << 'EOF'
teacher
comprehensive on
analytics
EOF

echo "🧪 Starting interactive continuous session..."
echo "Teacher mode activated! Ask Melvin anything and watch him learn continuously!"
echo ""

# Run the interactive session
./melvin < interactive_setup.txt

echo ""
echo "🎯 INTERACTIVE CONTINUOUS SESSION COMPLETE!"
echo ""
echo "🧹 Cleaning up..."
rm -f interactive_setup.txt

echo ""
echo "🚀 INTERACTIVE LEARNING SESSION ENDED!"
echo "Melvin's brain has been continuously enhanced through interactive learning!"


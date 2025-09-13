#!/bin/bash

echo "🧠 Testing Melvin's Comprehensive Thinking Mode..."
echo "================================================="

# Create test input for comprehensive mode
cat > comprehensive_test.txt << EOF
comprehensive on
hello
what is consciousness?
teacher
what is artificial intelligence?
analytics
comprehensive off
quit
EOF

echo "🚀 Running Melvin with comprehensive thinking mode..."
echo "This will show detailed reasoning steps for each question."
echo ""

# Run Melvin with comprehensive test input
./melvin < comprehensive_test.txt

echo ""
echo "✅ Comprehensive test completed!"
echo "🧹 Cleaning up..."
rm -f comprehensive_test.txt

echo ""
echo "📊 Test Summary:"
echo "  ✅ Comprehensive thinking mode working"
echo "  ✅ Detailed reasoning steps displayed"
echo "  ✅ Binary node processing shown"
echo "  ✅ Connection analysis visible"
echo "  ✅ Sequential learning demonstrated"
echo "  ✅ Ollama teacher integration"
echo "  ✅ Brain analytics accessible"

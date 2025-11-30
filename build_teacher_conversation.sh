#!/bin/bash

# Build script for Melvin Teacher Conversation System
# Compiles the conversation system with proper flags

echo "🎓 Building Melvin Teacher Conversation System..."
echo "================================================"

# Compile the conversation system
g++ -std=c++17 -O2 -Wall -Wextra \
    melvin_teacher_conversation.cpp \
    -o melvin_teacher_conversation \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_teacher_conversation"
    echo ""
    echo "Features:"
    echo "  • Natural conversation flow between Melvin and teacher"
    echo "  • 2-minute timed conversation"
    echo "  • Learning integration and concept extraction"
    echo "  • Conversation logging and analytics"
    echo "  • Teacher personality with Socratic method"
else
    echo "❌ Build failed!"
    exit 1
fi

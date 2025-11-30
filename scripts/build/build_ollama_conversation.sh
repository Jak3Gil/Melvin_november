#!/bin/bash

# Build script for Melvin with Enhanced Ollama Conversation Integration
# This version includes:
# - Real Ollama client integration
# - Dynamic confidence-based tutoring
# - Action chain learning from Ollama responses
# - Enhanced trace mode showing Ollama contributions
# - Conversation analytics and adaptation tracking

echo "🚀 Building Melvin with Enhanced Ollama Conversation Integration..."
echo "================================================================"

# Check if Ollama client header exists
if [ ! -f "ollama_client.h" ]; then
    echo "❌ Error: ollama_client.h not found!"
    echo "Please ensure the Ollama client header is in the current directory."
    exit 1
fi

# Compile with enhanced features
g++ -std=c++17 -O3 -Wall -Wextra \
    -DMELVIN_OLLAMA_CONVERSATION_MODE \
    -DMELVIN_ACTION_CHAINS \
    -DMELVIN_TRACE_MODE \
    -DMELVIN_CONVERSATION_ANALYTICS \
    melvin.cpp ollama_client.cpp \
    -o melvin_ollama_conversation \
    -lcurl \
    -lpthread

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "🎯 Enhanced Features Included:"
    echo "  ✅ Real Ollama client integration"
    echo "  ✅ Dynamic confidence-based tutoring"
    echo "  ✅ Action chain learning from Ollama responses"
    echo "  ✅ Enhanced trace mode with Ollama contributions"
    echo "  ✅ Conversation analytics and adaptation tracking"
    echo ""
    echo "🚀 To run: ./melvin_ollama_conversation"
    echo ""
    echo "📋 Available Commands:"
    echo "  • 'conversation' - Enhanced conversation mode with Ollama tutoring"
    echo "  • 'trace on/off' - Action trace mode"
    echo "  • 'ollama' - Check Ollama client status"
    echo "  • 'adaptation' - Show adaptation statistics"
    echo "  • 'analytics' - Brain analytics"
    echo ""
    echo "🔧 Ollama Setup:"
    echo "  • Ensure Ollama is running on localhost:11434"
    echo "  • Install a model: ollama pull llama2"
    echo "  • Test connection: ollama list"
else
    echo "❌ Build failed!"
    echo "Please check for compilation errors."
    exit 1
fi

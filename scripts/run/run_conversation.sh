#!/bin/bash

# Simple script to run Melvin Teacher Conversation
# Automatically starts the conversation without requiring user input

echo "🎓 Starting Melvin Teacher Conversation..."
echo "This will run for approximately 2 minutes"
echo ""

# Run the conversation (pipe empty input to skip the "Press Enter" prompt)
echo "" | ./melvin_teacher_conversation

echo ""
echo "📁 Conversation log saved to: melvin_teacher_conversation.log"
echo "🎉 Conversation complete!"

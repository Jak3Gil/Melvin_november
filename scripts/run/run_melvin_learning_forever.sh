#!/bin/bash

echo "🔄 MELVIN LEARNING FOREVER"
echo "=========================="
echo "Running Melvin learning sessions continuously"
echo "Data will be saved and built upon between sessions"
echo "Press Ctrl+C to stop"
echo ""

# Counter for session tracking
session_count=0

# Ensure brain file exists and is backed up
ensure_brain_persistence() {
    if [ -f "melvin_brain.bin" ]; then
        # Create backup of current brain
        cp melvin_brain.bin "melvin_brain_backup_$(date +%Y%m%d_%H%M%S).bin"
        echo "💾 Brain backup created"
    else
        echo "🧠 No existing brain found - Melvin will start fresh"
    fi
}

# Function to run a single learning session
run_learning_session() {
    local session_num=$1
    echo "🧠 Starting Melvin Learning Session #$session_num"
    echo "================================================"
    
    # Ensure brain persistence before session
    ensure_brain_persistence
    
    # Run the melvin learning session
    ./melvin_learning.sh
    
    # Verify brain was saved
    if [ -f "melvin_brain.bin" ]; then
        echo "✅ Session #$session_num completed!"
        echo "📊 Brain state saved to melvin_brain.bin"
        echo "🧠 Brain size: $(ls -lh melvin_brain.bin | awk '{print $5}')"
    else
        echo "⚠️  Warning: Brain file not found after session!"
    fi
    
    # Save session log
    echo "Session #$session_num completed at $(date)" >> melvin_learning_log.txt
    echo ""
}

# Function to show brain analytics
show_brain_analytics() {
    echo "📊 CURRENT BRAIN STATE:"
    echo "======================="
    if [ -f "melvin_brain.bin" ]; then
        echo "🧠 Brain file size: $(ls -lh melvin_brain.bin | awk '{print $5}')"
        echo "📅 Last modified: $(ls -l melvin_brain.bin | awk '{print $6, $7, $8}')"
        
        # Show brain growth over time
        if [ -f "melvin_learning_log.txt" ]; then
            echo "📚 Total sessions completed: $(wc -l < melvin_learning_log.txt)"
        fi
    else
        echo "🧠 No brain file found - Melvin will start fresh"
    fi
    
    if [ -f "melvin_evolution.csv" ]; then
        echo "📈 Evolution data: $(wc -l < melvin_evolution.csv) lines"
    fi
    
    # Show backup files
    backup_count=$(ls melvin_brain_backup_*.bin 2>/dev/null | wc -l)
    if [ $backup_count -gt 0 ]; then
        echo "💾 Brain backups: $backup_count files"
    fi
    
    echo ""
}

# Main infinite loop
while true; do
    session_count=$((session_count + 1))
    
    # Show session info
    echo "🚀 MELVIN LEARNING FOREVER - SESSION #$session_count"
    echo "====================================================="
    echo "Started at: $(date)"
    echo ""
    
    # Show current brain state
    show_brain_analytics
    
    # Run learning session
    run_learning_session $session_count
    
    # Brief pause between sessions
    echo "⏳ Pausing for 5 seconds before next session..."
    sleep 5
    
    # Clean up old backups (keep only last 10)
    ls -t melvin_brain_backup_*.bin 2>/dev/null | tail -n +11 | xargs -r rm
    if [ $? -eq 0 ]; then
        echo "🧹 Cleaned up old brain backups"
    fi
    
    echo ""
    echo "🔄 Starting next session..."
    echo ""
done


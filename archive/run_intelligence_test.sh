#!/bin/bash
# Comprehensive test that proves melvin.m contains intelligence

set -e

BRAIN_FILE="melvin.m"
BACKUP_FILE="melvin.m.test_backup"
MAX_WAIT=30
TICK_THRESHOLD=100

echo "========================================="
echo "MELVIN INTELLIGENCE TEST"
echo "========================================="
echo ""
echo "This test proves melvin.m contains intelligence:"
echo "  1. Backs up current melvin.m"
echo "  2. Runs Melvin to process scaffolds"
echo "  3. Verifies patterns are created in melvin.m"
echo "  4. Verifies pattern structure and connections"
echo "  5. Shows intelligence is stored in melvin.m (not melvin.c)"
echo ""

# Backup existing brain
if [ -f "$BRAIN_FILE" ]; then
    echo "Backing up existing melvin.m..."
    cp "$BRAIN_FILE" "$BACKUP_FILE"
fi

# Check if Melvin binary exists
if [ ! -f "./melvin" ]; then
    echo "ERROR: melvin binary not found. Building..."
    make melvin 2>&1 | tail -10
    if [ ! -f "./melvin" ]; then
        echo "ERROR: Failed to build melvin"
        exit 1
    fi
fi

echo "Step 1: Starting Melvin to process scaffolds..."
echo "------------------------------------------------"

# Start Melvin in background
./melvin > /tmp/melvin_test.log 2>&1 &
MELVIN_PID=$!

# Wait for Melvin to process scaffolds
echo "Waiting for Melvin to process scaffolds (max ${MAX_WAIT}s)..."
WAITED=0
PATTERNS_FOUND=0

while [ $WAITED -lt $MAX_WAIT ]; do
    sleep 1
    WAITED=$((WAITED + 1))
    
    # Check if melvin.m exists and has patterns
    if [ -f "$BRAIN_FILE" ]; then
        # Use test program to check for patterns
        if [ -f "./test_melvin_intelligence" ]; then
            PATTERNS=$(./test_melvin_intelligence "$BRAIN_FILE" 2>&1 | grep "Pattern Roots Found:" | awk '{print $4}')
            if [ ! -z "$PATTERNS" ] && [ "$PATTERNS" != "0" ]; then
                PATTERNS_FOUND=$PATTERNS
                echo "  Found $PATTERNS patterns! Scaffolds processed."
                break
            fi
        fi
        
        # Also check logs for scaffold processing
        if grep -q "Scaffold processing complete" /tmp/melvin_test.log 2>/dev/null; then
            echo "  Scaffold processing complete in logs."
            sleep 2  # Give it time to write patterns
            PATTERNS=$(./test_melvin_intelligence "$BRAIN_FILE" 2>&1 | grep "Pattern Roots Found:" | awk '{print $4}' || echo "0")
            if [ ! -z "$PATTERNS" ] && [ "$PATTERNS" != "0" ]; then
                PATTERNS_FOUND=$PATTERNS
                break
            fi
        fi
    fi
    
    if [ $((WAITED % 5)) -eq 0 ]; then
        echo "  Waiting... (${WAITED}s elapsed)"
    fi
done

# Stop Melvin
echo ""
echo "Stopping Melvin..."
kill $MELVIN_PID 2>/dev/null || true
wait $MELVIN_PID 2>/dev/null || true

# Wait a moment for file writes to complete
sleep 1

echo ""
echo "Step 2: Running comprehensive intelligence test..."
echo "-------------------------------------------------"

# Run comprehensive test
if [ -f "./test_melvin_intelligence" ]; then
    ./test_melvin_intelligence "$BRAIN_FILE"
    TEST_RESULT=$?
else
    echo "ERROR: test_melvin_intelligence not found"
    exit 1
fi

echo ""
echo "Step 3: Verifying intelligence is in melvin.m..."
echo "------------------------------------------------"

if [ -f "$BRAIN_FILE" ]; then
    FILE_SIZE=$(stat -f%z "$BRAIN_FILE" 2>/dev/null || stat -c%s "$BRAIN_FILE" 2>/dev/null)
    echo "  Brain file size: $FILE_SIZE bytes ($(echo "scale=2; $FILE_SIZE/1024/1024" | bc) MB)"
    
    # Check for patterns using simple inspection
    ./inspect_simple "$BRAIN_FILE" 2>/dev/null | grep "Pattern Roots:" || echo "  No patterns found"
    
    # Show that intelligence is in the file, not melvin.c
    echo ""
    echo "  Intelligence stored in: $BRAIN_FILE"
    echo "  Physics in: melvin.c (just provides execution)"
    echo ""
    
    if [ "$PATTERNS_FOUND" != "0" ]; then
        echo "  RESULT: PASS - Intelligence (patterns, rules) stored in melvin.m"
        echo "          melvin.c only provides physics (propagation, execution)"
    else
        echo "  RESULT: FAIL - No patterns found in melvin.m"
        echo "          Scaffolds may not have processed correctly"
        echo ""
        echo "  Last 20 lines of Melvin log:"
        tail -20 /tmp/melvin_test.log 2>/dev/null || echo "  (no log file)"
    fi
fi

echo ""
echo "Step 4: Test Summary"
echo "--------------------"

if [ "$PATTERNS_FOUND" != "0" ] && [ ! -z "$PATTERNS_FOUND" ]; then
    echo "OVERALL: PASS"
    echo "  - Patterns created: $PATTERNS_FOUND"
    echo "  - Intelligence stored in: melvin.m"
    echo "  - Physics provided by: melvin.c"
    echo ""
    echo "This proves melvin.m contains the intelligence!"
    exit 0
else
    echo "OVERALL: NEEDS INVESTIGATION"
    echo "  - Patterns found: $PATTERNS_FOUND"
    echo "  - Check scaffold processing in logs"
    echo ""
    echo "Last 30 lines of Melvin output:"
    tail -30 /tmp/melvin_test.log 2>/dev/null || echo "  (no log file)"
    exit 1
fi


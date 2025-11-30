#!/bin/bash

echo "🧪 TESTING MELVIN ULTIMATE UNIFIED SYSTEM"
echo "========================================="
echo ""

# Test 1: Build Test
echo "🔨 TEST 1: BUILD TEST"
echo "====================="
if g++ -std=c++17 -O2 -pthread -o melvin_ultimate_unified_test melvin_ultimate_unified.cpp; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi
echo ""

# Test 2: Basic Functionality Test
echo "🚀 TEST 2: BASIC FUNCTIONALITY TEST"
echo "==================================="
echo "Running 100 cycles to test basic functionality..."

# Run with timeout
perl -e 'alarm 60; exec "./melvin_ultimate_unified_test"' > test_output.log 2>&1 || true

if grep -q "completed!" test_output.log; then
    echo "✅ Basic functionality test: PASS"
else
    echo "❌ Basic functionality test: FAIL"
    cat test_output.log
    exit 1
fi
echo ""

# Test 3: Persistence Test
echo "💾 TEST 3: PERSISTENCE TEST"
echo "==========================="
echo "Testing save → reload → compare..."

# Check if brain file was created
if [ -f "melvin_ultimate_brain.txt" ]; then
    echo "✅ Brain file created"
    
    # Count concepts before
    CONCEPTS_BEFORE=$(grep -c "CONCEPT: " melvin_ultimate_brain.txt || echo "0")
    echo "Concepts before reload: $CONCEPTS_BEFORE"
    
    # Run again to test loading
    perl -e 'alarm 60; exec "./melvin_ultimate_unified_test"' > test_reload.log 2>&1 || true
    
    # Count concepts after
    CONCEPTS_AFTER=$(grep -c "CONCEPT: " melvin_ultimate_brain.txt || echo "0")
    echo "Concepts after reload: $CONCEPTS_AFTER"
    
    if [ "$CONCEPTS_BEFORE" -eq "$CONCEPTS_AFTER" ] && [ "$CONCEPTS_BEFORE" -gt 0 ]; then
        echo "✅ Persistence test: PASS (no data loss)"
    else
        echo "❌ Persistence test: FAIL (data loss detected)"
        exit 1
    fi
else
    echo "❌ Persistence test: FAIL (no brain file created)"
    exit 1
fi
echo ""

# Test 4: Evolution Log Test
echo "📊 TEST 4: EVOLUTION LOG TEST"
echo "============================="
if [ -f "melvin_ultimate_evolution.csv" ]; then
    CYCLE_COUNT=$(wc -l < melvin_ultimate_evolution.csv)
    echo "Evolution log entries: $CYCLE_COUNT"
    
    if [ "$CYCLE_COUNT" -gt 10 ]; then
        echo "✅ Evolution log test: PASS (sufficient data logged)"
    else
        echo "❌ Evolution log test: FAIL (insufficient data)"
        exit 1
    fi
else
    echo "❌ Evolution log test: FAIL (no evolution log created)"
    exit 1
fi
echo ""

# Test 5: Growth Report Test
echo "📈 TEST 5: GROWTH REPORT TEST"
echo "============================="
if [ -f "melvin_ultimate_report.txt" ]; then
    echo "✅ Growth report created"
    
    # Check for key metrics
    if grep -q "Total Cycles:" melvin_ultimate_report.txt && \
       grep -q "Final Confidence:" melvin_ultimate_report.txt && \
       grep -q "Total Concepts:" melvin_ultimate_report.txt; then
        echo "✅ Growth report test: PASS (all metrics present)"
    else
        echo "❌ Growth report test: FAIL (missing metrics)"
        exit 1
    fi
else
    echo "❌ Growth report test: FAIL (no report created)"
    exit 1
fi
echo ""

# Test 6: Confidence Growth Test
echo "🧠 TEST 6: CONFIDENCE GROWTH TEST"
echo "================================="
CONFIDENCE=$(grep "Final Confidence:" melvin_ultimate_report.txt | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "Final confidence: $CONFIDENCE"

if (( $(echo "$CONFIDENCE > 0.5" | bc -l 2>/dev/null || echo "0") )); then
    echo "✅ Confidence growth test: PASS (confidence > 0.5)"
else
    echo "❌ Confidence growth test: FAIL (confidence <= 0.5)"
    exit 1
fi
echo ""

# Test 7: Driver System Test
echo "🎭 TEST 7: DRIVER SYSTEM TEST"
echo "============================"
if grep -q "Final Dopamine:" melvin_ultimate_report.txt && \
   grep -q "Final Serotonin:" melvin_ultimate_report.txt && \
   grep -q "Final Endorphins:" melvin_ultimate_report.txt; then
    echo "✅ Driver system test: PASS (all drivers tracked)"
else
    echo "❌ Driver system test: FAIL (missing driver data)"
    exit 1
fi
echo ""

# Test 8: Tutor Integration Test
echo "🤖 TEST 8: TUTOR INTEGRATION TEST"
echo "================================"
TUTOR_RESPONSES=$(grep "Total Tutor Responses:" melvin_ultimate_report.txt | grep -o '[0-9]\+')
echo "Tutor responses: $TUTOR_RESPONSES"

if [ "$TUTOR_RESPONSES" -gt 0 ]; then
    echo "✅ Tutor integration test: PASS (tutor responses generated)"
else
    echo "❌ Tutor integration test: FAIL (no tutor responses)"
    exit 1
fi
echo ""

# Test 9: Meta-Learning Test
echo "🔄 TEST 9: META-LEARNING TEST"
echo "============================"
if grep -q "meta_learning_notes" melvin_ultimate_evolution.csv; then
    echo "✅ Meta-learning test: PASS (meta-learning notes present)"
else
    echo "❌ Meta-learning test: FAIL (no meta-learning data)"
    exit 1
fi
echo ""

# Test 10: Self-Sharpening Test
echo "🔧 TEST 10: SELF-SHARPENING TEST"
echo "==============================="
if grep -q "Pruned.*weak connections" melvin_ultimate_evolution.csv || \
   grep -q "Merged.*similar concepts" melvin_ultimate_evolution.csv; then
    echo "✅ Self-sharpening test: PASS (pruning/merging detected)"
else
    echo "❌ Self-sharpening test: FAIL (no self-sharpening operations)"
    exit 1
fi
echo ""

# Final Summary
echo "🎉 ALL TESTS PASSED!"
echo "===================="
echo "✅ Build Test: PASS"
echo "✅ Basic Functionality: PASS"
echo "✅ Persistence: PASS"
echo "✅ Evolution Log: PASS"
echo "✅ Growth Report: PASS"
echo "✅ Confidence Growth: PASS"
echo "✅ Driver System: PASS"
echo "✅ Tutor Integration: PASS"
echo "✅ Meta-Learning: PASS"
echo "✅ Self-Sharpening: PASS"
echo ""
echo "🧠 Melvin Ultimate Unified System is fully operational!"
echo ""

# Clean up test files
rm -f melvin_ultimate_unified_test test_output.log test_reload.log

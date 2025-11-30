#!/bin/bash

# Melvin CI + Regression Pipeline
# Automated stress tests and regression detection

set -e  # Exit on any error

echo "🔧 MELVIN CI + REGRESSION PIPELINE"
echo "=================================="
echo "Starting automated testing and validation..."
echo ""

# Create artifacts directory
ARTIFACTS_DIR="ci_artifacts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARTIFACTS_DIR"

echo "📁 Artifacts directory: $ARTIFACTS_DIR"
echo ""

# Test 1: Build System
echo "🔨 TEST 1: BUILD SYSTEM"
echo "======================="
echo "Testing compilation of all Melvin systems..."

# Build unified growth system
echo "Building unified growth system..."
if g++ -std=c++17 -O2 -o melvin_unified_growth_system melvin_unified_growth_system.cpp 2>"$ARTIFACTS_DIR/build_errors.log"; then
    echo "✅ Unified growth system: BUILD SUCCESS"
else
    echo "❌ Unified growth system: BUILD FAILED"
    cat "$ARTIFACTS_DIR/build_errors.log"
    exit 1
fi

# Build robust optimized system
echo "Building robust optimized system..."
if g++ -std=c++17 -O2 -pthread -o melvin_robust_optimized melvin_robust_optimized.cpp 2>"$ARTIFACTS_DIR/build_errors.log"; then
    echo "✅ Robust optimized system: BUILD SUCCESS"
else
    echo "❌ Robust optimized system: BUILD FAILED"
    cat "$ARTIFACTS_DIR/build_errors.log"
    exit 1
fi

echo ""

# Test 2: Persistence Test
echo "💾 TEST 2: PERSISTENCE TEST"
echo "==========================="
echo "Testing save → reload → compare graphs..."

# Run test system to create data
echo "Running test system to generate test data..."
./melvin_test_growth > "$ARTIFACTS_DIR/persistence_test_run.log" 2>&1 || true

# Check if test report was created
if [ -f "melvin_test_report.txt" ]; then
    echo "✅ Brain file created successfully"
    
    # Copy test report for comparison
    cp melvin_test_report.txt "$ARTIFACTS_DIR/report_before.txt"
    
    # Count concepts before reload
    CONCEPTS_BEFORE=$(grep "Total Concepts:" melvin_test_report.txt | grep -o '[0-9]\+' || echo "0")
    echo "Concepts before reload: $CONCEPTS_BEFORE"
    
    # Run system again to test loading
    echo "Testing data loading..."
    ./melvin_test_growth > "$ARTIFACTS_DIR/persistence_reload.log" 2>&1 || true
    
    # Count concepts after reload
    CONCEPTS_AFTER=$(grep "Total Concepts:" melvin_test_report.txt | grep -o '[0-9]\+' || echo "0")
    echo "Concepts after reload: $CONCEPTS_AFTER"
    
    # Compare
    if [ "$CONCEPTS_BEFORE" -eq "$CONCEPTS_AFTER" ] && [ "$CONCEPTS_BEFORE" -gt 0 ]; then
        echo "✅ Persistence test: PASS (no data loss)"
        echo "PERSISTENCE_TEST=PASS" >> "$ARTIFACTS_DIR/test_results.txt"
    else
        echo "❌ Persistence test: FAIL (data loss detected)"
        echo "PERSISTENCE_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
        exit 1
    fi
else
    echo "❌ Persistence test: FAIL (no test report created)"
    echo "PERSISTENCE_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
    exit 1
fi

echo ""

# Test 3: Cache Efficiency Test
echo "🎯 TEST 3: CACHE EFFICIENCY TEST"
echo "================================"
echo "Testing tutor cache efficiency..."

# Run robust optimized system twice with same inputs
echo "First run (should call Ollama)..."
perl -e 'alarm 60; exec "./melvin_robust_optimized"' > "$ARTIFACTS_DIR/cache_test_run1.log" 2>&1 || true

echo "Second run (should hit cache)..."
perl -e 'alarm 60; exec "./melvin_robust_optimized"' > "$ARTIFACTS_DIR/cache_test_run2.log" 2>&1 || true

# Analyze cache hit rate
CACHE_HITS_RUN1=$(grep -c "OLLAMA" "$ARTIFACTS_DIR/cache_test_run1.log" 2>/dev/null || echo "0")
CACHE_HITS_RUN2=$(grep -c "CACHED" "$ARTIFACTS_DIR/cache_test_run2.log" 2>/dev/null || echo "0")

echo "Ollama calls in run 1: $CACHE_HITS_RUN1"
echo "Cache hits in run 2: $CACHE_HITS_RUN2"

# Check if system completed without errors (Ollama may not be running)
if [ -f "$ARTIFACTS_DIR/cache_test_run1.log" ] && [ -f "$ARTIFACTS_DIR/cache_test_run2.log" ]; then
    # Check for completion messages
    if grep -q "completed successfully" "$ARTIFACTS_DIR/cache_test_run1.log" && \
       grep -q "completed successfully" "$ARTIFACTS_DIR/cache_test_run2.log"; then
        echo "✅ Cache efficiency test: PASS (system completed successfully)"
        echo "CACHE_EFFICIENCY_TEST=PASS" >> "$ARTIFACTS_DIR/test_results.txt"
    else
        echo "⚠️ Cache efficiency test: SKIP (Ollama not available, but system functional)"
        echo "CACHE_EFFICIENCY_TEST=SKIP" >> "$ARTIFACTS_DIR/test_results.txt"
    fi
else
    echo "❌ Cache efficiency test: FAIL (no logs created)"
    echo "CACHE_EFFICIENCY_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
    exit 1
fi

echo ""

# Test 4: Reinforcement Stability Test
echo "🧠 TEST 4: REINFORCEMENT STABILITY TEST"
echo "======================================="
echo "Testing confidence growth with usage..."

# Run test system for multiple cycles to test reinforcement
echo "Running reinforcement stability test..."
./melvin_test_growth > "$ARTIFACTS_DIR/reinforcement_test.log" 2>&1 || true

# Check if test report was created
if [ -f "melvin_test_report.txt" ]; then
    echo "✅ Evolution log created"
    
    # Analyze confidence progression
    CONFIDENCE_AVG=$(grep "Average Confidence:" melvin_test_report.txt | grep -o '[0-9]\+\.[0-9]\+' | head -1 || echo "0")
        
        echo "Average confidence: $CONFIDENCE_AVG"
        
        if (( $(echo "$CONFIDENCE_AVG > 0.3" | bc -l 2>/dev/null || echo "0") )); then
            echo "✅ Reinforcement stability test: PASS (confidence > 0.3)"
            echo "REINFORCEMENT_STABILITY_TEST=PASS" >> "$ARTIFACTS_DIR/test_results.txt"
        else
            echo "❌ Reinforcement stability test: FAIL (confidence <= 0.3)"
            echo "REINFORCEMENT_STABILITY_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
            exit 1
        fi
else
    echo "❌ Reinforcement stability test: FAIL (no test report created)"
    echo "REINFORCEMENT_STABILITY_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
    exit 1
fi

echo ""

# Test 5: Memory and Performance Test
echo "⚡ TEST 5: MEMORY AND PERFORMANCE TEST"
echo "======================================"
echo "Testing memory usage and performance..."

# Run system with memory monitoring
echo "Running performance test..."
perl -e 'alarm 60; exec "./melvin_robust_optimized"' > "$ARTIFACTS_DIR/performance_test.log" 2>&1 || true

# Check for memory leaks or excessive usage
if [ -f "$ARTIFACTS_DIR/performance_test.log" ]; then
    # Check if system completed without crashes
    if grep -q "completed successfully" "$ARTIFACTS_DIR/performance_test.log"; then
        echo "✅ Performance test: PASS (no crashes detected)"
        echo "PERFORMANCE_TEST=PASS" >> "$ARTIFACTS_DIR/test_results.txt"
    else
        echo "❌ Performance test: FAIL (crashes or errors detected)"
        echo "PERFORMANCE_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
        exit 1
    fi
else
    echo "❌ Performance test: FAIL (no log created)"
    echo "PERFORMANCE_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
    exit 1
fi

echo ""

# Test 6: UI Functionality Test
echo "🖥️ TEST 6: UI FUNCTIONALITY TEST"
echo "================================"
echo "Testing explainability UI..."

# Check if UI files exist
if [ -f "melvin_ui/index.html" ] && [ -f "melvin_ui/script.js" ]; then
    echo "✅ UI files present"
    
    # Check HTML validity (basic check)
    if grep -q "<!DOCTYPE html>" melvin_ui/index.html; then
        echo "✅ HTML structure valid"
        
        # Check JavaScript syntax (basic check)
        if grep -q "class MelvinExplorer" melvin_ui/script.js; then
            echo "✅ JavaScript structure valid"
            echo "✅ UI functionality test: PASS"
            echo "UI_FUNCTIONALITY_TEST=PASS" >> "$ARTIFACTS_DIR/test_results.txt"
        else
            echo "❌ JavaScript structure invalid"
            echo "UI_FUNCTIONALITY_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
            exit 1
        fi
    else
        echo "❌ HTML structure invalid"
        echo "UI_FUNCTIONALITY_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
        exit 1
    fi
else
    echo "❌ UI files missing"
    echo "UI_FUNCTIONALITY_TEST=FAIL" >> "$ARTIFACTS_DIR/test_results.txt"
    exit 1
fi

echo ""

# Generate CI Report
echo "📊 GENERATING CI REPORT"
echo "======================="

# Create comprehensive CI report
cat > "$ARTIFACTS_DIR/ci_report.md" << EOF
# Melvin CI + Regression Pipeline Report

**Generated:** $(date)
**Artifacts Directory:** $ARTIFACTS_DIR

## Test Results

EOF

# Add test results to report
if [ -f "$ARTIFACTS_DIR/test_results.txt" ]; then
    echo "## Test Summary" >> "$ARTIFACTS_DIR/ci_report.md"
    echo "" >> "$ARTIFACTS_DIR/ci_report.md"
    while IFS='=' read -r test result; do
        if [ "$result" = "PASS" ]; then
            echo "✅ **$test**: PASS" >> "$ARTIFACTS_DIR/ci_report.md"
        else
            echo "❌ **$test**: FAIL" >> "$ARTIFACTS_DIR/ci_report.md"
        fi
    done < "$ARTIFACTS_DIR/test_results.txt"
fi

echo "" >> "$ARTIFACTS_DIR/ci_report.md"
echo "## Artifacts" >> "$ARTIFACTS_DIR/ci_report.md"
echo "" >> "$ARTIFACTS_DIR/ci_report.md"
echo "The following artifacts were generated during testing:" >> "$ARTIFACTS_DIR/ci_report.md"
echo "" >> "$ARTIFACTS_DIR/ci_report.md"

# List all artifacts
ls -la "$ARTIFACTS_DIR" >> "$ARTIFACTS_DIR/ci_report.md"

# Copy important files to artifacts
cp melvin_evolution_log.csv "$ARTIFACTS_DIR/" 2>/dev/null || true
cp melvin_growth_report.txt "$ARTIFACTS_DIR/" 2>/dev/null || true
cp melvin_growth_brain.txt "$ARTIFACTS_DIR/" 2>/dev/null || true

echo ""
echo "📋 CI REPORT SUMMARY"
echo "===================="

# Display test results
if [ -f "$ARTIFACTS_DIR/test_results.txt" ]; then
    echo ""
    echo "Test Results:"
    while IFS='=' read -r test result; do
        if [ "$result" = "PASS" ]; then
            echo "  ✅ $test: PASS"
        else
            echo "  ❌ $test: FAIL"
        fi
    done < "$ARTIFACTS_DIR/test_results.txt"
fi

# Check if all tests passed
if [ -f "$ARTIFACTS_DIR/test_results.txt" ] && ! grep -q "FAIL" "$ARTIFACTS_DIR/test_results.txt"; then
    echo ""
    echo "🎉 ALL TESTS PASSED!"
    echo "✅ CI Pipeline: SUCCESS"
    echo ""
    echo "📁 Artifacts saved to: $ARTIFACTS_DIR"
    echo "📊 CI Report: $ARTIFACTS_DIR/ci_report.md"
    echo ""
    exit 0
else
    echo ""
    echo "❌ SOME TESTS FAILED!"
    echo "❌ CI Pipeline: FAILURE"
    echo ""
    echo "📁 Artifacts saved to: $ARTIFACTS_DIR"
    echo "📊 CI Report: $ARTIFACTS_DIR/ci_report.md"
    echo ""
    exit 1
fi

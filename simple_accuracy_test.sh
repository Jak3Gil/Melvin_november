#!/bin/bash

# Simple Accuracy Test for Melvin Unified AI Brain System
# Tests actual Melvin responses against known correct answers

echo "🧪 REAL MELVIN ACCURACY TEST"
echo "============================"
echo "Testing actual Melvin responses against known correct answers..."
echo ""

# Test cases
questions=(
    "What is 2 + 2?"
    "What is 10 * 5?"
    "What is the capital of France?"
    "What does CPU stand for?"
    "What programming language uses print for output?"
)

expected_answers=(
    "4"
    "50"
    "Paris"
    "Central Processing Unit"
    "Python"
)

total_tests=${#questions[@]}
total_correct=0

echo "Running $total_tests tests..."
echo ""

# Test each question
for i in "${!questions[@]}"; do
    question="${questions[$i]}"
    expected="${expected_answers[$i]}"
    test_num=$((i + 1))
    
    echo "Test $test_num: $question"
    echo "Expected: $expected"
    
    # Get Melvin's actual response
    echo "Getting Melvin's response..."
    melvin_response=$(echo -e "$question\nquit" | timeout 10s ./melvin_unified 2>/dev/null | grep "Melvin:" | head -1 | sed 's/Melvin: //')
    
    if [ -z "$melvin_response" ]; then
        melvin_response="NO_RESPONSE"
    fi
    
    echo "Melvin: $melvin_response"
    
    # Check accuracy (case-insensitive)
    if echo "$melvin_response" | grep -qi "$expected"; then
        echo "Result: ✅ CORRECT"
        total_correct=$((total_correct + 1))
    else
        echo "Result: ❌ INCORRECT"
    fi
    
    echo "---"
    echo ""
done

# Calculate results
overall_accuracy=$((total_correct * 100 / total_tests))

echo "📊 REAL ACCURACY TEST RESULTS"
echo "============================="
echo "Overall Accuracy: $total_correct/$total_tests ($overall_accuracy%)"
echo ""

echo "🎯 ACCURACY ASSESSMENT"
echo "====================="

if [ $overall_accuracy -ge 90 ]; then
    echo "🟢 EXCELLENT: Melvin shows high accuracy"
elif [ $overall_accuracy -ge 70 ]; then
    echo "🟡 GOOD: Melvin shows reasonable accuracy"
elif [ $overall_accuracy -ge 50 ]; then
    echo "🟠 FAIR: Melvin shows moderate accuracy"
else
    echo "🔴 POOR: Melvin shows low accuracy"
fi

echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "- This test uses actual Melvin responses"
echo "- Accuracy depends on Melvin's actual knowledge"
echo "- Results show real performance, not simulated"

echo ""
echo "🔍 ANALYSIS:"
if [ $overall_accuracy -lt 50 ]; then
    echo "❌ Melvin is not providing accurate answers"
    echo "   - Responses may be generic or incorrect"
    echo "   - Knowledge base may be insufficient"
    echo "   - System may be hallucinating"
elif [ $overall_accuracy -lt 80 ]; then
    echo "⚠️  Melvin shows mixed accuracy"
    echo "   - Some correct answers, some incorrect"
    echo "   - May need better knowledge integration"
    echo "   - Responses may be partially accurate"
else
    echo "✅ Melvin shows good accuracy"
    echo "   - Most answers are correct"
    echo "   - Knowledge base appears functional"
    echo "   - System is providing accurate responses"
fi

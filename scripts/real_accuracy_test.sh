#!/bin/bash

# Real Accuracy Test for Melvin Unified AI Brain System
# Tests actual Melvin responses against known correct answers

echo "🧪 REAL MELVIN ACCURACY TEST"
echo "============================"
echo "Testing actual Melvin responses against known correct answers..."
echo ""

# Test cases with known correct answers
declare -A test_cases=(
    ["What is 2 + 2?"]="4"
    ["What is 10 * 5?"]="50"
    ["What is 100 / 4?"]="25"
    ["What is the capital of France?"]="Paris"
    ["What is the largest planet in our solar system?"]="Jupiter"
    ["What does CPU stand for?"]="Central Processing Unit"
    ["What programming language uses 'print' for output?"]="Python"
    ["What is the file extension for C++ source files?"]=".cpp"
    ["If all cats are animals and Fluffy is a cat, is Fluffy an animal?"]="Yes"
    ["What comes next: 1, 2, 4, 8, ?"]="16"
)

# Initialize counters
total_tests=0
total_correct=0
math_correct=0
math_total=0
science_correct=0
science_total=0
tech_correct=0
tech_total=0
logic_correct=0
logic_total=0

echo "Running tests..."
echo ""

# Test each question
for question in "${!test_cases[@]}"; do
    expected="${test_cases[$question]}"
    total_tests=$((total_tests + 1))
    
    echo "Test $total_tests: $question"
    echo "Expected: $expected"
    
    # Get Melvin's actual response
    melvin_response=$(echo "$question" | timeout 10s ./melvin_unified 2>/dev/null | grep "Melvin:" | head -1 | sed 's/Melvin: //')
    
    if [ -z "$melvin_response" ]; then
        melvin_response="NO_RESPONSE"
    fi
    
    echo "Melvin: $melvin_response"
    
    # Check accuracy (case-insensitive)
    if echo "$melvin_response" | grep -qi "$expected"; then
        echo "Result: ✅ CORRECT"
        total_correct=$((total_correct + 1))
        
        # Categorize by question type
        if [[ "$question" == *"2 + 2"* ]] || [[ "$question" == *"10 * 5"* ]] || [[ "$question" == *"100 / 4"* ]]; then
            math_correct=$((math_correct + 1))
        fi
        if [[ "$question" == *"capital of France"* ]] || [[ "$question" == *"largest planet"* ]]; then
            science_correct=$((science_correct + 1))
        fi
        if [[ "$question" == *"CPU stand for"* ]] || [[ "$question" == *"programming language"* ]] || [[ "$question" == *"file extension"* ]]; then
            tech_correct=$((tech_correct + 1))
        fi
        if [[ "$question" == *"Fluffy is a cat"* ]] || [[ "$question" == *"1, 2, 4, 8"* ]]; then
            logic_correct=$((logic_correct + 1))
        fi
    else
        echo "Result: ❌ INCORRECT"
    fi
    
    # Update category totals
    if [[ "$question" == *"2 + 2"* ]] || [[ "$question" == *"10 * 5"* ]] || [[ "$question" == *"100 / 4"* ]]; then
        math_total=$((math_total + 1))
    fi
    if [[ "$question" == *"capital of France"* ]] || [[ "$question" == *"largest planet"* ]]; then
        science_total=$((science_total + 1))
    fi
    if [[ "$question" == *"CPU stand for"* ]] || [[ "$question" == *"programming language"* ]] || [[ "$question" == *"file extension"* ]]; then
        tech_total=$((tech_total + 1))
    fi
    if [[ "$question" == *"Fluffy is a cat"* ]] || [[ "$question" == *"1, 2, 4, 8"* ]]; then
        logic_total=$((logic_total + 1))
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

echo "Category Breakdown:"
if [ $math_total -gt 0 ]; then
    math_accuracy=$((math_correct * 100 / math_total))
    echo "  Math: $math_correct/$math_total ($math_accuracy%)"
fi
if [ $science_total -gt 0 ]; then
    science_accuracy=$((science_correct * 100 / science_total))
    echo "  Science: $science_correct/$science_total ($science_accuracy%)"
fi
if [ $tech_total -gt 0 ]; then
    tech_accuracy=$((tech_correct * 100 / tech_total))
    echo "  Technology: $tech_correct/$tech_total ($tech_accuracy%)"
fi
if [ $logic_total -gt 0 ]; then
    logic_accuracy=$((logic_correct * 100 / logic_total))
    echo "  Logic: $logic_correct/$logic_total ($logic_accuracy%)"
fi

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
echo "- Complex questions may fail gracefully"
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

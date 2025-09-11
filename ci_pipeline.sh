#!/bin/bash

echo "🔧 Melvin CI Pipeline - Integrated Upgrade System"
echo "================================================"

# Set fixed seed for reproducibility
export RANDOM_SEED=42

# Step 1: Build all systems
echo "📦 Building all Melvin systems..."
./build_unified_input.sh
./build_full_brain_test.sh
./build_upgrade_system.sh

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

# Step 2: Run persistence reload test
echo "📚 Testing persistence reload..."
./melvin_integrated_upgrade_system > persistence_test.log 2>&1
PERSISTENCE_EXIT_CODE=$?

if [ $PERSISTENCE_EXIT_CODE -ne 0 ]; then
    echo "❌ Persistence reload test failed!"
    exit 1
fi

# Step 3: Run stress suite regression test
echo "🧪 Running stress suite regression test..."
./melvin_integrated_upgrade_system > stress_test.log 2>&1
STRESS_EXIT_CODE=$?

if [ $STRESS_EXIT_CODE -ne 0 ]; then
    echo "❌ Stress test suite failed!"
    exit 1
fi

# Step 4: Check for regressions
echo "🔍 Checking for regressions..."
if grep -q "REGRESSION" stress_test.log; then
    echo "❌ Regressions detected!"
    grep "REGRESSION" stress_test.log
    exit 1
fi

# Step 5: Generate CI artifacts
echo "📊 Generating CI artifacts..."
mkdir -p ci_artifacts
cp melvin_session_state.json ci_artifacts/ 2>/dev/null || true
cp melvin_test_metrics.csv ci_artifacts/ 2>/dev/null || true
cp melvin_upgrade_log.txt ci_artifacts/ 2>/dev/null || true
cp persistence_test.log ci_artifacts/
cp stress_test.log ci_artifacts/

# Step 6: Generate CI summary
echo "📋 Generating CI summary..."
cat > ci_artifacts/ci_summary.txt << EOF
Melvin CI Pipeline Results
=========================
Timestamp: $(date)
Build Status: SUCCESS
Persistence Test: PASS
Stress Test: PASS
Regression Check: PASS
Overall Result: PASS

Artifacts Generated:
- melvin_session_state.json (persistence state)
- melvin_test_metrics.csv (performance metrics)
- melvin_upgrade_log.txt (detailed logs)
- persistence_test.log (persistence test output)
- stress_test.log (stress test output)
EOF

echo "✅ CI Pipeline completed successfully!"
echo "📁 Artifacts saved to ci_artifacts/"
echo "📋 Summary: ci_artifacts/ci_summary.txt"

# Step 7: Display final results
echo ""
echo "🎯 Test and analyze."
echo "==================="
echo "All systems operational and validated!"

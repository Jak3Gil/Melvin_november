#!/bin/bash
# Create Teachable Hardware Brain - Complete Setup Script
# Combines all tools to create a self-contained, teachable brain

BRAIN_PATH="${1:-hardware_brain.m}"

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  CREATE TEACHABLE HARDWARE BRAIN                   ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║  Complete setup: operations + patterns + edges     ║"
echo "║  Result: Self-contained brain ready for hardware!  ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Check if brain exists
if [ -f "$BRAIN_PATH" ]; then
    echo "⚠️  Brain file exists: $BRAIN_PATH"
    echo "   Delete it first or use a different name"
    echo ""
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    rm -f "$BRAIN_PATH"
fi

# Step 1: Create empty brain
echo "═══════════════════════════════════════════════════"
echo "STEP 1: Creating Empty Brain"
echo "═══════════════════════════════════════════════════"
echo ""

# Create using melvin_create_v2
echo "Creating $BRAIN_PATH..."
cat > /tmp/create_brain_tmp.c << 'EOF'
#include "src/melvin.h"
#include <stdio.h>

int main(int argc, char **argv) {
    melvin_create_v2(argv[1], 10000, 50000, 131072, 0);
    printf("✅ Created %s\n", argv[1]);
    return 0;
}
EOF

gcc -o /tmp/create_brain_tmp /tmp/create_brain_tmp.c src/melvin.c -I. -O2 -lm -lpthread 2>/dev/null
/tmp/create_brain_tmp "$BRAIN_PATH"
rm -f /tmp/create_brain_tmp /tmp/create_brain_tmp.c

if [ ! -f "$BRAIN_PATH" ]; then
    echo "❌ Failed to create brain"
    exit 1
fi

echo "✅ Empty brain created"
echo ""

# Step 2: Teach hardware operations
echo "═══════════════════════════════════════════════════"
echo "STEP 2: Teaching Hardware Operations"
echo "═══════════════════════════════════════════════════"
echo ""

# Build tools if needed
if [ ! -f "tools/teach_hardware_operations" ]; then
    echo "Building tools..."
    cd tools && make teach_hardware_operations && cd ..
fi

if [ ! -f "tools/teach_hardware_operations" ]; then
    echo "❌ Failed to build tools"
    exit 1
fi

./tools/teach_hardware_operations "$BRAIN_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Failed to teach operations"
    exit 1
fi

echo ""

# Step 3: Create port patterns
echo "═══════════════════════════════════════════════════"
echo "STEP 3: Creating Port Patterns"
echo "═══════════════════════════════════════════════════"
echo ""

if [ ! -f "tools/create_port_patterns" ]; then
    cd tools && make create_port_patterns && cd ..
fi

./tools/create_port_patterns "$BRAIN_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create patterns"
    exit 1
fi

echo ""

# Step 4: Bootstrap edges
echo "═══════════════════════════════════════════════════"
echo "STEP 4: Bootstrapping Reflex Edges"
echo "═══════════════════════════════════════════════════"
echo ""

if [ ! -f "tools/bootstrap_hardware_edges" ]; then
    cd tools && make bootstrap_hardware_edges && cd ..
fi

./tools/bootstrap_hardware_edges "$BRAIN_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Failed to bootstrap edges"
    exit 1
fi

echo ""

# Summary
echo "╔════════════════════════════════════════════════════╗"
echo "║  TEACHABLE HARDWARE BRAIN COMPLETE!                ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

ls -lh "$BRAIN_PATH"
echo ""

echo "Brain contains:"
echo "  ✅ ARM64 hardware control code (in blob)"
echo "  ✅ EXEC nodes (pointing to code)"
echo "  ✅ Port patterns (input/output structure)"
echo "  ✅ Semantic patterns (common labels)"
echo "  ✅ Bootstrap edges (weak reflexes)"
echo ""

echo "Brain is SELF-CONTAINED and TEACHABLE!"
echo ""

echo "Next steps:"
echo "  1. Deploy to Jetson: scp $BRAIN_PATH jetson:/home/melvin/"
echo "  2. Run on Jetson: ./melvin_hardware_runner $BRAIN_PATH"
echo "  3. Brain learns from hardware experience!"
echo ""

echo "✅ Complete!"
echo ""


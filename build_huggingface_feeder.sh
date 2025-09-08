#!/bin/bash

# ============================================================================
# BUILD HUGGING FACE DATA FEEDER FOR MELVIN
# ============================================================================

echo "🤗 Building Melvin Hugging Face Data Feeder..."
echo "=============================================="

# Set compiler flags
CXX_FLAGS="-std=c++17 -O2 -Wall -Wextra"
INCLUDES="-I."
LIBS="-pthread"

# Source files
SOURCES="melvin_huggingface_feeder.cpp melvin_fully_unified_brain.cpp"

# Output executable
OUTPUT="melvin_huggingface_feeder"

echo "🔧 Compiling sources..."
echo "  Sources: $SOURCES"
echo "  Output: $OUTPUT"
echo "  Flags: $CXX_FLAGS"

# Compile the program
g++ $CXX_FLAGS $INCLUDES -o $OUTPUT $SOURCES $LIBS

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "🎯 Executable created: $OUTPUT"
    echo ""
    echo "🚀 Ready to feed Melvin Hugging Face data!"
    echo "   Run: ./$OUTPUT"
    echo ""
    echo "📊 This will:"
    echo "   - Feed Melvin programming knowledge"
    echo "   - Feed Melvin machine learning concepts"
    echo "   - Feed Melvin NLP knowledge"
    echo "   - Feed Melvin code examples"
    echo "   - Feed Melvin reasoning examples"
    echo "   - Let Melvin think about his knowledge"
    echo "   - Create nodes and connections in unified brain"
    echo "   - Show comprehensive statistics"
else
    echo "❌ Compilation failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "🎉 Build complete! Melvin is ready to learn from Hugging Face data!"

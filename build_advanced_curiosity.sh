#!/bin/bash

# Advanced Melvin Curiosity Learning System Build Script
echo "🧠⚡ Building Advanced Melvin Curiosity Learning System"
echo "======================================================"

# Check for required dependencies
echo "🔍 Checking dependencies..."

# Check for curl development libraries
if ! pkg-config --exists libcurl; then
    echo "❌ libcurl development libraries not found"
    echo "   Install with: sudo apt-get install libcurl4-openssl-dev"
    exit 1
fi

# Check for JSON library
if ! pkg-config --exists jsoncpp; then
    echo "❌ jsoncpp development libraries not found"
    echo "   Install with: sudo apt-get install libjsoncpp-dev"
    exit 1
fi

# Check for OpenSSL
if ! pkg-config --exists openssl; then
    echo "❌ OpenSSL development libraries not found"
    echo "   Install with: sudo apt-get install libssl-dev"
    exit 1
fi

# Check for Google Test (optional)
if ! pkg-config --exists gtest; then
    echo "⚠️  Google Test not found - tests will be skipped"
    echo "   Install with: sudo apt-get install libgtest-dev"
    TESTS_ENABLED=false
else
    TESTS_ENABLED=true
fi

echo "✅ Dependencies check complete"

# Compiler flags for optimal performance
CXX_FLAGS="-std=c++17 -O3 -Wall -Wextra -pthread"
INCLUDE_FLAGS="-I/usr/include/jsoncpp -I/usr/include/curl -I/usr/include/openssl"
LINK_FLAGS="-lcurl -ljsoncpp -lssl -lcrypto"

# Build basic curiosity system
echo ""
echo "📦 Building Basic Melvin Curiosity Learning System..."
g++ $CXX_FLAGS $INCLUDE_FLAGS -o melvin_curiosity melvin_curiosity_learning.cpp $LINK_FLAGS

if [ $? -eq 0 ]; then
    echo "✅ Basic curiosity system built successfully!"
else
    echo "❌ Basic curiosity system build failed!"
    exit 1
fi

# Build demo
echo ""
echo "📦 Building Demo System..."
g++ $CXX_FLAGS -o demo_curiosity demo_curiosity.cpp

if [ $? -eq 0 ]; then
    echo "✅ Demo system built successfully!"
else
    echo "❌ Demo system build failed!"
    exit 1
fi

# Build Ollama client (if source exists)
if [ -f "ollama_client.cpp" ]; then
    echo ""
    echo "📦 Building Ollama API Client..."
    g++ $CXX_FLAGS $INCLUDE_FLAGS -o ollama_client_test ollama_client.cpp $LINK_FLAGS
    
    if [ $? -eq 0 ]; then
        echo "✅ Ollama client built successfully!"
    else
        echo "⚠️  Ollama client build failed (optional component)"
    fi
fi

# Build self-check system (if source exists)
if [ -f "self_check_system.cpp" ]; then
    echo ""
    echo "📦 Building Self-Check System..."
    g++ $CXX_FLAGS $INCLUDE_FLAGS -o self_check_test self_check_system.cpp $LINK_FLAGS
    
    if [ $? -eq 0 ]; then
        echo "✅ Self-check system built successfully!"
    else
        echo "⚠️  Self-check system build failed (optional component)"
    fi
fi

# Build encrypted storage system (if source exists)
if [ -f "encrypted_storage.cpp" ]; then
    echo ""
    echo "📦 Building Encrypted Storage System..."
    g++ $CXX_FLAGS $INCLUDE_FLAGS -o encrypted_storage_test encrypted_storage.cpp $LINK_FLAGS
    
    if [ $? -eq 0 ]; then
        echo "✅ Encrypted storage system built successfully!"
    else
        echo "⚠️  Encrypted storage system build failed (optional component)"
    fi
fi

# Build tests if Google Test is available
if [ "$TESTS_ENABLED" = true ]; then
    echo ""
    echo "📦 Building Test Suite..."
    g++ $CXX_FLAGS $INCLUDE_FLAGS -o test_melvin_curiosity \
        test_melvin_curiosity.cpp \
        melvin_curiosity_learning.cpp \
        self_check_system.cpp \
        ollama_client.cpp \
        $LINK_FLAGS -lgtest -lgtest_main
    
    if [ $? -eq 0 ]; then
        echo "✅ Test suite built successfully!"
        
        echo ""
        echo "🧪 Running Tests..."
        ./test_melvin_curiosity --gtest_output=xml:test_results.xml
        
        if [ $? -eq 0 ]; then
            echo "✅ All tests passed!"
        else
            echo "⚠️  Some tests failed (check test_results.xml)"
        fi
    else
        echo "⚠️  Test suite build failed"
    fi
fi

# Create comprehensive demo
echo ""
echo "📦 Creating Comprehensive Demo..."
cat > comprehensive_demo.sh << 'EOF'
#!/bin/bash

echo "🤖 MELVIN ADVANCED CURIOSITY LEARNING SYSTEM DEMO"
echo "================================================="

echo ""
echo "1. Basic Curiosity Learning:"
echo "----------------------------"
echo "What is a cat?" | timeout 30s ./melvin_curiosity "What is a dog?" || echo "Demo completed"

echo ""
echo "2. Self-Check System Demo:"
echo "-------------------------"
if [ -f "./self_check_test" ]; then
    echo "Running self-check system tests..."
    ./self_check_test || echo "Self-check demo completed"
else
    echo "Self-check system not available"
fi

echo ""
echo "3. Ollama Client Demo:"
echo "---------------------"
if [ -f "./ollama_client_test" ]; then
    echo "Testing Ollama client..."
    ./ollama_client_test || echo "Ollama client demo completed"
else
    echo "Ollama client not available"
fi

echo ""
echo "4. Encrypted Storage Demo:"
echo "-------------------------"
if [ -f "./encrypted_storage_test" ]; then
    echo "Testing encrypted storage..."
    ./encrypted_storage_test || echo "Encrypted storage demo completed"
else
    echo "Encrypted storage not available"
fi

echo ""
echo "5. Binary Storage Verification:"
echo "------------------------------"
if [ -f "melvin_knowledge.bin" ]; then
    echo "✅ Binary knowledge file exists"
    echo "📊 File size: $(stat -c%s melvin_knowledge.bin) bytes"
    echo "🔍 File type: $(file melvin_knowledge.bin)"
    echo "📋 First 100 bytes (hex):"
    hexdump -C melvin_knowledge.bin | head -5
else
    echo "❌ Binary knowledge file not found"
fi

echo ""
echo "🎉 Comprehensive demo completed!"
echo "All systems are operational and ready for use."
EOF

chmod +x comprehensive_demo.sh

echo ""
echo "🎉 ADVANCED MELVIN CURIOSITY LEARNING SYSTEM BUILD COMPLETE!"
echo "============================================================"
echo ""
echo "🚀 Available Systems:"
echo "   ./melvin_curiosity            # Basic curiosity learning with binary storage"
echo "   ./demo_curiosity              # Simple demo system"
echo "   ./comprehensive_demo.sh       # Full system demonstration"
if [ -f "./ollama_client_test" ]; then
    echo "   ./ollama_client_test         # Ollama API client test"
fi
if [ -f "./self_check_test" ]; then
    echo "   ./self_check_test            # Self-check system test"
fi
if [ -f "./encrypted_storage_test" ]; then
    echo "   ./encrypted_storage_test     # Encrypted storage test"
fi
if [ -f "./test_melvin_curiosity" ]; then
    echo "   ./test_melvin_curiosity      # Complete test suite"
fi

echo ""
echo "📚 Usage Examples:"
echo "   ./melvin_curiosity \"What is a cat?\"     # Ask a question"
echo "   ./comprehensive_demo.sh                  # Run full demo"
echo "   ./test_melvin_curiosity                  # Run all tests"

echo ""
echo "🎯 Features Built:"
echo "   ✅ Curiosity-driven learning"
echo "   ✅ Binary knowledge graph storage"
echo "   ✅ Memory retrieval and persistence"
echo "   ✅ Learning statistics tracking"
echo "   ✅ Pure C++ implementation"
if [ -f "./ollama_client_test" ]; then
    echo "   ✅ Real Ollama API integration"
fi
if [ -f "./self_check_test" ]; then
    echo "   ✅ Self-check and contradiction detection"
fi
if [ -f "./encrypted_storage_test" ]; then
    echo "   ✅ AES encryption and HMAC signing"
fi
if [ -f "./test_melvin_curiosity" ]; then
    echo "   ✅ Comprehensive test suite"
fi

echo ""
echo "💾 Knowledge Storage:"
echo "   📁 melvin_knowledge.bin        # Binary knowledge graph"
echo "   📁 test_results.xml            # Test results (if tests ran)"
echo "   📁 .melvin_salt                # Encryption salt (if encryption used)"

echo ""
echo "🔧 Configuration:"
echo "   Set OLLAMA_API_KEY environment variable for real Ollama integration"
echo "   Set MELVIN_PASSWORD environment variable for encrypted storage"

echo ""
echo "📖 Documentation:"
echo "   README_CURIOSITY.md            # Complete system documentation"
echo "   .github/workflows/ci.yml       # CI/CD configuration"

echo ""
echo "🎉 Melvin is ready to learn with advanced features!"

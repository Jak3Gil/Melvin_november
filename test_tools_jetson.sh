#!/bin/bash
# test_tools_jetson.sh - Test that installed tools work and can be syscalled

set -e

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "=========================================="
echo "Testing Pattern Generation Tools"
echo "=========================================="
echo "Target: $JETSON_USER@$JETSON_IP"
echo ""

# Check connection
if ! ping -c 1 -W 2 $JETSON_IP > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Jetson at $JETSON_IP"
    exit 1
fi
echo "✓ Connection OK"
echo ""

# Test tools
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
set -e

# Get tools directory
if [ -f ~/.melvin_tools_dir ]; then
    TOOLS_DIR=$(cat ~/.melvin_tools_dir)
else
    TOOLS_DIR="$HOME/melvin/tools"
fi

echo "Tools directory: $TOOLS_DIR"
echo ""

# Test 1: Ollama (LLM)
echo "=========================================="
echo "1. Testing Ollama (LLM)"
echo "=========================================="
if command -v ollama &> /dev/null; then
    echo "✓ Ollama installed"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is running"
        
        # Test LLM generation
        echo "Testing LLM generation..."
        RESPONSE=$(curl -s -m 10 http://localhost:11434/api/generate -d '{"model":"llama3.2:1b","prompt":"Say hello","stream":false}' | jq -r '.response' 2>/dev/null || echo "")
        if [ -n "$RESPONSE" ]; then
            echo "✓ LLM works! Response: ${RESPONSE:0:50}..."
        else
            echo "⚠ LLM test failed (may need model download)"
        fi
    else
        echo "⚠ Ollama not running - start with: ollama serve"
    fi
else
    echo "✗ Ollama not installed"
fi
echo ""

# Test 2: ONNX Runtime (Vision)
echo "=========================================="
echo "2. Testing ONNX Runtime (Vision)"
echo "=========================================="
if python3 -c "import onnxruntime" 2>/dev/null; then
    echo "✓ ONNX Runtime installed"
    python3 -c "import onnxruntime; print('  Version:', onnxruntime.__version__)"
    
    # Check for model
    if [ -f "$TOOLS_DIR/mobilenet.onnx" ]; then
        echo "✓ MobileNet model found"
        
        # Test model loading
        python3 << PYTHON
import onnxruntime as ort
try:
    session = ort.InferenceSession("$TOOLS_DIR/mobilenet.onnx")
    print("✓ Vision model loads successfully")
except Exception as e:
    print(f"⚠ Vision model load failed: {e}")
PYTHON
    else
        echo "⚠ MobileNet model not found at $TOOLS_DIR/mobilenet.onnx"
    fi
else
    echo "✗ ONNX Runtime not installed"
fi
echo ""

# Test 3: Whisper.cpp (STT)
echo "=========================================="
echo "3. Testing Whisper.cpp (STT)"
echo "=========================================="
if [ -f "$TOOLS_DIR/whisper.cpp/main" ]; then
    echo "✓ Whisper.cpp built"
    
    # Check for model
    if [ -f "$TOOLS_DIR/whisper.cpp/models/ggml-base.en.bin" ]; then
        echo "✓ Whisper model found"
        echo "✓ STT ready (test with audio file)"
    else
        echo "⚠ Whisper model not found"
    fi
else
    echo "✗ Whisper.cpp not built"
fi
echo ""

# Test 4: TTS (piper/eSpeak)
echo "=========================================="
echo "4. Testing TTS"
echo "=========================================="
if [ -f "$TOOLS_DIR/piper/piper" ]; then
    echo "✓ piper installed"
    if [ -f "$TOOLS_DIR/piper/en_US-lessac-medium.onnx" ]; then
        echo "✓ piper voice model found"
        echo "✓ TTS ready (piper)"
    else
        echo "⚠ piper voice model not found"
    fi
elif command -v espeak &> /dev/null; then
    echo "✓ eSpeak installed (fallback)"
    echo "✓ TTS ready (eSpeak)"
else
    echo "✗ No TTS available"
fi
echo ""

# Test 5: Syscall test (compile and test melvin_tools.c)
echo "=========================================="
echo "5. Testing Syscall Integration"
echo "=========================================="
cd ~/melvin 2>/dev/null || cd /mnt/melvin_ssd/melvin 2>/dev/null || cd ~

if [ -f src/melvin_tools.c ]; then
    echo "✓ melvin_tools.c found"
    
    # Try to compile
    if gcc -c -I. src/melvin_tools.c -o /tmp/melvin_tools_test.o 2>/dev/null; then
        echo "✓ melvin_tools.c compiles"
        rm -f /tmp/melvin_tools_test.o
    else
        echo "⚠ melvin_tools.c compilation failed (may need dependencies)"
    fi
else
    echo "⚠ melvin_tools.c not found"
fi
echo ""

# Summary
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo "Tools directory: $TOOLS_DIR"
echo ""
echo "Installed:"
[ -f "$TOOLS_DIR/whisper.cpp/main" ] && echo "  ✓ Whisper.cpp"
[ -f "$TOOLS_DIR/piper/piper" ] && echo "  ✓ piper"
command -v ollama &> /dev/null && echo "  ✓ Ollama"
python3 -c "import onnxruntime" 2>/dev/null && echo "  ✓ ONNX Runtime"
command -v espeak &> /dev/null && echo "  ✓ eSpeak"
echo ""
echo "Models:"
[ -f "$TOOLS_DIR/mobilenet.onnx" ] && echo "  ✓ MobileNet"
[ -f "$TOOLS_DIR/whisper.cpp/models/ggml-base.en.bin" ] && echo "  ✓ Whisper base.en"
[ -f "$TOOLS_DIR/piper/en_US-lessac-medium.onnx" ] && echo "  ✓ piper voice"
ollama list 2>/dev/null | grep -q llama3.2 && echo "  ✓ llama3.2:1b"
echo ""
echo "Ready to use tools via syscalls in melvin_tools.c!"
EOF

echo ""
echo "✓ Tool testing complete!"


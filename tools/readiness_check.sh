#!/bin/bash
# Melvin System Readiness Check

echo "=========================================="
echo "Melvin System Readiness Check"
echo "=========================================="
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

check() {
    if [ $? -eq 0 ]; then
        echo "✓ $1"
        ((CHECKS_PASSED++))
        return 0
    else
        echo "✗ $1"
        ((CHECKS_FAILED++))
        return 1
    fi
}

# 1. Core System
echo "1. Core System"
echo "   Checking Melvin build..."
[ -f "melvin_hardware_runner" ] && check "   melvin_hardware_runner exists" || echo "   ⚠ melvin_hardware_runner not built (run 'make melvin_hardware_runner')"

[ -f "src/melvin.c" ] && check "   melvin.c exists"
[ -f "src/melvin.h" ] && check "   melvin.h exists"
echo ""

# 2. Tools
echo "2. AI Tools"

# Check Ollama (can be in PATH or tools dir)
if command -v ollama &> /dev/null; then
    check "   Ollama installed ($(which ollama))"
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        check "   Ollama service running"
    else
        echo "   ⚠ Ollama installed but service not running (start with: ollama serve)"
    fi
else
    OLLAMA_PATH="$HOME/melvin/tools/ollama/ollama"
    [ -f "$OLLAMA_PATH" ] && check "   Ollama installed" || echo "   ⚠ Ollama not found"
fi

# Check Whisper
WHISPER_PATH="$HOME/melvin/tools/whisper.cpp/main"
[ -f "$WHISPER_PATH" ] && check "   Whisper installed" || echo "   ⚠ Whisper not found at $WHISPER_PATH"

# Check Piper
PIPER_PATH="$HOME/melvin/tools/piper/piper"
[ -f "$PIPER_PATH" ] && check "   Piper installed" || echo "   ⚠ Piper not found at $PIPER_PATH"
[ -f "$HOME/melvin/tools/piper/en_US-lessac-medium.onnx" ] && check "   Piper voice installed" || echo "   ⚠ Piper voice not found"

# Check Vision (ONNX)
if python3 -c "import onnxruntime" 2>/dev/null; then
    check "   ONNX Runtime installed"
else
    echo "   ⚠ ONNX Runtime not installed (pip3 install onnxruntime --user)"
fi

if [ -f "$HOME/melvin/tools/mobilenet.onnx" ] || [ -f "/mnt/melvin_ssd/melvin/tools/mobilenet.onnx" ]; then
    check "   MobileNet vision model found"
else
    echo "   ⚠ MobileNet model not found"
fi

echo ""

# 3. Hardware
echo "3. Hardware"
if command -v aplay &> /dev/null; then
    aplay -l 2>&1 | grep -q "USB\|Audio" && check "   USB audio device detected" || echo "   ⚠ No USB audio device found"
else
    echo "   ⚠ aplay not available"
fi

if [ -d "/dev/video0" ] || [ -c "/dev/video0" ]; then
    check "   Camera device available"
else
    echo "   ⚠ No camera device found"
fi
echo ""

# 4. Control System
echo "4. Control System"
[ -f "tools/melvin_service.sh" ] && check "   Service script exists"
[ -f "tools/melvin_control_api.py" ] && check "   Control API exists"
[ -f "tools/melvin.service" ] && check "   Systemd service exists"
python3 -c "import http.server, json" 2>/dev/null && check "   Python dependencies available" || echo "   ⚠ Python dependencies missing"
echo ""

# 5. Dashboard
echo "5. Dashboard"
[ -f "tools/melvin_dashboard.py" ] && check "   Dashboard server exists"
[ -f "tools/dashboard.html" ] && check "   Dashboard UI exists"
[ -f "tools/dashboard.js" ] && check "   Dashboard JS exists"
[ -f "tools/dashboard.css" ] && check "   Dashboard CSS exists"
echo ""

# 6. Patterns
echo "6. Soft Structure Patterns"
[ -f "src/melvin.c" ] && grep -q "MOTOR CONTROL PATTERNS" src/melvin.c && check "   Motor control patterns" || echo "   ⚠ Motor patterns not found"
[ -f "src/melvin.c" ] && grep -q "FILE I/O PATTERNS" src/melvin.c && check "   File I/O patterns" || echo "   ⚠ File I/O patterns not found"
[ -f "src/melvin.c" ] && grep -q "CONVERSATION DATA PATTERNS" src/melvin.c && check "   Conversation patterns" || echo "   ⚠ Conversation patterns not found"
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Passed: $CHECKS_PASSED"
echo "Failed: $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo "✓ SYSTEM READY!"
    echo ""
    echo "To start Melvin:"
    echo "  ./tools/melvin_service.sh start"
    echo ""
    echo "To use dashboard:"
    echo "  python3 tools/melvin_dashboard_app.py"
else
    echo "⚠ Some checks failed - see above"
    echo ""
    echo "Common fixes:"
    echo "  - Build: make melvin_hardware_runner"
    echo "  - Install tools: ./install_tools_jetson.sh"
    echo "  - Check hardware: aplay -l, ls /dev/video*"
fi
echo ""


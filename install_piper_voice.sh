#!/bin/bash
#
# install_piper_voice.sh - Install Piper TTS voice model
#
# Usage: ./install_piper_voice.sh [voice_name]
#
# Available voices:
#   en_US-lessac-medium (recommended - good balance)
#   en_US-lessac-low (faster, lower quality)
#   en_US-lessac-high (slower, higher quality)
#   en_US-amy-medium (different female voice)
#   en_US-ryan-medium (male voice)
#

set -e

VOICE_NAME="${1:-en_US-lessac-medium}"
PIPER_DIR="${HOME}/melvin/tools/piper"
VOICES_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US"

echo "=========================================="
echo "Installing Piper Voice: $VOICE_NAME"
echo "=========================================="
echo ""

# Create piper directory if it doesn't exist
mkdir -p "$PIPER_DIR"

# Check if piper binary exists
if [ ! -f "$PIPER_DIR/piper" ] && [ ! -f "$PIPER_DIR/piper-arm64" ] && ! command -v piper &> /dev/null; then
    echo "⚠ Piper binary not found!"
    echo ""
    echo "Installing Piper binary first..."
    
    # Download piper for ARM64
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz"
    echo "Downloading Piper binary..."
    cd "$PIPER_DIR"
    wget -q "$PIPER_URL" -O piper.tar.gz || curl -L "$PIPER_URL" -o piper.tar.gz
    tar -xzf piper.tar.gz
    chmod +x piper_arm64/piper
    mv piper_arm64/piper piper-arm64
    rm -rf piper_arm64 piper.tar.gz
    echo "✓ Piper binary installed"
    echo ""
fi

# Download voice model
echo "Downloading voice model: $VOICE_NAME"
echo ""

VOICE_MODEL="${VOICE_NAME}.onnx"
VOICE_JSON="${VOICE_NAME}.onnx.json"

cd "$PIPER_DIR"

# Download .onnx model file
echo "  Downloading ${VOICE_MODEL}..."
wget -q "${VOICES_URL}/${VOICE_NAME}/${VOICE_MODEL}" -O "$VOICE_MODEL" || \
curl -L "${VOICES_URL}/${VOICE_NAME}/${VOICE_MODEL}" -o "$VOICE_MODEL" || {
    echo "  ⚠ Download failed, trying alternative URL..."
    # Try direct GitHub release
    wget -q "https://github.com/rhasspy/piper/releases/download/v1.2.0/${VOICE_MODEL}" -O "$VOICE_MODEL" || \
    curl -L "https://github.com/rhasspy/piper/releases/download/v1.2.0/${VOICE_MODEL}" -o "$VOICE_MODEL" || {
        echo "  ❌ Failed to download model"
        exit 1
    }
}

# Download .json config file
echo "  Downloading ${VOICE_JSON}..."
wget -q "${VOICES_URL}/${VOICE_NAME}/${VOICE_JSON}" -O "$VOICE_JSON" || \
curl -L "${VOICES_URL}/${VOICE_NAME}/${VOICE_JSON}" -o "$VOICE_JSON" || {
    echo "  ⚠ JSON download failed (model may still work)"
}

if [ -f "$VOICE_MODEL" ]; then
    echo ""
    echo "✓ Voice model installed!"
    ls -lh "$VOICE_MODEL"
    echo ""
    echo "Voice location: $PIPER_DIR/$VOICE_MODEL"
    echo ""
    echo "Available voices in this script:"
    echo "  - en_US-lessac-medium (recommended)"
    echo "  - en_US-lessac-low"
    echo "  - en_US-lessac-high"
    echo "  - en_US-amy-medium"
    echo "  - en_US-ryan-medium"
    echo ""
    echo "To install a different voice, run:"
    echo "  ./install_piper_voice.sh <voice_name>"
else
    echo ""
    echo "❌ Installation failed"
    exit 1
fi


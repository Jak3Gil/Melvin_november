# Tools Installation Guide

## Overview

This guide installs local pattern generation tools on the Jetson:
- **Ollama** (LLM) - Local language models
- **ONNX Runtime + MobileNet** (Vision) - Image recognition
- **Whisper.cpp** (STT) - Speech-to-text
- **piper/eSpeak** (TTS) - Text-to-speech

All tools run locally on Jetson - no cloud dependencies.

## Quick Install

```bash
./install_tools_jetson.sh
```

This script:
1. Connects to Jetson via USB (169.254.123.100)
2. Installs all tools
3. Downloads models
4. Creates wrapper scripts
5. Tests installation

## Manual Installation

If you prefer to install manually on Jetson:

### 1. Ollama (LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Download model
ollama pull llama3.2:1b  # Lightweight for Jetson
```

### 2. ONNX Runtime (Vision)

```bash
# Install ONNX Runtime
pip3 install onnxruntime

# Download MobileNet model
cd /mnt/melvin_ssd/melvin/tools
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx -O mobilenet.onnx
```

### 3. Whisper.cpp (STT)

```bash
# Install dependencies
sudo apt-get install -y build-essential cmake ffmpeg libsndfile1

# Clone and build
cd /mnt/melvin_ssd/melvin/tools
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make

# Download model
./models/download-ggml-model.sh base.en
```

### 4. piper (TTS)

```bash
# Install via pip
pip3 install piper-tts

# Or download binary
cd /mnt/melvin_ssd/melvin/tools
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar -xzf piper_arm64.tar.gz

# Download voice model
cd piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
```

### 5. eSpeak (TTS Fallback)

```bash
# Lightweight TTS fallback
sudo apt-get install -y espeak espeak-data
```

## Tool Locations

All tools installed in: `/mnt/melvin_ssd/melvin/tools/`

```
tools/
├── mobilenet.onnx          # Vision model
├── whisper.cpp/            # STT
│   ├── main                # Whisper binary
│   └── models/
│       └── ggml-base.en.bin
├── piper/                  # TTS
│   ├── piper               # piper binary
│   └── en_US-lessac-medium.onnx
└── run_*.sh                # Wrapper scripts
```

## Testing Tools

### Test Ollama
```bash
curl http://localhost:11434/api/tags
ollama run llama3.2:1b "Hello"
```

### Test Vision
```bash
cd /mnt/melvin_ssd/melvin/tools
python3 -c "import onnxruntime; print('ONNX works')"
```

### Test STT
```bash
cd /mnt/melvin_ssd/melvin/tools
./whisper.cpp/main -m whisper.cpp/models/ggml-base.en.bin -f audio.wav
```

### Test TTS
```bash
cd /mnt/melvin_ssd/melvin/tools
echo "Hello" | ./piper/piper --model piper/en_US-lessac-medium.onnx --output_file output.wav
# Or with eSpeak:
espeak -s 150 -w output.wav "Hello"
```

## Integration with Melvin

The tools are called via syscalls in `melvin_tools.c`:
- `melvin_tool_llm_generate()` - Calls Ollama
- `melvin_tool_vision_identify()` - Uses ONNX Runtime
- `melvin_tool_audio_stt()` - Uses Whisper.cpp
- `melvin_tool_audio_tts()` - Uses piper/eSpeak

Update `melvin_tools.c` to use the actual tool binaries/models.

## Troubleshooting

### Ollama not starting
```bash
# Start manually
ollama serve

# Check status
systemctl status ollama
```

### ONNX model not found
```bash
# Download model manually
cd /mnt/melvin_ssd/melvin/tools
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx
```

### Whisper build fails
```bash
# Install more dependencies
sudo apt-get install -y libopenblas-dev libgomp1

# Try building again
cd whisper.cpp
make clean
make
```

### piper not working
```bash
# Use eSpeak fallback (always works)
espeak "Hello" -w output.wav
```

## Performance Notes

- **Ollama**: llama3.2:1b is lightweight but slower. Consider llama3.2:3b for better quality.
- **Whisper**: base.en is fast. Use small.en for better accuracy.
- **Vision**: MobileNet is fast but less accurate. Use ResNet for better accuracy.
- **TTS**: piper is better quality, eSpeak is faster.

All tools run locally - no network latency, but uses Jetson compute.


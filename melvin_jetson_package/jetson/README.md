# Melvin - Jetson Deployment Package

## Quick Start

1. **Install dependencies and tools:**
   ```bash
   ./install.sh
   ```

2. **Run Melvin:**
   ```bash
   ./run_melvin.sh
   ```

## Manual Installation

1. **Install dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libasound2-dev python3 python3-pip curl jq binutils
   ```

2. **Install AI tools:**
   ```bash
   bash tools/install_tools_jetson.sh
   ```

3. **Compile:**
   ```bash
   make
   ```

4. **Run:**
   ```bash
   ./melvin_hardware_runner ~/melvin_data/brains/melvin.m default default
   ```

## Files

- `melvin_hardware_runner` - Main binary (with hardware support)
- `melvin_run_continuous` - Binary (without hardware)
- `src/` - Source files
- `tools/` - Tool installation scripts
- `docs/` - Documentation

## Brain Files

Brain files (`.m` files) are stored in `~/melvin_data/brains/` by default.

## Audio Devices

To use specific audio devices:
```bash
./melvin_hardware_runner brain.m hw:0,0 hw:0,0
```

List available devices:
```bash
aplay -l
arecord -l
```

## Tools

- **Ollama** (LLM): `~/melvin/tools/ollama/`
- **Whisper** (STT): `~/melvin/tools/whisper.cpp/`
- **Piper** (TTS): `~/melvin/tools/piper/`
- **ONNX Runtime** (Vision): `~/melvin/tools/onnx/`

## Troubleshooting

- **Audio not working**: Check device permissions, try `default` device
- **Tools not found**: Run `bash tools/install_tools_jetson.sh`
- **Compilation errors**: Install build-essential and libasound2-dev

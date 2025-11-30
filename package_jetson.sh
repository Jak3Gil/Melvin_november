#!/bin/bash
#
# package_jetson.sh - Create deployment package for Jetson
#
# Packages only what's needed to run Melvin on Jetson:
# - Compiled binaries
# - Source files (for recompilation if needed)
# - Tools installation scripts
# - Configuration files
# - Essential documentation
#

set -e

PACKAGE_NAME="melvin_jetson_package"
PACKAGE_DIR="./${PACKAGE_NAME}"
JETSON_DIR="${PACKAGE_DIR}/jetson"

echo "=========================================="
echo "Creating Jetson Deployment Package"
echo "=========================================="
echo ""

# Clean previous package
rm -rf "${PACKAGE_DIR}"
mkdir -p "${JETSON_DIR}"

echo "1. Creating directory structure..."
mkdir -p "${JETSON_DIR}/bin"
mkdir -p "${JETSON_DIR}/src"
mkdir -p "${JETSON_DIR}/tools"
mkdir -p "${JETSON_DIR}/scripts"
mkdir -p "${JETSON_DIR}/docs"
mkdir -p "${JETSON_DIR}/config"

echo "2. Copying source files..."
# Core source files
cp src/melvin.h "${JETSON_DIR}/src/"
cp src/melvin.c "${JETSON_DIR}/src/"
cp src/host_syscalls.c "${JETSON_DIR}/src/"
cp src/melvin_tools.h "${JETSON_DIR}/src/"
cp src/melvin_tools.c "${JETSON_DIR}/src/"

# Hardware drivers
cp src/melvin_hardware_audio.c "${JETSON_DIR}/src/"
cp src/melvin_hardware_video.c "${JETSON_DIR}/src/"
cp src/melvin_hardware_runner.c "${JETSON_DIR}/src/"
# Hardware header (if exists, but runner doesn't need it)
if [ -f src/melvin_hardware.h ]; then
    cp src/melvin_hardware.h "${JETSON_DIR}/src/"
fi

# Runners
cp src/melvin_run_continuous.c "${JETSON_DIR}/src/"

echo "3. Copying tools installation script..."
cp install_tools_jetson.sh "${JETSON_DIR}/tools/" 2>/dev/null || echo "  ⚠ install_tools_jetson.sh not found"

echo "4. Copying essential documentation..."
mkdir -p "${JETSON_DIR}/docs"
cp docs/GRAPH_BASED_SOLUTIONS.md "${JETSON_DIR}/docs/" 2>/dev/null || true
cp docs/TOOLS_INSTALLATION.md "${JETSON_DIR}/docs/" 2>/dev/null || true
cp docs/PRODUCTION_READINESS.md "${JETSON_DIR}/docs/" 2>/dev/null || true

echo "5. Creating Makefile for Jetson..."
cat > "${JETSON_DIR}/Makefile" << 'MAKEFILE_EOF'
# Makefile for Melvin on Jetson

CC = gcc
CFLAGS = -std=c11 -Wall -I. -O2
LIBS = -lm -pthread -lasound

SRC_DIR = src
SRCS = $(SRC_DIR)/melvin.c \
       $(SRC_DIR)/host_syscalls.c \
       $(SRC_DIR)/melvin_tools.c

HARDWARE_SRCS = $(SRCS) \
                $(SRC_DIR)/melvin_hardware_audio.c \
                $(SRC_DIR)/melvin_hardware_video.c \
                $(SRC_DIR)/melvin_hardware_runner.c

CONTINUOUS_SRCS = $(SRCS) \
                  $(SRC_DIR)/melvin_run_continuous.c

all: melvin_hardware_runner melvin_run_continuous

melvin_hardware_runner: $(HARDWARE_SRCS)
	$(CC) $(CFLAGS) -o $@ $(HARDWARE_SRCS) $(LIBS)

melvin_run_continuous: $(CONTINUOUS_SRCS)
	$(CC) $(CFLAGS) -o $@ $(CONTINUOUS_SRCS) $(LIBS)

clean:
	rm -f melvin_hardware_runner melvin_run_continuous

install: all
	mkdir -p /usr/local/bin
	cp melvin_hardware_runner /usr/local/bin/
	cp melvin_run_continuous /usr/local/bin/

.PHONY: all clean install
MAKEFILE_EOF

echo "6. Creating installation script..."
cat > "${JETSON_DIR}/install.sh" << 'INSTALL_EOF'
#!/bin/bash
#
# install.sh - Install Melvin on Jetson
#

set -e

echo "=========================================="
echo "Installing Melvin on Jetson"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠ Warning: This doesn't appear to be a Jetson device"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo "1. Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libasound2-dev \
    python3 \
    python3-pip \
    curl \
    jq \
    binutils

# Install tools (if script exists)
if [ -f tools/install_tools_jetson.sh ]; then
    echo ""
    echo "2. Installing AI tools..."
    bash tools/install_tools_jetson.sh
else
    echo "  ⚠ Tools installation script not found"
fi

# Compile Melvin
echo ""
echo "3. Compiling Melvin..."
make clean
make

# Create data directory
echo ""
echo "4. Creating data directory..."
mkdir -p ~/melvin_data
mkdir -p ~/melvin_data/brains

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Binaries:"
echo "  ./melvin_hardware_runner  - Run with hardware (mic, speaker, camera)"
echo "  ./melvin_run_continuous   - Run without hardware"
echo ""
echo "Data directory: ~/melvin_data/brains/"
echo ""
INSTALL_EOF

chmod +x "${JETSON_DIR}/install.sh"

echo "7. Creating run script..."
cat > "${JETSON_DIR}/run_melvin.sh" << 'RUN_EOF'
#!/bin/bash
#
# run_melvin.sh - Run Melvin on Jetson
#

BRAIN_FILE="${1:-~/melvin_data/brains/melvin.m}"
AUDIO_DEVICE="${2:-default}"

echo "=========================================="
echo "Starting Melvin"
echo "=========================================="
echo ""
echo "Brain file: $BRAIN_FILE"
echo "Audio device: $AUDIO_DEVICE"
echo ""

# Create brain if it doesn't exist
if [ ! -f "$BRAIN_FILE" ]; then
    echo "Creating new brain file..."
    mkdir -p "$(dirname "$BRAIN_FILE")"
    ./melvin_hardware_runner "$BRAIN_FILE" "$AUDIO_DEVICE" "$AUDIO_DEVICE" &
    sleep 2
    pkill -f melvin_hardware_runner
    echo "Brain file created"
    echo ""
fi

# Run Melvin
echo "Starting Melvin..."
./melvin_hardware_runner "$BRAIN_FILE" "$AUDIO_DEVICE" "$AUDIO_DEVICE"
RUN_EOF

chmod +x "${JETSON_DIR}/run_melvin.sh"

echo "8. Creating README..."
cat > "${JETSON_DIR}/README.md" << 'README_EOF'
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
README_EOF

echo "9. Creating package archive..."
cd "${PACKAGE_DIR}"
tar -czf "../${PACKAGE_NAME}.tar.gz" .
cd ..

echo ""
echo "=========================================="
echo "Package Created!"
echo "=========================================="
echo ""
echo "Package directory: ${PACKAGE_DIR}/"
echo "Package archive: ${PACKAGE_NAME}.tar.gz"
echo ""
echo "Contents:"
echo "  bin/          - (empty, binaries compiled on Jetson)"
echo "  src/          - Source files"
echo "  tools/        - Tool installation scripts"
echo "  scripts/      - Utility scripts"
echo "  docs/         - Documentation"
echo "  config/       - Configuration files"
echo "  Makefile      - Build system"
echo "  install.sh    - Installation script"
echo "  run_melvin.sh - Run script"
echo "  README.md     - Documentation"
echo ""
echo "To deploy:"
echo "  1. Copy ${PACKAGE_NAME}.tar.gz to Jetson"
echo "  2. Extract: tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "  3. Run: ./install.sh"
echo ""


# Jetson Orin AGX Setup Guide

This guide helps you set up and run Melvin on NVIDIA Jetson Orin AGX with 64GB RAM.

## Prerequisites

- NVIDIA Jetson Orin AGX
- JetPack 5.x or later
- Ubuntu 20.04/22.04
- Internet connection (for initial setup)

## Quick Start

### 1. Initial Setup

```bash
# Run setup script to install dependencies
./setup_jetson.sh
```

This will install:
- Build tools (gcc, make, etc.)
- Development libraries
- Optional tools (curl, git, Ollama)

### 2. Build Melvin

```bash
# Build main executable and all plugins
./build_jetson.sh

# Or use Makefile
make -f Makefile.jetson
```

### 3. Initialize Brain File

```bash
# Create melvin.m brain file (10M nodes, 100M edges)
./init_melvin_jetson.sh

# Or specify custom size
./init_melvin_jetson.sh custom_brain.m
```

### 4. Run Melvin

```bash
# Run with default brain file
./run_jetson.sh

# Or specify brain file
./run_jetson.sh custom_brain.m
```

## Build Options

### Using Makefile

```bash
make -f Makefile.jetson          # Build everything
make -f Makefile.jetson plugins   # Build only plugins
make -f Makefile.jetson clean     # Clean build artifacts
```

### Manual Build

```bash
# Main executable
gcc -O3 -march=armv8.2-a+fp16+simd -mtune=cortex-a78 \
    -std=c11 -fPIC -o melvin melvin.c -lm -ldl -lpthread

# Plugins
for plugin in plugins/*.c; do
    gcc -O3 -march=armv8.2-a+fp16+simd -mtune=cortex-a78 \
        -shared -fPIC -I. -undefined dynamic_lookup \
        -o "${plugin%.c}.so" "$plugin"
done
```

## System Configuration

### Memory Settings

With 64GB RAM, Melvin can use:
- **Node capacity**: 10 million (default) - can be increased
- **Edge capacity**: 100 million (default) - can be increased
- **Memory limit**: 50GB (set in run_jetson.sh)

### CPU Settings

- **Threads**: 8 (Jetson Orin has 12 cores, using 8 for safety)
- **Optimization**: ARMv8.2-a with FP16 and SIMD

### Performance Tuning

Edit `run_jetson.sh` to adjust:
- `OMP_NUM_THREADS`: Number of CPU threads
- `ulimit -v`: Virtual memory limit
- `ulimit -m`: Physical memory limit

## Plugins

All plugins are automatically built and loaded:
- `mc_api.so` - Internet API connections
- `mc_fs.so` - Filesystem operations
- `mc_git.so` - Git repository cloning
- `mc_io.so` - I/O operations
- `mc_parse.so` - C file parsing
- `mc_build.so` - Compilation
- `mc_scaffold.so` - Pattern injection
- `mc_bootstrap.so` - Bootstrap operations

## Troubleshooting

### Build Errors

```bash
# Install missing dependencies
sudo apt-get update
sudo apt-get install build-essential

# Check compiler
gcc --version
```

### Runtime Errors

```bash
# Check brain file exists
ls -lh melvin.m

# Check permissions
chmod +x melvin
chmod +x plugins/*.so

# Run with debug
./melvin -d melvin.m
```

### Memory Issues

If you get out-of-memory errors:
1. Reduce node/edge capacity in `init_melvin_jetson.sh`
2. Check available memory: `free -h`
3. Close other applications

### Plugin Loading Errors

```bash
# Rebuild plugins
make -f Makefile.jetson clean
make -f Makefile.jetson plugins

# Check plugin dependencies
ldd plugins/*.so
```

## Network Setup

For API plugins to work:
- Ensure network connectivity
- Install curl: `sudo apt-get install curl`
- For LLM: Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`

## File Structure

```
.
├── melvin.c              # Main executable (DO NOT MODIFY)
├── melvin.h              # Header file
├── melvin.m              # Brain file (DO NOT MODIFY)
├── Makefile.jetson       # Build configuration
├── build_jetson.sh       # Build script
├── setup_jetson.sh       # Setup script
├── run_jetson.sh         # Run script
├── init_melvin_jetson.sh # Brain initialization
├── plugins/              # MC function plugins
│   ├── mc_api.c
│   ├── mc_fs.c
│   └── ...
└── scaffolds/           # Pattern injection files
```

## Next Steps

1. **Run Melvin**: `./run_jetson.sh`
2. **Monitor**: Use `htop` to watch CPU/memory usage
3. **Logs**: Check stderr for debug output
4. **Customize**: Edit scripts to adjust settings

## Support

For issues:
- Check logs: `./melvin -d melvin.m 2>&1 | tee melvin.log`
- Verify system: `uname -a`, `free -h`, `lscpu`
- Test plugins: `ldd plugins/*.so`


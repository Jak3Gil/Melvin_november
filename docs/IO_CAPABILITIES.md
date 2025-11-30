# Melvin I/O Capabilities

## INPUTS

### 1. Byte Ingestion (Primary Input)
```c
ingest_byte(rt, channel_id, byte_value, energy)
```

**What it accepts:**
- **Any byte stream**: Text, images, audio, sensor data, network packets, etc.
- **Channels (0-255)**: Separate data streams
- **Energy**: Input strength (typically 1.0)

**How it works:**
- Each byte creates/finds a DATA node
- Creates SEQ edges (byte-to-byte sequence)
- Creates CHAN edges (channel-to-data)
- Patterns form automatically from repeated sequences

**Examples:**
```c
// Text input
ingest_byte(rt, 0, 'H', 1.0f);
ingest_byte(rt, 0, 'e', 1.0f);
ingest_byte(rt, 0, 'l', 1.0f);

// Sensor reading
ingest_byte(rt, 1, sensor_value, 1.0f);

// Image pixel
ingest_byte(rt, 2, pixel_value, 1.0f);
```

### 2. Reward Signals (Feedback)
```c
inject_reward(rt, node_id, reward_value)
```

**What it accepts:**
- **Node ID**: Which node/circuit to reward
- **Reward value**: Positive (good) or negative (bad)

**How it works:**
- Modifies edge weights via free-energy rule
- Strengthens circuits that lead to reward
- Weakens circuits that lead to punishment

### 3. EXEC Input (via Machine Code)
EXEC nodes can read from:
- Files (via `read()` syscall)
- Network sockets (via `recv()` syscall)
- Hardware devices (via `read()` on device files)
- Sensors (via device-specific syscalls)
- GPU memory (via mmap/ioctl)

**Example EXEC code (ARM64) that reads:**
```assembly
mov x0, #0        // stdin
adr x1, buffer    // buffer address
mov x2, #256      // length
mov x8, #63       // sys_read
svc #0            // syscall
// x0 now contains bytes read
mov x0, x0        // return bytes read as energy
ret
```

## OUTPUTS

### 1. EXEC Output (Primary Output Mechanism)
EXEC nodes run machine code that can:

**Write to stdout/stderr:**
```assembly
// ARM64 example
mov x0, #1        // stdout
adr x1, message   // message
mov x2, #13       // length
mov x8, #64       // sys_write
svc #0
mov x0, #0x42     // return value → energy
ret
```

**Write to files:**
- `open()` syscall to get file descriptor
- `write()` syscall to write data
- `close()` syscall when done

**Network output:**
- `socket()` to create socket
- `connect()` to connect
- `send()` to send data

**GPU output:**
- `ioctl()` to GPU device
- `mmap()` GPU memory
- Launch CUDA/OpenCL kernels

**Hardware control:**
- `open()` GPIO/I2C/SPI device files
- `ioctl()` or `write()` to control hardware
- Direct register access (if running in kernel mode)

### 2. Node Activation Output
```c
float activation = melvin_get_output(rt, node_id);
uint8_t byte = melvin_get_output_byte(rt, node_id);
```

**What it provides:**
- Read any node's activation
- Convert to byte value (0-255)
- Poll periodically for continuous output

### 3. Output Node Callback
```c
melvin_register_output_callback(rt, callback, context);
```

**What it provides:**
- Callback triggered when `NODE_ID_OUTPUT` is activated
- Can write to stdout, files, network, etc.
- Integrated with graph physics

## HARDWARE ACCESS

### CPU Access
- **Direct**: EXEC nodes run raw machine code on CPU
- **Registers**: Full access to CPU registers
- **Instructions**: Any CPU instruction (add, mul, branch, etc.)
- **Syscalls**: Full syscall interface

### GPU Access
- **CUDA**: Launch kernels via CUDA runtime (from EXEC code)
- **OpenCL**: Launch kernels via OpenCL (from EXEC code)
- **Vulkan**: GPU compute via Vulkan API
- **Direct**: GPU device files via ioctl

**Example: EXEC node that launches GPU kernel**
```c
// EXEC code would:
// 1. Load CUDA library
// 2. Allocate GPU memory
// 3. Launch kernel
// 4. Return result as energy
```

### Hardware Control
- **GPIO**: `/dev/gpiochip*` device files
- **I2C**: `/dev/i2c-*` device files
- **SPI**: `/dev/spidev*` device files
- **Serial**: `/dev/tty*` device files
- **USB**: USB device files
- **PCI**: `/dev/pci*` or direct memory access

**Example: EXEC node that controls GPIO**
```assembly
// ARM64 example
mov x0, #0        // open GPIO device
adr x1, gpio_path
mov x2, #2        // O_RDWR
mov x8, #56       // sys_open
svc #0
// x0 = file descriptor
// Now can ioctl() to control GPIO
```

## CURRENT LIMITATIONS

1. **No direct C-side output API**: All output must go through EXEC nodes (by design - physics-only)
2. **EXEC nodes must be created by graph**: The graph must form EXEC nodes through pattern formation
3. **No built-in network stack**: EXEC code must implement networking via syscalls
4. **No built-in file I/O helpers**: EXEC code must use raw syscalls

## FUTURE ENHANCEMENTS

1. **Output node callbacks**: Already implemented, can be extended
2. **Streaming output**: Continuous output from high-activation nodes
3. **Multi-modal I/O**: Structured I/O for images, audio, etc.
4. **Network protocol**: Built-in network interface
5. **GPU integration**: Direct GPU kernel nodes

## SUMMARY

**Inputs:**
- ✅ Bytes (any stream)
- ✅ Reward signals
- ✅ EXEC can read (files, network, hardware)

**Outputs:**
- ✅ EXEC can write (stdout, files, network, hardware)
- ✅ Node activations (readable)
- ✅ Output node callbacks

**Hardware:**
- ✅ CPU (direct machine code execution)
- ✅ GPU (via syscalls from EXEC code)
- ✅ Hardware devices (via device files from EXEC code)

**Design Philosophy:**
- All I/O goes through the graph physics
- EXEC nodes are the bridge to the external world
- No magic shortcuts - everything is nodes, edges, energy, events


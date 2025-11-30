/*
 * MELVIN I/O INTERFACE
 * 
 * This defines how Melvin interacts with the external world:
 * 
 * INPUTS:
 * - Bytes (via ingest_byte) - any byte stream
 * - Channels - separate data streams (0-255)
 * - Reward signals - feedback from environment
 * 
 * OUTPUTS:
 * - EXEC nodes can run machine code that can:
 *   - Make syscalls (read/write files, network, etc.)
 *   - Access CPU/GPU directly
 *   - Write to stdout/stderr
 *   - Control hardware
 * 
 * CURRENT CAPABILITIES:
 * - Input: ingest_byte(rt, channel_id, byte_value, energy)
 * - Output: EXEC nodes return values → converted to energy
 * - No explicit output API yet (EXEC must use syscalls)
 */

#ifndef MELVIN_IO_H
#define MELVIN_IO_H

#include <stdint.h>
#include <stdio.h>

// Forward declarations
typedef struct MelvinRuntime MelvinRuntime;
typedef struct MelvinFile MelvinFile;

// ========================================================================
// INPUT INTERFACE
// ========================================================================

/*
 * Ingest a byte into Melvin
 * 
 * This is the PRIMARY input mechanism. All external data becomes bytes:
 * - Text → bytes
 * - Images → bytes
 * - Sensor readings → bytes
 * - Network packets → bytes
 * - Audio → bytes
 * 
 * Parameters:
 *   rt: Runtime pointer
 *   channel_id: Channel ID (0-255) - separate data streams
 *   byte_value: The byte to ingest (0-255)
 *   energy: Input energy (typically 1.0)
 * 
 * Returns: Node ID of the DATA node created/found
 */
uint64_t ingest_byte(MelvinRuntime *rt, uint64_t channel_id, uint8_t byte_value, float energy);

/*
 * Ingest a buffer of bytes
 * Convenience function for bulk ingestion
 */
void ingest_buffer(MelvinRuntime *rt, uint64_t channel_id, const uint8_t *buffer, size_t len, float energy);

/*
 * Ingest from file
 * Reads file and ingests all bytes
 */
int ingest_file(MelvinRuntime *rt, uint64_t channel_id, const char *filename, float energy);

/*
 * Ingest from stdin
 * Reads available bytes from stdin and ingests them
 */
size_t ingest_stdin(MelvinRuntime *rt, uint64_t channel_id, float energy);

// ========================================================================
// OUTPUT INTERFACE
// ========================================================================

/*
 * Output node - well-known node ID for output
 * When this node's activation crosses a threshold, it triggers output
 */
#define NODE_ID_OUTPUT 200ULL

/*
 * Register output callback
 * When NODE_ID_OUTPUT is activated, this callback is called
 * 
 * The callback receives:
 *   rt: Runtime pointer
 *   node_id: The output node ID (usually NODE_ID_OUTPUT)
 *   activation: Current activation value
 *   context: User-provided context pointer
 */
typedef void (*MelvinOutputCallback)(MelvinRuntime *rt, uint64_t node_id, float activation, void *context);

void melvin_register_output_callback(MelvinRuntime *rt, MelvinOutputCallback callback, void *context);

/*
 * Get output value from a node
 * Reads a node's activation and converts it to output
 * 
 * For continuous output, poll this periodically
 */
float melvin_get_output(MelvinRuntime *rt, uint64_t node_id);

/*
 * Get output as byte
 * Maps activation to byte value (0-255)
 */
uint8_t melvin_get_output_byte(MelvinRuntime *rt, uint64_t node_id);

// ========================================================================
// EXEC OUTPUT (via machine code)
// ========================================================================

/*
 * EXEC nodes can output via:
 * 
 * 1. Return values → converted to energy → injected into graph
 * 2. Direct syscalls from machine code:
 *    - write() to stdout/stderr
 *    - write() to files
 *    - send() to network sockets
 *    - GPU kernel launches (CUDA/OpenCL)
 *    - Hardware control (GPIO, I2C, SPI, etc.)
 * 
 * Example EXEC code (ARM64) that outputs:
 * 
 *   mov x0, #1        // stdout
 *   adr x1, message   // message address
 *   mov x2, #13       // length
 *   mov x8, #64       // sys_write
 *   svc #0            // syscall
 *   mov x0, #0x42     // return value
 *   ret
 * 
 * The return value (0x42) becomes energy injected into the graph.
 */

// ========================================================================
// REWARD INTERFACE
// ========================================================================

/*
 * Inject reward signal
 * 
 * This is how the environment provides feedback:
 * - Positive reward: Good behavior
 * - Negative reward: Bad behavior
 * 
 * Reward affects learning via the free-energy rule
 */
void inject_reward(MelvinRuntime *rt, uint64_t node_id, float reward_value);

/*
 * Inject reward to multiple nodes
 * Useful for rewarding entire circuits
 */
void inject_reward_to_nodes(MelvinRuntime *rt, uint64_t *node_ids, size_t count, float reward_value);

// ========================================================================
// CHANNEL MANAGEMENT
// ========================================================================

/*
 * Channels are separate data streams:
 * - Channel 0: Primary input stream
 * - Channel 1: Secondary input stream
 * - Channel 2-255: Additional streams
 * 
 * Each channel maintains its own:
 * - Channel node (CH_C)
 * - Sequence edges (SEQ)
 * - Pattern formation
 */

// Get channel node ID
uint64_t melvin_get_channel_node_id(uint64_t channel_id);

// ========================================================================
// I/O SUMMARY
// ========================================================================

/*
 * INPUTS:
 * 1. Bytes via ingest_byte() - any byte stream
 * 2. Reward via inject_reward() - feedback signals
 * 3. EXEC return values - machine code can read sensors, files, network
 * 
 * OUTPUTS:
 * 1. EXEC machine code - can make syscalls (write, send, etc.)
 * 2. Node activations - can be read via melvin_get_output()
 * 3. Output node (NODE_ID_OUTPUT) - callback-based output
 * 
 * HARDWARE ACCESS:
 * - EXEC nodes run raw machine code
 * - Machine code can make any syscall
 * - Machine code can access CPU registers directly
 * - Machine code can launch GPU kernels (via syscalls)
 * - Machine code can control hardware (GPIO, I2C, SPI via syscalls)
 * 
 * LIMITATIONS:
 * - All output must go through EXEC nodes (machine code)
 * - No direct C-side output API (by design - physics-only)
 * - EXEC nodes must be created and activated by the graph
 */

#endif // MELVIN_IO_H


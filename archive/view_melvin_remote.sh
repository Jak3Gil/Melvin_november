#!/bin/bash
# View Melvin's graph visualization remotely (headless mode)

set -e

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "=========================================="
echo "Melvin Remote Graph Viewer"
echo "=========================================="
echo ""
echo "Connecting to Jetson and displaying graph stats..."
echo "Press Ctrl+C to stop"
echo ""

# Function to display graph stats
display_graph() {
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'REMOTEEOF'
        # Read graph header
        python3 << 'PYEOF' 2>/dev/null || cat << 'CATEOF'
import struct
import sys

try:
    with open('/home/melvin/melvin_system/melvin.m', 'rb') as f:
        # Read header (first 40 bytes: 5 uint64_t)
        data = f.read(40)
        if len(data) == 40:
            num_nodes, num_edges, tick, node_cap, edge_cap = struct.unpack('QQQQQ', data)
            
            print(f"╔══════════════════════════════════════════════════════════╗")
            print(f"║           MELVIN GRAPH VISUALIZATION                     ║")
            print(f"╚══════════════════════════════════════════════════════════╝")
            print()
            print(f"Tick: {tick:,}")
            print(f"Nodes: {num_nodes:,} / {node_cap:,} ({num_nodes/node_cap*100 if node_cap > 0 else 0:.1f}%)")
            print(f"Edges: {num_edges:,} / {edge_cap:,} ({num_edges/edge_cap*100 if edge_cap > 0 else 0:.1f}%)")
            print()
            
            # Try to read some node data
            if num_nodes > 0 and num_nodes < 1000000:
                f.seek(40)  # Skip header
                # Read first few nodes (each node is ~64 bytes)
                node_data = f.read(min(64 * 10, 64 * num_nodes))
                if node_data:
                    active_count = 0
                    for i in range(0, min(len(node_data), 64 * 10), 64):
                        if i + 12 < len(node_data):
                            # Read activation (float at offset 0)
                            activation = struct.unpack('f', node_data[i:i+4])[0]
                            if activation > 0.1:
                                active_count += 1
                    print(f"Active nodes (sample): {active_count}")
        else:
            print("Graph file too small or invalid")
except Exception as e:
    print(f"Error: {e}")
PYEOF

# Fallback to C if Python not available
CATEOF
        # C fallback
        cat > /tmp/read_melvin.c << 'CEOF'
#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t tick;
    uint64_t node_cap;
    uint64_t edge_cap;
} BrainHeader;

int main() {
    int fd = open("/home/melvin/melvin_system/melvin.m", O_RDONLY);
    if (fd < 0) {
        printf("Cannot open melvin.m\n");
        return 1;
    }
    
    struct stat st;
    fstat(fd, &st);
    
    BrainHeader *header = (BrainHeader *)mmap(NULL, sizeof(BrainHeader), PROT_READ, MAP_SHARED, fd, 0);
    if (header == MAP_FAILED) {
        printf("Cannot mmap\n");
        close(fd);
        return 1;
    }
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║           MELVIN GRAPH VISUALIZATION                     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("Tick: %llu\n", (unsigned long long)header->tick);
    printf("Nodes: %llu / %llu (%.1f%%)\n",
           (unsigned long long)header->num_nodes,
           (unsigned long long)header->node_cap,
           header->node_cap > 0 ? (double)header->num_nodes / header->node_cap * 100.0 : 0.0);
    printf("Edges: %llu / %llu (%.1f%%)\n",
           (unsigned long long)header->num_edges,
           (unsigned long long)header->edge_cap,
           header->edge_cap > 0 ? (double)header->num_edges / header->edge_cap * 100.0 : 0.0);
    printf("\n");
    
    munmap(header, sizeof(BrainHeader));
    close(fd);
    return 0;
}
CEOF
        gcc -o /tmp/read_melvin /tmp/read_melvin.c 2>/dev/null && /tmp/read_melvin || echo "Cannot read graph"
REMOTEEOF
}

# Clear screen and loop
clear
while true; do
    display_graph
    echo ""
    echo "Refreshing in 2 seconds... (Ctrl+C to stop)"
    sleep 2
    clear
done


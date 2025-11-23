#!/bin/bash
# Initialize melvin.m brain file for Jetson Orin AGX
# Creates a memory-mapped brain file with appropriate size for 64GB RAM

set -e

echo "=========================================="
echo "Initializing Melvin Brain (Jetson)"
echo "=========================================="

BRAIN_FILE="${1:-melvin.m}"

if [ -f "$BRAIN_FILE" ]; then
    echo "Warning: $BRAIN_FILE already exists"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REOT =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
    rm -f "$BRAIN_FILE"
fi

# Brain size parameters - calculate based on available disk space
# The limit should be disk space, not an arbitrary number

# Get available disk space in bytes
AVAILABLE_SPACE=$(df -k . | tail -1 | awk '{print $4}')
AVAILABLE_SPACE=$((AVAILABLE_SPACE * 1024))  # Convert KB to bytes

# Reserve 10% of disk for other uses (safety margin)
USABLE_SPACE=$((AVAILABLE_SPACE * 90 / 100))

echo "Available disk space: $(numfmt --to=iec-i --suffix=B $AVAILABLE_SPACE 2>/dev/null || echo "$((AVAILABLE_SPACE / 1024 / 1024 / 1024)) GB")"
echo "Usable space (90%): $(numfmt --to=iec-i --suffix=B $USABLE_SPACE 2>/dev/null || echo "$((USABLE_SPACE / 1024 / 1024 / 1024)) GB")"
echo ""

# Node and Edge sizes (bytes per struct)
NODE_SIZE=40    # bytes per Node
EDGE_SIZE=32    # bytes per Edge
HEADER_SIZE=256 # bytes for header

# Calculate maximum possible nodes and edges
# We'll allocate 70% to edges, 30% to nodes (edges are more numerous)
NODE_SPACE=$((USABLE_SPACE * 30 / 100))
EDGE_SPACE=$((USABLE_SPACE * 70 / 100))

NODE_CAP=$((NODE_SPACE / NODE_SIZE))
EDGE_CAP=$((EDGE_SPACE / EDGE_SIZE))

# Minimum sizes (in case disk is very small)
MIN_NODE_CAP=1000000   # 1 million nodes minimum
MIN_EDGE_CAP=10000000  # 10 million edges minimum

if [ $NODE_CAP -lt $MIN_NODE_CAP ]; then
    echo "Warning: Calculated node capacity ($NODE_CAP) is below minimum ($MIN_NODE_CAP)"
    echo "Using minimum capacity instead"
    NODE_CAP=$MIN_NODE_CAP
fi

if [ $EDGE_CAP -lt $MIN_EDGE_CAP ]; then
    echo "Warning: Calculated edge capacity ($EDGE_CAP) is below minimum ($MIN_EDGE_CAP)"
    echo "Using minimum capacity instead"
    EDGE_CAP=$MIN_EDGE_CAP
fi

TOTAL_SIZE=$((HEADER_SIZE + NODE_CAP * NODE_SIZE + EDGE_CAP * EDGE_SIZE))
TOTAL_SIZE_MB=$((TOTAL_SIZE / 1024 / 1024))
TOTAL_SIZE_GB=$((TOTAL_SIZE / 1024 / 1024 / 1024))

echo "Creating brain file: $BRAIN_FILE"
echo "  Node capacity: $NODE_CAP ($(numfmt --to=iec-i --suffix=B $((NODE_CAP * NODE_SIZE)) 2>/dev/null || echo "$((NODE_CAP * NODE_SIZE / 1024 / 1024)) MB"))"
echo "  Edge capacity: $EDGE_CAP ($(numfmt --to=iec-i --suffix=B $((EDGE_CAP * EDGE_SIZE)) 2>/dev/null || echo "$((EDGE_CAP * EDGE_SIZE / 1024 / 1024)) MB"))"
echo "  Total size: ~${TOTAL_SIZE_MB} MB ($(numfmt --to=iec-i --suffix=B $TOTAL_SIZE 2>/dev/null || echo "${TOTAL_SIZE_GB} GB"))"
echo "  Limited by disk space: $(numfmt --to=iec-i --suffix=B $USABLE_SPACE 2>/dev/null || echo "$((USABLE_SPACE / 1024 / 1024 / 1024)) GB")"
echo ""

# Create initialization program
cat > /tmp/init_melvin.c << 'EOF'
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#define MELVIN_MAGIC 0x4D4C564E
#define MELVIN_VERSION 2

typedef struct {
    uint64_t num_nodes;
    uint64_t num_edges;
    uint64_t tick;
    uint64_t node_cap;
    uint64_t edge_cap;
    uint8_t padding[224];
} BrainHeader;

typedef struct {
    float a;
    float bias;
    float decay;
    uint32_t kind;
    uint32_t flags;
    float reliability;
    uint32_t success_count;
    uint32_t failure_count;
    uint32_t mc_id;
    uint16_t mc_flags;
    uint16_t mc_role;
    float value;
} Node;

typedef struct {
    uint64_t src;
    uint64_t dst;
    float w;
    uint32_t flags;
    float elig;
    uint32_t usage_count;
} Edge;

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <brain_file> <node_cap> <edge_cap>\n", argv[0]);
        return 1;
    }
    
    const char *brain_file = argv[1];
    uint64_t node_cap = strtoull(argv[2], NULL, 10);
    uint64_t edge_cap = strtoull(argv[3], NULL, 10);
    
    size_t header_size = sizeof(BrainHeader);
    size_t node_size = sizeof(Node);
    size_t edge_size = sizeof(Edge);
    size_t total_size = header_size + node_cap * node_size + edge_cap * edge_size;
    
    printf("Creating brain file: %s\n", brain_file);
    printf("  Node capacity: %llu\n", (unsigned long long)node_cap);
    printf("  Edge capacity: %llu\n", (unsigned long long)edge_cap);
    printf("  Total size: %zu bytes (%.2f MB)\n", total_size, total_size / 1024.0 / 1024.0);
    
    // Create file
    int fd = open(brain_file, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    // Extend file to full size
    if (ftruncate(fd, total_size) < 0) {
        perror("ftruncate");
        close(fd);
        return 1;
    }
    
    // Map file
    void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    
    // Initialize header
    BrainHeader *header = (BrainHeader *)map;
    memset(header, 0, sizeof(BrainHeader));
    header->num_nodes = 0;
    header->num_edges = 0;
    header->tick = 0;
    header->node_cap = node_cap;
    header->edge_cap = edge_cap;
    
    // Initialize nodes and edges to zero
    Node *nodes = (Node *)((uint8_t *)map + header_size);
    Edge *edges = (Edge *)((uint8_t *)nodes + node_cap * node_size);
    
    memset(nodes, 0, node_cap * node_size);
    memset(edges, 0, edge_cap * edge_size);
    
    // Unmap
    munmap(map, total_size);
    close(fd);
    
    printf("Brain file created successfully!\n");
    return 0;
}
EOF

# Compile and run initializer
echo "Compiling initializer..."
gcc -o /tmp/init_melvin /tmp/init_melvin.c -std=c11

echo "Initializing brain file..."
/tmp/init_melvin "$BRAIN_FILE" "$NODE_CAP" "$EDGE_CAP"

# Cleanup
rm -f /tmp/init_melvin /tmp/init_melvin.c

echo ""
echo "=========================================="
echo "Brain initialization complete!"
echo "=========================================="
echo "Brain file: $BRAIN_FILE"
echo "Ready to run: ./run_jetson.sh $BRAIN_FILE"
echo ""


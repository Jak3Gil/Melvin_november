#!/bin/bash
# Expand Melvin graph to use full 4TB SSD capacity
# This will backup current brain and create a new one with full capacity

set -e

echo "=========================================="
echo "Expanding Melvin to Full 4TB Capacity"
echo "=========================================="
echo ""

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "Step 1: Stopping Melvin service..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl stop melvin.service" || echo "Service already stopped"
echo ""

echo "Step 2: Backing up current brain..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    cd /home/melvin/melvin_system
    BACKUP_FILE="melvin.m.backup.$(date +%Y%m%d_%H%M%S)"
    if [ -f melvin.m ]; then
        cp melvin.m "$BACKUP_FILE"
        echo "✓ Backed up to: $BACKUP_FILE"
        ls -lh "$BACKUP_FILE"
    else
        echo "No existing brain file to backup"
    fi
EOF
echo ""

echo "Step 3: Calculating 4TB capacity..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    cd /home/melvin/melvin_system
    
    # Use the SSD for the brain file
    BRAIN_DIR="/mnt/melvin_ssd"
    BRAIN_FILE="$BRAIN_DIR/melvin.m"
    
    # Get available space on SSD
    AVAILABLE_SPACE=$(df -k "$BRAIN_DIR" | tail -1 | awk '{print $4}')
    AVAILABLE_SPACE=$((AVAILABLE_SPACE * 1024))  # Convert KB to bytes
    
    # Reserve 5% for safety (less than before since we want maximum capacity)
    USABLE_SPACE=$((AVAILABLE_SPACE * 95 / 100))
    
    echo "SSD available: $(numfmt --to=iec-i --suffix=B $AVAILABLE_SPACE 2>/dev/null || echo "$((AVAILABLE_SPACE / 1024 / 1024 / 1024)) GB")"
    echo "Usable (95%): $(numfmt --to=iec-i --suffix=B $USABLE_SPACE 2>/dev/null || echo "$((USABLE_SPACE / 1024 / 1024 / 1024)) GB")"
    echo ""
    
    # Node and Edge sizes
    NODE_SIZE=40
    EDGE_SIZE=32
    HEADER_SIZE=256
    
    # Allocate 30% to nodes, 70% to edges
    NODE_SPACE=$((USABLE_SPACE * 30 / 100))
    EDGE_SPACE=$((USABLE_SPACE * 70 / 100))
    
    NODE_CAP=$((NODE_SPACE / NODE_SIZE))
    EDGE_CAP=$((EDGE_SPACE / EDGE_SIZE))
    
    TOTAL_SIZE=$((HEADER_SIZE + NODE_CAP * NODE_SIZE + EDGE_CAP * EDGE_SIZE))
    
    echo "Calculated capacity:"
    echo "  Nodes: $NODE_CAP ($(numfmt --to=iec-i --suffix=B $((NODE_CAP * NODE_SIZE)) 2>/dev/null || echo "$((NODE_CAP * NODE_SIZE / 1024 / 1024 / 1024)) GB"))"
    echo "  Edges: $EDGE_CAP ($(numfmt --to=iec-i --suffix=B $((EDGE_CAP * EDGE_SIZE)) 2>/dev/null || echo "$((EDGE_CAP * EDGE_SIZE / 1024 / 1024 / 1024)) GB"))"
    echo "  Total: $(numfmt --to=iec-i --suffix=B $TOTAL_SIZE 2>/dev/null || echo "$((TOTAL_SIZE / 1024 / 1024 / 1024)) GB")"
    echo ""
    
    # Create initialization program
    cat > /tmp/init_melvin_4tb.c << 'INITEOF'
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
    printf("  Total size: %zu bytes (%.2f GB)\n", total_size, total_size / 1024.0 / 1024.0 / 1024.0);
    
    int fd = open(brain_file, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    if (ftruncate(fd, total_size) < 0) {
        perror("ftruncate");
        close(fd);
        return 1;
    }
    
    void *map = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    
    BrainHeader *header = (BrainHeader *)map;
    memset(header, 0, sizeof(BrainHeader));
    header->num_nodes = 0;
    header->num_edges = 0;
    header->tick = 0;
    header->node_cap = node_cap;
    header->edge_cap = edge_cap;
    
    Node *nodes = (Node *)((uint8_t *)map + header_size);
    Edge *edges = (Edge *)((uint8_t *)nodes + node_cap * node_size);
    
    memset(nodes, 0, node_cap * node_size);
    memset(edges, 0, edge_cap * edge_size);
    
    munmap(map, total_size);
    close(fd);
    
    printf("Brain file created successfully!\n");
    return 0;
}
INITEOF
    
    gcc -o /tmp/init_melvin_4tb /tmp/init_melvin_4tb.c -std=c11
    
    echo "Creating new brain file with full 4TB capacity..."
    /tmp/init_melvin_4tb "$BRAIN_FILE" "$NODE_CAP" "$EDGE_CAP"
    
    # Create symlink from old location
    if [ -f melvin.m ]; then
        mv melvin.m melvin.m.old
    fi
    ln -sf "$BRAIN_FILE" melvin.m
    
    echo "✓ Brain file created at: $BRAIN_FILE"
    echo "✓ Symlink created: melvin.m -> $BRAIN_FILE"
    ls -lh "$BRAIN_FILE" melvin.m
EOF
echo ""

echo "Step 4: Deploying updated melvin.c with optimization..."
cd /Users/jakegilbert/melvin_november/Melvin_november
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no melvin.c "$JETSON_HOST:/home/melvin/melvin_system/"
echo ""

echo "Step 5: Recompiling Melvin..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    cd /home/melvin/melvin_system
    echo "Compiling with optimization support..."
    gcc -O2 -o melvin melvin.c -ldl -lm -lpthread -I. 2>&1 | grep -E 'error|warning' | head -5 || echo "✓ Compilation successful"
    ls -lh melvin
EOF
echo ""

echo "Step 6: Starting Melvin service..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl start melvin.service"
sleep 2
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl status melvin.service --no-pager -l | head -10"
echo ""

echo "=========================================="
echo "Expansion Complete!"
echo "=========================================="
echo ""
echo "Melvin now has:"
echo "  - Full 4TB SSD capacity available"
echo "  - Automatic optimization when capacity is reached"
echo "  - Continuous growth and improvement"
echo ""
echo "The system will now optimize (prune low-value nodes/edges)"
echo "instead of stopping when capacity is reached."
echo ""


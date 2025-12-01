#!/bin/bash
# Create a new Melvin brain file

if [ $# -lt 1 ]; then
    echo "Usage: $0 <brain_file.m> [nodes] [edges] [blob_size_kb]"
    echo "  Default: 1000 nodes, 5000 edges, 64KB blob"
    echo ""
    echo "Example:"
    echo "  $0 data/brain.m"
    echo "  $0 data/brain.m 5000 25000 256"
    exit 1
fi

BRAIN_FILE="$1"
NODES=${2:-1000}
EDGES=${3:-5000}
BLOB_KB=${4:-64}
BLOB_SIZE=$((BLOB_KB * 1024))

echo "Creating new brain file: $BRAIN_FILE"
echo "  Nodes: $NODES"
echo "  Edges: $EDGES"
echo "  Blob size: ${BLOB_KB}KB ($BLOB_SIZE bytes)"
echo ""

# Check if file already exists
if [ -f "$BRAIN_FILE" ]; then
    echo "Warning: File already exists: $BRAIN_FILE"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
    rm -f "$BRAIN_FILE"
fi

# Create a simple C program to create the brain
cat > /tmp/create_brain.c << 'EOF'
#include "../src/melvin.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <file> <nodes> <edges> <blob_size>\n", argv[0]);
        return 1;
    }
    
    const char *path = argv[1];
    size_t nodes = (size_t)atol(argv[2]);
    size_t edges = (size_t)atol(argv[3]);
    size_t blob_size = (size_t)atol(argv[4]);
    
    printf("Creating brain: %s\n", path);
    printf("  Nodes: %zu\n", nodes);
    printf("  Edges: %zu\n", edges);
    printf("  Blob: %zu bytes\n", blob_size);
    
    Graph *g = melvin_open(path, nodes, edges, blob_size);
    if (!g) {
        fprintf(stderr, "Failed to create brain\n");
        return 1;
    }
    
    printf("  ✓ Brain created successfully\n");
    printf("  File size: %llu bytes\n", (unsigned long long)g->hdr->file_size);
    
    melvin_sync(g);
    melvin_close(g);
    
    printf("  ✓ Brain saved\n");
    return 0;
}
EOF

# Compile the brain creator
gcc -std=c11 -Wall -O2 -Isrc -o /tmp/create_brain /tmp/create_brain.c src/melvin.c -lm -pthread 2>&1 | grep -v "warning: unused function" || true

if [ ! -f /tmp/create_brain ]; then
    echo "Error: Failed to compile brain creator"
    exit 1
fi

# Create the brain
/tmp/create_brain "$BRAIN_FILE" "$NODES" "$EDGES" "$BLOB_SIZE"

if [ $? -eq 0 ] && [ -f "$BRAIN_FILE" ]; then
    FILE_SIZE=$(stat -f%z "$BRAIN_FILE" 2>/dev/null || stat -c%s "$BRAIN_FILE" 2>/dev/null || echo "unknown")
    echo ""
    echo "✓ Brain file created: $BRAIN_FILE"
    echo "  Size: $FILE_SIZE bytes"
    echo ""
    echo "Next steps:"
    echo "  # Seed patterns"
    echo "  melvin_seed_patterns $BRAIN_FILE corpus/basic/patterns.txt 0.6"
    echo ""
    echo "  # Seed math knowledge"
    echo "  ./scripts/seed_complete_math.sh $BRAIN_FILE"
    echo ""
    echo "  # Or run the learning test"
    echo "  ./test/test_learn_both_methods"
else
    echo "Error: Failed to create brain file"
    exit 1
fi

# Cleanup
rm -f /tmp/create_brain.c /tmp/create_brain


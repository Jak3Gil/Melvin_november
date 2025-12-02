#!/bin/bash
# Deploy preseeded Melvin to Jetson

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "=========================================="
echo "DEPLOYING PRESEEDED MELVIN TO JETSON"
echo "=========================================="
echo ""

# Build preseed tool
echo "1. Building preseed tool..."
gcc -std=c11 -O2 -I. -o preseed_melvin preseed_melvin.c src/melvin.c -lm -pthread || exit 1
echo "   ✓ Built"

# Copy to Jetson
echo ""
echo "2. Copying to Jetson..."
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
    preseed_melvin src/melvin.c src/melvin.h \
    "$JETSON_USER@$JETSON_IP:~/melvin/" || exit 1
echo "   ✓ Copied"

# Run preseeding on Jetson
echo ""
echo "3. Running preseeding on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << 'EOF'
cd ~/melvin

echo "Creating preseeded brain..."
./preseed_melvin brain_preseeded.m

echo ""
echo "Verifying brain..."
ls -lh brain_preseeded.m

echo ""
echo "Brain stats:"
cat > /tmp/check.c << 'CCODE'
#include "src/melvin.h"
#include <stdio.h>
int main() {
    Graph *g = melvin_open("brain_preseeded.m", 0, 0, 0);
    if (!g) return 1;
    printf("Nodes: %llu\n", (unsigned long long)g->node_count);
    printf("Edges: %llu\n", (unsigned long long)g->edge_count);
    printf("Edge capacity: %llu\n", (unsigned long long)g->edge_capacity);
    printf("Avg edge strength: %.4f\n", g->avg_edge_strength);
    
    // Count edges per node
    uint32_t nodes_with_edges = 0;
    for (uint32_t i = 0; i < g->node_count && i < 256; i++) {
        if (g->nodes[i].first_out != UINT32_MAX || g->nodes[i].first_in != UINT32_MAX) {
            nodes_with_edges++;
        }
    }
    printf("Nodes with connections: %u / 256 (first 256)\n", nodes_with_edges);
    
    melvin_close(g);
    return 0;
}
CCODE

gcc -std=c11 -I. -o /tmp/check /tmp/check.c src/melvin.c -lm -pthread 2>/dev/null
/tmp/check

EOF

echo ""
echo "=========================================="
echo "✓ PRESEEDED BRAIN READY ON JETSON"
echo "=========================================="
echo ""
echo "Next: Start continuous learning with AI tools"


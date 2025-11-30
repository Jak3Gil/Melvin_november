#!/bin/bash
# Simple one-time view of Melvin's current status

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "=========================================="
echo "Melvin Status (Current)"
echo "=========================================="
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    # Try Python first, then C fallback
    python3 << 'PYEOF' 2>/dev/null || {
import struct
import os
from datetime import datetime

try:
    with open('/home/melvin/melvin_system/melvin.m', 'rb') as f:
        data = f.read(40)
        if len(data) == 40:
            num_nodes, num_edges, tick, node_cap, edge_cap = struct.unpack('QQQQQ', data)
            
            print(f"╔══════════════════════════════════════════════════════════╗")
            print(f"║           MELVIN GRAPH STATUS                            ║")
            print(f"║           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                    ║")
            print(f"╚══════════════════════════════════════════════════════════╝")
            print()
            print(f"Tick: {tick:,}")
            print(f"Nodes: {num_nodes:,} / {node_cap:,} ({num_nodes/node_cap*100 if node_cap > 0 else 0:.1f}%)")
            print(f"Edges: {num_edges:,} / {edge_cap:,} ({num_edges/edge_cap*100 if edge_cap > 0 else 0:.1f}%)")
            print()
            
            # Calculate some stats
            node_usage = (num_nodes / node_cap * 100) if node_cap > 0 else 0
            edge_usage = (num_edges / edge_cap * 100) if edge_cap > 0 else 0
            
            print("Usage:")
            print(f"  Nodes: {'█' * int(node_usage/2)}{'░' * (50-int(node_usage/2))} {node_usage:.1f}%")
            print(f"  Edges: {'█' * int(edge_usage/2)}{'░' * (50-int(edge_usage/2))} {edge_usage:.1f}%")
            print()
            
            # Check service status
            import subprocess
            result = subprocess.run(['systemctl', 'is-active', 'melvin.service'], 
                                  capture_output=True, text=True)
            status = result.stdout.strip()
            print(f"Service: {'✓ Running' if status == 'active' else '✗ ' + status}")
        else:
            print("Graph file too small")
except Exception as e:
    print(f"Error: {e}")
PYEOF

    # C fallback
    cat > /tmp/melvin_status.c << 'CEOF'
#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

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
        printf("Cannot read graph\n");
        close(fd);
        return 1;
    }
    
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char time_str[100];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║           MELVIN GRAPH STATUS                            ║\n");
    printf("║           %s                    ║\n", time_str);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("Tick: %llu\n", (unsigned long long)header->tick);
    printf("Nodes: %llu / %llu (%.1f%%)\n",
           (unsigned long long)header->num_nodes,
           (unsigned long long)header->node_cap,
           header->node_cap > 0 ? (double)header->num_nodes / header->node_cap * 100.0 : 0.0);
    printf("Edges: %llu / %llu (%.1f%%)\n\n",
           (unsigned long long)header->num_edges,
           (unsigned long long)header->edge_cap,
           header->edge_cap > 0 ? (double)header->num_edges / header->edge_cap * 100.0 : 0.0);
    
    double node_pct = header->node_cap > 0 ? (double)header->num_nodes / header->node_cap * 100.0 : 0.0;
    double edge_pct = header->edge_cap > 0 ? (double)header->num_edges / header->edge_cap * 100.0 : 0.0;
    
    printf("Usage:\n");
    printf("  Nodes: ");
    for (int i = 0; i < 50; i++) {
        printf("%c", i < (int)(node_pct/2) ? '█' : '░');
    }
    printf(" %.1f%%\n", node_pct);
    
    printf("  Edges: ");
    for (int i = 0; i < 50; i++) {
        printf("%c", i < (int)(edge_pct/2) ? '█' : '░');
    }
    printf(" %.1f%%\n", edge_pct);
    
    munmap(header, sizeof(BrainHeader));
    close(fd);
    return 0;
}
CEOF
        gcc -o /tmp/melvin_status /tmp/melvin_status.c 2>/dev/null && /tmp/melvin_status
    }
    
    echo ""
    echo "Service Status:"
    systemctl is-active melvin.service >/dev/null 2>&1 && echo "  ✓ Running" || echo "  ✗ Not running"
EOF


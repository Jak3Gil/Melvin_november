#!/bin/bash
# Deploy Complete Teachable Hardware System to Jetson

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_DIR="/home/melvin/teachable_system"

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  DEPLOYING TEACHABLE HARDWARE SYSTEM TO JETSON     ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Create package directory
echo "📦 Creating deployment package..."
rm -rf /tmp/teachable_system
mkdir -p /tmp/teachable_system/{src,tools}

# Copy source
cp src/melvin.c src/melvin.h /tmp/teachable_system/src/
cp melvin.o /tmp/teachable_system/ 2>/dev/null || true

# Copy tools
cp tools/teach_hardware_operations.c /tmp/teachable_system/tools/
cp tools/create_port_patterns.c /tmp/teachable_system/tools/
cp tools/bootstrap_hardware_edges.c /tmp/teachable_system/tools/
cp tools/Makefile /tmp/teachable_system/tools/

# Copy scripts
cp create_teachable_hardware_brain.sh /tmp/teachable_system/

# Create Makefile for top level
cat > /tmp/teachable_system/Makefile << 'EOF'
CC=gcc
CFLAGS=-Wall -O2 -I.
LDFLAGS=-lm -lpthread

all: melvin.o tools

melvin.o: src/melvin.c src/melvin.h
	$(CC) $(CFLAGS) -c src/melvin.c -o melvin.o

tools: melvin.o
	cd tools && $(MAKE)

clean:
	rm -f melvin.o
	cd tools && $(MAKE) clean

.PHONY: all tools clean
EOF

echo "✅ Package created"
echo ""

# Deploy to Jetson
echo "🚀 Deploying to Jetson..."
echo "   Target: $JETSON_USER@$JETSON_IP:$JETSON_DIR"
echo ""

# Create directory
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "mkdir -p $JETSON_DIR" 2>&1 | grep -v "Warning:"

# Copy files
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no -r /tmp/teachable_system/* \
    "$JETSON_USER@$JETSON_IP:$JETSON_DIR/" 2>&1 | grep -v "Warning:"

echo "✅ Files deployed"
echo ""

# Build on Jetson
echo "🔨 Building on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "cd $JETSON_DIR && make" 2>&1 | tail -10

echo ""
echo "✅ Build complete on Jetson"
echo ""

# Create brain on Jetson
echo "🧠 Creating teachable brain on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "cd $JETSON_DIR && bash create_teachable_hardware_brain.sh hardware_brain.m" 2>&1

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT COMPLETE                                ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Teachable brain created on Jetson!"
echo ""
echo "To use:"
echo "  ssh $JETSON_USER@$JETSON_IP"
echo "  cd $JETSON_DIR"
echo "  ls -lh hardware_brain.m  # See the brain file"
echo ""
echo "Brain contains:"
echo "  ✅ ARM64 hardware control code"
echo "  ✅ Port patterns"
echo "  ✅ Semantic patterns"
echo "  ✅ Bootstrap edges"
echo ""
echo "Next: Run with hardware and let it learn!"
echo ""


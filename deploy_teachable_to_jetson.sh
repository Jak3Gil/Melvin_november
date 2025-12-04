#!/bin/bash
# Deploy Teachable EXEC System to Jetson for Testing

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_DIR="/home/melvin/melvin_teachable"

echo "╔════════════════════════════════════════════════════╗"
echo "║  Deploying Teachable EXEC to Jetson               ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Create deployment package
echo "📦 Creating deployment package..."
mkdir -p /tmp/melvin_teachable
cp src/melvin.c /tmp/melvin_teachable/
cp src/melvin.h /tmp/melvin_teachable/
cp test_teachable_exec.c /tmp/melvin_teachable/
cp test_blob_exec_proof.c /tmp/melvin_teachable/

# Create Makefile for Jetson
cat > /tmp/melvin_teachable/Makefile << 'EOF'
CC=gcc
CFLAGS=-Wall -O2 -I.
LDFLAGS=-lm -lpthread

all: test_blob_proof test_teachable

melvin.o: melvin.c melvin.h
	$(CC) $(CFLAGS) -c melvin.c -o melvin.o

test_blob_proof: test_blob_exec_proof.c
	$(CC) $(CFLAGS) test_blob_exec_proof.c -o test_blob_proof

test_teachable: test_teachable_exec.c melvin.o
	$(CC) $(CFLAGS) test_teachable_exec.c melvin.o $(LDFLAGS) -o test_teachable

clean:
	rm -f *.o test_blob_proof test_teachable *.m

.PHONY: all clean
EOF

# Create test runner script
cat > /tmp/melvin_teachable/run_tests.sh << 'EOF'
#!/bin/bash
echo "╔════════════════════════════════════════════════════╗"
echo "║  TEACHABLE EXEC - Jetson Test                      ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

echo "Building tests..."
make clean
make all

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build complete"
echo ""

echo "═══════════════════════════════════════════════════"
echo "TEST 1: Blob Execution Proof"
echo "═══════════════════════════════════════════════════"
echo ""

./test_blob_proof

echo ""
echo "═══════════════════════════════════════════════════"
echo "TEST 2: Teachable EXEC System"
echo "═══════════════════════════════════════════════════"
echo ""

./test_teachable

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  TESTS COMPLETE                                    ║"
echo "╚════════════════════════════════════════════════════╝"
EOF

chmod +x /tmp/melvin_teachable/run_tests.sh

echo "✅ Package created"
echo ""

# Deploy to Jetson
echo "🚀 Deploying to Jetson..."
echo "   Copying files to $JETSON_USER@$JETSON_IP:$JETSON_DIR"
echo ""

# Create directory on Jetson
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "mkdir -p $JETSON_DIR" 2>&1 | grep -v "Warning:"

# Copy files
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no -r /tmp/melvin_teachable/* \
    "$JETSON_USER@$JETSON_IP:$JETSON_DIR/" 2>&1 | grep -v "Warning:"

echo "✅ Files deployed"
echo ""

# Run tests on Jetson
echo "╔════════════════════════════════════════════════════╗"
echo "║  Running Tests on Jetson                           ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "cd $JETSON_DIR && bash run_tests.sh" 2>&1

echo ""
echo "═══════════════════════════════════════════════════"
echo "DEPLOYMENT COMPLETE"
echo "═══════════════════════════════════════════════════"
echo ""
echo "To connect manually:"
echo "  ./jetson_terminal.sh"
echo "  cd $JETSON_DIR"
echo ""


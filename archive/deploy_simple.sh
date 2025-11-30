#!/bin/bash
# Simple deploy script - transfers only melvin.c, melvin.h, melvin.m
# Uses SSH with password authentication

set -e

JETSON_USER="melvin"
JETSON_HOST="169.254.123.100"
JETSON_PASS="123456"
JETSON_DIR="~/melvin_system"

echo "=========================================="
echo "Deploying Melvin to Jetson"
echo "=========================================="
echo "Host: $JETSON_USER@$JETSON_HOST"
echo ""

# Check files exist
FILES=("melvin.c" "melvin.h" "melvin.m")
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: $file not found!"
        exit 1
    fi
done

echo "Files to deploy:"
ls -lh melvin.c melvin.h melvin.m | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Try sshpass if available, otherwise use expect
if command -v sshpass &> /dev/null; then
    echo "Using sshpass for authentication..."
    
    # Create directory
    echo "Creating directory on Jetson..."
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_HOST" "mkdir -p $JETSON_DIR" || {
        echo "Failed to create directory. Trying anyway..."
    }
    
    # Transfer files
    echo ""
    echo "Transferring files..."
    for file in "${FILES[@]}"; do
        echo "  → $file"
        sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no "$file" "$JETSON_USER@$JETSON_HOST:$JETSON_DIR/"
    done
    
elif command -v expect &> /dev/null; then
    echo "Using expect for authentication..."
    
    # Create directory
    expect << EOF
spawn ssh -o StrictHostKeyChecking=no $JETSON_USER@$JETSON_HOST "mkdir -p $JETSON_DIR"
expect "password:"
send "$JETSON_PASS\r"
expect eof
EOF
    
    # Transfer files
    echo ""
    echo "Transferring files..."
    for file in "${FILES[@]}"; do
        echo "  → $file"
        expect << EOF
spawn scp -o StrictHostKeyChecking=no $file $JETSON_USER@$JETSON_HOST:$JETSON_DIR/
expect "password:"
send "$JETSON_PASS\r"
expect eof
EOF
    done
    
else
    echo "Neither sshpass nor expect found."
    echo ""
    echo "Please install one of:"
    echo "  brew install hudochenkov/sshpass/sshpass"
    echo "  brew install expect"
    echo ""
    echo "Or manually copy files:"
    echo "  scp melvin.c melvin.h melvin.m $JETSON_USER@$JETSON_HOST:$JETSON_DIR/"
    echo ""
    echo "Password: $JETSON_PASS"
    exit 1
fi

echo ""
echo "✓ Files transferred successfully!"
echo ""
echo "Now building on Jetson..."
echo ""

# Build on Jetson
if command -v sshpass &> /dev/null; then
    sshpass -p "$JETSON_PASS" ssh "$JETSON_USER@$JETSON_HOST" << 'ENDSSH'
cd ~/melvin_system
echo "Building melvin..."
gcc -std=c11 -O3 -march=armv8-a -o melvin melvin.c -lm -ldl -lpthread
if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    ls -lh melvin
else
    echo "✗ Build failed!"
    exit 1
fi
ENDSSH
elif command -v expect &> /dev/null; then
    expect << 'EOF'
spawn ssh melvin@169.254.123.100 "cd ~/melvin_system && gcc -std=c11 -O3 -march=armv8.2-a+fp16+simd -mtune=cortex-a78 -o melvin melvin.c -lm -ldl -lpthread"
expect "password:"
send "123456\r"
expect eof
EOF
fi

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "To run Melvin on Jetson:"
echo "  ssh $JETSON_USER@$JETSON_HOST"
echo "  cd ~/melvin_system"
echo "  ./melvin"
echo ""


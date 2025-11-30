#!/bin/bash
# create_brain.sh - Create initial Melvin brain file on Jetson

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
BRAIN_PATH="/mnt/melvin_ssd/melvin/brain.m"
CORPUS_DIR="/mnt/melvin_ssd/melvin/corpus/basic"

echo "Creating Melvin brain file on Jetson..."
echo ""

# Create brain with corpus packed into cold_data
sshpass -p "$JETSON_PASS" ssh "$JETSON_USER@$JETSON_IP" << 'EOF'
cd /mnt/melvin_ssd/melvin

# Check if corpus exists
if [ ! -d "corpus/basic" ]; then
    echo "Warning: corpus/basic not found, creating empty brain"
    ./melvin_pack_corpus -i /tmp -o brain.m --hot-nodes 10000 --hot-edges 50000 --hot-blob-bytes 1048576 --cold-data-bytes 0
else
    echo "Packing corpus into brain..."
    ./melvin_pack_corpus -i corpus/basic -o brain.m --hot-nodes 10000 --hot-edges 50000 --hot-blob-bytes 1048576
fi

echo "Seeding instincts..."
./melvin_seed_instincts brain.m

echo ""
echo "Brain created: brain.m"
ls -lh brain.m
EOF

echo ""
echo "Brain file created successfully!"


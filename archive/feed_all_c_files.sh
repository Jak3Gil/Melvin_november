#!/bin/bash
# Feed ALL C files to Melvin on Jetson

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${1:-192.168.55.1}"
JETSON_DIR="~/melvin"
INGEST_DIR="${JETSON_DIR}/ingested_repos"

echo "=========================================="
echo "Feeding ALL C Files to Melvin on Jetson"
echo "=========================================="
echo "Jetson: ${JETSON_USER}@${JETSON_HOST}"
echo ""

# Find all C files
echo "Finding all C files..."
C_FILES=($(find . -name "*.c" -type f ! -path "*/backup/*" ! -path "*/.git/*" ! -path "*/test_*" ! -path "*/monitor_melvin.c" ! -path "*/init_melvin_simple.c"))

if [ ${#C_FILES[@]} -eq 0 ]; then
    echo "ERROR: No C files found"
    exit 1
fi

echo "Found ${#C_FILES[@]} C files:"
for file in "${C_FILES[@]}"; do
    echo "  - $file"
done
echo ""

# Create ingest directory on Jetson
echo "Creating ingest directory on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" \
    "mkdir -p ${INGEST_DIR}"

# Copy files to Jetson
echo "Copying files to Jetson..."
COPIED=0
FAILED=0

for file in "${C_FILES[@]}"; do
    filename=$(basename "$file")
    # Preserve directory structure in ingested_repos
    rel_path="${file#./}"
    dir_path=$(dirname "$rel_path")
    
    if [ "$dir_path" != "." ]; then
        # Create subdirectory on Jetson
        sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" \
            "mkdir -p ${INGEST_DIR}/${dir_path}" 2>/dev/null
        
        target_path="${INGEST_DIR}/${rel_path}"
    else
        target_path="${INGEST_DIR}/${filename}"
    fi
    
    echo "  → $rel_path"
    if sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        "$file" "${JETSON_USER}@${JETSON_HOST}:${target_path}" 2>/dev/null; then
        ((COPIED++))
    else
        echo "    ✗ Failed to copy"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo "Total files: ${#C_FILES[@]}"
echo "Copied: $COPIED"
echo "Failed: $FAILED"
echo ""
echo "✓ Files copied to ${INGEST_DIR} on Jetson"
echo ""
echo "Melvin will automatically parse these files when:"
echo "  - parse_c node activates (runs on startup and periodically)"
echo "  - Files are in ${INGEST_DIR}/"
echo ""
echo "To trigger parsing immediately:"
echo "  ssh ${JETSON_USER}@${JETSON_HOST}"
echo "  cd ${JETSON_DIR}"
echo "  # parse_c node should activate automatically"
echo ""


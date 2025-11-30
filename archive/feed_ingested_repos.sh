#!/bin/bash
# Feed all files from ingested_repos folder to Melvin on Jetson

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${1:-192.168.55.1}"
JETSON_DIR="~/melvin"
INGEST_DIR="${JETSON_DIR}/ingested_repos"

SOURCE_DIR="./ingested_repos"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: $SOURCE_DIR directory not found"
    exit 1
fi

echo "=========================================="
echo "Feeding ingested_repos to Melvin on Jetson"
echo "=========================================="
echo "Jetson: ${JETSON_USER}@${JETSON_HOST}"
echo "Source: $SOURCE_DIR"
echo ""

# Find all code files
echo "Finding code files..."
CODE_FILES=($(find "$SOURCE_DIR" -type f \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.txt" -o -name "*.md" \) ! -path "*/.git/*"))

if [ ${#CODE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No code files found in $SOURCE_DIR"
    exit 1
fi

echo "Found ${#CODE_FILES[@]} files:"
for file in "${CODE_FILES[@]}"; do
    echo "  - $file"
done | head -20
if [ ${#CODE_FILES[@]} -gt 20 ]; then
    echo "  ... and $(( ${#CODE_FILES[@]} - 20 )) more files"
fi
echo ""

# Create ingest directory on Jetson
echo "Creating ingest directory on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" \
    "mkdir -p ${INGEST_DIR}"

# Copy entire directory structure to Jetson using rsync if available, otherwise tar
echo "Copying files to Jetson (preserving directory structure)..."
if sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" "command -v rsync" >/dev/null 2>&1; then
    # Use rsync for efficient transfer
    echo "Using rsync..."
    sshpass -p "$JETSON_PASS" rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no" \
        "$SOURCE_DIR/" "${JETSON_USER}@${JETSON_HOST}:${INGEST_DIR}/" 2>&1 | tail -20
else
    # Fallback: use tar
    echo "Using tar (rsync not available)..."
    cd "$(dirname "$SOURCE_DIR")"
    tar czf /tmp/ingested_repos.tar.gz "$(basename "$SOURCE_DIR")" 2>/dev/null
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        /tmp/ingested_repos.tar.gz "${JETSON_USER}@${JETSON_HOST}:/tmp/"
    sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" \
        "cd ${INGEST_DIR} && tar xzf /tmp/ingested_repos.tar.gz && rm /tmp/ingested_repos.tar.gz" 2>/dev/null
    rm -f /tmp/ingested_repos.tar.gz
fi

# Also copy github_urls.txt if it exists
if [ -f "github_urls.txt" ]; then
    echo "Copying github_urls.txt..."
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        "github_urls.txt" "${JETSON_USER}@${JETSON_HOST}:${INGEST_DIR}/"
fi

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Files copied to ${INGEST_DIR} on Jetson"
echo ""
echo "Melvin will automatically parse these files when:"
echo "  - parse_c node activates (runs on startup and periodically)"
echo "  - Files are in ${INGEST_DIR}/"
echo ""
echo "To verify files on Jetson:"
echo "  ssh ${JETSON_USER}@${JETSON_HOST}"
echo "  ls -lh ${INGEST_DIR}/"
echo ""


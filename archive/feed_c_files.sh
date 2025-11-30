#!/bin/bash
# Feed C files to Melvin running on Jetson

JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${1:-jetson.local}"
JETSON_DIR="~/melvin"
INGEST_DIR="${JETSON_DIR}/ingested_repos"  # mc_parse looks here

# Parse arguments
FILES=()
for arg in "$@"; do
    if [ "$arg" != "$JETSON_HOST" ] && [ -f "$arg" ] && [[ "$arg" == *.c ]]; then
        FILES+=("$arg")
    fi
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "Usage: $0 [jetson_host] file1.c file2.c ..."
    echo "   or: $0 [jetson_host] --dir directory/"
    echo ""
    echo "Feeds C files to Melvin on Jetson for parsing/digestion"
    exit 1
fi

# Check for --dir option
if [ "$2" == "--dir" ] && [ -n "$3" ]; then
    DIR="$3"
    if [ ! -d "$DIR" ]; then
        echo "ERROR: Directory not found: $DIR"
        exit 1
    fi
    echo "Finding all .c files in $DIR..."
    FILES=($(find "$DIR" -name "*.c" -type f))
    echo "Found ${#FILES[@]} C files"
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No C files to feed"
    exit 1
fi

echo "=========================================="
echo "Feeding C Files to Melvin on Jetson"
echo "=========================================="
echo "Jetson: ${JETSON_USER}@${JETSON_HOST}"
echo "Files: ${#FILES[@]}"
echo ""

# Create ingest directory on Jetson
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "${JETSON_USER}@${JETSON_HOST}" \
    "mkdir -p ${INGEST_DIR}"

# Copy files to Jetson
echo "Copying files to Jetson..."
for file in "${FILES[@]}"; do
    filename=$(basename "$file")
    echo "  → $filename"
    sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no \
        "$file" "${JETSON_USER}@${JETSON_HOST}:${INGEST_DIR}/"
done

echo ""
echo "✓ Files copied to ${INGEST_DIR} on Jetson"
echo ""
echo "Melvin will automatically parse these files when:"
echo "  - parse_c node activates (runs on startup)"
echo "  - Files are in ${INGEST_DIR} (or plugins/ or current directory)"
echo ""
echo "To trigger parsing immediately, you can:"
echo "  ssh ${JETSON_USER}@${JETSON_HOST}"
echo "  cd ${JETSON_DIR}"
echo "  # Move files to plugins/ or wait for auto-parse"
echo ""


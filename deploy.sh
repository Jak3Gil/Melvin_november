#!/bin/bash

# Melvin Deployment Script
# Syncs code from PC to Jetson and deploys

set -e

# Configuration
JETSON_HOST="melvin-jetson"  # Change to your Jetson's hostname/IP
JETSON_USER="melvin"         # Change to your Jetson username
JETSON_PATH="/home/melvin/melvin"
BUILD_DIR="build"
SERVICE_NAME="melvin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if rsync is available
check_dependencies() {
    if ! command -v rsync &> /dev/null; then
        error "rsync is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v ssh &> /dev/null; then
        error "ssh is not available. Please check your SSH configuration."
        exit 1
    fi
}

# Test SSH connection
test_connection() {
    log "Testing SSH connection to Jetson..."
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${JETSON_USER}@${JETSON_HOST}" "echo 'SSH connection successful'" 2>/dev/null; then
        error "Cannot connect to Jetson. Check SSH configuration and network."
        exit 1
    fi
    log "SSH connection successful"
}

# Sync code to Jetson
sync_code() {
    log "Syncing code to Jetson..."
    
    # Create remote directory if it doesn't exist
    ssh "${JETSON_USER}@${JETSON_HOST}" "mkdir -p ${JETSON_PATH}"
    
    # Sync source code (exclude build artifacts and temporary files)
    rsync -avz --delete \
        --exclude='build/' \
        --exclude='.git/' \
        --exclude='*.o' \
        --exclude='*.so' \
        --exclude='*.a' \
        --exclude='CMakeCache.txt' \
        --exclude='CMakeFiles/' \
        --exclude='logs/*.log' \
        --exclude='node_modules/' \
        --exclude='.vscode/' \
        --exclude='*.swp' \
        --exclude='*.tmp' \
        ./ "${JETSON_USER}@${JETSON_HOST}:${JETSON_PATH}/"
    
    log "Code sync completed"
}

# Build on Jetson
build_on_jetson() {
    log "Building on Jetson..."
    
    ssh "${JETSON_USER}@${JETSON_HOST}" "cd ${JETSON_PATH} && \
        mkdir -p ${BUILD_DIR} && \
        cd ${BUILD_DIR} && \
        cmake .. && \
        make -j$(nproc)"
    
    log "Build completed successfully"
}

# Install and restart service
deploy_service() {
    log "Installing and restarting service..."
    
    ssh "${JETSON_USER}@${JETSON_HOST}" "cd ${JETSON_PATH}/${BUILD_DIR} && \
        sudo make install && \
        sudo systemctl daemon-reload && \
        sudo systemctl restart ${SERVICE_NAME} && \
        sudo systemctl enable ${SERVICE_NAME}"
    
    # Check service status
    sleep 2
    if ssh "${JETSON_USER}@${JETSON_HOST}" "sudo systemctl is-active --quiet ${SERVICE_NAME}"; then
        log "Service ${SERVICE_NAME} is running successfully"
    else
        warn "Service ${SERVICE_NAME} may not be running. Check logs:"
        ssh "${JETSON_USER}@${JETSON_HOST}" "sudo journalctl -u ${SERVICE_NAME} -n 20"
    fi
}

# Show logs
show_logs() {
    log "Recent logs from Jetson:"
    ssh "${JETSON_USER}@${JETSON_HOST}" "sudo journalctl -u ${SERVICE_NAME} -n 30 --no-pager"
}

# Main deployment flow
main() {
    log "Starting Melvin deployment..."
    
    check_dependencies
    test_connection
    sync_code
    build_on_jetson
    deploy_service
    
    log "Deployment completed successfully!"
    log "Melvin should now be running on Jetson"
    log "Access the UI at: http://${JETSON_HOST}:8080"
    
    # Ask if user wants to see logs
    read -p "Show recent logs? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        show_logs
    fi
}

# Handle command line arguments
case "${1:-}" in
    "sync")
        check_dependencies
        test_connection
        sync_code
        log "Code sync completed"
        ;;
    "build")
        check_dependencies
        test_connection
        build_on_jetson
        log "Build completed"
        ;;
    "deploy")
        check_dependencies
        test_connection
        deploy_service
        log "Service deployed"
        ;;
    "logs")
        check_dependencies
        test_connection
        show_logs
        ;;
    "test")
        check_dependencies
        test_connection
        log "Connection test successful"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  sync    - Sync code to Jetson only"
        echo "  build   - Build on Jetson only"
        echo "  deploy  - Deploy service only"
        echo "  logs    - Show recent logs"
        echo "  test    - Test SSH connection"
        echo "  help    - Show this help"
        echo "  (none)  - Full deployment (sync + build + deploy)"
        ;;
    "")
        main
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

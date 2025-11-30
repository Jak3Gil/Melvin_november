#!/bin/bash
# Install Electron for Melvin Dashboard app

cd "$(dirname "$0")"

echo "Installing Electron for Melvin Dashboard..."
echo ""

if ! command -v node &> /dev/null; then
    echo "⚠ Node.js not found!"
    echo "Please install Node.js first:"
    echo "  Mac: brew install node"
    echo "  Linux: sudo apt install nodejs npm"
    echo "  Or download from: https://nodejs.org/"
    exit 1
fi

echo "Node.js version: $(node --version)"
echo "npm version: $(npm --version)"
echo ""

# Install Electron and dependencies
npm install

echo ""
echo "✓ Electron installed!"
echo ""
echo "To launch the app:"
echo "  ./launch_dashboard.sh"
echo ""
echo "Or manually:"
echo "  npm start"
echo ""


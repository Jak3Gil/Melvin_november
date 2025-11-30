# Melvin Dashboard - Build and Launch

.PHONY: install launch electron python clean

# Install Electron dependencies
install:
	@echo "Installing Electron..."
	cd tools && npm install

# Launch as Python app (opens browser window)
python:
	@echo "Launching Melvin Dashboard (Python app)..."
	python3 tools/melvin_dashboard_app.py

# Launch as Electron app (true desktop app)
electron:
	@echo "Launching Melvin Dashboard (Electron app)..."
	cd tools && npm start

# Launch (auto-detects best option)
launch: python

# Clean build artifacts
clean:
	rm -rf tools/node_modules
	rm -rf tools/dist
	rm -rf tools/package-lock.json

# Help
help:
	@echo "Melvin Dashboard - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make install   - Install Electron dependencies"
	@echo "  make python   - Launch as Python app (opens browser)"
	@echo "  make electron - Launch as Electron app (desktop app)"
	@echo "  make launch   - Launch (defaults to Python app)"
	@echo "  make clean    - Clean build artifacts"
	@echo ""


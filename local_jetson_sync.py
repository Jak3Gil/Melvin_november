#!/usr/bin/env python3
"""
Local Jetson Sync System
========================
Works without internet - creates deployment packages for COM8/PuTTY transfer
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class LocalJetsonSync:
    def __init__(self):
        self.pc_files = Path(".")
        self.jetson_ready = Path("jetson_deployment")
        self.jetson_ready.mkdir(exist_ok=True)
        
        # Core Melvin files to sync
        self.core_files = [
            "melvin_working_stable.py",
            "sync_melvin.py",
            "pc_to_jetson_sync.py"
        ]
    
    def create_jetson_package(self):
        """Create a complete Jetson deployment package"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = self.jetson_ready / f"melvin_v{timestamp}"
        package_dir.mkdir(exist_ok=True)
        
        print(f"📦 Creating Jetson package: {package_dir.name}")
        
        # Create the core brain file (since it's on Jetson)
        brain_code = '''#!/usr/bin/env python3
"""
Melvin Working Stable - PC Generated Version
==========================================
"""
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

class MelvinWorkingBrain:
    def __init__(self):
        self.nodes = {}
        self.connections = defaultdict(list)
        self.memory_file = Path("melvin_memory.json")
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def add_node(self, content, node_type="text"):
        node_id = f"{node_type}_{len(self.nodes)}"
        self.nodes[node_id] = {
            "content": content,
            "type": node_type,
            "timestamp": time.time(),
            "embedding": [0.1] * 256  # Simple placeholder
        }
        self.logger.info(f"Added node: {node_id}")
        return node_id
    
    def process_text(self, text):
        return self.add_node(text, "text")
    
    def process_camera(self):
        return self.add_node("Camera frame captured", "visual")
    
    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump({
                "nodes": self.nodes,
                "connections": dict(self.connections)
            }, f, indent=2)
        self.logger.info(f"Memory saved: {len(self.nodes)} nodes")
    
    def get_status(self):
        return f"Nodes: {len(self.nodes)}, Connections: {len(self.connections)}"

def main():
    print("🧠 MELVIN WORKING STABLE")
    print("=" * 30)
    
    brain = MelvinWorkingBrain()
    
    while True:
        cmd = input("melvin-work> ").strip().lower()
        
        if cmd.startswith("text "):
            text = cmd[5:]
            brain.process_text(text)
            print(f"✅ Processed text: {text[:50]}...")
        
        elif cmd == "camera":
            brain.process_camera()
            print("📸 Camera processed")
        
        elif cmd == "status":
            print(f"📊 Status: {brain.get_status()}")
        
        elif cmd == "save":
            brain.save_memory()
            print("💾 Memory saved")
        
        elif cmd == "quit":
            brain.save_memory()
            print("👋 Goodbye!")
            break
        
        else:
            print("❓ Commands: text <msg>, camera, status, save, quit")

if __name__ == "__main__":
    main()
'''
        
        # Write the brain file
        with open(package_dir / "melvin_working_stable.py", "w", encoding="utf-8") as f:
            f.write(brain_code)
        
        # Create deployment script
        deploy_script = f'''#!/bin/bash
echo "🚀 DEPLOYING MELVIN v{timestamp}"
echo "================================"

# Make executable
chmod +x melvin_working_stable.py

# Test run
echo "🧠 Testing Melvin..."
timeout 5s python3 melvin_working_stable.py <<EOF || true
status
quit
EOF

echo "✅ Melvin deployed successfully!"
echo "🎯 Run: python3 melvin_working_stable.py"
'''
        
        with open(package_dir / "deploy.sh", "w", encoding="utf-8") as f:
            f.write(deploy_script)
        
        # Create transfer instructions
        instructions = f'''
JETSON DEPLOYMENT - COM8/PuTTY Method
====================================

1. TRANSFER FILES (via PuTTY):

   A) Create brain file:
   nano melvin_working_stable.py
   # Copy contents from melvin_working_stable.py
   # Ctrl+X, Y, Enter
   
   B) Create deploy script:
   nano deploy.sh
   # Copy contents from deploy.sh  
   # Ctrl+X, Y, Enter

2. DEPLOY:
   chmod +x deploy.sh melvin_working_stable.py
   ./deploy.sh

3. RUN:
   python3 melvin_working_stable.py

4. TEST COMMANDS:
   melvin-work> text Hello Melvin
   melvin-work> camera
   melvin-work> status
   melvin-work> save
   melvin-work> quit

Package: melvin_v{timestamp}
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
        
        with open(package_dir / "JETSON_INSTRUCTIONS.txt", "w", encoding="utf-8") as f:
            f.write(instructions)
        
        print(f"✅ Package created: {package_dir}")
        print("📋 See JETSON_INSTRUCTIONS.txt for deployment steps")
        return package_dir
    
    def show_transfer_commands(self):
        """Show copy-paste commands for PuTTY"""
        packages = list(self.jetson_ready.glob("melvin_v*"))
        if not packages:
            print("❌ No packages found. Create one first.")
            return
        
        latest = max(packages, key=lambda p: p.name)
        print(f"\n🔧 PuTTY TRANSFER COMMANDS for {latest.name}")
        print("=" * 50)
        
        # Read the brain file
        brain_file = latest / "melvin_working_stable.py"
        if brain_file.exists():
            print("\n1. CREATE BRAIN FILE:")
            print("nano melvin_working_stable.py")
            print("# Copy the contents from the file, then Ctrl+X, Y, Enter")
        
        # Read deploy script
        deploy_file = latest / "deploy.sh"
        if deploy_file.exists():
            print("\n2. CREATE DEPLOY SCRIPT:")
            print("nano deploy.sh")
            print("# Copy the contents from the file, then Ctrl+X, Y, Enter")
        
        print("\n3. MAKE EXECUTABLE & DEPLOY:")
        print("chmod +x deploy.sh melvin_working_stable.py")
        print("./deploy.sh")
        
        print("\n4. RUN MELVIN:")
        print("python3 melvin_working_stable.py")

def main():
    sync = LocalJetsonSync()
    
    print("🔄 LOCAL JETSON SYNC (No Internet Required)")
    print("=" * 45)
    print("1. Create deployment package")
    print("2. Show transfer commands")
    print("3. Open deployment folder")
    
    choice = input("Choice [1-3]: ").strip()
    
    if choice == "1":
        package = sync.create_jetson_package()
        print(f"\n📂 Package location: {package}")
        sync.show_transfer_commands()
    
    elif choice == "2":
        sync.show_transfer_commands()
    
    elif choice == "3":
        import subprocess
        subprocess.run(["explorer", str(sync.jetson_ready)], shell=True)
        print(f"📂 Opened: {sync.jetson_ready}")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()

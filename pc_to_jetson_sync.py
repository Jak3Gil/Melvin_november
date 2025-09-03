#!/usr/bin/env python3
"""
PC to Jetson Sync System
========================
Simple file sync for COM8/PuTTY connections
"""

import os
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime

class PCJetsonSync:
    def __init__(self):
        self.pc_workspace = Path(".")
        self.jetson_files = [
            "melvin_working_stable.py",
            "sync_melvin.py", 
            "melvin_global_brain.py",
            "melvin_clean_brain.py"
        ]
        self.sync_folder = Path("jetson_sync")
        self.sync_folder.mkdir(exist_ok=True)
        
    def create_sync_package(self):
        """Create a package ready for Jetson deployment"""
        print("📦 Creating Jetson sync package...")
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = self.sync_folder / f"package_{timestamp}"
        package_dir.mkdir(exist_ok=True)
        
        # Copy files
        copied_files = []
        for file in self.jetson_files:
            if Path(file).exists():
                shutil.copy2(file, package_dir)
                copied_files.append(file)
                print(f"✅ Copied: {file}")
        
        # Create deployment script
        deploy_script = f"""#!/bin/bash
# Auto-generated deployment script
echo "🚀 Deploying Melvin update..."
echo "📅 Package: {timestamp}"

# Backup existing files
mkdir -p backups/backup_{timestamp}
"""
        
        for file in copied_files:
            deploy_script += f"""
if [ -f "{file}" ]; then
    cp "{file}" backups/backup_{timestamp}/
    echo "📋 Backed up: {file}"
fi
cp package_{timestamp}/{file} .
chmod +x {file}
echo "✅ Updated: {file}"
"""
        
        deploy_script += """
echo "🎯 Deployment complete!"
echo "🧠 Run: python3 melvin_working_stable.py"
"""
        
        with open(package_dir / "deploy.sh", "w", encoding="utf-8") as f:
            f.write(deploy_script)
        
        # Create instructions
        instructions = f"""
JETSON DEPLOYMENT INSTRUCTIONS
==============================

1. Transfer this entire folder to Jetson:
   package_{timestamp}/

2. On Jetson, run:
   cd package_{timestamp}
   chmod +x deploy.sh
   ./deploy.sh

3. Test:
   python3 melvin_working_stable.py

Files included:
{chr(10).join(f"- {f}" for f in copied_files)}
"""
        
        with open(package_dir / "INSTRUCTIONS.txt", "w", encoding="utf-8") as f:
            f.write(instructions)
        
        print(f"📦 Package created: {package_dir}")
        print("📋 See INSTRUCTIONS.txt for deployment steps")
        return package_dir
    
    def create_transfer_commands(self, package_dir):
        """Create copy-paste commands for PuTTY"""
        print("\n🔧 PuTTY Transfer Commands:")
        print("=" * 40)
        
        # Create individual file transfer commands
        for file in package_dir.glob("*"):
            if file.is_file() and file.name != "INSTRUCTIONS.txt":
                print(f"\n# Transfer {file.name}:")
                print(f"nano {file.name}")
                print("# Paste file contents, then Ctrl+X, Y, Enter")
        
        print(f"\n# Make executable:")
        print("chmod +x deploy.sh melvin_working_stable.py")
        print("\n# Deploy:")
        print("./deploy.sh")

def main():
    sync = PCJetsonSync()
    
    print("🔄 PC TO JETSON SYNC")
    print("=" * 30)
    print("1. Create sync package")
    print("2. Show transfer commands")
    
    choice = input("Choice [1-2]: ").strip()
    
    if choice == "1":
        package = sync.create_sync_package()
        sync.create_transfer_commands(package)
    elif choice == "2":
        # Find latest package
        packages = list(sync.sync_folder.glob("package_*"))
        if packages:
            latest = max(packages, key=lambda p: p.name)
            sync.create_transfer_commands(latest)
        else:
            print("❌ No packages found. Create one first.")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()

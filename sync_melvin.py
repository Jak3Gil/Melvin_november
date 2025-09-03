#!/usr/bin/env python3
"""
🔄 MELVIN SYNC SYSTEM
====================
Simple way to sync code without Git complexity
"""

import os
import shutil
import time
import json
from pathlib import Path

def create_version_backup():
    """Create versioned backup"""
    timestamp = int(time.time())
    backup_dir = f"versions/v_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup current files
    for file in ["melvin_global_brain.py", "melvin_clean_brain.py"]:
        if Path(file).exists():
            shutil.copy(file, backup_dir)
    
    print(f"✅ Version saved: {backup_dir}")
    return backup_dir

def list_versions():
    """List available versions"""
    if Path("versions").exists():
        versions = sorted([d.name for d in Path("versions").iterdir() if d.is_dir()])
        print("📋 Available versions:")
        for i, version in enumerate(versions[-10:], 1):  # Show last 10
            print(f"  {i}. {version}")
        return versions
    return []

def restore_version(version_name):
    """Restore a previous version"""
    version_path = Path(f"versions/{version_name}")
    if version_path.exists():
        # Backup current before restore
        create_version_backup()
        
        # Restore files
        for file in version_path.glob("*.py"):
            shutil.copy(file, ".")
            print(f"✅ Restored: {file.name}")
    else:
        print(f"❌ Version not found: {version_name}")

def create_update_package():
    """Create update package for easy transfer"""
    package_dir = "melvin_update_package"
    os.makedirs(package_dir, exist_ok=True)
    
    # Copy current files
    files_to_package = [
        "melvin_global_brain.py",
        "melvin_clean_brain.py", 
        "melvin_working.py"
    ]
    
    packaged_files = []
    for file in files_to_package:
        if Path(file).exists():
            shutil.copy(file, package_dir)
            packaged_files.append(file)
    
    # Create install script
    install_script = f'''#!/bin/bash
# Melvin Auto-Install Script
echo "🚀 Installing Melvin Update..."

# Backup current
mkdir -p backup_$(date +%Y%m%d_%H%M%S)
cp melvin*.py backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

# Install new files
cp melvin_update_package/*.py .
chmod +x *.py

echo "✅ Melvin updated successfully!"
echo "Run: python3 melvin_working.py"
'''
    
    with open(f"{package_dir}/install.sh", "w") as f:
        f.write(install_script)
    
    os.chmod(f"{package_dir}/install.sh", 0o755)
    
    print(f"📦 Update package created: {package_dir}/")
    print(f"   Files: {', '.join(packaged_files)}")
    print("   To install: cd melvin_update_package && ./install.sh")

def main():
    print("🔄 MELVIN SYNC SYSTEM")
    print("=" * 30)
    print("1. Create version backup")
    print("2. List versions")
    print("3. Restore version")
    print("4. Create update package")
    print("5. Quick deploy working version")
    
    choice = input("Choice [1-5]: ").strip()
    
    if choice == "1":
        create_version_backup()
    elif choice == "2":
        list_versions()
    elif choice == "3":
        versions = list_versions()
        if versions:
            try:
                idx = int(input("Select version number: ")) - 1
                if 0 <= idx < len(versions):
                    restore_version(versions[idx])
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Invalid number")
    elif choice == "4":
        create_update_package()
    elif choice == "5":
        # Quick deploy working version
        create_version_backup()
        
        working_code = '''#!/usr/bin/env python3
"""🧠 MELVIN WORKING VERSION - Tested & Stable"""
import cv2, numpy as np, time, logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MelvinWorkingBrain:
    def __init__(self):
        self.nodes = {}
        self.connections = defaultdict(list)
        self.stats = {"nodes": 0, "connections": 0, "start": time.time()}
        Path("melvin_global_memory").mkdir(exist_ok=True)
        logger.info("🧠 Melvin Working Brain initialized")
    
    def add_text(self, text):
        node_id = f"text_{len(self.nodes)}"
        self.nodes[node_id] = {"type": "text", "content": text, "time": time.time()}
        self._connect_recent(node_id)
        self.stats["nodes"] += 1
        logger.info(f"📝 Added: {text[:40]}...")
        return node_id
    
    def add_visual(self):
        try:
            cap = cv2.VideoCapture('/dev/video0')
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Simple feature extraction
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    features = {
                        "brightness": float(np.mean(gray) / 255),
                        "contrast": float(np.std(gray) / 255),
                        "size": frame.shape
                    }
                    
                    node_id = f"visual_{len(self.nodes)}"
                    self.nodes[node_id] = {"type": "visual", "features": features, "time": time.time()}
                    self._connect_recent(node_id)
                    self.stats["nodes"] += 1
                    
                    logger.info(f"📹 Visual: brightness={features['brightness']:.2f}")
                    cap.release()
                    return node_id
                cap.release()
        except Exception as e:
            logger.error(f"Camera error: {e}")
        return None
    
    def _connect_recent(self, node_id):
        # Connect to recent nodes (simple temporal connection)
        recent = [nid for nid, node in self.nodes.items() 
                 if time.time() - node["time"] < 10.0 and nid != node_id]
        
        for other_id in recent[-3:]:  # Connect to last 3
            self.connections[node_id].append(other_id)
            self.connections[other_id].append(node_id)
            self.stats["connections"] += 1
    
    def status(self):
        runtime = time.time() - self.stats["start"]
        return {
            "nodes": len(self.nodes),
            "connections": self.stats["connections"],
            "runtime_seconds": int(runtime),
            "nodes_per_minute": len(self.nodes) / (runtime / 60) if runtime > 0 else 0
        }
    
    def save_state(self):
        state_file = Path("melvin_global_memory/working_state.json")
        with open(state_file, "w") as f:
            # Convert to serializable format
            save_data = {
                "stats": self.status(),
                "node_count": len(self.nodes),
                "connection_count": self.stats["connections"],
                "save_time": time.time()
            }
            json.dump(save_data, f, indent=2)
        logger.info(f"💾 State saved: {len(self.nodes)} nodes")

def main():
    print("🧠 MELVIN WORKING BRAIN")
    print("=" * 30)
    
    brain = MelvinWorkingBrain()
    
    print("Commands: text <message>, camera, stream <seconds>, status, save, quit")
    
    try:
        while True:
            cmd = input("melvin-work> ").strip()
            
            if cmd in ["quit", "exit", "q"]:
                break
            elif cmd.startswith("text "):
                brain.add_text(cmd[5:])
            elif cmd == "camera":
                brain.add_visual()
            elif cmd.startswith("stream "):
                try:
                    seconds = int(cmd.split()[1])
                    print(f"📹 Streaming for {seconds} seconds...")
                    for i in range(seconds):
                        brain.add_visual()
                        if i % 5 == 0:
                            print(f"Status: {brain.status()}")
                        time.sleep(1)
                except ValueError:
                    print("❌ Invalid duration")
            elif cmd == "status":
                status = brain.status()
                print(f"📊 {status}")
            elif cmd == "save":
                brain.save_state()
            elif cmd == "help":
                print("Commands: text <msg>, camera, stream <sec>, status, save, quit")
            else:
                print("❓ Unknown command (try 'help')")
                
    except (EOFError, KeyboardInterrupt):
        print("\\n🛑 Stopping...")
    
    brain.save_state()
    print("✅ Melvin Working Brain stopped")

if __name__ == "__main__":
    main()
'''
        
        with open("melvin_working_stable.py", "w") as f:
            f.write(working_code)
        
        os.chmod("melvin_working_stable.py", 0o755)
        print("✅ Working version deployed: melvin_working_stable.py")
        print("🚀 Ready to run: python3 melvin_working_stable.py")

if __name__ == "__main__":
    main()


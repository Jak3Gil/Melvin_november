
#!/usr/bin/env python3
"""🧠 MELVIN WORKING VERSION - Tested & Stable"""
import cv2, numpy as np, time, logging
from pathlib import Path
from collections import defaultdict
import json

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
        print("\n🛑 Stopping...")
    
    brain.save_state()
    print("✅ Melvin Working Brain stopped")

if __name__ == "__main__":
    main()

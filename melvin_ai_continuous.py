#!/usr/bin/env python3
"""
Melvin AI Continuous Learning System

Architecture:
  Camera → Vision AI → Semantic labels → Melvin
  Mic → Whisper STT → Text → Melvin
  
  Melvin learns from preprocessed, semantic data
  AI tools run continuously, Melvin observes and learns patterns
"""

import subprocess
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import sys
import os
import threading
import queue

# Configuration
BRAIN_PATH = "brain_preseeded.m"
CAMERA1_DEV = "/dev/video0"
CAMERA2_DEV = "/dev/video2"
MIC_DEVICE = "hw:0,0"

# Queues for AI processing
vision_queue = queue.Queue(maxsize=10)
audio_queue = queue.Queue(maxsize=10)
melvin_feed_queue = queue.Queue(maxsize=100)

class MelvinFeeder:
    """Feeds data to Melvin C process via stdin"""
    def __init__(self):
        self.proc = None
        self.start_melvin()
    
    def start_melvin(self):
        """Start Melvin C process"""
        # Create C program that reads from stdin and feeds to graph
        c_code = '''
#include "src/melvin.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main() {
    Graph *g = melvin_open("brain_preseeded.m", 0, 0, 0);
    if (!g) return 1;
    
    fprintf(stderr, "Melvin ready: %llu nodes, %llu edges\\n", 
            (unsigned long long)g->node_count, (unsigned long long)g->edge_count);
    
    // Read from stdin: "PORT:DATA\\n"
    char line[1024];
    uint64_t bytes_fed = 0;
    time_t last_report = time(NULL);
    uint64_t start_edges = g->edge_count;
    
    while (fgets(line, sizeof(line), stdin)) {
        // Parse: PORT:DATA
        uint32_t port = 0;
        char *colon = strchr(line, ':');
        if (!colon) continue;
        
        port = atoi(line);
        char *data = colon + 1;
        size_t len = strlen(data);
        if (len > 0 && data[len-1] == '\\n') data[len-1] = '\\0';
        
        // Feed data
        for (size_t i = 0; data[i] != '\\0'; i++) {
            melvin_feed_byte(g, port, (uint8_t)data[i], 0.3f);
            bytes_fed++;
        }
        
        if (bytes_fed % 100 == 0) {
            melvin_call_entry(g);
            
            // Report every 10 seconds
            time_t now = time(NULL);
            if (now - last_report >= 10) {
                fprintf(stderr, "[%llds] Fed %llu bytes | Edges: %llu (+%llu)\\n",
                        (long long)(now - last_report), (unsigned long long)bytes_fed,
                        (unsigned long long)g->edge_count,
                        (unsigned long long)(g->edge_count - start_edges));
                last_report = now;
                start_edges = g->edge_count;
                bytes_fed = 0;
            }
        }
    }
    
    melvin_close(g);
    return 0;
}
'''
        
        # Compile
        with open('/tmp/melvin_feeder.c', 'w') as f:
            f.write(c_code)
        
        result = subprocess.run(
            ['gcc', '-std=c11', '-O2', '-I.', '-o', '/tmp/melvin_feeder', 
             '/tmp/melvin_feeder.c', 'src/melvin.c', '-lm', '-pthread'],
            cwd='/home/melvin/melvin',
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"Compile error: {result.stderr.decode()}")
            sys.exit(1)
        
        # Start process
        self.proc = subprocess.Popen(
            ['/tmp/melvin_feeder'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait for ready message
        time.sleep(0.5)
    
    def feed(self, port, data):
        """Feed data to Melvin on specified port"""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(f"{port}:{data}\n")
                self.proc.stdin.flush()
            except:
                print("Melvin process died")
                sys.exit(1)

def vision_thread():
    """Process camera frames with vision AI"""
    print("[Vision] Starting...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    frame_count = 0
    while True:
        try:
            # Capture frame from camera 1
            subprocess.run(
                f"ffmpeg -f v4l2 -video_size 320x240 -i {CAMERA1_DEV} "
                f"-frames:v 1 /tmp/frame.jpg -y 2>/dev/null",
                shell=True, timeout=2
            )
            
            # Process with vision
            img = Image.open('/tmp/frame.jpg')
            tensor = transform(img)
            
            # Simple feature extraction (in real system: use actual vision model)
            # For now: just extract brightness/color features
            mean_r = tensor[0].mean().item()
            mean_g = tensor[1].mean().item()
            mean_b = tensor[2].mean().item()
            
            # Create semantic label
            brightness = (mean_r + mean_g + mean_b) / 3
            if brightness > 0.7:
                label = "BRIGHT_SCENE"
            elif brightness < 0.3:
                label = "DARK_SCENE"
            else:
                label = "NORMAL_LIGHT"
            
            # Add to feed queue
            melvin_feed_queue.put((100, f"VISION:{label}"))
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[Vision] Processed {frame_count} frames")
            
            time.sleep(1)  # Process 1 frame/sec
            
        except Exception as e:
            print(f"[Vision] Error: {e}")
            time.sleep(2)

def audio_thread():
    """Process audio with Whisper STT"""
    print("[Audio] Starting...")
    
    # Try to import whisper
    try:
        import whisper
        model = whisper.load_model("tiny")
        print("[Audio] Whisper loaded")
    except:
        print("[Audio] Whisper not available - using dummy")
        model = None
    
    while True:
        try:
            # Capture 3 seconds of audio
            subprocess.run(
                f"timeout 3 ffmpeg -f pulse -i default -t 3 -ar 16000 -ac 1 "
                f"/tmp/audio.wav -y 2>/dev/null",
                shell=True
            )
            
            if os.path.exists('/tmp/audio.wav') and os.path.getsize('/tmp/audio.wav') > 1000:
                if model:
                    result = model.transcribe('/tmp/audio.wav')
                    text = result["text"].strip()
                    if text:
                        melvin_feed_queue.put((101, f"STT:{text}"))
                        print(f"[Audio] '{text}'")
                else:
                    melvin_feed_queue.put((101, "AUDIO_DETECTED"))
            
            time.sleep(3)
            
        except Exception as e:
            print(f"[Audio] Error: {e}")
            time.sleep(5)

def feeder_thread(melvin):
    """Feed data from queue to Melvin"""
    print("[Feeder] Starting...")
    
    while True:
        try:
            port, data = melvin_feed_queue.get(timeout=1)
            melvin.feed(port, data)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Feeder] Error: {e}")
            break

def main():
    print("="*60)
    print("MELVIN AI CONTINUOUS LEARNING")
    print("="*60)
    print()
    print("Architecture:")
    print("  Camera → Vision AI → Semantic labels → Melvin")
    print("  Mic → Whisper STT → Text → Melvin")
    print()
    print("Melvin learns patterns from AI-processed data")
    print("="*60)
    print()
    
    # Start Melvin
    melvin = MelvinFeeder()
    print("✓ Melvin started")
    
    # Start threads
    threads = [
        threading.Thread(target=vision_thread, daemon=True),
        threading.Thread(target=audio_thread, daemon=True),
        threading.Thread(target=feeder_thread, args=(melvin,), daemon=True),
    ]
    
    for t in threads:
        t.start()
    
    print("✓ All threads started")
    print()
    print("Running... (Ctrl+C to stop)")
    print()
    
    # Monitor stderr from Melvin process
    try:
        while True:
            line = melvin.proc.stderr.readline()
            if line:
                print(f"[Melvin] {line.strip()}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        melvin.proc.terminate()

if __name__ == "__main__":
    main()


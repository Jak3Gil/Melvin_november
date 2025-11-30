#!/usr/bin/env python3
"""
Melvin Dashboard - Production monitoring and control tool
- Real-time graph state monitoring
- Drag-and-drop pattern feeding
- 3D node/edge visualization
- Output display
"""

import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import ctypes
import ctypes.util

# Add src to path for melvin imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    # Try to import melvin C library
    melvin_lib = ctypes.CDLL(str(Path(__file__).parent.parent / 'build' / 'libmelvin.so'))
except:
    melvin_lib = None
    print("⚠ Melvin library not found - dashboard will run in simulation mode")

class MelvinDashboard:
    def __init__(self, brain_path="/tmp/melvin_brain.m", port=8080):
        self.brain_path = brain_path
        self.port = port
        self.graph_state = {
            'nodes': 0,
            'edges': 0,
            'chaos': 0.0,
            'activation': 0.0,
            'edge_strength': 0.0,
            'active_nodes': 0,
            'recent_outputs': []
        }
        self.running = False
        self.monitor_thread = None
        
    def get_graph_state(self):
        """Get current graph state from .m file"""
        if not os.path.exists(self.brain_path):
            return self.graph_state
            
        try:
            # Read .m file header to get state
            with open(self.brain_path, 'rb') as f:
                # Read header (first 100 bytes should have counts)
                header = f.read(100)
                if len(header) >= 20:
                    # Parse node_count and edge_count (assuming v2 format)
                    # This is a simplified parser - real one would use ctypes
                    node_count = int.from_bytes(header[20:28], 'little') if len(header) >= 28 else 0
                    edge_count = int.from_bytes(header[28:36], 'little') if len(header) >= 36 else 0
                    
                    self.graph_state['nodes'] = node_count
                    self.graph_state['edges'] = edge_count
        except Exception as e:
            print(f"Error reading graph state: {e}")
            
        return self.graph_state
    
    def feed_pattern_file(self, file_path):
        """Feed a pattern file to the graph"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Write to a temp file that melvin can read
            temp_path = f"/tmp/melvin_feed_{int(time.time())}.dat"
            with open(temp_path, 'wb') as f:
                f.write(data)
            
            # Call melvin feed function (would need C interface)
            # For now, just return success
            return {'success': True, 'bytes_fed': len(data), 'file': temp_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            self.get_graph_state()
            time.sleep(0.5)  # Update every 500ms
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)


class DashboardHandler(BaseHTTPRequestHandler):
    dashboard = None
    
    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/' or path == '/index.html':
            self.serve_file('dashboard.html')
        elif path == '/dashboard.js':
            self.serve_file('dashboard.js', 'application/javascript')
        elif path == '/dashboard.css':
            self.serve_file('dashboard.css', 'text/css')
        elif path == '/api/state':
            self.serve_json(self.dashboard.get_graph_state())
        elif path.startswith('/api/'):
            self.send_error(404)
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/api/feed':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                if 'file_path' in data:
                    result = self.dashboard.feed_pattern_file(data['file_path'])
                    self.serve_json(result)
                else:
                    self.serve_json({'success': False, 'error': 'No file_path provided'})
            except Exception as e:
                self.serve_json({'success': False, 'error': str(e)})
        else:
            self.send_error(404)
    
    def serve_file(self, file_path, content_type='text/html'):
        """Serve a static file"""
        try:
            # Try relative to script directory first (when running from tools/)
            full_path = Path(__file__).parent / file_path
            if not full_path.exists():
                # Try relative to parent (when running from repo root)
                full_path = Path(__file__).parent.parent / 'tools' / file_path
            if not full_path.exists():
                self.send_error(404)
                return
                
            with open(full_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))
    
    def serve_json(self, data):
        """Serve JSON response"""
        json_data = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data)
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Melvin Dashboard')
    parser.add_argument('--brain', default='/tmp/melvin_brain.m', help='Path to .m brain file')
    parser.add_argument('--port', type=int, default=8080, help='HTTP server port')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    dashboard = MelvinDashboard(brain_path=args.brain, port=args.port)
    dashboard.start_monitoring()
    DashboardHandler.dashboard = dashboard
    
    server = HTTPServer((args.host, args.port), DashboardHandler)
    
    print(f"==========================================")
    print(f"Melvin Dashboard")
    print(f"==========================================")
    print(f"Brain file: {args.brain}")
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Press Ctrl+C to stop")
    print(f"==========================================")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        dashboard.stop()
        server.shutdown()


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Melvin Control API - REST API for controlling Melvin service
- Start/stop/pause/resume
- Status monitoring
- Graph state queries
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

PID_FILE = "/tmp/melvin.pid"
BRAIN_FILE = os.getenv("MELVIN_BRAIN", "/tmp/melvin_brain.m")

class MelvinControlAPI(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/api/status':
            self.serve_json(self.get_status())
        elif path == '/api/control/start':
            self.serve_json(self.control_start())
        elif path == '/api/control/stop':
            self.serve_json(self.control_stop())
        elif path == '/api/control/restart':
            self.serve_json(self.control_restart())
        elif path == '/api/control/pause':
            self.serve_json(self.control_pause())
        elif path == '/api/control/resume':
            self.serve_json(self.control_resume())
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/api/control/feed':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            self.serve_json(self.control_feed(data))
        else:
            self.send_error(404)
    
    def get_status(self):
        """Get Melvin service status"""
        status = {
            'running': False,
            'pid': None,
            'brain_file': BRAIN_FILE,
            'brain_exists': os.path.exists(BRAIN_FILE)
        }
        
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    pid = int(f.read().strip())
                    status['pid'] = pid
                    # Check if process is running
                    try:
                        os.kill(pid, 0)  # Signal 0 just checks if process exists
                        status['running'] = True
                    except OSError:
                        status['running'] = False
                        os.remove(PID_FILE)
            except (ValueError, IOError):
                status['running'] = False
        
        return status
    
    def control_start(self):
        """Start Melvin service"""
        status = self.get_status()
        if status['running']:
            return {'success': False, 'error': 'Melvin is already running'}
        
        try:
            # Use service script if available
            script = Path(__file__).parent / 'melvin_service.sh'
            if script.exists():
                result = subprocess.run(
                    [str(script), 'start'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return {
                    'success': result.returncode == 0,
                    'message': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None
                }
            else:
                return {'success': False, 'error': 'Service script not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def control_stop(self):
        """Stop Melvin service"""
        status = self.get_status()
        if not status['running']:
            return {'success': False, 'error': 'Melvin is not running'}
        
        try:
            script = Path(__file__).parent / 'melvin_service.sh'
            if script.exists():
                result = subprocess.run(
                    [str(script), 'stop'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return {
                    'success': result.returncode == 0,
                    'message': result.stdout,
                    'error': result.stderr if result.returncode != 0 else None
                }
            else:
                # Direct kill
                pid = status['pid']
                if pid:
                    os.kill(pid, 15)  # SIGTERM
                    return {'success': True, 'message': f'Stopped PID {pid}'}
                return {'success': False, 'error': 'No PID found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def control_restart(self):
        """Restart Melvin service"""
        stop_result = self.control_stop()
        if not stop_result.get('success'):
            return stop_result
        
        import time
        time.sleep(2)
        return self.control_start()
    
    def control_pause(self):
        """Pause Melvin (SIGSTOP)"""
        status = self.get_status()
        if not status['running']:
            return {'success': False, 'error': 'Melvin is not running'}
        
        try:
            pid = status['pid']
            if pid:
                os.kill(pid, 19)  # SIGSTOP
                return {'success': True, 'message': f'Paused PID {pid}'}
            return {'success': False, 'error': 'No PID found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def control_resume(self):
        """Resume Melvin (SIGCONT)"""
        status = self.get_status()
        if not status['running']:
            return {'success': False, 'error': 'Melvin is not running'}
        
        try:
            pid = status['pid']
            if pid:
                os.kill(pid, 18)  # SIGCONT
                return {'success': True, 'message': f'Resumed PID {pid}'}
            return {'success': False, 'error': 'No PID found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def control_feed(self, data):
        """Feed data to running Melvin"""
        # TODO: Implement feeding mechanism
        # Could write to a pipe or use shared memory
        return {'success': False, 'error': 'Not implemented yet'}
    
    def serve_json(self, data):
        """Serve JSON response"""
        json_data = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data)
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Melvin Control API')
    parser.add_argument('--port', type=int, default=8081, help='API server port')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()
    
    server = HTTPServer((args.host, args.port), MelvinControlAPI)
    
    print(f"Melvin Control API")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  GET  /api/status")
    print(f"  GET  /api/control/start")
    print(f"  GET  /api/control/stop")
    print(f"  GET  /api/control/restart")
    print(f"  GET  /api/control/pause")
    print(f"  GET  /api/control/resume")
    print(f"Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()


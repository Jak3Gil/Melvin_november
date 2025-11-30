#!/usr/bin/env python3
"""
Melvin Dashboard App - Launches dashboard in separate window
"""

import os
import sys
import time
import webbrowser
import threading
import subprocess
from pathlib import Path
import argparse

def open_browser(url, delay=1.0):
    """Open browser in a new window after delay"""
    time.sleep(delay)
    # Try to open in a new window
    # On Mac, this opens in a new window if browser supports it
    try:
        # Try Chrome/Chromium first (better window handling)
        if sys.platform == 'darwin':
            # Mac: use open command to force new window
            subprocess.run(['open', '-na', 'Google Chrome', '--args', '--new-window', url], 
                         check=False, capture_output=True)
            subprocess.run(['open', '-na', 'Chromium', '--args', '--new-window', url], 
                         check=False, capture_output=True)
        elif sys.platform.startswith('linux'):
            # Linux: try chromium/chrome
            subprocess.run(['chromium-browser', '--new-window', url], 
                         check=False, capture_output=True)
            subprocess.run(['google-chrome', '--new-window', url], 
                         check=False, capture_output=True)
        elif sys.platform == 'win32':
            # Windows: use start command
            subprocess.run(['start', 'chrome', '--new-window', url], 
                         check=False, shell=True, capture_output=True)
    except:
        pass
    
    # Fallback to default browser
    webbrowser.open_new(url)

def main():
    parser = argparse.ArgumentParser(description='Melvin Dashboard App')
    parser.add_argument('--brain', default='/tmp/melvin_brain.m', help='Path to .m brain file')
    parser.add_argument('--port', type=int, default=8080, help='HTTP server port')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--no-browser', action='store_true', help="Don't open browser automatically")
    args = parser.parse_args()
    
    # Add tools directory to path
    tools_dir = Path(__file__).parent
    sys.path.insert(0, str(tools_dir))
    
    # Start dashboard server in background
    print("Starting Melvin Dashboard Server...")
    print(f"Brain: {args.brain}")
    print(f"Port: {args.port}")
    print("")
    
    # Import and start dashboard
    import importlib.util
    dashboard_path = tools_dir / 'melvin_dashboard.py'
    spec = importlib.util.spec_from_file_location("melvin_dashboard", dashboard_path)
    melvin_dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(melvin_dashboard)
    
    MelvinDashboard = melvin_dashboard.MelvinDashboard
    DashboardHandler = melvin_dashboard.DashboardHandler
    from http.server import HTTPServer
    
    dashboard = MelvinDashboard(brain_path=args.brain, port=args.port)
    dashboard.start_monitoring()
    DashboardHandler.dashboard = dashboard
    
    server = HTTPServer((args.host, args.port), DashboardHandler)
    
    # Open browser in new window
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        print(f"Opening dashboard in browser: {url}")
        browser_thread = threading.Thread(target=open_browser, args=(url,), daemon=True)
        browser_thread.start()
    
    print("=" * 50)
    print("Melvin Dashboard is running!")
    print(f"URL: {url}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    print("")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        dashboard.stop()
        server.shutdown()

if __name__ == '__main__':
    main()


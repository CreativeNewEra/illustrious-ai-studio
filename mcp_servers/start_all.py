#!/usr/bin/env python3
"""
Simple script to start all MCP servers.
"""

import subprocess
import sys
import time
from pathlib import Path

def start_server(script_path: str, port: int) -> subprocess.Popen:
    """Start a single MCP server."""
    env = {
        'MCP_SERVER_PORT': str(port),
        'PYTHONPATH': str(Path(__file__).parent.parent),
    }
    
    return subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

def main():
    """Start all MCP servers."""
    servers = {
        'filesystem': ('/home/ant/AI/Project/mcp_servers/filesystem_server.py', 8001),
        'web-fetch': ('/home/ant/AI/Project/mcp_servers/web_fetch_server.py', 8002),
        'git': ('/home/ant/AI/Project/mcp_servers/git_server.py', 8003),
        'image-analysis': ('/home/ant/AI/Project/mcp_servers/image_analysis_server.py', 8004),
    }
    
    processes = {}
    
    print("Starting MCP servers...")
    
    for name, (script, port) in servers.items():
        try:
            process = start_server(script, port)
            processes[name] = process
            print(f"Started {name} server on port {port} (PID: {process.pid})")
            time.sleep(0.5)  # Brief delay between starts
        except Exception as e:
            print(f"Failed to start {name}: {e}")
    
    print(f"\nStarted {len(processes)} MCP servers.")
    print("Press Ctrl+C to stop all servers...")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if any servers died
            for name, process in list(processes.items()):
                if process.poll() is not None:
                    print(f"Server {name} has stopped")
                    del processes[name]
            
            if not processes:
                print("All servers have stopped")
                break
                
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        
        for name, process in processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"Force killed {name}")
            except Exception as e:
                print(f"Error stopping {name}: {e}")

if __name__ == "__main__":
    main()
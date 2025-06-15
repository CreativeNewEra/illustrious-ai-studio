#!/usr/bin/env python3
"""
MCP Server Manager
Manages multiple MCP servers for the Illustrious AI Studio.
"""

import asyncio
import json
import logging
import os
import subprocess
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-manager")

class MCPServerManager:
    """Manages multiple MCP servers."""

    def __init__(self, config_path: str | None = None):
        default_path = os.getenv("MCP_CONFIG", "mcp_servers/config.json")
        self.config_path = Path(config_path or default_path)
        self.servers: Dict[str, subprocess.Popen] = {}
        self.config = self.load_config()
        self.running = False
        
    def load_config(self) -> Dict[str, Any]:
        """Load the MCP server configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            data = json.load(f)

        def expand_env(obj: Any) -> Any:
            if isinstance(obj, str):
                return os.path.expandvars(obj)
            if isinstance(obj, list):
                return [expand_env(v) for v in obj]
            if isinstance(obj, dict):
                return {k: expand_env(v) for k, v in obj.items()}
            return obj

        return expand_env(data)
    
    def start_server(self, name: str, config: Dict[str, Any]) -> bool:
        """Start a single MCP server."""
        if not config.get('enabled', True):
            logger.info(f"Server {name} is disabled, skipping")
            return False
        
        if name in self.servers:
            logger.warning(f"Server {name} is already running")
            return True
        
        try:
            # Prepare command
            command = [config['command']] + config.get('args', [])
            
            # Set environment variables
            env = {
                'MCP_SERVER_NAME': name,
                'MCP_SERVER_PORT': str(config.get('port', 8000)),
                'PYTHONPATH': str(Path(__file__).parent.parent),
            }
            
            # Start the process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            self.servers[name] = process
            logger.info(f"Started MCP server '{name}' (PID: {process.pid})")
            
            # Give it a moment to start
            time.sleep(1)
            
            # Check if it's still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server {name} failed to start:")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                del self.servers[name]
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server {name}: {str(e)}")
            return False
    
    def stop_server(self, name: str) -> bool:
        """Stop a single MCP server."""
        if name not in self.servers:
            logger.warning(f"Server {name} is not running")
            return False
        
        try:
            process = self.servers[name]
            
            # Try graceful shutdown first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning(f"Force killing server {name}")
                process.kill()
                process.wait()
            
            del self.servers[name]
            logger.info(f"Stopped MCP server '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop server {name}: {str(e)}")
            return False
    
    def restart_server(self, name: str) -> bool:
        """Restart a single MCP server."""
        logger.info(f"Restarting server {name}")
        
        # Stop the server
        self.stop_server(name)
        
        # Wait a moment
        time.sleep(1)
        
        # Start it again
        server_config = self.config['mcpServers'].get(name)
        if server_config:
            return self.start_server(name, server_config)
        else:
            logger.error(f"No configuration found for server {name}")
            return False
    
    def get_server_status(self, name: str) -> Dict[str, Any]:
        """Get the status of a specific server."""
        if name not in self.servers:
            return {'status': 'stopped', 'pid': None}
        
        process = self.servers[name]
        if process.poll() is None:
            return {'status': 'running', 'pid': process.pid}
        else:
            # Process has died
            del self.servers[name]
            return {'status': 'died', 'pid': None}
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all configured servers."""
        status = {}
        
        for name in self.config['mcpServers']:
            status[name] = self.get_server_status(name)
            status[name]['config'] = self.config['mcpServers'][name]
        
        return status
    
    def start_all(self) -> None:
        """Start all configured MCP servers."""
        logger.info("Starting all MCP servers...")
        
        success_count = 0
        for name, config in self.config['mcpServers'].items():
            if self.start_server(name, config):
                success_count += 1
        
        logger.info(f"Started {success_count}/{len(self.config['mcpServers'])} MCP servers")
        self.running = True
    
    def stop_all(self) -> None:
        """Stop all running MCP servers."""
        logger.info("Stopping all MCP servers...")
        
        for name in list(self.servers.keys()):
            self.stop_server(name)
        
        logger.info("All MCP servers stopped")
        self.running = False
    
    def monitor_servers(self) -> None:
        """Monitor server health and restart if needed."""
        while self.running:
            try:
                for name in list(self.servers.keys()):
                    process = self.servers[name]
                    
                    if process.poll() is not None:
                        logger.warning(f"Server {name} has died, attempting restart...")
                        
                        # Remove from servers dict
                        del self.servers[name]
                        
                        # Restart if auto_restart is enabled
                        if self.config.get('settings', {}).get('auto_restart', True):
                            server_config = self.config['mcpServers'].get(name)
                            if server_config:
                                self.start_server(name, server_config)
                
                # Check every 10 seconds
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(5)
    
    def run(self) -> None:
        """Run the MCP server manager."""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.stop_all()
            sys.exit(0)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start all servers
            self.start_all()
            
            # Start monitor in a separate thread
            monitor_thread = threading.Thread(target=self.monitor_servers, daemon=True)
            monitor_thread.start()
            
            logger.info("MCP Server Manager is running. Press Ctrl+C to stop.")
            
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop_all()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Manager")
    parser.add_argument('--config', '-c', default=os.getenv('MCP_CONFIG', 'mcp_servers/config.json'),
                       help='Path to configuration file')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Show server status and exit')
    parser.add_argument('--start', help='Start a specific server')
    parser.add_argument('--stop', help='Stop a specific server')
    parser.add_argument('--restart', help='Restart a specific server')
    
    args = parser.parse_args()
    
    manager = MCPServerManager(args.config)
    
    if args.status:
        status = manager.get_all_status()
        print("\nMCP Server Status:")
        print("=" * 50)
        for name, info in status.items():
            print(f"{name}: {info['status']}")
            if info['pid']:
                print(f"  PID: {info['pid']}")
            print(f"  Port: {info['config'].get('port', 'N/A')}")
            print(f"  Enabled: {info['config'].get('enabled', True)}")
            print()
        return
    
    if args.start:
        config = manager.config['mcpServers'].get(args.start)
        if config:
            if manager.start_server(args.start, config):
                print(f"Started server {args.start}")
            else:
                print(f"Failed to start server {args.start}")
        else:
            print(f"Server {args.start} not found in configuration")
        return
    
    if args.stop:
        if manager.stop_server(args.stop):
            print(f"Stopped server {args.stop}")
        else:
            print(f"Failed to stop server {args.stop}")
        return
    
    if args.restart:
        if manager.restart_server(args.restart):
            print(f"Restarted server {args.restart}")
        else:
            print(f"Failed to restart server {args.restart}")
        return
    
    # Default: run the manager
    manager.run()

if __name__ == "__main__":
    main()

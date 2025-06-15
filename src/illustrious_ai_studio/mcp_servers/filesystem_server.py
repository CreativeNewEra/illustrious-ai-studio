#!/usr/bin/env python3
"""
MCP Filesystem Server
Provides secure filesystem operations with configurable access controls.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filesystem-server")

# Initialize the FastMCP server
mcp = FastMCP("Filesystem Server")

# Configuration
BASE_DIR = Path(os.getenv("WORKSPACE_DIR", Path(__file__).resolve().parents[1]))
ALLOWED_DIRECTORIES = [
    str(BASE_DIR),
    "/tmp/illustrious_ai",
    str(BASE_DIR / "gallery"),
    str(BASE_DIR / "examples"),
]

def is_path_allowed(path: Path) -> bool:
    """Check if the given path is within allowed directories."""
    try:
        resolved_path = path.resolve()
        for allowed_dir in ALLOWED_DIRECTORIES:
            if resolved_path.is_relative_to(Path(allowed_dir).resolve()):
                return True
        return False
    except (OSError, ValueError):
        return False

@mcp.tool()
def read_file(path: str) -> str:
    """
    Read and return the contents of a file.
    
    Args:
        path: The path to the file to read
        
    Returns:
        The contents of the file as a string
    """
    file_path = Path(path)
    
    if not is_path_allowed(file_path):
        raise ValueError(f"Access denied: {path} is not in allowed directories")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try to read as binary and return base64 for non-text files
        import base64
        content = file_path.read_bytes()
        return f"[Binary file - base64 encoded]\n{base64.b64encode(content).decode()}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        path: The path to the file to write
        content: The content to write to the file
        
    Returns:
        Success message
    """
    file_path = Path(path)
    
    if not is_path_allowed(file_path):
        raise ValueError(f"Access denied: {path} is not in allowed directories")
    
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_path.write_text(content, encoding='utf-8')
    return f"Successfully wrote to {path}"

@mcp.tool()
def list_directory(path: str) -> List[Dict[str, Any]]:
    """
    List the contents of a directory.
    
    Args:
        path: The path to the directory to list
        
    Returns:
        A list of dictionaries containing file/directory information
    """
    dir_path = Path(path)
    
    if not is_path_allowed(dir_path):
        raise ValueError(f"Access denied: {path} is not in allowed directories")
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    items = []
    for item in sorted(dir_path.iterdir()):
        try:
            stat = item.stat()
            items.append({
                "name": item.name,
                "path": str(item),
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else None,
                "modified": stat.st_mtime,
            })
        except (OSError, ValueError):
            # Skip items that can't be accessed
            continue
    
    return items

@mcp.tool()
def create_directory(path: str) -> str:
    """
    Create a new directory.
    
    Args:
        path: The path of the directory to create
        
    Returns:
        Success message
    """
    dir_path = Path(path)
    
    if not is_path_allowed(dir_path):
        raise ValueError(f"Access denied: {path} is not in allowed directories")
    
    dir_path.mkdir(parents=True, exist_ok=True)
    return f"Successfully created directory {path}"

@mcp.tool()
def delete_file(path: str) -> str:
    """
    Delete a file.
    
    Args:
        path: The path to the file to delete
        
    Returns:
        Success message
    """
    file_path = Path(path)
    
    if not is_path_allowed(file_path):
        raise ValueError(f"Access denied: {path} is not in allowed directories")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if file_path.is_file():
        file_path.unlink()
        return f"Successfully deleted file {path}"
    else:
        raise ValueError(f"Path is not a file: {path}")

@mcp.tool()
def file_exists(path: str) -> bool:
    """
    Check if a file or directory exists.
    
    Args:
        path: The path to check
        
    Returns:
        True if the path exists, False otherwise
    """
    file_path = Path(path)
    
    if not is_path_allowed(file_path):
        return False
    
    return file_path.exists()

# Resource endpoints can be added later if needed

if __name__ == "__main__":
    mcp.run()

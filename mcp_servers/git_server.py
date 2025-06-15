#!/usr/bin/env python3
"""
MCP Git Server
Provides Git repository operations and information.
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("git-server")

# Initialize the FastMCP server
mcp = FastMCP("Git Server")

# Configuration
BASE_DIR = Path(os.getenv("WORKSPACE_DIR", Path(__file__).resolve().parents[1]))
ALLOWED_REPOSITORIES = [
    str(BASE_DIR),
]

def is_repo_allowed(repo_path: Path) -> bool:
    """Check if the given repository path is allowed."""
    try:
        resolved_path = repo_path.resolve()
        for allowed_repo in ALLOWED_REPOSITORIES:
            if resolved_path.is_relative_to(Path(allowed_repo).resolve()):
                return True
        return False
    except (OSError, ValueError):
        return False

def run_git_command(repo_path: str, command: List[str]) -> Dict[str, Any]:
    """Run a git command in the specified repository."""
    repo = Path(repo_path)
    
    if not is_repo_allowed(repo):
        raise ValueError(f"Access denied: {repo_path} is not in allowed repositories")
    
    if not repo.exists():
        raise FileNotFoundError(f"Repository not found: {repo_path}")
    
    # Check if it's a git repository
    git_dir = repo / ".git"
    if not git_dir.exists():
        raise ValueError(f"Not a git repository: {repo_path}")
    
    try:
        result = subprocess.run(
            ["git"] + command,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        raise Exception("Git command timed out")
    except Exception as e:
        raise Exception(f"Error running git command: {str(e)}")

@mcp.tool()
def git_status(repo_path: str = str(BASE_DIR)) -> Dict[str, Any]:
    """
    Get the status of a Git repository.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        Git status information
    """
    result = run_git_command(repo_path, ["status", "--porcelain", "-b"])
    
    if not result["success"]:
        raise Exception(f"Git status failed: {result['stderr']}")
    
    lines = result["stdout"].strip().split('\n') if result["stdout"].strip() else []
    
    # Parse branch info
    branch_info = {}
    files = []
    
    for line in lines:
        if line.startswith('##'):
            # Branch information
            branch_line = line[3:]  # Remove '## '
            if '...' in branch_line:
                local, remote = branch_line.split('...')[0], branch_line.split('...')[1]
                branch_info['local'] = local
                branch_info['remote'] = remote.split(' ')[0]
            else:
                branch_info['local'] = branch_line
        else:
            # File status
            status = line[:2]
            filename = line[3:]
            files.append({
                'status': status,
                'filename': filename,
                'staged': status[0] != ' ' and status[0] != '?',
                'modified': status[1] != ' ',
            })
    
    return {
        'branch': branch_info,
        'files': files,
        'clean': len(files) == 0,
    }

@mcp.tool()
def git_log(repo_path: str = str(BASE_DIR), max_count: int = 10) -> List[Dict[str, Any]]:
    """
    Get the commit history of a Git repository.
    
    Args:
        repo_path: Path to the git repository
        max_count: Maximum number of commits to return
        
    Returns:
        List of commit information
    """
    result = run_git_command(repo_path, [
        "log", 
        f"--max-count={max_count}",
        "--pretty=format:%H|%an|%ae|%ad|%s",
        "--date=iso"
    ])
    
    if not result["success"]:
        raise Exception(f"Git log failed: {result['stderr']}")
    
    commits = []
    for line in result["stdout"].strip().split('\n'):
        if line:
            parts = line.split('|', 4)
            if len(parts) == 5:
                commits.append({
                    'hash': parts[0],
                    'author_name': parts[1],
                    'author_email': parts[2],
                    'date': parts[3],
                    'message': parts[4],
                })
    
    return commits

@mcp.tool()
def git_diff(repo_path: str = str(BASE_DIR), filename: Optional[str] = None, staged: bool = False) -> str:
    """
    Get the diff of changes in a Git repository.
    
    Args:
        repo_path: Path to the git repository
        filename: Specific file to diff (optional)
        staged: Whether to show staged changes only
        
    Returns:
        Git diff output
    """
    command = ["diff"]
    if staged:
        command.append("--cached")
    if filename:
        command.append(filename)
    
    result = run_git_command(repo_path, command)
    
    if not result["success"]:
        raise Exception(f"Git diff failed: {result['stderr']}")
    
    return result["stdout"]

@mcp.tool()
def git_branch_list(repo_path: str = str(BASE_DIR)) -> List[Dict[str, Any]]:
    """
    List all branches in a Git repository.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        List of branch information
    """
    result = run_git_command(repo_path, ["branch", "-a", "-v"])
    
    if not result["success"]:
        raise Exception(f"Git branch failed: {result['stderr']}")
    
    branches = []
    for line in result["stdout"].strip().split('\n'):
        if line:
            line = line.strip()
            current = line.startswith('*')
            if current:
                line = line[2:]  # Remove '* '
            
            parts = line.split()
            if len(parts) >= 2:
                branches.append({
                    'name': parts[0],
                    'hash': parts[1],
                    'current': current,
                    'remote': parts[0].startswith('remotes/'),
                })
    
    return branches

@mcp.tool()
def git_show_commit(repo_path: str = str(BASE_DIR), commit_hash: str = "HEAD") -> Dict[str, Any]:
    """
    Show detailed information about a specific commit.
    
    Args:
        repo_path: Path to the git repository
        commit_hash: The commit hash to show (default: HEAD)
        
    Returns:
        Detailed commit information
    """
    # Get commit info
    result = run_git_command(repo_path, [
        "show", 
        "--pretty=format:%H|%an|%ae|%ad|%s|%b",
        "--date=iso",
        "--name-status",
        commit_hash
    ])
    
    if not result["success"]:
        raise Exception(f"Git show failed: {result['stderr']}")
    
    lines = result["stdout"].split('\n')
    
    # Parse commit info (first line)
    commit_line = lines[0]
    parts = commit_line.split('|', 5)
    
    commit_info = {
        'hash': parts[0],
        'author_name': parts[1],
        'author_email': parts[2],
        'date': parts[3],
        'subject': parts[4],
        'body': parts[5] if len(parts) > 5 else '',
        'files': []
    }
    
    # Parse file changes
    for line in lines[1:]:
        if line and '\t' in line:
            status, filename = line.split('\t', 1)
            commit_info['files'].append({
                'status': status,
                'filename': filename
            })
    
    return commit_info

@mcp.tool()
def git_remote_info(repo_path: str = str(BASE_DIR)) -> List[Dict[str, str]]:
    """
    Get information about remote repositories.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        List of remote repository information
    """
    result = run_git_command(repo_path, ["remote", "-v"])
    
    if not result["success"]:
        raise Exception(f"Git remote failed: {result['stderr']}")
    
    remotes = []
    for line in result["stdout"].strip().split('\n'):
        if line:
            parts = line.split()
            if len(parts) >= 3:
                remotes.append({
                    'name': parts[0],
                    'url': parts[1],
                    'type': parts[2].strip('()')
                })
    
    return remotes

@mcp.tool()
def git_file_history(filename: str, repo_path: str = str(BASE_DIR), max_count: int = 10) -> List[Dict[str, Any]]:
    """
    Get the commit history for a specific file.
    
    Args:
        repo_path: Path to the git repository
        filename: The file to get history for
        max_count: Maximum number of commits to return
        
    Returns:
        List of commits that modified the file
    """
    result = run_git_command(repo_path, [
        "log",
        f"--max-count={max_count}",
        "--pretty=format:%H|%an|%ae|%ad|%s",
        "--date=iso",
        "--",
        filename
    ])
    
    if not result["success"]:
        raise Exception(f"Git log failed: {result['stderr']}")
    
    commits = []
    for line in result["stdout"].strip().split('\n'):
        if line:
            parts = line.split('|', 4)
            if len(parts) == 5:
                commits.append({
                    'hash': parts[0],
                    'author_name': parts[1],
                    'author_email': parts[2],
                    'date': parts[3],
                    'message': parts[4],
                    'filename': filename,
                })
    
    return commits

if __name__ == "__main__":
    mcp.run()
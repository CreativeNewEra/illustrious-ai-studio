# MCP Servers for Illustrious AI Studio

This directory contains Model Context Protocol (MCP) servers that extend the capabilities of the Illustrious AI Studio with additional tools and resources.

## Available MCP Servers

### 1. Filesystem Server (Port 8001)
**File:** `filesystem_server.py`

Provides secure filesystem operations with configurable access controls.

**Tools:**
- `read_file(path)` - Read file contents
- `write_file(path, content)` - Write content to file
- `list_directory(path)` - List directory contents
- `create_directory(path)` - Create new directory
- `delete_file(path)` - Delete a file
- `file_exists(path)` - Check if file/directory exists

**Security:** Only allows access to specified directories for safety.

### 2. Web Fetch Server (Port 8002)
**File:** `web_fetch_server.py`

Provides web content fetching and conversion for efficient LLM usage.

**Tools:**
- `fetch_url(url, extract_content=True)` - Fetch and clean web content
- `extract_links(url, filter_domain=False)` - Extract all links from a webpage
- `search_content(url, search_term, context_chars=200)` - Search within webpage content
- `get_page_metadata(url)` - Extract page metadata (title, description, etc.)

**Features:** Converts HTML to markdown, cleans content, handles timeouts.

### 3. Git Server (Port 8003)
**File:** `git_server.py`

Provides Git repository operations and information.

**Tools:**
- `git_status(repo_path)` - Get repository status
- `git_log(repo_path, max_count=10)` - Get commit history
- `git_diff(repo_path, filename=None, staged=False)` - Show diffs
- `git_branch_list(repo_path)` - List all branches
- `git_show_commit(repo_path, commit_hash="HEAD")` - Show commit details
- `git_remote_info(repo_path)` - Get remote repository info
- `git_file_history(repo_path, filename, max_count=10)` - Get file commit history

**Security:** Only allows access to specified repository paths.

### 4. Image Analysis Server (Port 8004)
**File:** `image_analysis_server.py`

Provides image analysis capabilities for the AI Studio.

**Tools:**
- `analyze_image_properties(image_path/image_base64)` - Analyze basic image properties
- `extract_image_colors(image_path/image_base64, num_colors=5)` - Extract dominant colors
- `compare_images(image1, image2)` - Compare two images
- `create_image_thumbnail(image, size=256, output_path=None)` - Create thumbnails
- `get_image_metadata(image_path)` - Extract image metadata and EXIF data

**Features:** Supports both file paths and base64 input, color analysis, metadata extraction.

## Setup and Configuration

### Installation

The required dependencies are installed automatically:

```bash
pip install mcp beautifulsoup4 markdownify
```

### Configuration

MCP servers are configured in `config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["${MCP_DIR}/filesystem_server.py"],
      "description": "Secure filesystem operations",
      "port": 8001,
      "enabled": true
    }
  }
}
```

### Security Configuration

The servers include security controls:

```json
{
  "settings": {
    "allowed_directories": [
      "${WORKSPACE_DIR}",
      "/tmp/illustrious_ai",
      "${WORKSPACE_DIR}/gallery",
      "${WORKSPACE_DIR}/examples"
    ]
  }
}
```

## Running MCP Servers

### Option 1: Start All Servers (Recommended)

```bash
cd mcp_servers
python start_all.py
```

This starts all enabled servers and monitors them.

### Option 2: Use the Manager

```bash
cd mcp_servers

# Start all servers
python manager.py

# Check status
python manager.py --status

# Start specific server
python manager.py --start filesystem

# Stop specific server
python manager.py --stop filesystem

# Restart specific server
python manager.py --restart filesystem
```

### Option 3: Start Individual Servers

```bash
cd mcp_servers

# Start filesystem server
python filesystem_server.py

# Start web fetch server  
python web_fetch_server.py

# Start git server
python git_server.py

# Start image analysis server
python image_analysis_server.py
```

## Integration with AI Studio

Once the MCP servers are running, they can be accessed:

1. **Via the main AI Studio** - The web interface can integrate with MCP servers
2. **Via API calls** - Make HTTP requests to the server endpoints
3. **Via MCP protocol** - Use standard MCP client libraries

### Example API Usage

```python
import httpx

# Using filesystem server
response = httpx.post("http://localhost:8001/tools/read_file",
                     json={"arguments": {"path": "${WORKSPACE_DIR}/README.md"}})

# Using web fetch server
response = httpx.post("http://localhost:8002/tools/fetch_url",
                     json={"arguments": {"url": "https://example.com"}})

# Using git server
response = httpx.post("http://localhost:8003/tools/git_status",
                     json={"arguments": {"repo_path": "${WORKSPACE_DIR}"}})
```

### Using MCP Tools from Chat

In the AI Studio chat box you can call these tools directly:

```text
/tool filesystem.read_file path=/tmp/foo.txt
```

The command format is `/tool <server>.<method> key=value`.

## Monitoring and Logging

All servers include logging and can be monitored:

```bash
# Check if servers are running
netstat -tlnp | grep -E '800[1-4]'

# View logs
journalctl -f | grep mcp

# Check individual server status
curl http://localhost:8001/health
curl http://localhost:8002/health  
curl http://localhost:8003/health
curl http://localhost:8004/health
```

## Troubleshooting

### Common Issues

1. **Port conflicts** - Change ports in `config.json` if needed
2. **Permission errors** - Check `allowed_directories` in config
3. **Import errors** - Ensure all dependencies are installed
4. **Server crashes** - Check logs for error details

### Server Health Checks

Each server provides basic health endpoints and error handling. The manager script includes automatic restart capabilities for failed servers.

### Dependencies

- **mcp** - Model Context Protocol SDK
- **beautifulsoup4** - HTML parsing for web fetch server
- **markdownify** - HTML to markdown conversion
- **PIL (Pillow)** - Image processing for image analysis server
- **httpx** - HTTP client for web requests

## Development

To add new MCP servers:

1. Create a new Python file in this directory
2. Use the FastMCP framework for quick setup
3. Add configuration to `config.json`
4. Update this README with the new server details

The existing servers serve as good templates for new implementations.
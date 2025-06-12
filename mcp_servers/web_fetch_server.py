#!/usr/bin/env python3
"""
MCP Web Fetch Server
Provides web content fetching and conversion for efficient LLM usage.
Enhanced with rate limiting and configurable user agent spoofing.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin
import re
import json
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-fetch-server")

# Initialize the FastMCP server
mcp = FastMCP("Web Fetch Server")

# Configuration - can be loaded from config file
CONFIG_FILE = Path("mcp_servers/web_fetch_config.json")

# Default configuration
DEFAULT_CONFIG = {
    "user_agents": {
        "default": "Illustrious AI Studio MCP Web Fetch Server/1.0",
        "browser": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "mobile": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    },
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 30,
        "requests_per_hour": 500
    },
    "timeout": 30,
    "max_content_length": 10485760,  # 10MB
    "allowed_domains": [],  # Empty means all domains allowed
    "blocked_domains": ["localhost", "127.0.0.1", "0.0.0.0"]
}

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from file or return defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults
            config = DEFAULT_CONFIG.copy()
            config.update(user_config)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}, using defaults")
    return DEFAULT_CONFIG.copy()

# Create config file if it doesn't exist
def create_default_config():
    """Create default configuration file."""
    if not CONFIG_FILE.exists():
        CONFIG_FILE.parent.mkdir(exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        logger.info(f"Created default config file: {CONFIG_FILE}")

# Initialize config
create_default_config()
CONFIG = load_config()

# Rate limiting storage
class RateLimiter:
    def __init__(self, requests_per_minute: int, requests_per_hour: int):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = []
        self.hour_requests = []
    
    def can_make_request(self) -> bool:
        """Check if a request can be made based on rate limits."""
        now = time.time()
        
        # Clean old requests
        self.minute_requests = [req_time for req_time in self.minute_requests if now - req_time < 60]
        self.hour_requests = [req_time for req_time in self.hour_requests if now - req_time < 3600]
        
        # Check limits
        if len(self.minute_requests) >= self.requests_per_minute:
            return False
        if len(self.hour_requests) >= self.requests_per_hour:
            return False
        
        return True
    
    def record_request(self):
        """Record a new request."""
        now = time.time()
        self.minute_requests.append(now)
        self.hour_requests.append(now)

# Initialize rate limiter
rate_limiter = RateLimiter(
    CONFIG['rate_limiting']['requests_per_minute'],
    CONFIG['rate_limiting']['requests_per_hour']
) if CONFIG['rate_limiting']['enabled'] else None

def clean_html_content(html: str) -> str:
    """Clean and extract meaningful content from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script, style, and other non-content elements
    for script in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement"]):
        script.decompose()
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
        comment.extract()
    
    # Convert to markdown for better LLM consumption
    markdown_content = md(str(soup), heading_style="ATX")
    
    # Clean up excessive whitespace
    cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
    cleaned_content = re.sub(r' +', ' ', cleaned_content)
    
    return cleaned_content.strip()

@mcp.tool()
async def fetch_url(url: str, extract_content: bool = True, user_agent_type: str = "default") -> Dict[str, Any]:
    """
    Fetch content from a URL and optionally extract meaningful text.
    
    Args:
        url: The URL to fetch
        extract_content: Whether to extract and clean content (default: True)
        user_agent_type: Type of user agent to use ("default", "browser", "mobile")
        
    Returns:
        Dictionary containing the fetched content and metadata
    """
    # Check rate limiting
    if rate_limiter and not rate_limiter.can_make_request():
        raise Exception("Rate limit exceeded. Please wait before making more requests.")
    
    # Validate URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {url}")
    
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}")
    
    # Check domain restrictions
    domain = parsed_url.netloc.lower()
    if CONFIG['blocked_domains'] and any(blocked in domain for blocked in CONFIG['blocked_domains']):
        raise ValueError(f"Access to domain {domain} is blocked")
    
    if CONFIG['allowed_domains'] and not any(allowed in domain for allowed in CONFIG['allowed_domains']):
        raise ValueError(f"Access to domain {domain} is not allowed")
    
    # Select user agent
    user_agent = CONFIG['user_agents'].get(user_agent_type, CONFIG['user_agents']['default'])
    
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
    }
    
    async with httpx.AsyncClient(timeout=CONFIG['timeout']) as client:
        try:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            # Record successful request for rate limiting
            if rate_limiter:
                rate_limiter.record_request()
            
            # Check content length
            content_length = len(response.content)
            if content_length > CONFIG['max_content_length']:
                raise ValueError(f"Content too large: {content_length} bytes (max: {CONFIG['max_content_length']})")
            
            # Get content type
            content_type = response.headers.get('content-type', '').lower()
            
            result = {
                'url': str(response.url),
                'status_code': response.status_code,
                'content_type': content_type,
                'content_length': content_length,
                'title': None,
                'raw_content': response.text,
                'cleaned_content': None,
                'user_agent_used': user_agent,
            }
            
            if extract_content and 'text/html' in content_type:
                # Extract title
                soup = BeautifulSoup(response.text, 'html.parser')
                title_tag = soup.find('title')
                if title_tag:
                    result['title'] = title_tag.get_text().strip()
                
                # Clean and extract content
                result['cleaned_content'] = clean_html_content(response.text)
            
            logger.info(f"Successfully fetched {url} ({content_length} bytes)")
            return result
            
        except httpx.TimeoutException:
            raise Exception(f"Request timeout while fetching: {url}")
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error {e.response.status_code} while fetching: {url}")
        except Exception as e:
            raise Exception(f"Error fetching {url}: {str(e)}")

@mcp.tool()
async def extract_links(url: str, filter_domain: bool = False) -> List[Dict[str, str]]:
    """
    Extract all links from a webpage.
    
    Args:
        url: The URL to extract links from
        filter_domain: Whether to only return links from the same domain
        
    Returns:
        List of dictionaries containing link information
    """
    content_data = await fetch_url(url, extract_content=False)
    
    soup = BeautifulSoup(content_data['raw_content'], 'html.parser')
    base_domain = urlparse(url).netloc
    
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        absolute_url = urljoin(url, href)
        link_domain = urlparse(absolute_url).netloc
        
        # Filter by domain if requested
        if filter_domain and link_domain != base_domain:
            continue
        
        link_text = link.get_text().strip()
        if not link_text:
            link_text = href
        
        links.append({
            'url': absolute_url,
            'text': link_text,
            'domain': link_domain,
        })
    
    return links

@mcp.tool()
async def search_content(url: str, search_term: str, context_chars: int = 200) -> List[Dict[str, str]]:
    """
    Search for specific content within a webpage.
    
    Args:
        url: The URL to search within
        search_term: The term to search for
        context_chars: Number of characters to include around matches
        
    Returns:
        List of dictionaries containing search results with context
    """
    content_data = await fetch_url(url, extract_content=True)
    
    if not content_data['cleaned_content']:
        raise ValueError("No content available to search")
    
    content = content_data['cleaned_content']
    search_term_lower = search_term.lower()
    content_lower = content.lower()
    
    matches = []
    start = 0
    
    while True:
        pos = content_lower.find(search_term_lower, start)
        if pos == -1:
            break
        
        # Get context around the match
        context_start = max(0, pos - context_chars)
        context_end = min(len(content), pos + len(search_term) + context_chars)
        
        context = content[context_start:context_end]
        
        # Highlight the match (using markdown bold)
        highlighted_context = context.replace(
            content[pos:pos + len(search_term)],
            f"**{content[pos:pos + len(search_term)]}**"
        )
        
        matches.append({
            'position': pos,
            'context': highlighted_context,
            'url': content_data['url'],
        })
        
        start = pos + 1
    
    return matches

@mcp.tool()
async def get_page_metadata(url: str) -> Dict[str, Any]:
    """
    Extract metadata from a webpage (title, description, etc.).
    
    Args:
        url: The URL to extract metadata from
        
    Returns:
        Dictionary containing page metadata
    """
    content_data = await fetch_url(url, extract_content=False)
    
    soup = BeautifulSoup(content_data['raw_content'], 'html.parser')
    
    metadata = {
        'url': content_data['url'],
        'title': None,
        'description': None,
        'keywords': None,
        'author': None,
        'og_title': None,
        'og_description': None,
        'og_image': None,
    }
    
    # Extract basic metadata
    title = soup.find('title')
    if title:
        metadata['title'] = title.get_text().strip()
    
    # Meta tags
    for meta in soup.find_all('meta'):
        name = meta.get('name', '').lower()
        property_name = meta.get('property', '').lower()
        content = meta.get('content', '')
        
        if name == 'description':
            metadata['description'] = content
        elif name == 'keywords':
            metadata['keywords'] = content
        elif name == 'author':
            metadata['author'] = content
        elif property_name == 'og:title':
            metadata['og_title'] = content
        elif property_name == 'og:description':
            metadata['og_description'] = content
        elif property_name == 'og:image':
            metadata['og_image'] = content
    
    return metadata

if __name__ == "__main__":
    mcp.run()
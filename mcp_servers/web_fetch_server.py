#!/usr/bin/env python3
"""
MCP Web Fetch Server
Provides web content fetching and conversion for efficient LLM usage.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin
import re

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-fetch-server")

# Initialize the FastMCP server
mcp = FastMCP("Web Fetch Server")

# Configuration
USER_AGENT = "Illustrious AI Studio MCP Web Fetch Server/1.0"
TIMEOUT = 30
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB limit

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
async def fetch_url(url: str, extract_content: bool = True) -> Dict[str, Any]:
    """
    Fetch content from a URL and optionally extract meaningful text.
    
    Args:
        url: The URL to fetch
        extract_content: Whether to extract and clean content (default: True)
        
    Returns:
        Dictionary containing the fetched content and metadata
    """
    # Validate URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {url}")
    
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}")
    
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            # Check content length
            content_length = len(response.content)
            if content_length > MAX_CONTENT_LENGTH:
                raise ValueError(f"Content too large: {content_length} bytes (max: {MAX_CONTENT_LENGTH})")
            
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
            }
            
            if extract_content and 'text/html' in content_type:
                # Extract title
                soup = BeautifulSoup(response.text, 'html.parser')
                title_tag = soup.find('title')
                if title_tag:
                    result['title'] = title_tag.get_text().strip()
                
                # Clean and extract content
                result['cleaned_content'] = clean_html_content(response.text)
            
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
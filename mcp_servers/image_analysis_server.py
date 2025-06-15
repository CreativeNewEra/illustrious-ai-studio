#!/usr/bin/env python3
"""
MCP Image Analysis Server
Provides image analysis capabilities for the AI Studio.
"""

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageStat
import torch
from mcp.server.fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-analysis-server")

# Initialize the FastMCP server
mcp = FastMCP("Image Analysis Server")

# Configuration
BASE_DIR = Path(os.getenv("WORKSPACE_DIR", Path(__file__).resolve().parents[1]))
ALLOWED_DIRECTORIES = [
    str(BASE_DIR / "gallery"),
    "/tmp/illustrious_ai/gallery",
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

def load_image_from_path(image_path: str) -> Image.Image:
    """Load an image from a file path."""
    path = Path(image_path)
    
    if not is_path_allowed(path):
        raise ValueError(f"Access denied: {image_path} is not in allowed directories")
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        return Image.open(path)
    except Exception as e:
        raise ValueError(f"Unable to load image: {str(e)}")

def load_image_from_base64(base64_data: str) -> Image.Image:
    """Load an image from base64 data."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]
        
        image_data = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Unable to decode base64 image: {str(e)}")

@mcp.tool()
def analyze_image_properties(image_path: Optional[str] = None, image_base64: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze basic properties of an image.
    
    Args:
        image_path: Path to the image file (optional)
        image_base64: Base64 encoded image data (optional)
        
    Returns:
        Dictionary containing image properties
    """
    if not image_path and not image_base64:
        raise ValueError("Either image_path or image_base64 must be provided")
    
    if image_path and image_base64:
        raise ValueError("Only one of image_path or image_base64 should be provided")
    
    # Load image
    if image_path:
        image = load_image_from_path(image_path)
    else:
        image = load_image_from_base64(image_base64)
    
    # Basic properties
    properties = {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format,
        'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
        'aspect_ratio': round(image.width / image.height, 3),
    }
    
    # Color analysis
    if image.mode in ('RGB', 'RGBA'):
        # Convert to RGB if RGBA
        rgb_image = image.convert('RGB')
        
        # Get color statistics
        stat = ImageStat.Stat(rgb_image)
        properties['color_stats'] = {
            'mean_rgb': [round(val, 2) for val in stat.mean],
            'median_rgb': [round(val, 2) for val in stat.median],
            'stddev_rgb': [round(val, 2) for val in stat.stddev],
        }
        
        # Brightness analysis
        grayscale = rgb_image.convert('L')
        brightness = ImageStat.Stat(grayscale).mean[0]
        properties['brightness'] = round(brightness, 2)
        properties['brightness_category'] = (
            'dark' if brightness < 85 else
            'medium' if brightness < 170 else
            'bright'
        )
    
    # File size if from path
    if image_path:
        file_size = Path(image_path).stat().st_size
        properties['file_size_bytes'] = file_size
        properties['file_size_mb'] = round(file_size / (1024 * 1024), 2)
    
    return properties

@mcp.tool()
def extract_image_colors(image_path: Optional[str] = None, image_base64: Optional[str] = None, num_colors: int = 5) -> List[Dict[str, Any]]:
    """
    Extract dominant colors from an image.
    
    Args:
        image_path: Path to the image file (optional)
        image_base64: Base64 encoded image data (optional)
        num_colors: Number of dominant colors to extract
        
    Returns:
        List of dominant colors with RGB values and percentages
    """
    if not image_path and not image_base64:
        raise ValueError("Either image_path or image_base64 must be provided")
    
    if image_path and image_base64:
        raise ValueError("Only one of image_path or image_base64 should be provided")
    
    # Load and convert image
    if image_path:
        image = load_image_from_path(image_path)
    else:
        image = load_image_from_base64(image_base64)
    
    # Convert to RGB
    rgb_image = image.convert('RGB')
    
    # Resize for faster processing
    max_size = 150
    if rgb_image.width > max_size or rgb_image.height > max_size:
        rgb_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Get all pixels
    pixels = list(rgb_image.getdata())
    
    # Count color frequencies
    color_counts = {}
    for pixel in pixels:
        color_counts[pixel] = color_counts.get(pixel, 0) + 1
    
    # Sort by frequency and get top colors
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    top_colors = sorted_colors[:num_colors]
    
    total_pixels = len(pixels)
    
    result = []
    for color, count in top_colors:
        percentage = round((count / total_pixels) * 100, 2)
        result.append({
            'rgb': list(color),
            'hex': '#{:02x}{:02x}{:02x}'.format(*color),
            'percentage': percentage,
            'pixel_count': count,
        })
    
    return result

@mcp.tool()
def compare_images(image1_path: Optional[str] = None, image1_base64: Optional[str] = None,
                   image2_path: Optional[str] = None, image2_base64: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare two images and analyze their differences.
    
    Args:
        image1_path: Path to the first image (optional)
        image1_base64: Base64 data for the first image (optional)
        image2_path: Path to the second image (optional)
        image2_base64: Base64 data for the second image (optional)
        
    Returns:
        Dictionary containing comparison results
    """
    # Validate inputs
    if not (image1_path or image1_base64) or not (image2_path or image2_base64):
        raise ValueError("Both images must be provided")
    
    # Load images
    if image1_path:
        image1 = load_image_from_path(image1_path)
    else:
        image1 = load_image_from_base64(image1_base64)
    
    if image2_path:
        image2 = load_image_from_path(image2_path)
    else:
        image2 = load_image_from_base64(image2_base64)
    
    # Basic comparison
    comparison = {
        'same_dimensions': image1.size == image2.size,
        'same_mode': image1.mode == image2.mode,
        'image1_size': image1.size,
        'image2_size': image2.size,
        'size_difference': {
            'width_diff': image2.width - image1.width,
            'height_diff': image2.height - image1.height,
        }
    }
    
    # If same size, we can do pixel comparison
    if image1.size == image2.size:
        # Convert both to RGB for comparison
        rgb1 = image1.convert('RGB')
        rgb2 = image2.convert('RGB')
        
        # Calculate pixel differences
        pixels1 = list(rgb1.getdata())
        pixels2 = list(rgb2.getdata())
        
        total_pixels = len(pixels1)
        different_pixels = 0
        total_difference = 0
        
        for p1, p2 in zip(pixels1, pixels2):
            if p1 != p2:
                different_pixels += 1
                # Calculate Euclidean distance between RGB values
                diff = sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2)) ** 0.5
                total_difference += diff
        
        comparison['pixel_comparison'] = {
            'identical_pixels': total_pixels - different_pixels,
            'different_pixels': different_pixels,
            'similarity_percentage': round(((total_pixels - different_pixels) / total_pixels) * 100, 2),
            'average_pixel_difference': round(total_difference / total_pixels, 2) if total_pixels > 0 else 0,
        }
    
    return comparison

@mcp.tool()
def create_image_thumbnail(image_path: Optional[str] = None, image_base64: Optional[str] = None,
                          size: int = 256, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a thumbnail of an image.
    
    Args:
        image_path: Path to the source image (optional)
        image_base64: Base64 encoded source image (optional)
        size: Maximum size for the thumbnail (width or height)
        output_path: Path to save the thumbnail (optional)
        
    Returns:
        Dictionary containing thumbnail information and base64 data
    """
    if not image_path and not image_base64:
        raise ValueError("Either image_path or image_base64 must be provided")
    
    # Load image
    if image_path:
        image = load_image_from_path(image_path)
    else:
        image = load_image_from_base64(image_base64)
    
    # Create thumbnail
    thumbnail = image.copy()
    thumbnail.thumbnail((size, size), Image.Resampling.LANCZOS)
    
    # Save to output path if provided
    if output_path:
        output_path_obj = Path(output_path)
        if not is_path_allowed(output_path_obj.parent):
            raise ValueError(f"Access denied: {output_path} is not in allowed directories")
        
        thumbnail.save(output_path)
    
    # Convert to base64
    buffer = io.BytesIO()
    thumbnail.save(buffer, format='PNG')
    thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        'original_size': image.size,
        'thumbnail_size': thumbnail.size,
        'thumbnail_base64': thumbnail_base64,
        'saved_to': output_path if output_path else None,
    }

@mcp.tool()
def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract metadata from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing image metadata
    """
    image = load_image_from_path(image_path)
    
    metadata = {
        'filename': Path(image_path).name,
        'format': image.format,
        'mode': image.mode,
        'size': image.size,
        'info': dict(image.info) if image.info else {},
    }
    
    # EXIF data for JPEG images
    if hasattr(image, '_getexif') and image._getexif():
        try:
            from PIL.ExifTags import TAGS
            exif_data = {}
            for tag_id, value in image._getexif().items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
            metadata['exif'] = exif_data
        except Exception:
            metadata['exif'] = {}
    
    return metadata

if __name__ == "__main__":
    mcp.run()
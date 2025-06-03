def create_docker_requirements():
    """Create requirements.txt for Docker"""
    
    requirements_content = '''torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
diffusers==0.24.0
transformers==4.36.0
accelerate==0.25.0
gradio==4.8.0
fastapi==0.104.1
uvicorn==0.24.0
Pillow==10.1.0
requests==2.31.0
pydantic==2.5.0
safetensors==0.4.0
'''
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Created requirements.txt")

def create_deployment_guide():
    """Create deployment guide"""
    
    guide_content = '''# 🐳 Docker Deployment Guide

## Quick Start

1. **Prepare your models:**
   ```bash
   mkdir -p models/sdxl
   # Copy your SDXL model (.safetensors) to models/sdxl/
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Access AI Studio:**
   - Web UI: http://localhost:7860
   - API: http://localhost:8000

## Detailed Setup

### Prerequisites
- Docker with GPU support (nvidia-docker2)
- CUDA-compatible GPU with 12GB+ VRAM
- 20GB+ disk space

### Build Options

**Option 1: Docker Compose (Recommended)**
```bash
# Build and start
docker-compose up --build -d

# View logs
docker-compose logs -f ai-studio

# Stop
docker-compose down
```

**Option 2: Docker Build**
```bash
# Build image
docker build -t illustrious-ai-studio .

# Run container
docker run -p 7860:7860 -p 8000:8000 -p 11434:11434 \\
  --gpus all \\
  -v ./models:/app/models:ro \\
  -v ./gallery:/app/tmp/illustrious_ai/gallery \\
  illustrious-ai-studio
```

### Environment Variables

```bash
# GPU selection
CUDA_VISIBLE_DEVICES=0

# Ollama configuration
OLLAMA_HOST=0.0.0.0
OLLAMA_MODELS=/app/models/ollama

# Application settings
MODEL_PATH=/app/models/sdxl/your_model.safetensors
```

### Volume Mounts

- `./models:/app/models:ro` - Model files (read-only)
- `./gallery:/app/tmp/illustrious_ai/gallery` - Generated images
- `./examples:/app/examples:ro` - Example files

### Health Checks

```bash
# Check if services are running
curl http://localhost:7860  # Gradio UI
curl http://localhost:8000/status  # MCP Server
curl http://localhost:11434/api/version  # Ollama
```

### Troubleshooting

**GPU not detected:**
```bash
# Check nvidia-docker installation
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

**Out of memory:**
```bash
# Reduce model precision or use CPU offloading
# Update config.yaml before building
```

**Ollama models not downloading:**
```bash
# Manual model pull
docker exec -it illustrious-ai-studio ollama pull qwen2.5:7b
```
'''
    
    with open("deployment_guide.md", "w") as f:
        f.write(guide_content)
    
    print("✅ Created deployment_guide.md")

def create_production_config():
    """Create production configuration"""
    
    prod_config = '''# Production Configuration for AI Studio

# Nginx reverse proxy configuration
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # Gradio UI
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Gradio
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # API endpoints
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for image generation
        proxy_read_timeout 300s;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
    limit_req zone=api burst=5 nodelay;
    
    # File upload size limit
    client_max_body_size 50M;
}

# Systemd service file
[Unit]
Description=AI Studio Container
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ai-studio
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
'''
    
    with open("production_config.conf", "w") as f:
        f.write(prod_config)
    
    print("✅ Created production_config.conf")

def main():
    """Generate all Docker deployment files"""
    print("🐳 Creating Docker deployment configuration...")
    print("=" * 50)
    
    create_dockerfile()
    create_docker_compose()
    create_startup_script()
    create_docker_requirements()
    create_deployment_guide()
    create_production_config()
    
    print("\n🎉 Docker deployment files created successfully!")
    print("\nNext steps:")
    print("1. Copy your SDXL model to ./models/sdxl/")
    print("2. Run: docker-compose up --build")
    print("3. Access AI Studio at http://localhost:7860")
    print("\nFor production deployment, see deployment_guide.md")

if __name__ == "__main__":
    main()
```

---

## 📚 **8. Documentation Examples**

### **examples/docs/api_reference.md**
```markdown
# 🔌 AI Studio API Reference

Complete reference for the AI Studio MCP (Model Context Protocol) server API.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required for local deployment.

---

## Endpoints

### GET /status
**Check server and model status**

**Response:**
```json
{
  "status": "running",
  "models": {
    "sdxl": true,
    "ollama": true,
    "multimodal": false
  },
  "cuda_available": true
}
```

**Example:**
```bash
curl http://localhost:8000/status
```

---

### POST /generate-image
**Generate images from text prompts**

**Request Body:**
```json
{
  "prompt": "string (required)",
  "negative_prompt": "string (optional)",
  "steps": "integer (10-100, default: 30)",
  "guidance": "float (1.0-20.0, default: 7.5)",
  "seed": "integer (-1 for random, default: -1)"
}
```

**Response:**
```json
{
  "success": true,
  "image_base64": "string (base64 encoded PNG)",
  "message": "Image generated successfully! Seed: 12345"
}
```

**Error Response:**
```json
{
  "detail": "Error message"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/generate-image \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "steps": 30,
    "guidance": 7.5
  }'
```

---

### POST /chat
**Chat with the language model**

**Request Body:**
```json
{
  "message": "string (required)",
  "session_id": "string (default: 'default')",
  "temperature": "float (0.0-2.0, default: 0.7)",
  "max_tokens": "integer (1-2048, default: 256)"
}
```

**Response:**
```json
{
  "response": "AI response text",
  "session_id": "session_identifier"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Create a creative prompt for a dragon",
    "temperature": 0.8
  }'
```

---

### POST /analyze-image
**Analyze images with multimodal AI** *(Requires vision-capable model)*

**Request Body:**
```json
{
  "image_base64": "string (base64 encoded image)",
  "question": "string (default: 'Describe this image in detail')"
}
```

**Response:**
```json
{
  "analysis": "Detailed description of the image..."
}
```

**Example:**
```python
import base64
import requests

# Encode image
with open("image.png", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Analyze
response = requests.post("http://localhost:8000/analyze-image", json={
    "image_base64": img_base64,
    "question": "What objects are in this image?"
})
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 503 | Service Unavailable - Model not loaded |
| 500 | Internal Server Error - Generation failed |

---

## Rate Limits
No rate limiting in default configuration. For production, implement reverse proxy rate limiting.

---

## Examples

### Python Client
```python
import requests
import base64
from PIL import Image
import io

class AIStudioClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def generate(self, prompt, **kwargs):
        response = requests.post(f"{self.base_url}/generate-image", json={
            "prompt": prompt,
            **kwargs
        })
        
        if response.status_code == 200:
            data = response.json()
            img_data = base64.b64decode(data["image_base64"])
            return Image.open(io.BytesIO(img_data))
        return None
    
    def chat(self, message, **kwargs):
        response = requests.post(f"{self.base_url}/chat", json={
            "message": message,
            **kwargs
        })
        
        if response.status_code == 200:
            return response.json()["response"]
        return None

# Usage
client = AIStudioClient()
image = client.generate("a cute robot cat")
response = client.chat("Hello, how are you?")
```

### JavaScript Client
```javascript
class AIStudioClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async generate(prompt, options = {}) {
        const response = await fetch(this.baseUrl + '/generate-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, ...options })
        });
        
        if (response.ok) {
            const data = await response.json();
            return 'data:image/png;base64,' + data.image_base64;
        }
        return null;
    }
    
    async chat(message, options = {}) {
        const response = await fetch(this.baseUrl + '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, ...options })
        });
        
        if (response.ok) {
            const data = await response.json();
            return data.response;
        }
        return null;
    }
}

// Usage
const client = new AIStudioClient();
client.generate('a beautiful landscape').then(imageUrl => {
    document.getElementById('image').src = imageUrl;
});
```
```

### **examples/docs/troubleshooting.md**
```markdown
# 🛠️ Troubleshooting Guide

Common issues and solutions for AI Studio setup and usage.

---

## Installation Issues

### CUDA Out of Memory
**Problem:** GPU runs out of memory during model loading or generation.

**Solutions:**
```python
# 1. Enable model CPU offloading (in app.py)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()  # More aggressive

# 2. Use lower precision
pipe = pipe.to(torch.float16)  # or torch.bfloat16

# 3. Reduce batch size and resolution
# Generate smaller images first

# 4. Set memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Ollama Connection Failed
**Problem:** Cannot connect to Ollama server.

**Check:**
```bash
# 1. Is Ollama running?
ollama list

# 2. Start Ollama service
ollama serve

# 3. Test connection
curl http://localhost:11434/api/version

# 4. Check model availability
ollama list
ollama pull qwen2.5:7b  # if missing
```

### Python Dependencies Issues
**Problem:** Import errors or version conflicts.

**Solutions:**
```bash
# 1. Create fresh virtual environment
python -m venv ai_studio_env
source ai_studio_env/bin/activate  # or ai_studio_env\Scripts\activate on Windows

# 2. Install specific versions
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Clear pip cache if needed
pip cache purge
```

---

## Model Loading Issues

### SDXL Model Not Found
**Problem:** "SDXL model not found" error.

**Solutions:**
```python
# 1. Check file path
import os
model_path = "/path/to/your/model.safetensors"
print(f"File exists: {os.path.exists(model_path)}")

# 2. Update config.yaml
CONFIG = {
    "sd_model": "/correct/path/to/model.safetensors",  # Update this
    "ollama_model": "josiefied",
    "ollama_base_url": "http://localhost:11434"
}

# 3. Use absolute paths
CONFIG["sd_model"] = os.path.abspath("./models/your_model.safetensors")
```

### Model Loading Timeout
**Problem:** Models take too long to load.

**Solutions:**
```python
# 1. Increase timeout in requests
response = requests.post(url, json=data, timeout=300)  # 5 minutes

# 2. Use smaller/quantized models
# Download 4-bit or 8-bit quantized versions

# 3. Pre-load models
# Start Ollama and pull models before running app.py
```

---

## Runtime Issues

### Image Generation Fails
**Problem:** Generation returns None or errors.

**Debug Steps:**
```python
# 1. Check debug output in terminal
# Look for specific error messages

# 2. Test with simple prompt
prompt = "cat"  # Very simple
image, status = generate_image(prompt)
print(f"Status: {status}")

# 3. Check VRAM usage
nvidia-smi  # Monitor GPU memory

# 4. Try different settings
generate_image(prompt, steps=10, guidance=5.0)  # Minimal settings
```

### Chat Not Responding
**Problem:** Chat messages don't generate responses.

**Debug:**
```bash
# 1. Check terminal for debug output
# Should see "DEBUG: Chat wrapper called..."

# 2. Test Ollama directly
curl -X POST http://localhost:11434/api/chat -d '{
  "model": "josiefied",
  "messages": [{"role": "user", "content": "test"}],
  "stream": false
}'

# 3. Check model name
ollama list  # Verify "josiefied" exists
```

### Gradio Interface Issues
**Problem:** Web interface not loading or responding.

**Solutions:**
```bash
# 1. Check if port is available
netstat -an | grep 7860

# 2. Try different port
# In app.py: gradio_app.launch(server_port=7861)

# 3. Clear browser cache
# Or try incognito/private mode

# 4. Check firewall settings
# Ensure ports 7860 and 8000 are open
```

---

## Performance Issues

### Slow Image Generation
**Problem:** Generation takes too long.

**Optimizations:**
```python
# 1. Reduce steps
steps = 20  # Instead of 30-50

# 2. Lower guidance
guidance = 6.0  # Instead of 7.5-8.0

# 3. Use DPM++ scheduler (if available)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 4. Enable attention slicing
pipe.enable_attention_slicing()

# 5. Use xFormers (if installed)
pipe.enable_xformers_memory_efficient_attention()
```

### High Memory Usage
**Problem:** System using too much RAM/VRAM.

**Solutions:**
```python
# 1. Enable CPU offloading
pipe.enable_model_cpu_offload()

# 2. Clear CUDA cache periodically
import torch
torch.cuda.empty_cache()

# 3. Use gradient checkpointing
pipe.unet.enable_gradient_checkpointing()

# 4. Reduce precision
pipe = pipe.to(torch.float16)
```

---

## Network Issues

### API Not Accessible
**Problem:** Cannot reach API endpoints.

**Check:**
```bash
# 1. Server running?
ps aux | grep python
ps aux | grep app.py

# 2. Ports listening?
netstat -tlnp | grep :8000
netstat -tlnp | grep :7860

# 3. Firewall settings
sudo ufw status
sudo ufw allow 7860
sudo ufw allow 8000

# 4. Test locally first
curl http://localhost:8000/status
curl http://127.0.0.1:8000/status
```

### CORS Issues
**Problem:** Browser CORS errors when accessing API.

**Solutions:**
```python
# Add CORS middleware to FastAPI app
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Docker Issues

### Container Won't Start
**Problem:** Docker container exits or fails to start.

**Debug:**
```bash
# 1. Check logs
docker-compose logs ai-studio

# 2. Run interactive shell
docker run -it --rm illustrious-ai-studio /bin/bash

# 3. Check GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# 4. Verify mounts
docker run --rm -v ./models:/app/models illustrious-ai-studio ls /app/models
```

### Models Not Loading in Docker
**Problem:** Models not found inside container.

**Solutions:**
```bash
# 1. Check volume mounts
docker inspect illustrious-ai-studio | grep Mounts -A 10

# 2. Verify file permissions
ls -la ./models/

# 3. Use absolute paths in docker-compose.yml
volumes:
  - /absolute/path/to/models:/app/models:ro
```

---

## Getting Help

### Collect Debug Information
When reporting issues, include:

```bash
# System info
nvidia-smi
python --version
pip list | grep -E "(torch|diffusers|gradio|transformers)"

# GPU info
lspci | grep -i nvidia

# Disk space
df -h

# Process info
ps aux | grep -E "(python|ollama)"
netstat -tlnp | grep -E "(7860|8000|11434)"

# Docker info (if using Docker)
docker --version
docker-compose --version
docker images
docker ps -a
```

### Log Files
Check these locations for detailed logs:
- Terminal output where you ran `python main.py`
- Docker logs: `docker-compose logs`
- System logs: `/var/log/syslog` or `journalctl`

### Community Support
- GitHub Issues: Report bugs and request features
- Discord/Forums: Community discussions
- Documentation: Check README.md for updates

Remember to redact sensitive information (API keys, file paths) when sharing logs!
```

---

## 🎯 **Usage Summary**

### **Setting Up All Examples**
```bash
# Create directory structure
mkdir -p examples/{prompts,configs,api_examples,scripts,advanced,integrations,deployment,docs}

# Copy all example files to their respective directories
# (Use the content above to create each file)

# Make scripts executable
chmod +x examples/scripts/*.sh
chmod +x examples/scripts/*.py
chmod +x examples/api_examples/*.py
chmod +x examples/advanced/*.py
```

### **Quick Test Commands**
```bash
# Test your setup
python examples/api_examples/test_api.py

# Generate batch images
python examples/api_examples/batch_generate.py

# Run interactive chat
python examples/api_examples/chat_bot_example.py

# Create Docker deployment
python examples/deployment/docker_setup.py

# Test prompt variations
python examples/advanced/prompt_engineering.py portrait_photography
```

**🎉 Your AI Studio now has comprehensive examples covering:**
- ✅ **Prompt templates** and artistic styles
- ✅ **API integration** examples for Python/JavaScript
- ✅ **Advanced workflows** like story generation
- ✅ **Discord bot** and web service integrations
- ✅ **Jupyter notebook** support
- ✅ **Docker deployment** configurations  
- ✅ **Complete documentation** and troubleshooting guides

**This makes your project incredibly Claude Code-friendly** - any AI assistant can now understand your full architecture, extend features, debug issues, and help with development! 🚀
        if len(response) > 2000:
            # Discord has a 2000 character limit
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await ctx.send(f"💬 {chunk}")
        else:
            await ctx.send(f"💬 {response}")

@bot.command(name='help_ai')
async def help_command(ctx):
    """Show AI Studio bot help"""
    embed = discord.Embed(
        title="🤖 AI Studio Bot Commands",
        description="Generate images and chat with AI!",
        color=0x0099ff
    )
    
    embed.add_field(
        name="🎨 Image Generation",
        value="`!generate <prompt>` - Generate an image\n`!gen cute robot` - Short version",
        inline=False
    )
    
    embed.add_field(
        name="💬 Chat",
        value="`!chat <message>` - Chat with AI\n`!ask how are you?` - Short version",
        inline=False
    )
    
    embed.add_field(
        name="📋 Examples",
        value="`!generate magical forest scene`\n`!chat create a prompt for a dragon`",
        inline=False
    )
    
    await ctx.send(embed=embed)

# Error handling
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ Missing required argument. Use `!help_ai` for help.")
    elif isinstance(error, commands.CommandNotFound):
        pass  # Ignore unknown commands
    else:
        print(f"Error: {error}")
        await ctx.send("❌ An error occurred. Please try again.")

def run_discord_bot():
    """Run the Discord bot"""
    if DISCORD_BOT_TOKEN == "your_discord_bot_token_here":
        print("❌ Please set your Discord bot token in DISCORD_BOT_TOKEN")
        print("Get a token from: https://discord.com/developers/applications")
        return
    
    try:
        print("🚀 Starting Discord bot...")
        bot.run(DISCORD_BOT_TOKEN)
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    run_discord_bot()
```

### **examples/integrations/web_api_client.py**
```python
#!/usr/bin/env python3
"""
Web API client example for external applications
Demonstrates how to integrate AI Studio into other web services
"""

import requests
import json
import base64
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from pathlib import Path

class AIStudioClient:
    """Client for interacting with AI Studio MCP server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_status(self) -> Dict[str, Any]:
        """Check if AI Studio server is running and what models are available"""
        try:
            async with self.session.get(f"{self.base_url}/status") as response:
                if response.status == 200:
                    return await response.json()
                return {"error": f"Status check failed: {response.status}"}
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    async def generate_image(self, 
                           prompt: str,
                           negative_prompt: str = "",
                           steps: int = 30,
                           guidance: float = 7.5,
                           seed: int = -1) -> Optional[bytes]:
        """Generate an image and return raw image bytes"""
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance": guidance,
            "seed": seed
        }
        
        try:
            async with self.session.post(f"{self.base_url}/generate-image", 
                                       json=data, 
                                       timeout=aiohttp.ClientTimeout(total=180)) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("image_base64"):
                        return base64.b64decode(result["image_base64"])
                return None
        except Exception as e:
            print(f"Image generation error: {e}")
            return None
    
    async def chat(self, 
                   message: str,
                   session_id: str = "default",
                   temperature: float = 0.7,
                   max_tokens: int = 256) -> Optional[str]:
        """Chat with the AI model"""
        data = {
            "message": message,
            "session_id": session_id,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            async with self.session.post(f"{self.base_url}/chat", 
                                       json=data,
                                       timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response")
                return None
        except Exception as e:
            print(f"Chat error: {e}")
            return None
    
    async def analyze_image(self, 
                          image_bytes: bytes,
                          question: str = "Describe this image in detail") -> Optional[str]:
        """Analyze an image (requires multimodal model)"""
        image_base64 = base64.b64encode(image_bytes).decode()
        
        data = {
            "image_base64": image_base64,
            "question": question
        }
        
        try:
            async with self.session.post(f"{self.base_url}/analyze-image", 
                                       json=data,
                                       timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("analysis")
                return None
        except Exception as e:
            print(f"Image analysis error: {e}")
            return None

class ImageGenerator:
    """High-level image generation wrapper"""
    
    def __init__(self, client: AIStudioClient):
        self.client = client
        
    async def generate_with_enhancement(self, basic_prompt: str) -> Optional[bytes]:
        """Generate image with AI-enhanced prompt"""
        # First, enhance the prompt
        enhanced_prompt = await self.client.chat(
            f"Enhance this image prompt with artistic details: {basic_prompt}",
            temperature=0.8
        )
        
        if enhanced_prompt:
            # Generate image with enhanced prompt
            return await self.client.generate_image(enhanced_prompt)
        else:
            # Fallback to original prompt
            return await self.client.generate_image(basic_prompt)
    
    async def generate_style_variations(self, base_prompt: str, styles: list) -> Dict[str, bytes]:
        """Generate multiple style variations of the same prompt"""
        results = {}
        
        for style in styles:
            styled_prompt = f"{base_prompt}, {style} style"
            image_bytes = await self.client.generate_image(styled_prompt)
            if image_bytes:
                results[style] = image_bytes
        
        return results

# Example usage functions
async def demo_basic_usage():
    """Demonstrate basic API usage"""
    async with AIStudioClient() as client:
        # Check status
        print("🔍 Checking AI Studio status...")
        status = await client.check_status()
        print(f"Status: {status}")
        
        if status.get("error"):
            print("❌ AI Studio not available")
            return
        
        # Generate an image
        print("🎨 Generating image...")
        image_bytes = await client.generate_image(
            "a cute robot cat, anime style, masterpiece",
            steps=25
        )
        
        if image_bytes:
            # Save image
            with open("demo_generated.png", "wb") as f:
                f.write(image_bytes)
            print("✅ Image saved as demo_generated.png")
        
        # Chat with AI
        print("💬 Chatting with AI...")
        response = await client.chat("Create a creative prompt for a magical forest scene")
        print(f"AI Response: {response}")

async def demo_advanced_workflow():
    """Demonstrate advanced workflow with multiple API calls"""
    async with AIStudioClient() as client:
        generator = ImageGenerator(client)
        
        # Generate style variations
        print("🎨 Generating style variations...")
        styles = ["anime", "realistic", "oil painting", "cyberpunk"]
        
        variations = await generator.generate_style_variations(
            "a majestic dragon in a mountain landscape", 
            styles
        )
        
        # Save all variations
        for style, image_bytes in variations.items():
            filename = f"dragon_{style.replace(' ', '_')}.png"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            print(f"✅ Saved: {filename}")
        
        print(f"📊 Generated {len(variations)} style variations")

async def demo_image_analysis():
    """Demonstrate image analysis (requires vision model)"""
    async with AIStudioClient() as client:
        # First generate an image
        print("🎨 Generating image for analysis...")
        image_bytes = await client.generate_image(
            "a beautiful landscape with mountains and a lake"
        )
        
        if image_bytes:
            # Analyze the generated image
            print("🔍 Analyzing generated image...")
            analysis = await client.analyze_image(
                image_bytes,
                "Describe the composition, colors, and mood of this landscape"
            )
            
            if analysis:
                print(f"🤖 Analysis: {analysis}")
            else:
                print("❌ Image analysis not available (requires vision model)")

# Flask web service example
try:
    from flask import Flask, request, jsonify, send_file
    import io
    
    app = Flask(__name__)
    
    @app.route('/api/generate', methods=['POST'])
    async def api_generate():
        """Web API endpoint for image generation"""
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing prompt"}), 400
        
        async with AIStudioClient() as client:
            image_bytes = await client.generate_image(
                data['prompt'],
                data.get('negative_prompt', ''),
                data.get('steps', 30),
                data.get('guidance', 7.5),
                data.get('seed', -1)
            )
            
            if image_bytes:
                # Return image as response
                return send_file(
                    io.BytesIO(image_bytes),
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='generated.png'
                )
            else:
                return jsonify({"error": "Generation failed"}), 500
    
    @app.route('/api/chat', methods=['POST'])
    async def api_chat():
        """Web API endpoint for chat"""
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Missing message"}), 400
        
        async with AIStudioClient() as client:
            response = await client.chat(
                data['message'],
                data.get('session_id', 'web_api'),
                data.get('temperature', 0.7),
                data.get('max_tokens', 256)
            )
            
            if response:
                return jsonify({"response": response})
            else:
                return jsonify({"error": "Chat failed"}), 500
    
    def run_web_service():
        """Run the Flask web service"""
        print("🌐 Starting web service on http://localhost:5000")
        print("Endpoints:")
        print("• POST /api/generate - Generate images")
        print("• POST /api/chat - Chat with AI")
        app.run(host='0.0.0.0', port=5000, debug=True)

except ImportError:
    def run_web_service():
        print("❌ Flask not installed. Install with: pip install flask")

async def main():
    """Main demo function"""
    import sys
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        
        if demo_type == "basic":
            await demo_basic_usage()
        elif demo_type == "advanced":
            await demo_advanced_workflow()
        elif demo_type == "analysis":
            await demo_image_analysis()
        elif demo_type == "web":
            run_web_service()
        else:
            print(f"❌ Unknown demo type: {demo_type}")
    else:
        print("🚀 AI Studio Client Demo")
        print("=" * 30)
        print("Available demos:")
        print("• basic - Basic API usage")
        print("• advanced - Advanced workflow")
        print("• analysis - Image analysis")
        print("• web - Web service wrapper")
        print(f"\nUsage: python web_api_client.py [demo_type]")

if __name__ == "__main__":
    asyncio.run(main())
```

### **examples/integrations/jupyter_notebook.py**
```python
#!/usr/bin/env python3
"""
Jupyter Notebook integration examples
For interactive AI Studio usage in notebooks
"""

# Install required packages:
# pip install jupyter ipywidgets matplotlib requests pillow

import requests
import base64
import json
from PIL import Image
import matplotlib.pyplot as plt
import io
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

class JupyterAIStudio:
    """AI Studio integration for Jupyter notebooks"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = "jupyter_session"
        
    def check_connection(self):
        """Check if AI Studio is accessible"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print("✅ AI Studio Connected!")
                print(f"🎨 SDXL: {'Available' if status.get('models', {}).get('sdxl') else 'Not Available'}")
                print(f"🤖 Ollama: {'Available' if status.get('models', {}).get('ollama') else 'Not Available'}")
                print(f"👁️ Vision: {'Available' if status.get('models', {}).get('multimodal') else 'Not Available'}")
                return True
            else:
                print(f"❌ Connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to AI Studio: {e}")
            print("Make sure AI Studio is running on http://localhost:8000")
            return False
    
    def generate_image(self, prompt, **kwargs):
        """Generate image and display in notebook"""
        data = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", "blurry, low quality"),
            "steps": kwargs.get("steps", 30),
            "guidance": kwargs.get("guidance", 7.5),
            "seed": kwargs.get("seed", -1)
        }
        
        print(f"🎨 Generating: {prompt}")
        
        try:
            response = requests.post(f"{self.base_url}/generate-image", json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("image_base64"):
                    # Decode and display image
                    img_data = base64.b64decode(result["image_base64"])
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Display with matplotlib
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    plt.axis('off')
                    plt.title(f"Generated: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"✅ {result.get('message', 'Generated successfully!')}")
                    return image
                else:
                    print("❌ No image data received")
            else:
                print(f"❌ Generation failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return None
    
    def chat(self, message, **kwargs):
        """Chat with AI and return response"""
        data = {
            "message": message,
            "session_id": self.session_id,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512)
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat", json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No response")
                
                # Display formatted response
                display(HTML(f"""
                <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>🤖 AI Response:</strong><br>
                    {response_text.replace('\n', '<br>')}
                </div>
                """))
                
                return response_text
            else:
                print(f"❌ Chat failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return None
    
    def create_interactive_generator(self):
        """Create interactive widget for image generation"""
        
        # Widgets
        prompt_widget = widgets.Textarea(
            value="a beautiful landscape, masterpiece, detailed",
            placeholder="Enter your image prompt here...",
            description="Prompt:",
            layout=widgets.Layout(width='100%', height='100px')
        )
        
        negative_widget = widgets.Text(
            value="blurry, low quality, distorted",
            placeholder="Negative prompt...",
            description="Negative:",
            layout=widgets.Layout(width='100%')
        )
        
        steps_widget = widgets.IntSlider(
            value=30,
            min=10,
            max=100,
            step=5,
            description="Steps:",
            style={'description_width': 'initial'}
        )
        
        guidance_widget = widgets.FloatSlider(
            value=7.5,
            min=1.0,
            max=15.0,
            step=0.5,
            description="Guidance:",
            style={'description_width': 'initial'}
        )
        
        seed_widget = widgets.IntText(
            value=-1,
            description="Seed (-1 for random):",
            style={'description_width': 'initial'}
        )
        
        generate_button = widgets.Button(
            description="🎨 Generate Image",
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        output_area = widgets.Output()
        
        def on_generate_click(b):
            with output_area:
                clear_output(wait=True)
                self.generate_image(
                    prompt_widget.value,
                    negative_prompt=negative_widget.value,
                    steps=steps_widget.value,
                    guidance=guidance_widget.value,
                    seed=seed_widget.value if seed_widget.value != -1 else -1
                )
        
        generate_button.on_click(on_generate_click)
        
        # Layout
        ui = widgets.VBox([
            widgets.HTML("<h3>🎨 Interactive Image Generator</h3>"),
            prompt_widget,
            negative_widget,
            widgets.HBox([steps_widget, guidance_widget]),
            seed_widget,
            generate_button,
            output_area
        ])
        
        return ui
    
    def create_chat_interface(self):
        """Create interactive chat interface"""
        
        chat_history = widgets.HTML(
            value="<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px;'>💬 Chat started! Ask me anything or request image generation.</div>",
            layout=widgets.Layout(height='400px', overflow='auto')
        )
        
        message_input = widgets.Text(
            placeholder="Type your message here...",
            layout=widgets.Layout(width='80%')
        )
        
        send_button = widgets.Button(
            description="Send",
            button_style='primary',
            layout=widgets.Layout(width='15%')
        )
        
        def send_message(b=None):
            message = message_input.value.strip()
            if not message:
                return
            
            # Add user message to history
            current_history = chat_history.value
            user_msg = f"<div style='margin: 10px 0; padding: 10px; background-color: #e3f2fd; border-radius: 10px;'><strong>👤 You:</strong> {message}</div>"
            
            # Get AI response
            ai_response = self.chat(message)
            
            if ai_response:
                ai_msg = f"<div style='margin: 10px 0; padding: 10px; background-color: #f1f8e9; border-radius: 10px;'><strong>🤖 AI:</strong> {ai_response.replace(chr(10), '<br>')}</div>"
                chat_history.value = current_history + user_msg + ai_msg
            
            # Clear input
            message_input.value = ""
        
        send_button.on_click(send_message)
        
        def on_enter(change):
            if change['new'] and message_input.value.strip():
                send_message()
        
        # Layout
        ui = widgets.VBox([
            widgets.HTML("<h3>💬 AI Chat Interface</h3>"),
            chat_history,
            widgets.HBox([message_input, send_button])
        ])
        
        return ui

def create_example_notebook():
    """Create example notebook cells as string"""
    
    notebook_code = '''
# AI Studio Jupyter Integration Examples

## 1. Setup and Connection Test
```python
# Import the integration class
from jupyter_notebook import JupyterAIStudio

# Initialize AI Studio connection
ai = JupyterAIStudio()

# Test connection
ai.check_connection()
```

## 2. Simple Image Generation
```python
# Generate a basic image
ai.generate_image("a cute robot cat, anime style, masterpiece")
```

## 3. Advanced Image Generation with Parameters
```python
# Generate with custom parameters
ai.generate_image(
    "cyberpunk cityscape, neon lights, futuristic, detailed",
    negative_prompt="blurry, low quality, people",
    steps=40,
    guidance=8.0,
    seed=12345
)
```

## 4. Chat with AI
```python
# Ask AI to create prompts
response = ai.chat("Create 5 creative prompts for fantasy artwork")
print(response)
```

## 5. Interactive Widgets
```python
# Create interactive image generator
generator_ui = ai.create_interactive_generator()
display(generator_ui)
```

```python
# Create interactive chat interface
chat_ui = ai.create_chat_interface()
display(chat_ui)
```

## 6. Batch Generation Example
```python
# Generate multiple images
prompts = [
    "a magical forest, ethereal lighting",
    "a steampunk airship, detailed machinery", 
    "a crystal cave, glowing gems",
    "a futuristic city, flying cars"
]

images = []
for prompt in prompts:
    img = ai.generate_image(prompt, steps=25)  # Faster generation
    if img:
        images.append(img)

print(f"Generated {len(images)} images!")
```

## 7. Style Comparison
```python
base_prompt = "a majestic dragon"
styles = ["anime style", "realistic", "oil painting", "watercolor"]

for style in styles:
    full_prompt = f"{base_prompt}, {style}, masterpiece"
    ai.generate_image(full_prompt)
```
'''
    
    return notebook_code

# Standalone functions for direct notebook use
def quick_generate(prompt, **kwargs):
    """Quick image generation function"""
    ai = JupyterAIStudio()
    return ai.generate_image(prompt, **kwargs)

def quick_chat(message, **kwargs):
    """Quick chat function"""
    ai = JupyterAIStudio()
    return ai.chat(message, **kwargs)

def setup_ai_studio():
    """Quick setup function"""
    ai = JupyterAIStudio()
    if ai.check_connection():
        print("\n🎉 AI Studio is ready!")
        print("Try these commands:")
        print("• quick_generate('your prompt here')")
        print("• quick_chat('your message here')")
        print("• ai = JupyterAIStudio(); ai.create_interactive_generator()")
        return ai
    else:
        return None

if __name__ == "__main__":
    print("📓 Jupyter Notebook Integration for AI Studio")
    print("=" * 50)
    print(create_example_notebook())
```

---

## 🔧 **7. Deployment Examples**

### **examples/deployment/docker_setup.py**
```python
#!/usr/bin/env python3
"""
Docker deployment configuration generator
Creates Docker files and configurations for AI Studio deployment
"""

import os
from pathlib import Path

def create_dockerfile():
    """Create Dockerfile for AI Studio"""
    
    dockerfile_content = '''# AI Studio Docker Configuration
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    curl \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models/sdxl \\
    && mkdir -p /app/tmp/illustrious_ai/gallery \\
    && mkdir -p /app/examples

# Expose ports
EXPOSE 7860 8000 11434

# Create startup script
COPY docker_startup.sh /app/
RUN chmod +x /app/docker_startup.sh

# Start command
CMD ["/app/docker_startup.sh"]
'''
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile")

def create_docker_compose():
    """Create docker-compose.yml"""
    
    compose_content = '''version: '3.8'

services:
  ai-studio:
    build: .
    container_name: illustrious-ai-studio
    ports:
      - "7860:7860"  # Gradio UI
      - "8000:8000"  # MCP Server
      - "11434:11434" # Ollama
    volumes:
      - ./models:/app/models:ro  # Model files (read-only)
      - ./gallery:/app/tmp/illustrious_ai/gallery  # Generated images
      - ./examples:/app/examples:ro  # Examples (read-only)
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - OLLAMA_HOST=0.0.0.0
    runtime: nvidia  # For GPU support
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  # Optional: Separate Ollama service
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    ports:
      - "11435:11434"  # Alternative port
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    runtime: nvidia
    restart: unless-stopped
    profiles: ["separate-ollama"]  # Optional service

volumes:
  ollama_data:
    driver: local

networks:
  default:
    name: ai-studio-network
'''
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    print("✅ Created docker-compose.yml")

def create_startup_script():
    """Create Docker startup script"""
    
    startup_content = '''#!/bin/bash
# Docker startup script for AI Studio

echo "🚀 Starting AI Studio in Docker..."

# Start Ollama in background
echo "📦 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
while ! curl -s http://localhost:11434/api/version > /dev/null; do
    sleep 2
done

# Pull default models if they don't exist
echo "📥 Checking/pulling Ollama models..."
if ! ollama list | grep -q "qwen2.5:7b"; then
    echo "Pulling qwen2.5:7b..."
    ollama pull qwen2.5:7b
fi

# Create model alias if needed
ollama cp qwen2.5:7b josiefied 2>/dev/null || true

echo "✅ Ollama ready!"

# Check for SDXL model
if [ ! -f "/app/models/sdxl/*.safetensors" ]; then
    echo "⚠️  Warning: No SDXL model found in /app/models/sdxl/"
    echo "Please mount your model files to /app/models/"
fi

# Start AI Studio
echo "🎨 Starting AI Studio application..."
python3 app.py

# Cleanup on exit
trap "kill $OLLAMA_PID" EXIT
'''
    
    with open("docker_startup.sh", "w") as f:
        f.write(startup_content)
    
    os.chmod("docker_startup.sh", 0o755)
    print("✅ Created docker_startup.sh")

def create_docker_requirements():
    """Create requirements.txt for Docker"""
    
    requirements_content = '''torch==2.1.0# 🎨 Illustrious AI Studio - Examples Collection

Complete examples for getting the most out of your AI Studio setup.

## 📁 File Structure
Create these files in your project directory:

```
illustrious-ai-studio/
├── examples/
│   ├── prompts/
│   │   ├── artistic_styles.json
│   │   ├── character_prompts.json
│   │   └── scene_prompts.json
│   ├── configs/
│   │   ├── model_configs.json
│   │   └── generation_presets.json
│   ├── api_examples/
│   │   ├── test_api.py
│   │   ├── batch_generate.py
│   │   └── chat_bot_example.py
│   └── scripts/
│       ├── setup_models.sh
│       ├── backup_gallery.py
│       └── model_switcher.py
```

---

## 🎯 **1. Prompt Examples**

### **examples/prompts/artistic_styles.json**
```json
{
  "artistic_styles": {
    "anime": {
      "prompt_prefix": "anime style, manga artwork,",
      "quality_tags": "masterpiece, best quality, highly detailed, vibrant colors",
      "negative": "realistic, photographic, 3d render, blurry",
      "guidance": 8.0,
      "steps": 35
    },
    "photorealistic": {
      "prompt_prefix": "photorealistic, ultra realistic, high resolution,",
      "quality_tags": "8k uhd, professional photography, sharp focus",
      "negative": "anime, cartoon, drawing, painting, art",
      "guidance": 7.0,
      "steps": 40
    },
    "cyberpunk": {
      "prompt_prefix": "cyberpunk style, neon lights, futuristic,",
      "quality_tags": "dark atmosphere, glowing, detailed, cinematic",
      "negative": "medieval, fantasy, nature, bright daylight",
      "guidance": 8.5,
      "steps": 35
    },
    "fantasy": {
      "prompt_prefix": "fantasy art, magical, ethereal,",
      "quality_tags": "enchanted, mystical, detailed, beautiful",
      "negative": "modern, technology, urban, realistic",
      "guidance": 7.5,
      "steps": 30
    },
    "minimalist": {
      "prompt_prefix": "minimalist style, clean, simple,",
      "quality_tags": "elegant, geometric, modern, pristine",
      "negative": "cluttered, complex, detailed, ornate",
      "guidance": 6.0,
      "steps": 25
    }
  }
}
```

### **examples/prompts/character_prompts.json**
```json
{
  "character_templates": {
    "anime_girl": {
      "base": "1girl, anime style, beautiful face, detailed eyes,",
      "variations": {
        "magical": "holding staff, wizard robes, glowing magic, fantasy background",
        "modern": "casual clothes, city background, smartphone, trendy",
        "warrior": "armor, sword, battle stance, medieval setting"
      }
    },
    "fantasy_creature": {
      "base": "mythical creature, detailed, majestic,",
      "variations": {
        "dragon": "dragon, wings spread, breathing fire, scales, powerful",
        "unicorn": "unicorn, horn, flowing mane, forest, magical aura",
        "phoenix": "phoenix, fire wings, rebirth, golden flames, soaring"
      }
    },
    "robot": {
      "base": "robot, mechanical, futuristic,",
      "variations": {
        "humanoid": "android, human-like, sleek design, glowing eyes",
        "industrial": "heavy machinery, construction robot, utilitarian",
        "companion": "cute robot, friendly, small size, helpful"
      }
    }
  }
}
```

### **examples/prompts/scene_prompts.json**
```json
{
  "scene_templates": {
    "landscapes": [
      "mountain range at sunset, golden hour, misty peaks, dramatic clouds, cinematic",
      "tropical beach, crystal clear water, palm trees, white sand, paradise",
      "enchanted forest, ancient trees, magical lighting, fairy lights, mystical",
      "cyberpunk city, neon signs, rain reflections, night scene, futuristic skyline",
      "desert oasis, sand dunes, palm trees, clear blue water, serene"
    ],
    "interiors": [
      "cozy library, wooden shelves, warm lighting, leather chairs, peaceful",
      "modern kitchen, sleek design, marble counters, natural light, clean",
      "wizard's study, ancient books, magical artifacts, candlelight, mysterious",
      "spaceship bridge, control panels, screens, futuristic, high-tech",
      "coffee shop, warm atmosphere, books, plants, comfortable seating"
    ],
    "abstract": [
      "swirling colors, fluid motion, ethereal, dreamlike, abstract art",
      "geometric patterns, symmetrical, mathematical, precise, colorful",
      "fractal design, infinite complexity, psychedelic, mesmerizing",
      "light rays, prismatic, rainbow colors, magical, luminous",
      "organic forms, flowing, natural patterns, harmonious, peaceful"
    ]
  }
}
```

---

## ⚙️ **2. Configuration Examples**

### **examples/configs/model_configs.json**
```json
{
  "model_configurations": {
    "high_quality": {
      "description": "Best quality, slower generation",
      "steps": 50,
      "guidance": 8.0,
      "scheduler": "DPMSolverMultistepScheduler",
      "negative": "blurry, low quality, distorted, deformed, ugly, bad anatomy"
    },
    "balanced": {
      "description": "Good quality, reasonable speed",
      "steps": 30,
      "guidance": 7.5,
      "scheduler": "EulerAncestralDiscreteScheduler",
      "negative": "blurry, low quality, text, watermark"
    },
    "fast": {
      "description": "Quick generation, lower quality",
      "steps": 20,
      "guidance": 6.0,
      "scheduler": "LMSDiscreteScheduler",
      "negative": "blurry, low quality"
    },
    "experimental": {
      "description": "Creative settings for unique results",
      "steps": 35,
      "guidance": 9.0,
      "scheduler": "DDIMScheduler",
      "negative": "boring, generic, simple"
    }
  },
  "ollama_models": {
    "creative": {
      "model": "josiefied",
      "temperature": 0.8,
      "description": "Creative responses, good for prompts"
    },
    "precise": {
      "model": "qwen2.5:7b",
      "temperature": 0.3,
      "description": "Precise, factual responses"
    },
    "vision": {
      "model": "llava:latest",
      "temperature": 0.5,
      "description": "Image analysis and description"
    }
  }
}
```

### **examples/configs/generation_presets.json**
```json
{
  "generation_presets": {
    "portrait": {
      "prompt_suffix": "portrait, detailed face, professional lighting, high quality",
      "negative": "full body, landscape, multiple people, blurry face",
      "aspect_ratio": "3:4",
      "steps": 35,
      "guidance": 7.5
    },
    "landscape": {
      "prompt_suffix": "wide shot, panoramic view, detailed environment, cinematic",
      "negative": "portrait, close-up, indoor, people",
      "aspect_ratio": "16:9",
      "steps": 30,
      "guidance": 7.0
    },
    "concept_art": {
      "prompt_suffix": "concept art, detailed, artistic, professional illustration",
      "negative": "photo, realistic, amateur, sketch",
      "steps": 40,
      "guidance": 8.0
    },
    "character_sheet": {
      "prompt_suffix": "character design, multiple views, turnaround, reference sheet",
      "negative": "single view, landscape, environment",
      "steps": 45,
      "guidance": 8.5
    }
  }
}
```

---

## 💻 **3. API Usage Examples**

### **examples/api_examples/test_api.py**
```python
#!/usr/bin/env python3
"""
Test script for Illustrious AI Studio API
Run this to verify your MCP server is working correctly
"""

import requests
import json
import base64
from PIL import Image
import io

# Configuration
BASE_URL = "http://localhost:8000"

def test_server_status():
    """Test if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            print("✅ Server is running!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False

def test_image_generation():
    """Test image generation endpoint"""
    print("\n🎨 Testing image generation...")
    
    data = {
        "prompt": "a cute robot cat, anime style, masterpiece, highly detailed",
        "negative_prompt": "blurry, low quality, deformed",
        "steps": 25,
        "guidance": 7.5,
        "seed": 42
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate-image", json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image generated successfully!")
            
            # Save the image
            if result.get("image_base64"):
                img_data = base64.b64decode(result["image_base64"])
                image = Image.open(io.BytesIO(img_data))
                image.save("test_generated_image.png")
                print("💾 Image saved as 'test_generated_image.png'")
            
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (this is normal for slow GPUs)")
        return False

def test_chat():
    """Test chat endpoint"""
    print("\n💬 Testing chat...")
    
    data = {
        "message": "Create a creative prompt for a magical forest scene",
        "session_id": "test_session",
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat working!")
            print(f"Response: {result['response']}")
            return True
        else:
            print(f"❌ Chat failed: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Chat request timed out")
        return False

def main():
    print("🚀 Testing Illustrious AI Studio API")
    print("=" * 50)
    
    # Test each endpoint
    server_ok = test_server_status()
    if not server_ok:
        return
    
    chat_ok = test_chat()
    image_ok = test_image_generation()
    
    print("\n📊 Test Results:")
    print(f"Server Status: {'✅' if server_ok else '❌'}")
    print(f"Chat API: {'✅' if chat_ok else '❌'}")
    print(f"Image Generation: {'✅' if image_ok else '❌'}")

if __name__ == "__main__":
    main()
```

### **examples/api_examples/batch_generate.py**
```python
#!/usr/bin/env python3
"""
Batch image generation example
Generate multiple images with different prompts and settings
"""

import requests
import json
import base64
from PIL import Image
import io
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
OUTPUT_DIR = Path("batch_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Batch generation prompts
BATCH_PROMPTS = [
    {
        "name": "fantasy_dragon",
        "prompt": "majestic dragon, fantasy art, detailed scales, breathing fire, mountain background, epic, cinematic lighting",
        "steps": 35,
        "guidance": 8.0
    },
    {
        "name": "cyberpunk_city",
        "prompt": "cyberpunk cityscape, neon lights, rain, futuristic, night scene, detailed architecture, atmospheric",
        "steps": 30,
        "guidance": 7.5
    },
    {
        "name": "magical_forest",
        "prompt": "enchanted forest, magical lighting, fairy lights, ancient trees, mystical atmosphere, detailed foliage",
        "steps": 30,
        "guidance": 7.0
    },
    {
        "name": "space_station",
        "prompt": "futuristic space station, detailed sci-fi architecture, stars, space, high-tech, cinematic",
        "steps": 35,
        "guidance": 8.0
    }
]

def generate_image(prompt_config):
    """Generate a single image"""
    data = {
        "prompt": prompt_config["prompt"],
        "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
        "steps": prompt_config.get("steps", 30),
        "guidance": prompt_config.get("guidance", 7.5),
        "seed": -1  # Random seed
    }
    
    print(f"🎨 Generating: {prompt_config['name']}")
    
    try:
        response = requests.post(f"{BASE_URL}/generate-image", json=data, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            
            # Save the image
            if result.get("image_base64"):
                img_data = base64.b64decode(result["image_base64"])
                image = Image.open(io.BytesIO(img_data))
                
                filename = f"{prompt_config['name']}_{int(time.time())}.png"
                filepath = OUTPUT_DIR / filename
                image.save(filepath)
                
                print(f"✅ Saved: {filepath}")
                return True
            else:
                print("❌ No image data received")
                return False
        else:
            print(f"❌ Generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 Batch Image Generation")
    print("=" * 50)
    
    success_count = 0
    total_count = len(BATCH_PROMPTS)
    
    for i, prompt_config in enumerate(BATCH_PROMPTS, 1):
        print(f"\n[{i}/{total_count}] Processing...")
        
        if generate_image(prompt_config):
            success_count += 1
        
        # Small delay between generations
        time.sleep(2)
    
    print(f"\n📊 Batch Complete!")
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"📁 Images saved to: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
```

### **examples/api_examples/chat_bot_example.py**
```python
#!/usr/bin/env python3
"""
Interactive chat bot example using the Illustrious AI Studio API
Demonstrates session management and continuous conversation
"""

import requests
import json
import uuid

BASE_URL = "http://localhost:8000"

class IllustriousChat:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.base_url = BASE_URL
        
    def send_message(self, message, temperature=0.7):
        """Send a message to the chat API"""
        data = {
            "message": message,
            "session_id": self.session_id,
            "temperature": temperature,
            "max_tokens": 512
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat", json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response")
            else:
                return f"Error {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"Connection error: {e}"
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("🤖 Illustrious AI Chat Bot")
        print("Type 'quit' to exit, 'help' for commands")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if not user_input:
                    continue
                
                print("🤔 Thinking...")
                response = self.send_message(user_input)
                print(f"🤖 AI: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    def show_help(self):
        """Show available commands"""
        help_text = """
Available commands:
• Just type normally to chat
• '#generate [description]' - Generate an image
• 'quit' or 'exit' - End the chat
• 'help' - Show this help

Example prompts:
• "Create a prompt for a magical castle"
• "#generate a cute robot playing guitar"
• "What makes a good image prompt?"
• "Help me brainstorm fantasy character ideas"
        """
        print(help_text)

def main():
    # Test connection first
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=5)
        if response.status_code != 200:
            print("❌ Cannot connect to Illustrious AI Studio API")
            print("Make sure the server is running on http://localhost:8000")
            return
    except:
        print("❌ Cannot connect to Illustrious AI Studio API")
        print("Make sure the server is running on http://localhost:8000")
        return
    
    # Start chat
    chat_bot = IllustriousChat()
    chat_bot.chat_loop()

if __name__ == "__main__":
    main()
```

---

## 🔧 **4. Utility Scripts**

### **examples/scripts/setup_models.sh**
```bash
#!/bin/bash
# Auto-setup script for Illustrious AI Studio models

echo "🚀 Setting up Illustrious AI Studio Models"
echo "=========================================="

# Create directories
echo "📁 Creating directories..."
mkdir -p models/sdxl
mkdir -p models/ollama
mkdir -p tmp/illustrious_ai/gallery

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama already installed"
fi

# Start Ollama service
echo "🔄 Starting Ollama service..."
ollama serve &
sleep 5

# Pull recommended models
echo "📥 Pulling recommended Ollama models..."
echo "This may take a while depending on your internet connection..."

# Creative model for prompt enhancement
ollama pull qwen2.5:7b
echo "✅ Qwen2.5 7B downloaded"

# Vision model for image analysis
ollama pull llava:latest
echo "✅ LLaVA vision model downloaded"

# Create model alias
echo "🔗 Creating model aliases..."
ollama cp qwen2.5:7b creative-assistant

echo "📋 Available models:"
ollama list

echo ""
echo "✅ Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Download a Stable Diffusion XL model (.safetensors format)"
echo "2. Edit config.yaml with your model path"
echo "3. Run: python main.py"
echo ""
echo "Recommended SDXL models:"
echo "• https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
echo "• https://civitai.com/ (for specialized models)"
```

### **examples/scripts/backup_gallery.py**
```python
#!/usr/bin/env python3
"""
Backup script for generated images and metadata
Creates organized backups with timestamps
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
import zipfile

# Configuration
GALLERY_DIR = Path("tmp/illustrious_ai/gallery")
BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)

def create_backup():
    """Create a timestamped backup of the gallery"""
    if not GALLERY_DIR.exists():
        print("❌ Gallery directory not found")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"gallery_backup_{timestamp}"
    backup_path = BACKUP_DIR / f"{backup_name}.zip"
    
    print(f"📦 Creating backup: {backup_name}")
    
    # Count files
    image_files = list(GALLERY_DIR.glob("*.png"))
    metadata_files = list(GALLERY_DIR.glob("*.json"))
    
    print(f"📊 Found {len(image_files)} images and {len(metadata_files)} metadata files")
    
    # Create zip archive
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in GALLERY_DIR.iterdir():
            if file.is_file():
                zipf.write(file, file.name)
    
    # Create backup info
    info = {
        "timestamp": timestamp,
        "image_count": len(image_files),
        "metadata_count": len(metadata_files),
        "total_size_mb": round(backup_path.stat().st_size / (1024*1024), 2)
    }
    
    info_path = BACKUP_DIR / f"{backup_name}_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Backup created: {backup_path}")
    print(f"📈 Archive size: {info['total_size_mb']} MB")
    
    return True

def list_backups():
    """List all available backups"""
    backup_files = list(BACKUP_DIR.glob("gallery_backup_*.zip"))
    
    if not backup_files:
        print("📭 No backups found")
        return
    
    print("📋 Available backups:")
    print("-" * 50)
    
    for backup_file in sorted(backup_files):
        info_file = backup_file.with_name(backup_file.stem + "_info.json")
        
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            
            print(f"📦 {backup_file.name}")
            print(f"   Date: {info['timestamp']}")
            print(f"   Images: {info['image_count']}")
            print(f"   Size: {info['total_size_mb']} MB")
            print()

def cleanup_old_backups(keep_count=5):
    """Keep only the most recent N backups"""
    backup_files = sorted(BACKUP_DIR.glob("gallery_backup_*.zip"))
    
    if len(backup_files) <= keep_count:
        print(f"📦 {len(backup_files)} backups found, no cleanup needed")
        return
    
    to_remove = backup_files[:-keep_count]
    
    print(f"🗑️ Removing {len(to_remove)} old backups (keeping {keep_count} most recent)")
    
    for backup_file in to_remove:
        info_file = backup_file.with_name(backup_file.stem + "_info.json")
        
        backup_file.unlink()
        if info_file.exists():
            info_file.unlink()
        
        print(f"   Removed: {backup_file.name}")

def main():
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "create":
            create_backup()
        elif command == "list":
            list_backups()
        elif command == "cleanup":
            keep_count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            cleanup_old_backups(keep_count)
        else:
            print("Usage: python backup_gallery.py [create|list|cleanup]")
    else:
        # Default: create backup
        create_backup()

if __name__ == "__main__":
    main()
```

### **examples/scripts/model_switcher.py**
```python
#!/usr/bin/env python3
"""
Model switcher utility for quickly changing between different model configurations
"""

import json
import shutil
from pathlib import Path

CONFIG_FILE = Path("current_model_config.json")

# Model configurations
MODEL_CONFIGS = {
    "anime": {
        "name": "Anime/Manga Style",
        "sd_model": "/path/to/anime_model.safetensors",
        "ollama_model": "josiefied",
        "description": "Optimized for anime and manga style artwork"
    },
    "realistic": {
        "name": "Photorealistic",
        "sd_model": "/path/to/realistic_model.safetensors",
        "ollama_model": "qwen2.5:7b",
        "description": "Best for realistic and photographic images"
    },
    "artistic": {
        "name": "Artistic/Concept Art",
        "sd_model": "/path/to/artistic_model.safetensors",
        "ollama_model": "creative-assistant",
        "description": "Great for concept art and artistic styles"
    },
    "general": {
        "name": "General Purpose",
        "sd_model": "/path/to/general_model.safetensors",
        "ollama_model": "qwen2.5:7b",
        "description": "Balanced model for various styles"
    }
}

def list_configs():
    """List available model configurations"""
    print("📋 Available Model Configurations:")
    print("=" * 50)
    
    for key, config in MODEL_CONFIGS.items():
        print(f"🎯 {key}: {config['name']}")
        print(f"   {config['description']}")
        print(f"   SDXL: {Path(config['sd_model']).name}")
        print(f"   LLM: {config['ollama_model']}")
        print()

def get_current_config():
    """Get currently active configuration"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return None

def switch_config(config_key):
    """Switch to a specific model configuration"""
    if config_key not in MODEL_CONFIGS:
        print(f"❌ Configuration '{config_key}' not found")
        return False
    
    config = MODEL_CONFIGS[config_key]
    
    # Check if SDXL model exists
    if not Path(config["sd_model"]).exists():
        print(f"❌ SDXL model not found: {config['sd_model']}")
        print("Please update the path in MODEL_CONFIGS")
        return False
    
    # Save configuration
    current_config = {
        "active_config": config_key,
        "config": config,
        "switched_at": str(Path().cwd())
    }
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(current_config, f, indent=2)
    
    print(f"✅ Switched to: {config['name']}")
    print(f"🎨 SDXL Model: {Path(config['sd_model']).name}")
    print(f"🤖 LLM Model: {config['ollama_model']}")
    print()
    print("🔄 Please restart main.py to apply changes")

    # Generate updated config block
    model_paths = {
        "sd_model": config["sd_model"],
        "ollama_model": config["ollama_model"],
        "ollama_base_url": "http://localhost:11434"
    }

    print("\n📋 Update your config.yaml with:")
    print(json.dumps(model_paths, indent=2))
    
    return True

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("🔧 Model Switcher for Illustrious AI Studio")
        print("=" * 50)
        
        current = get_current_config()
        if current:
            print(f"🎯 Current: {current['config']['name']}")
            print()
        
        list_configs()
        print("Usage: python model_switcher.py [config_name]")
        print("Example: python model_switcher.py anime")
        return
    
    config_key = sys.argv[1].lower()
    
    if config_key == "list":
        list_configs()
    else:
        switch_config(config_key)

if __name__ == "__main__":
    main()
```

---

## 📝 **Usage Instructions**

### **Setting Up Examples**
```bash
# Create examples directory structure
mkdir -p examples/{prompts,configs,api_examples,scripts}

# Copy the JSON files to their locations
# (Create each file with the content above)

# Make scripts executable
chmod +x examples/scripts/*.sh
chmod +x examples/scripts/*.py
```

### **Using the Examples**

#### **1. Test Your Setup**
```bash
python examples/api_examples/test_api.py
```

#### **2. Generate Batch Images**
```bash
python examples/api_examples/batch_generate.py
```

#### **3. Interactive Chat**
```bash
python examples/api_examples/chat_bot_example.py
```

#### **4. Model Management**
```bash
# List available model configs
python examples/scripts/model_switcher.py list

# Switch to anime model
python examples/scripts/model_switcher.py anime
```

#### **5. Backup Your Gallery**
```bash
# Create backup
python examples/scripts/backup_gallery.py create

# List backups
python examples/scripts/backup_gallery.py list
```

---

## 🎨 **5. Advanced Examples**

### **examples/advanced/workflow_automation.py**
```python
#!/usr/bin/env python3
"""
Automated workflow example: Generate images based on story prompts
Creates a series of images that tell a visual story
"""

import requests
import json
import base64
import time
from PIL import Image
import io
from pathlib import Path

BASE_URL = "http://localhost:8000"
OUTPUT_DIR = Path("story_output")
OUTPUT_DIR.mkdir(exist_ok=True)

class StoryImageGenerator:
    def __init__(self):
        self.session_id = "story_session"
        
    def create_story_prompt(self, scene_description, story_context=""):
        """Use AI to enhance a scene description with story context"""
        message = f"""
        Create a detailed visual prompt for this scene: {scene_description}
        
        Story context: {story_context}
        
        Make it cinematic and detailed, suitable for image generation.
        Include artistic style, lighting, composition, and mood.
        Keep it under 150 words.
        """
        
        data = {
            "message": message,
            "session_id": self.session_id,
            "temperature": 0.8,
            "max_tokens": 200
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=data, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", scene_description)
        return scene_description
    
    def generate_story_image(self, prompt, scene_name, story_name):
        """Generate an image for a story scene"""
        data = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, text, watermark, deformed, ugly",
            "steps": 35,
            "guidance": 7.5,
            "seed": -1
        }
        
        print(f"🎨 Generating: {scene_name}")
        print(f"📝 Prompt: {prompt[:100]}...")
        
        response = requests.post(f"{BASE_URL}/generate-image", json=data, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("image_base64"):
                img_data = base64.b64decode(result["image_base64"])
                image = Image.open(io.BytesIO(img_data))
                
                # Create story-specific directory
                story_dir = OUTPUT_DIR / story_name
                story_dir.mkdir(exist_ok=True)
                
                filename = f"{len(list(story_dir.glob('*.png'))):02d}_{scene_name}.png"
                filepath = story_dir / filename
                image.save(filepath)
                
                # Save metadata
                metadata = {
                    "scene_name": scene_name,
                    "story_name": story_name,
                    "prompt": prompt,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                metadata_file = filepath.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✅ Saved: {filepath}")
                return True
        
        print(f"❌ Failed to generate {scene_name}")
        return False
    
    def generate_story_sequence(self, story_config):
        """Generate a complete story sequence"""
        story_name = story_config["name"]
        scenes = story_config["scenes"]
        context = story_config.get("context", "")
        
        print(f"📚 Generating story: {story_name}")
        print(f"🎬 Scenes to generate: {len(scenes)}")
        print("=" * 50)
        
        success_count = 0
        
        for i, scene in enumerate(scenes, 1):
            print(f"\n[{i}/{len(scenes)}] Processing scene...")
            
            # Enhance the scene description
            enhanced_prompt = self.create_story_prompt(
                scene["description"], 
                f"{context}. This is scene {i} of {len(scenes)}"
            )
            
            # Generate the image
            if self.generate_story_image(enhanced_prompt, scene["name"], story_name):
                success_count += 1
            
            # Brief pause between generations
            time.sleep(3)
        
        print(f"\n📊 Story Complete!")
        print(f"✅ Generated: {success_count}/{len(scenes)} scenes")
        print(f"📁 Story saved to: {OUTPUT_DIR / story_name}")
        
        # Create story summary
        self.create_story_summary(story_name, story_config, success_count)
    
    def create_story_summary(self, story_name, story_config, success_count):
        """Create a summary file for the generated story"""
        summary = {
            "story_name": story_name,
            "description": story_config.get("description", ""),
            "context": story_config.get("context", ""),
            "total_scenes": len(story_config["scenes"]),
            "generated_scenes": success_count,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenes": [
                {
                    "name": scene["name"],
                    "description": scene["description"]
                }
                for scene in story_config["scenes"]
            ]
        }
        
        summary_file = OUTPUT_DIR / story_name / "story_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

# Example story configurations
EXAMPLE_STORIES = {
    "dragon_quest": {
        "name": "dragon_quest",
        "description": "A hero's journey to defeat an ancient dragon",
        "context": "Epic fantasy adventure with medieval setting, heroic themes, and magical elements",
        "scenes": [
            {
                "name": "village_beginning",
                "description": "A peaceful medieval village at dawn, with a young hero preparing for a journey"
            },
            {
                "name": "dark_forest",
                "description": "The hero traveling through a mysterious dark forest with ancient trees"
            },
            {
                "name": "mountain_path",
                "description": "Climbing a treacherous mountain path leading to the dragon's lair"
            },
            {
                "name": "dragon_lair",
                "description": "Inside a massive cave filled with treasure and a sleeping ancient dragon"
            },
            {
                "name": "epic_battle",
                "description": "The hero fighting the dragon with sword and magic in an epic confrontation"
            },
            {
                "name": "victorious_return",
                "description": "The hero returning to the village as a celebrated champion"
            }
        ]
    },
    "space_odyssey": {
        "name": "space_odyssey",
        "description": "A journey through space to discover alien civilizations",
        "context": "Science fiction space adventure with futuristic technology and alien encounters",
        "scenes": [
            {
                "name": "space_station",
                "description": "A massive space station orbiting Earth, bustling with activity"
            },
            {
                "name": "starship_launch",
                "description": "A sleek starship launching into the depths of space"
            },
            {
                "name": "alien_planet",
                "description": "Landing on a colorful alien planet with strange flora and twin suns"
            },
            {
                "name": "alien_city",
                "description": "Discovering an advanced alien city with crystalline architecture"
            },
            {
                "name": "first_contact",
                "description": "Meeting peaceful alien beings for the first time"
            },
            {
                "name": "return_journey",
                "description": "The starship traveling back to Earth through a beautiful nebula"
            }
        ]
    }
}

def main():
    import sys
    
    generator = StoryImageGenerator()
    
    if len(sys.argv) > 1:
        story_key = sys.argv[1]
        if story_key in EXAMPLE_STORIES:
            generator.generate_story_sequence(EXAMPLE_STORIES[story_key])
        else:
            print(f"❌ Story '{story_key}' not found")
            print("Available stories:", list(EXAMPLE_STORIES.keys()))
    else:
        print("🎬 Story Image Generator")
        print("=" * 30)
        print("Available stories:")
        for key, story in EXAMPLE_STORIES.items():
            print(f"• {key}: {story['description']}")
        print(f"\nUsage: python workflow_automation.py [story_name]")
        print(f"Example: python workflow_automation.py dragon_quest")

if __name__ == "__main__":
    main()
```

### **examples/advanced/prompt_engineering.py**
```python
#!/usr/bin/env python3
"""
Advanced prompt engineering examples and techniques
Demonstrates different approaches to crafting effective prompts
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

class PromptEngineer:
    def __init__(self):
        self.session_id = "prompt_engineer"
        
    def test_prompt_variations(self, base_concept, variations):
        """Test different prompt variations for the same concept"""
        print(f"🧪 Testing prompt variations for: {base_concept}")
        print("=" * 60)
        
        results = []
        
        for i, variation in enumerate(variations, 1):
            print(f"\n[{i}/{len(variations)}] Testing variation...")
            print(f"📝 Prompt: {variation['name']}")
            
            # Enhance the prompt
            enhanced = self.enhance_prompt(variation['prompt'])
            
            # Generate image
            result = self.generate_test_image(enhanced, f"{base_concept}_{variation['name']}")
            
            results.append({
                "name": variation['name'],
                "original_prompt": variation['prompt'],
                "enhanced_prompt": enhanced,
                "success": result,
                "description": variation.get('description', '')
            })
            
            time.sleep(2)  # Brief pause
        
        # Save results
        self.save_test_results(base_concept, results)
        
        return results
    
    def enhance_prompt(self, base_prompt):
        """Use AI to enhance a basic prompt"""
        message = f"""
        Enhance this image generation prompt to be more detailed and effective:
        "{base_prompt}"
        
        Add specific details about:
        - Artistic style and technique
        - Lighting and atmosphere
        - Composition and framing
        - Quality indicators
        - Color palette and mood
        
        Keep it focused and under 200 words.
        """
        
        data = {
            "message": message,
            "session_id": self.session_id,
            "temperature": 0.7,
            "max_tokens": 250
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=data, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", base_prompt)
        return base_prompt
    
    def generate_test_image(self, prompt, filename):
        """Generate a test image"""
        data = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
            "steps": 30,
            "guidance": 7.5,
            "seed": 42  # Fixed seed for consistency
        }
        
        try:
            response = requests.post(f"{BASE_URL}/generate-image", json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Image generated successfully")
                return True
            else:
                print(f"❌ Generation failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def save_test_results(self, concept, results):
        """Save test results to JSON file"""
        output_dir = Path("prompt_tests")
        output_dir.mkdir(exist_ok=True)
        
        test_data = {
            "concept": concept,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_variations": len(results),
            "successful_generations": sum(1 for r in results if r['success']),
            "results": results
        }
        
        filename = f"{concept.replace(' ', '_')}_test_results.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"📊 Test results saved to: {filepath}")

# Test configurations
PROMPT_TESTS = {
    "portrait_photography": {
        "concept": "portrait photography",
        "variations": [
            {
                "name": "basic",
                "prompt": "portrait of a person",
                "description": "Minimal prompt to test baseline"
            },
            {
                "name": "professional",
                "prompt": "professional portrait photography, studio lighting, detailed face, high quality",
                "description": "Professional photography style"
            },
            {
                "name": "artistic",
                "prompt": "artistic portrait, dramatic lighting, oil painting style, masterpiece",
                "description": "Artistic interpretation"
            },
            {
                "name": "cinematic",
                "prompt": "cinematic portrait, film photography, bokeh, golden hour lighting",
                "description": "Cinematic movie-like quality"
            }
        ]
    },
    "fantasy_landscape": {
        "concept": "fantasy landscape",
        "variations": [
            {
                "name": "simple",
                "prompt": "fantasy landscape",
                "description": "Basic fantasy scene"
            },
            {
                "name": "detailed",
                "prompt": "epic fantasy landscape, magical forest, ancient ruins, mystical atmosphere, detailed",
                "description": "More specific elements"
            },
            {
                "name": "atmospheric",
                "prompt": "ethereal fantasy realm, glowing magical energy, misty atmosphere, otherworldly beauty",
                "description": "Focus on mood and atmosphere"
            },
            {
                "name": "cinematic",
                "prompt": "cinematic fantasy landscape, wide shot, dramatic lighting, movie poster style",
                "description": "Cinematic composition"
            }
        ]
    }
}

def analyze_prompt_patterns():
    """Analyze common patterns in effective prompts"""
    
    effective_patterns = {
        "Quality Enhancers": [
            "masterpiece", "best quality", "highly detailed", "professional",
            "8k resolution", "ultra realistic", "sharp focus", "intricate details"
        ],
        "Style Indicators": [
            "oil painting", "digital art", "concept art", "photorealistic",
            "anime style", "watercolor", "sketch", "3d render"
        ],
        "Lighting Terms": [
            "dramatic lighting", "soft lighting", "golden hour", "studio lighting",
            "natural lighting", "rim lighting", "volumetric lighting", "ambient light"
        ],
        "Composition": [
            "close-up", "wide shot", "portrait", "landscape orientation",
            "rule of thirds", "depth of field", "bokeh", "symmetrical"
        ],
        "Atmosphere": [
            "moody", "ethereal", "mystical", "serene", "dramatic",
            "peaceful", "energetic", "mysterious", "warm", "cool"
        ]
    }
    
    print("📋 Effective Prompt Patterns Analysis")
    print("=" * 50)
    
    for category, terms in effective_patterns.items():
        print(f"\n🎯 {category}:")
        for term in terms:
            print(f"   • {term}")
    
    print("\n💡 Tips for Better Prompts:")
    print("• Combine 2-3 style indicators")
    print("• Always include quality enhancers")
    print("• Specify lighting for mood")
    print("• Add composition details")
    print("• Use negative prompts to avoid unwanted elements")
    print("• Test different orderings of elements")

def main():
    import sys
    
    engineer = PromptEngineer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "analyze":
            analyze_prompt_patterns()
        elif command in PROMPT_TESTS:
            test_config = PROMPT_TESTS[command]
            engineer.test_prompt_variations(test_config["concept"], test_config["variations"])
        else:
            print(f"❌ Unknown command or test: {command}")
            print("Available commands: analyze")
            print("Available tests:", list(PROMPT_TESTS.keys()))
    else:
        print("🎯 Prompt Engineering Tool")
        print("=" * 30)
        print("Available commands:")
        print("• analyze - Show effective prompt patterns")
        print("\nAvailable tests:")
        for key, test in PROMPT_TESTS.items():
            print(f"• {key} - Test {test['concept']} variations")
        print(f"\nUsage: python prompt_engineering.py [command/test]")
        print(f"Example: python prompt_engineering.py portrait_photography")

if __name__ == "__main__":
    main()
```

### **examples/advanced/image_comparison.py**
```python
#!/usr/bin/env python3
"""
Image comparison and A/B testing tool
Compare different generation settings and models
"""

import requests
import json
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import io
from pathlib import Path

BASE_URL = "http://localhost:8000"
COMPARISON_DIR = Path("image_comparisons")
COMPARISON_DIR.mkdir(exist_ok=True)

class ImageComparator:
    def __init__(self):
        self.comparison_id = int(time.time())
        
    def generate_comparison_set(self, prompt, settings_list, test_name):
        """Generate multiple images with different settings for comparison"""
        print(f"🔬 Running comparison test: {test_name}")
        print(f"📝 Prompt: {prompt}")
        print("=" * 60)
        
        results = []
        images = []
        
        for i, settings in enumerate(settings_list, 1):
            print(f"\n[{i}/{len(settings_list)}] Testing: {settings['name']}")
            
            # Generate image
            image_data = self.generate_with_settings(prompt, settings)
            
            if image_data:
                images.append({
                    "name": settings['name'],
                    "image": image_data,
                    "settings": settings
                })
                print("✅ Generated successfully")
            else:
                print("❌ Generation failed")
            
            time.sleep(2)
        
        if images:
            # Create comparison grid
            comparison_path = self.create_comparison_grid(images, test_name, prompt)
            
            # Save detailed results
            self.save_comparison_results(images, test_name, prompt, comparison_path)
            
            print(f"\n📊 Comparison Complete!")
            print(f"🖼️ Grid saved: {comparison_path}")
            print(f"📋 Results saved: {COMPARISON_DIR / f'{test_name}_results.json'}")
        
        return images
    
    def generate_with_settings(self, prompt, settings):
        """Generate image with specific settings"""
        data = {
            "prompt": prompt,
            "negative_prompt": settings.get("negative_prompt", "blurry, low quality"),
            "steps": settings.get("steps", 30),
            "guidance": settings.get("guidance", 7.5),
            "seed": settings.get("seed", -1)
        }
        
        try:
            response = requests.post(f"{BASE_URL}/generate-image", json=data, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("image_base64"):
                    img_data = base64.b64decode(result["image_base64"])
                    return Image.open(io.BytesIO(img_data))
            
        except Exception as e:
            print(f"Error: {e}")
        
        return None
    
    def create_comparison_grid(self, images, test_name, prompt):
        """Create a grid comparison of all generated images"""
        if not images:
            return None
        
        # Calculate grid dimensions
        num_images = len(images)
        cols = min(3, num_images)  # Max 3 columns
        rows = (num_images + cols - 1) // cols
        
        # Get image dimensions (assuming all same size)
        img_width, img_height = images[0]["image"].size
        
        # Create grid
        grid_width = cols * img_width + (cols + 1) * 20  # 20px padding
        grid_height = rows * img_height + (rows + 1) * 20 + 100  # Extra space for labels
        
        grid = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title
        title = f"Comparison: {test_name}"
        draw.text((20, 10), title, fill='black', font=font)
        draw.text((20, 35), f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}", 
                 fill='gray', font=small_font)
        
        # Add images
        for i, img_data in enumerate(images):
            row = i // cols
            col = i % cols
            
            x = col * (img_width + 20) + 20
            y = row * (img_height + 20) + 80
            
            # Paste image
            grid.paste(img_data["image"], (x, y))
            
            # Add label
            label = img_data["name"]
            settings = img_data["settings"]
            details = f"Steps: {settings.get('steps', 30)}, Guidance: {settings.get('guidance', 7.5)}"
            
            draw.text((x, y + img_height + 5), label, fill='black', font=small_font)
            draw.text((x, y + img_height + 25), details, fill='gray', font=small_font)
        
        # Save grid
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}_comparison.png"
        filepath = COMPARISON_DIR / filename
        grid.save(filepath)
        
        return filepath
    
    def save_comparison_results(self, images, test_name, prompt, comparison_path):
        """Save detailed comparison results"""
        results = {
            "test_name": test_name,
            "prompt": prompt,
            "comparison_grid": str(comparison_path),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(images),
            "settings_compared": [
                {
                    "name": img["name"],
                    "settings": img["settings"]
                }
                for img in images
            ]
        }
        
        filename = f"{test_name}_results.json"
        filepath = COMPARISON_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

# Comparison test configurations
COMPARISON_TESTS = {
    "steps_comparison": {
        "name": "steps_comparison",
        "prompt": "beautiful anime girl, detailed face, colorful hair, masterpiece, best quality",
        "settings": [
            {"name": "Low_Steps", "steps": 15, "guidance": 7.5},
            {"name": "Medium_Steps", "steps": 30, "guidance": 7.5},
            {"name": "High_Steps", "steps": 50, "guidance": 7.5}
        ]
    },
    "guidance_comparison": {
        "name": "guidance_comparison", 
        "prompt": "cyberpunk cityscape, neon lights, futuristic, detailed, night scene",
        "settings": [
            {"name": "Low_Guidance", "steps": 30, "guidance": 5.0},
            {"name": "Medium_Guidance", "steps": 30, "guidance": 7.5},
            {"name": "High_Guidance", "steps": 30, "guidance": 10.0}
        ]
    },
    "negative_prompt_test": {
        "name": "negative_prompt_test",
        "prompt": "portrait of a woman, realistic, detailed",
        "settings": [
            {"name": "No_Negative", "steps": 30, "guidance": 7.5, "negative_prompt": ""},
            {"name": "Basic_Negative", "steps": 30, "guidance": 7.5, "negative_prompt": "blurry, low quality"},
            {"name": "Detailed_Negative", "steps": 30, "guidance": 7.5, 
             "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, extra limbs"}
        ]
    }
}

def main():
    import sys
    
    comparator = ImageComparator()
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        
        if test_name in COMPARISON_TESTS:
            test_config = COMPARISON_TESTS[test_name]
            comparator.generate_comparison_set(
                test_config["prompt"],
                test_config["settings"],
                test_config["name"]
            )
        else:
            print(f"❌ Test '{test_name}' not found")
            print("Available tests:", list(COMPARISON_TESTS.keys()))
    else:
        print("🔬 Image Comparison Tool")
        print("=" * 30)
        print("Available tests:")
        for key, test in COMPARISON_TESTS.items():
            print(f"• {key}: Compare {test['name'].replace('_', ' ')}")
        print(f"\nUsage: python image_comparison.py [test_name]")
        print(f"Example: python image_comparison.py steps_comparison")

if __name__ == "__main__":
    main()
```

---

## 🌟 **6. Integration Examples**

### **examples/integrations/discord_bot.py**
```python
#!/usr/bin/env python3
"""
Discord bot integration example
Allows users to generate images through Discord commands
"""

import discord
from discord.ext import commands
import requests
import base64
import io
from PIL import Image
import asyncio

# Configuration
DISCORD_BOT_TOKEN = "your_discord_bot_token_here"
AI_STUDIO_URL = "http://localhost:8000"

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

class AIStudioBot:
    def __init__(self):
        self.session_id = "discord_bot"
    
    async def generate_image(self, prompt, user_id):
        """Generate image via AI Studio API"""
        data = {
            "prompt": f"{prompt}, masterpiece, best quality, detailed",
            "negative_prompt": "blurry, low quality, nsfw, inappropriate",
            "steps": 25,  # Faster for Discord
            "guidance": 7.5,
            "seed": -1
        }
        
        try:
            response = requests.post(f"{AI_STUDIO_URL}/generate-image", json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("image_base64"):
                    img_data = base64.b64decode(result["image_base64"])
                    return io.BytesIO(img_data)
            
        except Exception as e:
            print(f"Error generating image: {e}")
        
        return None
    
    async def chat_with_ai(self, message, user_id):
        """Chat with AI via API"""
        data = {
            "message": message,
            "session_id": f"discord_{user_id}",
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(f"{AI_STUDIO_URL}/chat", json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Sorry, I couldn't generate a response.")
            
        except Exception as e:
            print(f"Error chatting with AI: {e}")
        
        return "Sorry, I'm having trouble connecting to the AI."

ai_studio = AIStudioBot()

@bot.event
async def on_ready():
    print(f'🤖 {bot.user} is connected to Discord!')
    print(f'🎨 AI Studio Bot is ready!')

@bot.command(name='generate', aliases=['gen', 'create'])
async def generate_image_command(ctx, *, prompt):
    """Generate an image from a text prompt"""
    if not prompt:
        await ctx.send("Please provide a prompt! Example: `!generate cute robot cat`")
        return
    
    # Send "typing" indicator
    async with ctx.typing():
        await ctx.send(f"🎨 Generating image: `{prompt}`\nThis may take a moment...")
        
        # Generate image
        image_data = await ai_studio.generate_image(prompt, ctx.author.id)
        
        if image_data:
            # Send image
            file = discord.File(image_data, filename="generated_image.png")
            embed = discord.Embed(
                title="🎨 Generated Image",
                description=f"Prompt: `{prompt}`",
                color=0x00ff00
            )
            embed.set_image(url="attachment://generated_image.png")
            embed.set_footer(text=f"Generated for {ctx.author.display_name}")
            
            await ctx.send(file=file, embed=embed)
        else:
            await ctx.send("❌ Sorry, I couldn't generate that image. Please try again!")

@bot.command(name='chat', aliases=['ask'])
async def chat_command(ctx, *, message):
    """Chat with the AI"""
    if not message:
        await ctx.send("Please provide a message! Example: `!chat Hello, how are you?`")
        return
    
    async with ctx.typing():
        response = await ai_studio.chat_with_ai(message, ctx.author.id)
        
        # Split long responses
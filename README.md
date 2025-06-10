# Illustrious AI Studio

A powerful local AI application that combines **Stable Diffusion XL (SDXL)** image generation with **Ollama LLM** chat capabilities in an intuitive web interface.

## 🌟 Features

### 🎨 **Dual AI System**
- **SDXL Image Generation**: High-quality 1024x1024 image generation using Stable Diffusion XL
- **Ollama LLM Chat**: Interactive chat with local language models
- **Cross-Tab Integration**: Generate images directly from chat using `#generate` commands
- **Dynamic Model Switching**: Swap SDXL or Ollama models without restarting

### 🚀 **Advanced Memory Management**
- **Automatic CUDA Memory Management**: Smart memory clearing and retry logic
- **Out-of-Memory Protection**: Up to 2 automatic retries with memory clearing on CUDA OOM errors
- **Memory Fragmentation Prevention**: Optimized PyTorch memory allocation
- **Performance Monitoring**: Real-time CUDA memory status and optimization tips

### 🌐 **Dual Interface**
- **Gradio Web UI** (Port 7860): Interactive web interface with tabs for image generation, chat, and analysis
- **FastAPI MCP Server** (Port 8000): RESTful API for programmatic access and integration
- **Additional MCP Servers** (Ports 8001-8004): Specialized servers for filesystem, web, git, and image operations

### 📸 **Gallery & Session Management**
- **Automatic Gallery**: All generated images saved with metadata to `/tmp/illustrious_ai/gallery/`
- **Session Persistence**: Chat history maintained per session
- **Image Analysis**: Analyze images with vision-capable models (if available)

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **CUDA-compatible GPU** (12GB+ VRAM recommended)
- **Ollama** installed and running locally
- **SDXL Model** in .safetensors format

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd illustrious-ai-studio
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n ai-studio python=3.11
   conda activate ai-studio
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Install CUDA-compatible PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Configure models**
   Copy `config.yaml` to your workspace and edit it or set environment variables:
   ```yaml
   sd_model: "/path/to/your/sdxl-model.safetensors"
   ollama_model: "qwen2.5:7b"  # Your Ollama model
   ollama_base_url: "http://localhost:11434"
   ```
   Environment overrides are `SD_MODEL`, `OLLAMA_MODEL`, and `OLLAMA_BASE_URL`.

5. **Start Ollama** (in separate terminal)
   ```bash
   ollama serve
   ollama pull qwen2.5:7b  # or your preferred model
   ```

## 🚀 Usage

### Starting the Application
```bash
conda activate ai-studio
python main.py
```

**Access Points:**
- **Web Interface**: http://localhost:7860
- **Main API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MCP Servers**: Ports 8001-8004 (filesystem, web, git, image analysis)

### Web Interface

#### 🎨 Image Generation Tab
- Enter prompts for image generation
- Adjust parameters: steps (20-50), guidance scale (5-12), seed
- View generation status and automatic memory management
- Images automatically saved to gallery

#### 💬 Chat Tab  
- Interactive chat with your Ollama model
- Use `#generate <prompt>` to create images from chat
- Session-based conversation history
- Clean responses with internal processing removed

#### 🔍 Image Analysis Tab
- Upload images for AI analysis
- Ask questions about image content
- Requires vision-capable Ollama model (e.g., llava, moondream)

#### ℹ️ Status Tab
- Real-time model status monitoring
- CUDA memory management information
- MCP server endpoint documentation
- Performance optimization tips

### API Usage

#### Main API Server (Port 8000)

**Image Generation:**
```bash
curl -X POST http://localhost:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape",
    "steps": 20,
    "guidance": 7.5
  }'
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "session_id": "my-session"
  }'
```

**Server Status:**
```bash
curl http://localhost:8000/status
```

**Switch Models:**
```bash
curl -X POST http://localhost:8000/switch-models \
  -H "Content-Type: application/json" \
  -d '{"sd_model": "/path/to/model.safetensors", "ollama_model": "qwen:7b"}'
```

#### MCP Servers

**Start All MCP Servers:**
```bash
cd mcp_servers && python start_all.py
```

**Filesystem Operations (Port 8001):**
```bash
curl -X POST http://localhost:8001/tools/read_file \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"path": "/home/ant/AI/Project/README.md"}}'
```

**Web Content Fetching (Port 8002):**
```bash
curl -X POST http://localhost:8002/tools/fetch_url \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"url": "https://example.com"}}'
```

**Git Operations (Port 8003):**
```bash
curl -X POST http://localhost:8003/tools/git_status \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"repo_path": "/home/ant/AI/Project"}}'
```

**Image Analysis (Port 8004):**
```bash
curl -X POST http://localhost:8004/tools/analyze_image_properties \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"image_path": "/path/to/image.jpg"}}'
```

## 🧪 Testing

Before running the test suite, install the dependencies:
```bash
pip install -r requirements.txt httpx
```
Skipping these installations will result in import errors when running `pytest`.

### Automated Testing
```bash
cd examples/api_examples
python test_api.py
```

### Manual Testing
```bash
# Test endpoints individually
python examples/api_examples/batch_generate.py
```

## ⚙️ Configuration

### Model Paths
Copy `config.yaml` and adjust the paths to your environment, or use environment variables:
- `sd_model` (`SD_MODEL`): Path to SDXL .safetensors file
- `ollama_model` (`OLLAMA_MODEL`): Ollama model name
- `ollama_base_url` (`OLLAMA_BASE_URL`): Ollama server URL

### Memory Optimization
The application includes automatic CUDA memory management, but you can optimize further:
- **Reduce Parameters**: Lower steps (20-30), guidance (5-8) for faster generation
- **Close GPU Apps**: Free VRAM by closing other GPU-intensive applications
- **Monitor Memory**: Use `nvidia-smi` to check GPU memory usage

## 🔧 Troubleshooting

### CUDA Out of Memory
**Automatic Fixes (No Action Required):**
- System automatically clears CUDA cache and retries (up to 2 attempts)
- Memory fragmentation prevention enabled by default
- Garbage collection integrated with memory management

**Manual Solutions (if automatic fixes fail):**
- Reduce image parameters (steps: 20-30, guidance: 5-8)
- Close other GPU applications
- Restart application if memory leaks persist
- Ensure 6GB+ VRAM available

### API Errors
- **HTTP 500**: General generation error - check model paths and dependencies
- **HTTP 507**: Memory insufficient - automatic retry will be triggered
- **HTTP 503**: Model not available - verify model loading in status tab

### Chat Issues
- Verify Ollama is running: `ollama serve`
- Check model is loaded: `ollama list`
- Test connection via status endpoint

## 📁 Project Structure

```
Project/
├── app.py                 # Main application file
├── CLAUDE.md             # Detailed development documentation
├── requirements.txt      # Python dependencies
├── mcp_servers/          # Model Context Protocol servers
│   ├── filesystem_server.py    # Filesystem operations
│   ├── web_fetch_server.py     # Web content fetching
│   ├── git_server.py           # Git repository operations
│   ├── image_analysis_server.py # Image analysis tools
│   ├── manager.py              # Server management
│   ├── start_all.py            # Quick start script
│   ├── config.json             # Server configuration
│   └── README.md               # MCP server documentation
├── examples/             # API examples and testing
│   ├── api_examples/     # API usage examples
│   ├── configs/          # Configuration presets
│   └── prompts/          # Prompt templates
└── gallery/              # Generated images (auto-created)
```

## 🔗 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/status` | GET | Server and model status |
| `/generate-image` | POST | Generate images with SDXL |
| `/chat` | POST | Chat with Ollama LLM |
| `/analyze-image` | POST | Analyze images (if vision model available) |
| `/switch-models` | POST | Switch SDXL and/or Ollama models |

### MCP Server Endpoints

| Server | Port | Description |
|--------|------|-------------|
| Filesystem | 8001 | File operations (read, write, list, etc.) |
| Web Fetch | 8002 | Web content fetching and analysis |
| Git | 8003 | Git repository operations |
| Image Analysis | 8004 | Advanced image processing and analysis |

### Response Codes
- `200`: Success
- `500`: General error
- `503`: Service unavailable (model not loaded)
- `507`: Insufficient storage (CUDA memory issue)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## ⚠️ Usage and Risks

This project is provided for personal and research use under the terms of the
[MIT License](LICENSE). Output generated by the included AI models may be
unpredictable or inappropriate. Use the software responsibly and comply with all
applicable laws.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Stable Diffusion XL** for high-quality image generation
- **Ollama** for local LLM capabilities
- **Gradio** for the intuitive web interface
- **FastAPI** for the robust API server

---

**🚀 Ready to create amazing AI-generated content locally? Get started now!**
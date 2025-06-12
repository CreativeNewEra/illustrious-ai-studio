# Illustrious AI Studio

A powerful local AI application that combines **Stable Diffusion XL (SDXL)** image generation with **Ollama LLM** capabilities, including vision models for image analysis.

## 🆕 Latest Improvements

-### 🚀 **Automated Setup & Model Management**
- **One-Command Setup**: `python setup_env.py` handles everything
- **Automatic Downloads**: Recommended models downloaded and configured
- **Smart Configuration**: Auto-updating config.yaml with optimal settings

### 📝 **Prompt Template System**
- **Save & Reuse**: Save your best prompts as reusable templates
- **Organization**: Categories, tags, and search functionality  
- **Import/Export**: Share templates as JSON files
- **Usage Analytics**: Track popular templates and usage statistics

### 🛡️ **Enhanced Security & Reliability**
- **Centralized Logging**: Automatic log rotation and error tracking
- **Rate Limiting**: Configurable request limits for web fetching
- **User Agent Spoofing**: Multiple user agent options for better web compatibility
- **Domain Security**: Allowed/blocked domain lists for MCP servers
- **Robust Error Handling**: User-friendly error messages throughout

### 📊 **Better User Experience**
- **Detailed Documentation**: Enhanced README with direct links and examples
- **Configuration Comments**: Inline explanations for all config options
- **Status Monitoring**: Real-time feedback and progress indicators
- **Performance Insights**: Detailed logging of model operations

## 🌟 Features

### 🎨 **Triple AI System**
- **SDXL Image Generation**: High-quality 1024x1024 image generation using Stable Diffusion XL
- **Ollama LLM Chat**: Interactive chat with local language models
- **Vision Analysis**: Image understanding and Q&A with vision-capable models
- **Cross-Feature Integration**: Generate images from chat, analyze generated images

### 🚀 **Advanced Memory Management**
- **Automatic GPU Memory Management**: Smart memory clearing and retry logic
- **Model Manager Tool**: Switch between image/LLM modes to optimize 16GB VRAM usage
- **Out-of-Memory Protection**: Automatic retries with memory clearing
- **Performance Optimization**: FP16 precision, TF32 enabled for RTX 4090M

### 🌐 **Dual Interface**
- **Gradio Web UI** (Port 7860): Interactive web interface with tabs for all features
- **FastAPI MCP Server** (Port 8000): RESTful API for programmatic access

### 📸 **Features**
- **Automatic Gallery**: Generated images saved with metadata
- **Prompt Enhancement**: LLM improves your image prompts
- **Session Management**: Persistent chat history
- **Model Switching**: Change models without restarting

## 🛠️ Installation

### Prerequisites
- **Python 3.10+**
- **GPU** with CUDA or ROCm support (NVIDIA or AMD, 16GB+ VRAM recommended)
  - Install the appropriate drivers for your hardware
- **Ollama** installed and running ([Installation Guide](https://ollama.ai/download))
- **SDXL Model** (.safetensors format) - we recommend [Illustrious-XL](https://huggingface.co/OnomaAI/Illustrious-xl) for anime-style generation

#### Recommended SDXL Model
For best results, we recommend the **Illustrious-XL** model:
- **Download**: [Hugging Face - Illustrious-XL](https://huggingface.co/OnomaAI/Illustrious-xl/blob/main/Illustrious-xl-v0.1.safetensors)
- **Size**: ~6.6GB
- **Strengths**: Excellent anime/manga style generation with high detail

### Supported GPUs
Illustrious AI Studio works with both **NVIDIA CUDA** and **AMD ROCm** GPUs. We
recommend at least **16GB** of VRAM for smooth 1024x1024 generation. Lower VRAM
(8-12GB) can work with reduced image sizes and steps. Verify your GPU with
`nvidia-smi` on NVIDIA hardware or `rocm-smi`/`rocminfo` on AMD.

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd illustrious-ai-studio
   ```

2. **Run the environment setup script** ✨ **NEW**
   ```bash
   python setup_env.py
   ```
   Select your GPU type (CUDA, ROCm, or CPU) and wait while the script:
   - Creates `./venv` and installs all requirements
   - Downloads the recommended models
   - Updates your `config.yaml` automatically

3. **Verify setup**
   ```bash
   python verify_setup.py
   ```

4. **Configure models** (already configured for your setup)
   - SDXL: `models/Illustrious.safetensors`
   - LLM: `goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k`
   - Vision: `qwen2.5vl:7b`

## 🚀 Usage

### Starting the Application
```bash
python main.py
```
Add `--lazy-load` to defer model initialization until first use:
```bash
python main.py --lazy-load
```
Then open: http://localhost:7860

Use the **Model Loader** section on the System Info tab to load SDXL or Ollama
models on demand if you started with `--lazy-load`.

### Memory Management
```bash
# For image generation heavy workloads
python model_manager.py --image-mode

# For LLM/chat heavy workloads  
python model_manager.py --llm-mode

# Balanced usage
python model_manager.py --balanced
```

### Testing
```bash
# Quick test of all features
python test_simple.py

# Comprehensive testing
python test_full_functionality.py
```

#### Developer Testing
Install the extra packages needed by the test suite (`pytest`, `Pillow`, `PyYAML`, etc.):

```bash
pip install -r requirements-test.txt
```

Then run:

```bash
pytest
```

## 💻 Web Interface

### 🎨 Text-to-Image Tab
- Enter prompts or use "Enhance Prompt" for AI improvement
- Adjust parameters: steps, guidance scale, seed
- Images auto-saved to gallery
- **New**: Use saved prompt templates for consistent results

### 💬 AI Chat Tab
- Chat with the LLM
- Use `#generate <description>` to create images
- Call MCP tools with `/tool <server>.<method> key=value`
- Maintains conversation history

### 🔍 Image Analysis Tab
- Upload any image for AI analysis
- Ask specific questions about images
- Works with generated images too

### 📝 Prompt Templates Tab ✨ **NEW**
- **Save Templates**: Save your best prompts for reuse
- **Organize**: Categorize templates and add tags
- **Search**: Find templates by name, content, or tags
- **Export/Import**: Share templates with others (JSON format)
- **Statistics**: Track usage and find popular templates
- **Quick Apply**: One-click template application

### 📊 System Info Tab
- Model configuration and status
- Switch models on the fly
- View API documentation
 - **Load Models On Demand**: Use the *Model Loader* checkboxes to select SDXL,
   the Ollama text model, and/or the vision model, then click **Load Selected**
   to initialize them

## 🔧 API Usage

### Enhanced MCP Servers ✨ **NEW**

#### Web Fetch Server
- **Rate Limiting**: Configurable requests per minute/hour
- **User Agent Spoofing**: Multiple user agent options (default, browser, mobile)
- **Domain Security**: Allowed/blocked domain lists
- **Content Filtering**: Automatic content cleaning and size limits

#### Filesystem Server
- **Secure Access**: Configurable allowed directories
- **Path Validation**: Prevents directory traversal attacks
- **Detailed Logging**: All file operations logged

### Image Generation
```bash
curl -X POST http://localhost:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cyberpunk cat with neon lights",
    "steps": 30,
    "guidance": 7.5
  }'
```

### Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about cats",
    "session_id": "default"
  }'
```
### MCP Tool Example
```bash
curl -X POST http://localhost:8001/tools/read_file \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"path": "/tmp/foo.txt"}}'
```

### Image Analysis
```bash
curl -X POST http://localhost:8000/analyze-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.png",
    "question": "What do you see?"
  }'
```

## ⚙️ Configuration

### config.yaml
```yaml
sd_model: "models/Illustrious.safetensors"
ollama_model: "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k"
ollama_vision_model: "qwen2.5vl:7b"
ollama_base_url: "http://localhost:11434"
gpu_backend: "cuda"  # "cuda", "rocm", or "cpu"
load_models_on_startup: true  # Set false for on-demand loading

cuda_settings:
  device: "cuda:0"
  dtype: "float16"
  enable_tf32: true
  memory_fraction: 0.95

generation_defaults:
  steps: 30
  guidance_scale: 7.5
  width: 1024
  height: 1024
```
Model paths can also be set via environment variables, e.g. `SD_MODEL` for the SDXL model or `MCP_CONFIG` for MCP servers.

## 🎯 Performance Tips

### For GPUs with 16GB+ VRAM
- Use `model_manager.py` to switch modes when needed
- Default 1024x1024 works well, use 768x768 for faster generation
- 20-30 steps for quality, 10-15 for speed
- Set `export OLLAMA_KEEP_ALIVE=0` to free memory faster
- Monitor usage with `nvidia-smi` or `rocm-smi`

### Troubleshooting GPU OOM
1. Run `python model_manager.py --image-mode` before generating
2. Reduce image size or steps
3. Check GPU usage with `nvidia-smi` (CUDA) or `rocm-smi`/`rocminfo` (ROCm)
4. Restart if memory fragmentation occurs

### Logging and Debugging ✨ **NEW**
- **Log Files**: Automatic logging to `logs/illustrious_ai_studio.log`
- **Log Rotation**: Automatic rotation (10MB max, 5 backups)
- **Console Output**: Real-time status updates
- **Error Tracking**: Detailed error messages and stack traces
- **Performance Monitoring**: Model loading and generation timing

**View Logs:**
```bash
# View recent logs
tail -f logs/illustrious_ai_studio.log

# Search for errors
grep -i error logs/illustrious_ai_studio.log
```

## 📁 Project Structure

```
illustrious-ai-studio/
├── main.py               # Main application entry
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── setup_env.py         # Automated environment setup
├── verify_setup.py      # Setup verification tool
├── test_simple.py       # Simple test suite
├── model_manager.py     # GPU memory management
├── core/               # Core modules
│   ├── sdxl.py         # Image generation
│   ├── ollama.py       # LLM integration
│   ├── state.py        # Application state
│   ├── config.py       # Configuration handler
│   └── memory.py       # Memory management
├── ui/                 # User interface
│   └── web.py          # Gradio interface
├── server/             # API server
│   └── api.py          # FastAPI endpoints
├── models/             # Model storage
│   └── Illustrious.safetensors
└── test_outputs/       # Test results
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `python test_simple.py`
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **Stable Diffusion XL** by Stability AI
- **Ollama** for local LLM deployment
- **Gradio** for the web interface
- **FastAPI** for the API server
- **Illustrious** model for anime-style generation

---

**Ready to create with AI? Your studio is set up and optimized for your RTX 4090M!**

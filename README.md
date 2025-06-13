# Illustrious AI Studio

A powerful local AI application that combines **Stable Diffusion XL (SDXL)** image generation with **Ollama LLM** capabilities, including vision models for image analysis.

## 🆕 Latest Improvements

### 🚀 **Automated Setup & Model Management**
- **One-Command Setup**: `python setup.py` handles everything
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
- **Memory Guardian System**: Real-time GPU memory monitoring and automatic OOM prevention
- **Intelligent Intervention**: Automatic memory cleanup when pressure thresholds are reached
- **Adaptive Generation**: Dynamic resolution and step adjustment based on available memory
- **Progressive Degradation**: Graceful fallback strategies instead of crashes
- **Model Manager Tool**: Switch between image/LLM modes to optimize 16GB VRAM usage
- **Performance Optimization**: FP16 precision, TF32 enabled for RTX 4090M

### 🌐 **Dual Interface**
- **Gradio Web UI** (Port 7860): Interactive web interface with tabs for all features
- **FastAPI MCP Server** (Port 8000): RESTful API for programmatic access

### 📸 **Enhanced User Experience** ✨ **NEW**
- **Recent Prompts**: Quick access to your last 20 prompts for faster iteration
- **Quick Style Buttons**: One-click application of popular styles (Anime, Realistic, Artistic, Fantasy, Cyberpunk)
- **Smart Resolution Selector**: 7 optimized resolution presets with quality indicators
- **Automatic Gallery**: Generated images saved with metadata
- **Gallery Viewer Tab**: Browse saved images and metadata
- **Project Galleries**: Organize images under named projects
- **Prompt Enhancement**: LLM improves your image prompts
- **Session Management**: Chat history automatically saved under your system's
  temporary directory and reloaded at startup
- **Model Switching**: Change models without restarting
- **Improved Error Handling**: Better recovery from generation failures

## 🛠️ Installation

### Prerequisites
- **Conda/Miniconda/Anaconda** ([Installation Guide](https://docs.conda.io/en/latest/miniconda.html))
- **GPU** with CUDA or ROCm support (NVIDIA or AMD, 16GB+ VRAM recommended)
  - Install the appropriate drivers for your hardware
- **Ollama** installed and running ([Installation Guide](https://ollama.ai/download)) - Optional for LLM features

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

### 🚀 One-Click Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/CreativeNewEra/illustrious-ai-studio.git
   cd illustrious-ai-studio
   ```

2. **Run the one-click setup** ✨
   ```bash
   python setup.py
   ```
   
   The setup will automatically:
   - ✓ Detect and use conda (Miniconda/Anaconda)
   - ✓ Create 'illustrious' conda environment with Python 3.12
   - ✓ Install PyTorch with CUDA/ROCm/CPU support
   - ✓ Install all dependencies
   - ✓ Download Illustrious SDXL model (6.6GB)
   - ✓ Configure Ollama models (if installed)
   - ✓ Generate launch scripts (`run.sh` for Linux/Mac and `run.bat` for Windows) after running `python setup.py`
   - ✓ Verify the installation

3. **Start the application**
   ```bash
   # Easy launch (Linux/Mac)
   ./run.sh
   
   # Easy launch (Windows) - `run.bat` is created after running `python setup.py`
   run.bat
   
   # Or manually:
   conda activate illustrious
   python main.py
   ```

## 🚀 Usage

### Starting the Application
Run with defaults:
```bash
python main.py
```

Advanced options:
```bash
# Lazy load models and use custom ports
python main.py --lazy-load --web-port 8080 --api-port 8081

# Require authentication and open a browser window
python main.py --auth user:pass --open-browser

# Enable memory optimizations
python main.py --optimize-memory
```

Open the UI at `http://localhost:<web-port>` (7860 by default).

Use the **Model Loader** section on the System Info tab to load SDXL or Ollama
models on demand when starting with `--lazy-load`.
Create new projects from the header and switch between them to view individual galleries.

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

### 🎨 Text-to-Image Tab ✨ **ENHANCED**
- **Recent Prompts**: Quick access to your last 20 prompts with dropdown selection
- **Quick Style Buttons**: One-click style application (Anime, Realistic, Artistic, Fantasy, Cyberpunk)
- **Resolution Selector**: Choose from 7 optimized resolutions (512x512 to 1024x1024, portrait/landscape)
- Enter prompts or use "Enhance Prompt" for AI improvement
- Press **Ctrl+Enter** (or **Cmd+Enter** on macOS) to generate
- Adjust parameters: steps, guidance scale, seed
- Images auto-saved to gallery
- **New**: Use saved prompt templates for consistent results

### 💬 AI Chat Tab
- Chat with the LLM
- Use `#generate <description>` to create images
- Call MCP tools with `/tool <server>.<method> key=value`
- Maintains conversation history across restarts

### 🔍 Image Analysis Tab
- Upload any image for AI analysis
- Ask specific questions about images
- Works with generated images too

### 🖼️ Gallery Tab ✨ **NEW**
- Browse saved images in a gallery view
- View metadata, open files, or copy their paths
- Use the project selector to switch between different galleries
- Search images by name or metadata
- Filter by tags to focus on specific categories
- Filter settings persist across sessions via files in `TEMP_DIR`

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

### Memory Report
```bash
curl http://localhost:8000/memory-report
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
  batch_size: 1
gallery_dir: "/tmp/illustrious_ai/gallery"
```
These defaults are optimized for GPUs with **16GB or more VRAM**. If you have a lower-memory card (8–12GB), reduce `width`, `height`, and `steps` in your `config.yaml` – for example `512x512` at 15 steps.

Model paths can also be set via environment variables, e.g. `SD_MODEL` for the SDXL model or `MCP_CONFIG` for MCP servers. Use `GALLERY_DIR` to customize where generated images are saved.

## 🎯 Performance Tips

### For GPUs with 16GB+ VRAM
- Use `model_manager.py` to switch modes when needed
- Default 1024x1024 works well, use 768x768 for faster generation
- 20-30 steps for quality, 10-15 for speed
- Launch with `--optimize-memory` to automatically set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
  `OLLAMA_KEEP_ALIVE=0`
- Monitor usage with `nvidia-smi` or `rocm-smi`

### Memory Guardian System ✨ **NEW**

The **Memory Guardian** provides automatic OOM prevention with real-time monitoring:

```bash
# Monitor memory status
python memory_manager.py --status

# Interactive memory monitoring
python memory_manager.py --monitor

# Generate comprehensive memory report
python memory_manager.py --report

# Test memory pressure handling
python memory_manager.py --test-pressure critical

# Clear GPU memory manually
python memory_manager.py --clear
```

**Key Features:**
- **Real-time Monitoring**: Continuous GPU memory tracking
- **Automatic Intervention**: Proactive memory cleanup at configurable thresholds
- **Adaptive Generation**: Dynamic resolution/step adjustment when memory is low
- **Progressive Degradation**: Graceful fallback instead of crashes
- **Configurable Thresholds**: Customize intervention points (70%, 85%, 95%, 98%)

**Memory Thresholds:**
- **Low (70%)**: Start close monitoring
- **Medium (85%)**: Begin preventive actions (cache clearing, garbage collection)
- **High (95%)**: Aggressive management (model unloading)
- **Critical (98%)**: Emergency intervention (force unload all models)

**Profiles:** configure `memory_guardian.profile` to quickly set these thresholds.
- **Conservative** – early intervention (60/75/90/95)
- **Balanced** – default thresholds (70/85/95/98)
- **Aggressive** – waits longer before intervening (80/90/97/99)

**Testing the System:**
```bash
# Test the Memory Guardian functionality
python test_memory_guardian.py
```

### Troubleshooting GPU OOM
With Memory Guardian active, OOM issues should be automatically prevented. If you still encounter problems:

1. Check Memory Guardian status: `python memory_manager.py --status`
2. Run `python model_manager.py --image-mode` before generating
3. Reduce image size or steps
4. Check GPU usage with `nvidia-smi` (CUDA) or `rocm-smi`/`rocminfo` (ROCm)
5. Enable memory optimizations: `python main.py --optimize-memory`

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
├── main.py                   # Main application entry
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── setup.py                 # Unified environment and model setup
├── verify_setup.py          # Setup verification tool (used by setup.py)
├── test_simple.py           # Simple test suite
├── model_manager.py         # Legacy GPU memory management
├── memory_manager.py        # NEW: Memory Guardian CLI tool
├── test_memory_guardian.py  # NEW: Memory Guardian test suite
├── core/                   # Core modules
│   ├── sdxl.py             # Image generation with OOM prevention
│   ├── ollama.py           # LLM integration
│   ├── state.py            # Application state
│   ├── config.py           # Configuration handler
│   ├── memory.py           # Basic memory utilities
│   └── memory_guardian.py  # NEW: Automatic OOM prevention system
├── ui/                     # User interface
│   └── web.py              # Gradio interface
├── server/                 # API server
│   └── api.py              # FastAPI endpoints
├── models/                 # Model storage
│   └── Illustrious.safetensors
├── logs/                   # NEW: Automatic logging
│   └── illustrious_ai_studio.log
└── test_outputs/           # Test results
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

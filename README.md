# Illustrious AI Studio

A powerful local AI application that combines **Stable Diffusion XL (SDXL)** image generation with **Ollama LLM** capabilities, including vision models for image analysis.

## 🌟 Features

### 🎨 **Triple AI System**
- **SDXL Image Generation**: High-quality 1024x1024 image generation using Stable Diffusion XL
- **Ollama LLM Chat**: Interactive chat with local language models
- **Vision Analysis**: Image understanding and Q&A with vision-capable models
- **Cross-Feature Integration**: Generate images from chat, analyze generated images

### 🚀 **Advanced Memory Management**
- **Automatic CUDA Memory Management**: Smart memory clearing and retry logic
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
- **NVIDIA GPU** with CUDA support (16GB+ VRAM recommended)
- **Ollama** installed and running
- **SDXL Model** (.safetensors format)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd illustrious-ai-studio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

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
Then open: http://localhost:7860

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

## 💻 Web Interface

### 🎨 Text-to-Image Tab
- Enter prompts or use "Enhance Prompt" for AI improvement
- Adjust parameters: steps, guidance scale, seed
- Images auto-saved to gallery

### 💬 AI Chat Tab
- Chat with the LLM
- Use `#generate <description>` to create images
- Maintains conversation history

### 🔍 Image Analysis Tab
- Upload any image for AI analysis
- Ask specific questions about images
- Works with generated images too

### 📊 System Info Tab
- Model configuration and status
- Switch models on the fly
- View API documentation

## 🔧 API Usage

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
sd_model: "/home/ant/AI/illustrious-ai-studio/models/Illustrious.safetensors"
ollama_model: "goekdenizguelmez/JOSIEFIED-Qwen3:8b-q6_k"
ollama_vision_model: "qwen2.5vl:7b"
ollama_base_url: "http://localhost:11434"

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

## 🎯 Performance Tips

### For RTX 4090M (16GB VRAM)
- Use `model_manager.py` to switch modes when needed
- Default 1024x1024 works well, use 768x768 for faster generation
- 20-30 steps for quality, 10-15 for speed
- Set `export OLLAMA_KEEP_ALIVE=0` to free memory faster

### Troubleshooting CUDA OOM
1. Run `python model_manager.py --image-mode` before generating
2. Reduce image size or steps
3. Check GPU usage with `nvidia-smi`
4. Restart if memory fragmentation occurs

## 📁 Project Structure

```
illustrious-ai-studio/
├── main.py               # Main application entry
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
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

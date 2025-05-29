# 🎨 Illustrious AI Studio

A powerful local AI application combining Stable Diffusion XL image generation with Ollama LLM chat capabilities. Create stunning artwork through both direct prompting and intelligent chat-based generation.

## ✨ Features

### 🎨 **Text-to-Image Generation**
- **Stable Diffusion XL** integration with custom models
- **Advanced prompt crafting** using AI enhancement
- **Gallery management** with automatic saving and metadata
- **Customizable parameters** (steps, guidance, seed, negative prompts)
- **Download functionality** with organized file management

### 💬 **AI Chat & Prompt Crafting** 
- **Local LLM integration** via Ollama (supports any Ollama model)
- **Intelligent prompt enhancement** for better image generation
- **Chat-based image generation** using `#generate [description]` commands
- **Cross-tab functionality** (generate from chat, display in image tab)
- **Session management** with persistent chat history

### 🔍 **Image Analysis** (Multimodal Models)
- **Vision-capable AI** analysis of uploaded images
- **Custom questions** about image content
- **Detailed descriptions** and image understanding

### 🖥️ **MCP Server Integration**
- **RESTful API** for external tool integration
- **Model Context Protocol** server on port 8000
- **Programmatic access** to all AI capabilities

## 🛠️ Requirements

### **System Requirements**
- **Python 3.8+**
- **CUDA-capable GPU** (recommended, 12GB+ VRAM)
- **16GB+ RAM** (system memory)
- **10GB+ storage** for models and outputs

### **Dependencies**
```bash
# Core ML Libraries
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
diffusers transformers accelerate
gradio fastapi uvicorn

# Utility Libraries  
Pillow requests pydantic pathlib
```

### **External Requirements**
- **Ollama** installed and running (https://ollama.ai)
- **Stable Diffusion XL model** (.safetensors format)

## 🚀 Installation

### **1. Clone Repository**
```bash
git clone <your-repo-url>
cd illustrious-ai-studio
```

### **2. Setup Conda Environment**
```bash
# Create and activate conda environment
conda create -n ai-studio python=3.10
conda activate ai-studio

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate gradio fastapi uvicorn
pip install Pillow requests pydantic
```

### **3. Install Ollama**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull your preferred model (example)
ollama pull qwen2.5:7b
# For vision capabilities:
ollama pull llava:latest
```

### **4. Download SDXL Model**
Download your preferred Stable Diffusion XL model (.safetensors format):
- Place in `/path/to/your/models/` directory
- Update `MODEL_PATHS` in `app.py`

### **5. Configure Application**
Edit the `MODEL_PATHS` configuration in `app.py`:
```python
MODEL_PATHS = {
    "sd_model": "/path/to/your/sdxl/model.safetensors",
    "ollama_model": "your-ollama-model-name", 
    "ollama_base_url": "http://localhost:11434"
}
```

## 🎯 Usage

### **Start the Application**
```bash
# Activate conda environment
conda activate ai-studio

# Run the application
python app.py
```

**Access Points:**
- **Web UI:** http://localhost:7860
- **MCP Server:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### **Web Interface Tabs**

#### **🎨 Text-to-Image**
1. Enter your image description in the prompt field
2. Optionally use "✨ Enhance Prompt" for AI improvement
3. Adjust advanced settings (steps, guidance, seed)
4. Click "🎨 Generate Image"
5. Download or save to gallery

#### **💬 AI Chat & Prompt Crafting**
1. Chat naturally with the AI assistant
2. Use `#generate [description]` for image generation
3. AI will enhance prompts and create images automatically
4. Generated images appear in both chat and Text-to-Image tab

#### **🔍 Image Analysis** (if multimodal model available)
1. Upload an image
2. Ask questions about the image
3. Get detailed AI analysis and descriptions

### **Chat Commands**
- **Normal chat:** Just type your message
- **Image generation:** `#generate a magical forest scene`
- **Alternative:** `generate image of a cyberpunk city`

## 🔌 API Documentation

### **MCP Server Endpoints**

#### **GET /status**
Check server and model status
```bash
# Make sure conda environment is activated
conda activate ai-studio

curl http://localhost:8000/status
```

#### **POST /generate-image**
Generate images programmatically
```bash
curl -X POST http://localhost:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape",
    "steps": 30,
    "guidance": 7.5,
    "seed": -1
  }'
```

#### **POST /chat**
Chat with the LLM
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "session_id": "my-session",
    "temperature": 0.7
  }'
```

#### **POST /analyze-image**
Analyze images (requires multimodal model)
```bash
curl -X POST http://localhost:8000/analyze-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-encoded-image>",
    "question": "What do you see in this image?"
  }'
```

## 📁 Project Structure

```
illustrious-ai-studio/
├── app.py                  # Main application file
├── README.md              # This file
├── requirements.txt       # Python dependencies (create this)
├── /tmp/illustrious_ai/   # Temporary files and gallery
│   └── gallery/          # Generated images and metadata
└── examples/             # Example prompts and configs
```

## 🔧 Configuration Options

### **Model Configuration**
```python
MODEL_PATHS = {
    "sd_model": "/path/to/model.safetensors",    # SDXL model path
    "ollama_model": "qwen2.5:7b",               # Ollama model name  
    "ollama_base_url": "http://localhost:11434"  # Ollama server URL
}
```

### **Generation Defaults**
- **Steps:** 30 (quality vs speed trade-off)
- **Guidance:** 7.5 (prompt adherence)
- **Negative Prompt:** "blurry, low quality, text, watermark, deformed"

### **Advanced Options**
- **CPU Offloading:** Enabled for memory efficiency
- **Gallery Auto-save:** All images saved with metadata
- **Session Management:** Persistent chat history per session

## 🐛 Troubleshooting

### **Common Issues**

#### **CUDA Out of Memory**
```bash
# Activate conda environment
conda activate ai-studio

# Set memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python app.py
```

#### **Ollama Connection Failed**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Test model
ollama run your-model-name
```

#### **Model Loading Errors**
- Verify SDXL model path is correct
- Check model format (.safetensors)
- Ensure sufficient VRAM (12GB+ recommended)

#### **Chat Not Responding**
- Check terminal for debug output
- Verify Ollama model is loaded
- Test with `curl http://localhost:11434/api/tags`

### **Performance Optimization**

#### **Memory Management**
- Use model CPU offloading for lower VRAM
- Enable sequential CPU offloading for extreme memory constraints
- Lower steps/resolution for faster generation

#### **Speed Improvements**
- Use smaller/quantized models
- Reduce inference steps
- Enable xFormers attention (if available)

## 🤝 Development

### **Code Structure**
- **`init_sdxl()`** - Stable Diffusion initialization
- **`init_ollama()`** - Ollama connection setup
- **`generate_image()`** - Image generation pipeline
- **`chat_completion()`** - LLM chat interface
- **`handle_chat()`** - Chat session management
- **`create_gradio_app()`** - Web interface setup

### **Adding New Features**
1. **New Ollama Models:** Update `MODEL_PATHS["ollama_model"]`
2. **Custom SDXL Models:** Update `MODEL_PATHS["sd_model"]`
3. **Additional Endpoints:** Add to MCP server section
4. **UI Modifications:** Edit `create_gradio_app()` function

### **Claude Code Integration**
This README provides comprehensive context for Claude Code to understand:
- Project architecture and dependencies
- Configuration patterns and options
- API interfaces and data structures
- Common issues and solutions
- Development patterns and extension points

Use Claude Code to:
- Add new image generation features
- Implement additional LLM integrations
- Extend the MCP server capabilities
- Optimize performance and memory usage
- Add new UI components or workflows

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- **Stability AI** for Stable Diffusion XL
- **Ollama** for local LLM infrastructure
- **Gradio** for the web interface framework
- **Anthropic** for Claude Code integration

---

**Happy AI Creating!** 🎨✨
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is an **Illustrious AI Studio** - a local AI application that combines Stable Diffusion XL image generation with Ollama LLM chat capabilities. The application consists of two main components:

1. **Gradio Web Interface** (port 7860) - Interactive web UI with tabs for image generation, chat, and image analysis
2. **FastAPI MCP Server** (port 8000) - RESTful API for programmatic access to AI capabilities

### Core System Design

- **Model Integration**: Dual AI system with SDXL for image generation and Ollama for chat/text processing
- **Cross-Tab Functionality**: Users can generate images from chat using `#generate` commands that appear in both chat and image tabs
- **Gallery Management**: Automatic saving of generated images with metadata in `/tmp/illustrious_ai/gallery/`
- **Session Management**: Persistent chat history per session with context preservation

### Key Components

- **`init_sdxl()`** - Handles SDXL model loading with GPU optimization and error handling
- **`init_ollama()`** - Establishes connection to local Ollama server and validates models
- **`generate_image()`** - Main image generation pipeline with automatic CUDA memory management
- **`clear_cuda_memory()`** - Automatic CUDA cache clearing and garbage collection
- **`chat_completion()`** - LLM interface for Ollama API communication
- **`handle_chat()`** - Session management and command parsing (including `#generate` detection)
- **`create_gradio_app()`** - Complex multi-tab web interface setup

## Development Commands

### Running the Application
```bash
# Activate the conda environment first
conda activate ai-studio

# Run the application
python app.py
```
This starts both the Gradio web interface (port 7860) and FastAPI MCP server (port 8000) in background threads.

### Requirements Installation
```bash
# Activate the conda environment first
conda activate ai-studio

# Install requirements
pip install -r requirements.txt
# Note: Requires manual CUDA PyTorch installation:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Model Configuration
Update `MODEL_PATHS` in app.py:72:37:
- `sd_model`: Path to SDXL .safetensors model
- `ollama_model`: Ollama model name (e.g., "qwen2.5:7b")
- `ollama_base_url`: Ollama server URL (default: http://localhost:11434)

### Testing Endpoints

#### Automated Testing Suite
```bash
# Run comprehensive API tests
cd examples/api_examples
python test_api.py
```

#### Manual Testing Commands
```bash
# Activate conda environment first
conda activate ai-studio

# Check server status
curl http://localhost:8000/status

# Test image generation API
curl -X POST http://localhost:8000/generate-image -H "Content-Type: application/json" -d '{"prompt": "test image", "steps": 20, "guidance": 7.5}'

# Test chat API  
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "Hello", "session_id": "test"}'

# Test image analysis API (requires vision-capable model)
curl -X POST http://localhost:8000/analyze-image -H "Content-Type: application/json" -d '{"image_base64": "base64_encoded_image", "question": "What do you see?"}'
```

#### Batch Generation Examples
```bash
# Run batch generation examples
cd examples/api_examples
python batch_generate.py
```

### MCP Server Status
The MCP server provides the following status information:
- **SDXL Model**: Image generation capability
- **Ollama Model**: Chat/text processing capability  
- **CUDA**: GPU acceleration availability
- **Vision/Multimodal**: Image analysis capability (model dependent)

Expected response format:
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

## Important Implementation Details

### Model Loading and Error Handling
- SDXL loading includes comprehensive error handling for missing models, CUDA availability, and memory constraints
- Ollama connection validates server availability, model existence, and tests with sample requests
- Model status tracking via global `model_status` dict with states for sdxl, ollama, and multimodal capabilities

### Image Generation Workflow
1. **Pre-generation memory clearing** - Automatic CUDA cache clearing before generation
2. **Prompt enhancement** via LLM (optional)
3. **SDXL pipeline execution** with configurable parameters and automatic retry logic
4. **CUDA memory management** - Automatic out-of-memory error handling with retry (up to 2 attempts)
5. **Post-generation cleanup** - Memory clearing after successful generation
6. **Automatic gallery saving** with metadata JSON files
7. **Cross-reference** between chat and image generation tabs

### Chat System Architecture
- Session-based chat history storage in `chat_history_store` global dict
- Command detection for `#generate` triggers image generation workflow
- Response cleaning to remove internal thinking tags (`<think>...</think>`)
- Integration with image generation system for seamless user experience

### MCP Server Integration
- FastAPI server runs concurrently with Gradio interface
- Three main endpoints: `/generate-image`, `/chat`, `/analyze-image`
- Base64 encoding for image data transfer
- Comprehensive error handling with appropriate HTTP status codes

## Dependencies and External Services

### Required External Services
- **Ollama**: Must be running on localhost:11434 with target model loaded
- **CUDA**: Recommended for SDXL performance (12GB+ VRAM)

### Python Dependencies (requirements.txt)
Key packages: torch, diffusers, transformers, gradio, fastapi, uvicorn, PIL, requests

### Model Requirements
- SDXL model in .safetensors format
- Ollama model (any supported model, vision models enable image analysis)

## Configuration Patterns

### Model Path Configuration
All model paths are centralized in `MODEL_PATHS` dict at app.py:33. Update these paths before running.

### Performance Optimization
- **Automatic CUDA Memory Management**: PyTorch memory fragmentation prevention via `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Dynamic Memory Clearing**: Automatic CUDA cache clearing before/after generation and on OOM errors
- **Smart Retry Logic**: Automatic retry with memory clearing on CUDA out-of-memory errors (up to 2 attempts)
- **Generator seed management** for reproducible results
- **Timeout handling** for Ollama requests (30-60 seconds)
- **Garbage collection** integration with CUDA memory management

### Gallery and File Management
- Temp directory creation with automatic cleanup
- UUID-based filename generation for uniqueness
- Metadata JSON files alongside image files for full generation context

## Troubleshooting

### Common Issues

#### CUDA Out of Memory Errors
**Symptoms:** "CUDA out of memory" errors during image generation.

**Automatic Fixes (No User Action Required):**
- System automatically clears CUDA memory cache and retries generation (up to 2 attempts)
- PyTorch memory fragmentation prevention enabled by default
- Garbage collection runs automatically with memory clearing

**Manual Solutions (if automatic fixes fail):**
1. **Reduce image parameters:** Lower steps (20-30), guidance scale (5-8), or image size
2. **Close other GPU applications** to free VRAM
3. **Restart application** if memory leaks persist
4. **Check available VRAM:** Ensure 6GB+ free for SDXL generation

#### Image Generation API Returns 500/507 Error
**Symptoms:** Image generation via API returns HTTP 500 (general error) or 507 (insufficient storage) errors.

**Cause:** SDXL pipeline memory allocation issue or model loading problem.

**Solutions:**
1. **Check logs for automatic retry attempts** - system will try up to 2 times with memory clearing
2. **HTTP 507 specifically indicates memory issues** - automatic fixes should handle this
3. **Verify model path:** Confirm SDXL model file exists at configured path
4. **Monitor CUDA memory:** Use `nvidia-smi` to check GPU memory usage

#### Chat Not Working
**Symptoms:** Chat endpoint returns errors or empty responses.

**Solutions:**
1. **Verify Ollama:** Ensure Ollama server is running (`ollama serve`)
2. **Check model:** Confirm target model is loaded (`ollama list`)
3. **Test connection:** Use `/status` endpoint to verify Ollama connectivity

#### Model Loading Failures
**Symptoms:** Status shows models as not loaded.

**Solutions:**
1. **Check paths:** Verify `MODEL_PATHS` in app.py point to existing files
2. **Memory issues:** Ensure sufficient system/GPU memory
3. **Dependencies:** Verify all required packages are installed
4. **Permissions:** Check file permissions for model files

### Performance Optimization
- **CUDA:** Ensure CUDA-compatible PyTorch installation for GPU acceleration
- **Memory Management:** Built-in automatic CUDA memory management handles most OOM issues
- **Memory Monitoring:** Use `nvidia-smi` to monitor GPU memory usage
- **Application Isolation:** Close other GPU-intensive applications for optimal performance
- **Model Optimization:** Consider using quantized models for lower memory usage
- **Parameter Tuning:** Reduce steps (20-30) and guidance scale (5-8) for faster generation with less memory
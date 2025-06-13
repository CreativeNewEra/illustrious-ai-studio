# CLAUDE.md

This file provides guidance to Claude (or other AI assistants) when working with code in this repository.

## Project Overview

**Illustrious AI Studio** - A local AI application that combines Stable Diffusion XL image generation with Ollama LLM capabilities, including vision models for image analysis.

### Core Components

1. **Gradio Web Interface** (port 7860) - Interactive web UI with tabs for:
   - Text-to-Image generation
   - AI Chat with prompt enhancement
   - Image Analysis (vision models)
   - System Info and configuration

2. **FastAPI MCP Server** (port 8000) - RESTful API for programmatic access

### Architecture

- **Triple AI System**: SDXL (images), Ollama LLM (text), Vision models (analysis)
- **Memory Management**: Automatic GPU memory handling (CUDA/ROCm) with retry logic, plus manual model manager
- **Cross-Feature Integration**: Generate images from chat, analyze generated images
- **Session Management**: Persistent chat history with context preservation

## Key Files and Functions

### Core Modules (`core/`)

- **`sdxl.py`**: Image generation with SDXL
  - `init_sdxl()` - Model loading with GPU optimization
  - `generate_image()` - Generation pipeline with automatic memory management
  - `switch_sdxl_model()` - Dynamic model switching

- **`ollama.py`**: LLM and vision model integration
  - `init_ollama()` - Connects to Ollama, loads text and vision models
  - `chat_completion()` - LLM chat interface
  - `analyze_image()` - Vision model image analysis
  - `generate_prompt()` - AI prompt enhancement

 - **`memory.py`**: GPU memory management
  - `clear_gpu_memory()` - Cache clearing and garbage collection
  - `get_model_status()` - System status reporting

- **`config.py`**: Configuration management
  - Loads from `config.yaml` and environment variables
  - Supports CUDA settings and generation defaults

- **`state.py`**: Application state management
  - `AppState` class holds models and session data

### UI and Server

- **`ui/web.py`**: Gradio interface setup
  - Multi-tab interface with event handlers
  - Cross-tab communication for integrated experience

- **`server/api.py`**: FastAPI endpoints
  - `/generate-image` - Image generation
  - `/chat` - LLM chat
  - `/analyze-image` - Vision analysis
  - `/status` - System status

### Utilities

- **`model_manager.py`**: GPU memory optimization tool
  - Switch between image/LLM modes
  - Unload models to free VRAM
  - Interactive and CLI modes

- **`verify_setup.py`**: System verification
  - Checks Python, CUDA/ROCm, dependencies
  - Verifies Ollama and model files
  - Generates setup report

- **`test_simple.py`**: Functional testing
  - Tests each component separately
  - Manages memory between tests
  - Provides clear pass/fail results

## Development Commands

### Running the Application
```bash
# Start the full application
python main.py

# Memory management before heavy tasks
python model_manager.py --image-mode  # For image generation
python model_manager.py --llm-mode    # For chat/vision
python model_manager.py --balanced    # For mixed usage
```

### Testing
```bash
# Verify setup
python verify_setup.py

# Install test dependencies
pip install -r requirements-test.txt

# Test functionality
python test_simple.py              # Recommended
python test_full_functionality.py  # Comprehensive

# API testing
cd examples/api_examples
python test_api.py
```

### Configuration

Edit `config.yaml`:
```yaml
sd_model: "/path/to/model.safetensors"
ollama_model: "model-name"
ollama_vision_model: "vision-model-name"
ollama_base_url: "http://localhost:11434"
gpu_backend: "cuda"  # "cuda", "rocm", or "cpu"

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
```

Environment variables override config:
- `SD_MODEL`
- `OLLAMA_MODEL`
- `OLLAMA_BASE_URL`

## Important Implementation Details

### Memory Management

1. **Automatic handling**:
   - Pre/post generation GPU cache clearing
   - Retry on OOM (up to 2 attempts)
   - Garbage collection integration

2. **Manual optimization**:
   - `model_manager.py` for mode switching
   - Unload Ollama models when not needed
   - Clear cache between heavy operations

### Error Handling

- **GPU OOM**: Automatic retry with memory clearing
- **Model loading**: Comprehensive error messages
- **API errors**: Proper HTTP status codes (500, 503, 507)
- **Ollama connection**: Timeout and validation

### Performance Optimization

- **FP16 precision** for speed
- **TF32 enabled** for RTX 30/40 series
- **Expandable segments** to prevent fragmentation
- **95% VRAM utilization** for maximum performance

## Common Issues and Solutions

### GPU Out of Memory (CUDA/ROCm)
1. Run `python model_manager.py --image-mode` before generation
2. Set `export OLLAMA_KEEP_ALIVE=0`
3. Reduce image size or steps
4. Monitor with `nvidia-smi` (CUDA) or `rocm-smi`/`rocminfo` (ROCm)

### Ollama Connection Failed
1. Verify Ollama is running: `ollama serve`
2. Check model exists: `ollama list`
3. Test: `curl http://localhost:11434/api/tags`

### Model Loading Errors
1. Verify file paths in config.yaml
2. Check file permissions
3. Ensure sufficient disk space
4. Validate model format (.safetensors)

## API Endpoints

### Image Generation
```bash
curl -X POST http://localhost:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cyberpunk cat",
    "steps": 30,
    "guidance": 7.5,
    "seed": -1
  }'
```

### Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello",
    "session_id": "default",
    "temperature": 0.7
  }'
```

### Image Analysis
```bash
curl -X POST http://localhost:8000/analyze-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64-data>",
    "question": "What is in this image?"
  }'
```

## Best Practices

1. **Memory**: Use model_manager.py for optimal VRAM usage
2. **Testing**: Run test_simple.py after changes
3. **Models**: Keep models in the `models/` directory
4. **Logs**: Check terminal output for detailed errors
5. **Performance**: 20-30 steps for quality/speed balance

## Hardware Requirements

 - **GPU**: NVIDIA (CUDA) or AMD (ROCm) with 16GB+ VRAM
- **CPU**: Modern multi-core processor
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ for models and outputs

## Dependencies

Core packages:
- PyTorch with CUDA or ROCm
- Diffusers, Transformers, Accelerate
- Gradio, FastAPI, Uvicorn
- Pillow, Requests, PyYAML

See `requirements.txt` for full list.

### Recent Updates (2025-06-13) ✨ **NEW**
```bash
# Test the latest fixes
python test_fixes.py

# The application now includes:
# - Enhanced error handling for image generation
# - Recent prompts system with persistence
# - Quick style buttons for faster workflow
# - Resolution selector with 7 optimized options
# - Improved API reliability and error recovery
```

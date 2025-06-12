# Changelog

All notable changes to the Illustrious AI Studio project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-06-11

### Added
- **Vision Model Support**: Full integration with qwen2.5vl:7b for image analysis
- **Memory Management Tools**:
  - `model_manager.py` - Interactive GPU memory management tool
  - Automatic model unloading and mode switching (image/LLM/balanced)
  - CLI support for quick mode changes
- **Setup and Testing Tools**:
  - `verify_setup.py` - Comprehensive system verification
  - `test_simple.py` - Component-by-component testing with memory management
  - `test_full_functionality.py` - Complete integration testing
- **Enhanced Configuration**:
  - CUDA settings for RTX 4090M optimization
  - Vision model configuration support
  - Performance settings (FP16, TF32, memory fraction)
- **Documentation**:
  - `SETUP_COMPLETE.md` - Quick reference guide for verified setup
  - Updated README.md with current features and usage
  - Modernized CLAUDE.md for AI assistant guidance

### Changed
- **Ollama Integration**: 
  - Support for separate text and vision models
  - Automatic vision model detection and loading
  - Improved model initialization with validation
- **Configuration Structure**:
  - Added `ollama_vision_model` field
  - Added `cuda_settings` section
  - Added `generation_defaults` section
- **Memory Management**:
  - Enhanced CUDA OOM handling with retry logic
  - Automatic memory clearing between operations
  - Environment variable support for memory control

### Fixed
- CUDA out of memory errors with 16GB VRAM limitation
- Ollama model persistence issues (OLLAMA_KEEP_ALIVE)
- Vision model initialization and state tracking
- Cross-model memory conflicts

### Improved
- Test coverage for all components
- Error handling and user feedback
- GPU memory utilization (95% efficiency)
- Model switching without restarts

## [1.0.0] - 2025-01-28

### Added
- **Core Features**
  - Illustrious AI Studio application with dual AI system integration
  - Stable Diffusion XL (SDXL) image generation capability
  - Ollama LLM chat and text processing integration
  - Cross-tab functionality with `#generate` commands
  - Automatic gallery management with metadata storage
  - Session-based chat history with context preservation

- **Web Interface**
  - Gradio web interface on port 7860
  - Multi-tab layout: Text-to-Image, AI Chat, Image Analysis, System Info
  - Real-time image generation with customizable parameters
  - Interactive chat interface with prompt crafting capabilities
  - System information and model status display

- **MCP Server Implementation**
  - FastAPI MCP server on port 8000
  - RESTful API endpoints for programmatic access
  - `/status` - Server and model status endpoint
  - `/generate-image` - Image generation API
  - `/chat` - LLM chat completion API
  - `/analyze-image` - Image analysis API

- **Model Integration**
  - SDXL model loading with GPU optimization
  - Ollama connection validation
  - Comprehensive model status tracking
  - CUDA acceleration support

### Technical Details
- **Architecture**: Dual-threaded application (Gradio + FastAPI)
- **Image Format**: PNG with JSON metadata files
- **Storage**: Temporary directory with automatic cleanup
- **Dependencies**: PyTorch, Diffusers, Transformers, Gradio, FastAPI
- **Models Supported**: 
  - SDXL .safetensors format for image generation
  - Any Ollama-compatible model for chat/text processing

## Version History Notes

### Version Numbering
- **Major.Minor.Patch** format following semantic versioning
- Major: Breaking changes or significant new features
- Minor: New features, backwards compatible
- Patch: Bug fixes, small improvements

### Development Roadmap
- [ ] Additional SDXL model support and recommendations
- [ ] Batch image generation interface
- [ ] Advanced prompt templates and presets
- [ ] Model quantization support for lower VRAM usage
- [ ] Export formats for generated content
- [ ] Plugin system for custom tools

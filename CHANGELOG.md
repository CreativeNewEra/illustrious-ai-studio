# Changelog

All notable changes to the Illustrious AI Studio project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CHANGELOG.md to track project changes and version history
- Enhanced MCP server testing documentation with status endpoint details
- Conda environment activation instructions throughout documentation

### Changed
- Updated CLAUDE.md with comprehensive MCP server testing procedures
- Improved documentation structure for better developer experience

### Fixed
- MCP server endpoints verified and working correctly
- All API endpoints tested and confirmed operational

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
  - Image analysis tab (for vision-capable models)
  - System information and model status display

- **MCP Server Implementation**
  - FastAPI MCP server on port 8000
  - RESTful API endpoints for programmatic access
  - `/status` - Server and model status endpoint
  - `/generate-image` - Image generation API
  - `/chat` - LLM chat completion API
  - `/analyze-image` - Image analysis API (vision models)

- **Model Integration**
  - SDXL model loading with GPU optimization and error handling
  - Ollama connection validation and model testing
  - Comprehensive model status tracking
  - CUDA acceleration support
  - CPU fallback for non-CUDA environments

- **Development Infrastructure**
  - Project documentation in CLAUDE.md
  - Requirements.txt with all necessary dependencies
  - Model configuration system via MODEL_PATHS
  - Comprehensive error handling and logging
  - Gallery and file management with UUID-based naming

### Technical Details
- **Architecture**: Dual-threaded application (Gradio + FastAPI)
- **Image Format**: PNG with JSON metadata files
- **Storage**: Temporary directory with automatic cleanup
- **Dependencies**: PyTorch, Diffusers, Transformers, Gradio, FastAPI, Ollama integration
- **Models Supported**: 
  - SDXL .safetensors format for image generation
  - Any Ollama-compatible model for chat/text processing
  - Vision models for image analysis (optional)

### Configuration
- Centralized model configuration in `MODEL_PATHS` dictionary
- Conda environment support (`ai-studio`)
- Flexible Ollama server URL configuration
- Customizable generation parameters (steps, guidance, seed)

### Performance Features
- CPU offloading for memory efficiency
- Generator seed management for reproducible results
- Timeout handling for Ollama requests (30-60 seconds)
- Base64 encoding for efficient API image transfer

---

## Version History Notes

### Version Numbering
- **Major.Minor.Patch** format following semantic versioning
- Major: Breaking changes or significant new features
- Minor: New features, backwards compatible
- Patch: Bug fixes, small improvements

### Change Categories
- **Added**: New features
- **Changed**: Changes in existing functionality  
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

### Development Milestones
- Initial release focused on core AI integration
- Future releases will expand model support and UI features
- Planned enhancements include additional AI models and export formats
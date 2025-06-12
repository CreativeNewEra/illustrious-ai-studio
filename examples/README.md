# 📁 Examples Directory

This directory contains comprehensive examples, documentation, and configuration files for the Illustrious AI Studio.

## 📋 Directory Structure

```
examples/
├── README.md                     # This file
├── API_DOCUMENTATION.md          # Complete API reference
├── ai_studio_todo_checklist.md   # Development roadmap and checklist
├── future_development_plan.md    # Long-term development vision
├── examples_folder.md            # Additional examples and utilities
├── api_examples/                 # API integration examples
│   ├── test_api.py              # Comprehensive API testing suite
│   ├── batch_generate.py        # Batch image generation examples
│   └── TXT\ 2.txt              # Additional text examples
├── configs/                      # Configuration examples
│   ├── generation_presets.json  # Image generation presets
│   └── model_configs.json       # Model configuration examples
├── prompts/                      # Prompt libraries and examples
│   ├── artistic_styles.json     # Art style prompt templates
│   ├── character_prompts.json   # Character generation prompts
│   └── scene_prompts.json       # Scene and environment prompts
└── scripts/                      # Utility scripts (empty, for future use)
```

## 🚀 Quick Start

Before running the examples, make sure the main project is installed using:
```bash
python ../setup_env.py
```

### 1. Test the API
```bash
cd api_examples
python test_api.py
```

### 2. Try Batch Generation
```bash
cd api_examples
python batch_generate.py
```

### 3. Explore Configuration Examples
```bash
cat configs/generation_presets.json
cat configs/model_configs.json
```

## 📚 Documentation Files

### 📖 API_DOCUMENTATION.md
Complete REST API reference including:
- Endpoint specifications
- Request/response formats
- Code examples in Python and JavaScript
- Error handling
- Quick start guide

### ✅ ai_studio_todo_checklist.md
Development checklist organized by phases:
- **Phase 1:** Immediate improvements
- **Phase 2:** Smart automation
- **Phase 3:** Creative intelligence
- **Phase 4:** Workflow automation
- **Phase 5:** Advanced features

### 🔮 future_development_plan.md
Long-term vision and technical roadmap:
- Intelligent quality engine
- Multi-model support
- AI creative director
- Personalization features

## 🧪 API Examples

### test_api.py
Comprehensive testing suite that validates all API endpoints:
- Status checks
- Image generation tests
- Chat functionality tests
- Image analysis tests (when available)
- Automatic error detection and reporting

**Usage:**
```bash
python test_api.py
```

### batch_generate.py
Examples of efficient batch image generation:
- Character portrait series
- Landscape collections
- Style variation sets
- Concurrent processing with progress tracking

**Usage:**
```bash
python batch_generate.py
# Follow interactive prompts to select generation type
```

## ⚙️ Configuration Examples

### generation_presets.json
Pre-configured settings for different image types:
- Quality presets (draft, standard, high-quality)
- Style-specific settings (anime, realistic, artistic)
- Resolution and optimization settings

### model_configs.json
Example model configurations:
- SDXL model settings
- Ollama model configurations
- Hardware optimization settings

## 📝 Prompt Libraries

### artistic_styles.json
Curated prompt templates for different art styles:
- Classical art styles
- Modern digital art
- Photography styles
- Abstract and experimental

### character_prompts.json
Character generation templates:
- Fantasy characters
- Modern characters
- Historical figures
- Character traits and attributes

### scene_prompts.json
Environment and scene templates:
- Natural landscapes
- Urban environments
- Interior scenes
- Fantasy and sci-fi settings

## 🔧 Usage Examples

### Python API Integration
```python
# Import the test suite functions
from api_examples.test_api import test_status_endpoint, test_image_generation

# Check server status
status = test_status_endpoint()

# Generate a single image
test_image_generation()
```

### Bash/cURL Examples
```bash
# Check server status
curl http://localhost:8000/status

# Generate an image
curl -X POST http://localhost:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "beautiful landscape", "steps": 25}'

# Chat with AI
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_id": "test"}'
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Make sure you're in the examples directory
   cd examples/api_examples
   python test_api.py
   ```

2. **Connection Refused:**
   ```bash
   # Verify the server is running
   curl http://localhost:8000/status
   
   # Start the server if needed
   python ../../app.py
   ```

3. **Image Generation Fails:**
   - Check GPU availability (`nvidia-smi` or `rocm-smi`)
   - Verify SDXL model is loaded
   - Restart application to reload models

## 📈 Performance Tips

### Batch Generation
- Use `batch_generate.py` for multiple images
- Adjust `MAX_WORKERS` based on GPU memory
- Monitor GPU usage during generation

### API Optimization
- Use appropriate timeout values for long operations
- Implement proper error handling and retries
- Cache results when possible

## 🔄 Next Steps

1. **Explore the checklist:** Review `ai_studio_todo_checklist.md` for planned features
2. **Read the API docs:** See `API_DOCUMENTATION.md` for complete reference
3. **Try examples:** Run the test scripts to understand capabilities
4. **Customize configs:** Modify preset files for your needs

## 🤝 Contributing

When adding new examples:

1. **API Examples:** Add to `api_examples/` directory
2. **Configurations:** Add to `configs/` directory  
3. **Prompts:** Add to `prompts/` directory
4. **Documentation:** Update this README

## 📞 Support

For issues and questions:
- Check the main `CLAUDE.md` documentation
- Review API documentation
- Test with the provided examples
- Check server logs for detailed error messages

---

*Last updated: [Current Date] - See git history for detailed changes*
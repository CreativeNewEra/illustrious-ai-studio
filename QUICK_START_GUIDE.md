# 🚀 Quick Start Guide for Illustrious AI Studio

## Problem Solved

This guide addresses the common issue where the Illustrious AI Studio app hangs during startup, particularly during model initialization. The solution provides multiple launch options with clear progress feedback.

## 🎯 Fastest Way to Launch

### Option 1: Use the Interactive Launcher (Recommended)
```bash
python launch.py
```
Then select option **1** for Quick Start mode.

### Option 2: Direct Quick Start
```bash
python main.py --quick-start --open-browser
```

## 🚀 Launch Options Explained

### 1. Quick Start Mode (Fastest)
- **Command**: `python main.py --quick-start --open-browser`
- **What it does**: Launches the web interface immediately without loading any models
- **Best for**: Getting the app running quickly, troubleshooting, or when you want to load models later
- **Startup time**: ~10-30 seconds

### 2. Lazy Load Mode
- **Command**: `python main.py --lazy-load --open-browser`
- **What it does**: Starts the UI first, then loads models only when needed
- **Best for**: Normal usage with faster startup
- **Startup time**: ~30-60 seconds

### 3. Full Initialization
- **Command**: `python main.py --open-browser`
- **What it does**: Loads all models at startup (traditional behavior)
- **Best for**: When you want everything ready immediately
- **Startup time**: 2-10 minutes (depending on hardware)

## 🔧 Troubleshooting Options

If you're having issues with specific components:

### Skip SDXL Model Loading
```bash
python main.py --no-sdxl --open-browser
```

### Skip Ollama Model Loading
```bash
python main.py --no-ollama --open-browser
```

### Skip Both Models (UI Only)
```bash
python main.py --no-sdxl --no-ollama --open-browser
```

### Memory Optimization
```bash
python main.py --quick-start --optimize-memory --open-browser
```

## 📊 Progress Feedback

The new version provides detailed progress information:

- ✅ **Pre-flight checks**: Validates models and connections before loading
- 🔄 **Loading progress**: Shows what's being loaded and how long it takes
- ⚠️ **Error handling**: Clear error messages with suggested solutions
- 💡 **Helpful tips**: Guidance on what to do if something fails

## 🎮 Interactive Launcher Features

The `launch.py` script provides:

1. **Menu-driven interface**: Easy selection of launch modes
2. **Real-time output**: See exactly what's happening during startup
3. **Custom options**: Enter your own command-line arguments
4. **Built-in help**: Access to all available options

## 📋 Common Issues and Solutions

### Issue: "SDXL model not found"
**Solution**: 
```bash
python main.py --no-sdxl --open-browser
```
Or check your `config.yaml` file for the correct model path.

### Issue: "Cannot connect to Ollama server"
**Solutions**:
1. Start Ollama: `ollama serve`
2. Or skip Ollama: `python main.py --no-ollama --open-browser`

### Issue: Out of memory errors
**Solution**:
```bash
python main.py --quick-start --optimize-memory --open-browser
```

### Issue: Still hangs during startup
**Solution**:
```bash
python main.py --quick-start --no-api --open-browser
```

## 🔄 Loading Models Later

When using `--quick-start`, you can load models later through the web interface:

1. Navigate to the "Model Management" section
2. Click "Load SDXL Model" or "Initialize Ollama"
3. Monitor progress in the interface

## 🛠️ Advanced Usage

### Custom Port
```bash
python main.py --quick-start --web-port 8080 --open-browser
```

### No Browser Auto-Open
```bash
python main.py --quick-start
```

### Debug Mode
```bash
python main.py --quick-start --log-level DEBUG
```

## 📞 Getting Help

View all available options:
```bash
python main.py --help
```

Or use the interactive launcher:
```bash
python launch.py
```
Then select option **5** for full help information.

---

**💡 Tip**: The `--quick-start` flag is your best friend for getting the app running quickly without any hanging issues!

# 🚀 Illustrious AI Studio - API Documentation

Complete reference for the FastAPI REST endpoints providing programmatic access to AI image generation and chat capabilities.

## 📋 Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Status Endpoint](#status-endpoint)
- [Image Generation](#image-generation)
- [Chat Completion](#chat-completion)
- [Image Analysis](#image-analysis)
- [Error Handling](#error-handling)
- [Code Examples](#code-examples)

---

## 🌐 Base URL

**Development Server:** `http://localhost:8000`

**Production:** Configure your deployment URL

---

## 🔐 Authentication

Currently, no authentication is required for local development. For production deployments, implement appropriate authentication middleware.

---

## 📊 Status Endpoint

### `GET /status`

Check server status and model availability.

**Response:**
```json
{
  "status": "running",
  "models": {
    "sdxl": true,
    "ollama": true,
    "multimodal": false
  },
  "gpu_available": true,
  "gpu_backend": "cuda"
}
```

**Example:**
```bash
curl http://localhost:8000/status
```

---

## 🎨 Image Generation

### `POST /generate-image`

Generate images using Stable Diffusion XL.

**Request Body:**
```json
{
  "prompt": "string (required)",
  "negative_prompt": "string (optional, default: '')",
  "steps": "integer (optional, default: 30, range: 10-100)",
  "guidance": "float (optional, default: 7.5, range: 1.0-20.0)",
  "seed": "integer (optional, default: -1 for random)"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "image_base64": "base64_encoded_image_data",
  "message": "✅ Image generated successfully! Seed: 12345"
}
```

**Error Responses:**
- **503:** SDXL model not available
- **500:** Generation failed

**Examples:**

**Basic Generation:**
```bash
curl -X POST http://localhost:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape with mountains and lakes",
    "steps": 25,
    "guidance": 7.5
  }'
```

**Advanced Generation:**
```bash
curl -X POST http://localhost:8000/generate-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "anime girl with blue hair, detailed face, studio lighting",
    "negative_prompt": "blurry, low quality, bad anatomy",
    "steps": 30,
    "guidance": 8.0,
    "seed": 42
  }'
```

**Python Example:**
```python
import requests
import base64

response = requests.post(
    "http://localhost:8000/generate-image",
    json={
        "prompt": "fantasy castle in the clouds",
        "negative_prompt": "blurry, low quality",
        "steps": 25,
        "guidance": 7.5,
        "seed": 123
    }
)

if response.status_code == 200:
    result = response.json()
    if result['success']:
        # Decode and save the image
        image_data = base64.b64decode(result['image_base64'])
        with open('generated_image.png', 'wb') as f:
            f.write(image_data)
        print(f"Image saved! {result['message']}")
```

---

## 💬 Chat Completion

### `POST /chat`

Chat with the AI using Ollama LLM integration.

**Request Body:**
```json
{
  "message": "string (required)",
  "session_id": "string (optional, default: 'default')",
  "temperature": "float (optional, default: 0.7, range: 0.0-2.0)",
  "max_tokens": "integer (optional, default: 256)"
}
```

**Success Response (200):**
```json
{
  "response": "AI generated response text",
  "session_id": "session_identifier"
}
```

**Error Responses:**
- **503:** Ollama model not available
- **500:** Chat completion failed

**Examples:**

**Basic Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! How are you today?",
    "session_id": "user_123"
  }'
```

**Creative Writing:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a short story about a magical forest",
    "session_id": "creative_session",
    "temperature": 0.9,
    "max_tokens": 512
  }'
```

**Image Generation Command:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "#generate a mystical dragon breathing fire",
    "session_id": "art_session"
  }'
```

**Python Example:**
```python
import requests

# Start a conversation
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "Can you help me write a prompt for a fantasy artwork?",
        "session_id": "art_consultation",
        "temperature": 0.8
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"AI: {result['response']}")
    print(f"Session: {result['session_id']}")
```

---

## 🔍 Image Analysis

### `POST /analyze-image`

Analyze images using vision-capable models (requires multimodal model).

**Request Body:**
```json
{
  "image_base64": "string (required) - base64 encoded image",
  "question": "string (optional, default: 'Describe this image in detail')"
}
```

**Success Response (200):**
```json
{
  "analysis": "Detailed description or answer to the question"
}
```

**Error Responses:**
- **503:** Vision model not available
- **500:** Analysis failed

**Example:**
```python
import requests
import base64
from PIL import Image
import io

# Prepare image
image = Image.open("your_image.jpg")
buffer = io.BytesIO()
image.save(buffer, format='PNG')
image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Analyze image
response = requests.post(
    "http://localhost:8000/analyze-image",
    json={
        "image_base64": image_base64,
        "question": "What objects do you see in this image?"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Analysis: {result['analysis']}")
```

---

## ⚠️ Error Handling

### Error Response Format
```json
{
  "detail": "Error description message"
}
```

### Common HTTP Status Codes

- **200:** Success
- **422:** Validation Error (invalid request body)
- **500:** Internal Server Error
- **503:** Service Unavailable (model not loaded)

### Error Types

**Validation Errors:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "prompt"],
      "msg": "Field required"
    }
  ]
}
```

**Model Unavailable:**
```json
{
  "detail": "❌ SDXL model not loaded. Please check your model path."
}
```

**Generation Failed:**
```json
{
  "detail": "❌ Generation failed: [Errno 32] Broken pipe"
}
```

---

## 📝 Code Examples

### Complete Python Integration

```python
import requests
import base64
import json
from PIL import Image
import io

class IllustriousAIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def check_status(self):
        """Check server status and model availability."""
        response = requests.get(f"{self.base_url}/status")
        return response.json()
    
    def generate_image(self, prompt, **kwargs):
        """Generate an image with SDXL."""
        data = {"prompt": prompt, **kwargs}
        response = requests.post(
            f"{self.base_url}/generate-image",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                return base64.b64decode(result['image_base64'])
        return None
    
    def chat(self, message, session_id="default", **kwargs):
        """Chat with the AI."""
        data = {
            "message": message,
            "session_id": session_id,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/chat", json=data)
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def analyze_image(self, image_path, question=None):
        """Analyze an image."""
        image = Image.open(image_path)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        data = {"image_base64": image_base64}
        if question:
            data["question"] = question
            
        response = requests.post(f"{self.base_url}/analyze-image", json=data)
        
        if response.status_code == 200:
            return response.json()['analysis']
        return None

# Usage example
client = IllustriousAIClient()

# Check if everything is running
status = client.check_status()
print(f"Server status: {status}")

# Generate an image
image_data = client.generate_image(
    "a beautiful sunset over the ocean",
    steps=25,
    guidance=7.5
)

if image_data:
    with open("sunset.png", "wb") as f:
        f.write(image_data)
    print("Image saved as sunset.png")

# Chat with AI
chat_response = client.chat(
    "Create a prompt for a fantasy landscape",
    session_id="my_session"
)

if chat_response:
    print(f"AI: {chat_response['response']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const fs = require('fs');

class IllustriousAIClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async checkStatus() {
        try {
            const response = await axios.get(`${this.baseUrl}/status`);
            return response.data;
        } catch (error) {
            console.error('Status check failed:', error.message);
            return null;
        }
    }

    async generateImage(prompt, options = {}) {
        try {
            const response = await axios.post(`${this.baseUrl}/generate-image`, {
                prompt,
                ...options
            });
            
            if (response.data.success) {
                return Buffer.from(response.data.image_base64, 'base64');
            }
        } catch (error) {
            console.error('Image generation failed:', error.message);
        }
        return null;
    }

    async chat(message, sessionId = 'default', options = {}) {
        try {
            const response = await axios.post(`${this.baseUrl}/chat`, {
                message,
                session_id: sessionId,
                ...options
            });
            return response.data;
        } catch (error) {
            console.error('Chat failed:', error.message);
            return null;
        }
    }
}

// Usage
(async () => {
    const client = new IllustriousAIClient();
    
    // Check status
    const status = await client.checkStatus();
    console.log('Status:', status);
    
    // Generate image
    const imageBuffer = await client.generateImage(
        'a cyberpunk cityscape at night',
        { steps: 30, guidance: 8.0 }
    );
    
    if (imageBuffer) {
        fs.writeFileSync('cyberpunk.png', imageBuffer);
        console.log('Image saved as cyberpunk.png');
    }
    
    // Chat
    const chatResponse = await client.chat('Hello, how are you?');
    if (chatResponse) {
        console.log('AI:', chatResponse.response);
    }
})();
```

---

## 🚀 Quick Start

1. **Start the server:**
   ```bash
   python main.py
   ```

2. **Test the connection:**
   ```bash
   curl http://localhost:8000/status
   ```

3. **Run example tests:**
   ```bash
   cd examples/api_examples
   python test_api.py
   ```

4. **Try batch generation:**
   ```bash
   python batch_generate.py
   ```

---

## 📚 Additional Resources

- **Testing Suite:** `examples/api_examples/test_api.py`
- **Batch Examples:** `examples/api_examples/batch_generate.py`
- **Configuration:** See `CLAUDE.md` for detailed setup instructions
- **Troubleshooting:** Check `CLAUDE.md` for common issues and solutions

---

*For more information about the project architecture and development setup, see the main [CLAUDE.md](../CLAUDE.md) documentation.*
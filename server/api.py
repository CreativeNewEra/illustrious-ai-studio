import base64
import io
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch

from core import sdxl, ollama
from core.sdxl import generate_image
from core.ollama import chat_completion, analyze_image
from core.memory import model_status

logger = logging.getLogger(__name__)


class GenerateImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    guidance: float = 7.5
    seed: int = -1


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    temperature: float = 0.7
    max_tokens: int = 256


class AnalyzeImageRequest(BaseModel):
    image_base64: str
    question: str = "Describe this image in detail"


app = FastAPI(title="Illustrious AI MCP Server", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    if sdxl.sdxl_pipe is None:
        sdxl.init_sdxl()
    if ollama.ollama_model is None:
        ollama.init_ollama()


@app.get("/status")
async def server_status():
    return {
        "status": "running",
        "models": model_status,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/generate-image")
async def mcp_generate_image(request: GenerateImageRequest):
    if sdxl.sdxl_pipe is None:
        raise HTTPException(status_code=503, detail="SDXL model not available")
    image, status = generate_image(
        request.prompt,
        request.negative_prompt,
        request.steps,
        request.guidance,
        request.seed,
    )
    if image is None:
        code = 507 if "out of memory" in status.lower() else 500
        raise HTTPException(status_code=code, detail=status)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return {"success": True, "image_base64": img_base64, "message": status}


@app.post("/chat")
async def mcp_chat(request: ChatRequest):
    if ollama.ollama_model is None:
        raise HTTPException(status_code=503, detail="Ollama model not available")
    messages = [{"role": "user", "content": request.message}]
    response = chat_completion(messages, request.temperature, request.max_tokens)
    return {"response": response, "session_id": request.session_id}


@app.post("/analyze-image")
async def mcp_analyze_image(request: AnalyzeImageRequest):
    if ollama.ollama_model is None or not model_status["multimodal"]:
        raise HTTPException(status_code=503, detail="Ollama vision model not available")
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    analysis = analyze_image(image, request.question)
    return {"analysis": analysis}


def run_mcp_server():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

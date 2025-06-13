import base64
import io
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from PIL import Image
import torch

from core import sdxl, ollama
from core.sdxl import generate_image
from core.ollama import chat_completion, analyze_image
from core.state import AppState
from core.config import CONFIG
from core.memory_guardian import get_memory_guardian

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


class SwitchModelsRequest(BaseModel):
    sd_model: str | None = None
    ollama_model: str | None = None


def create_api_app(state: AppState, auto_load: bool = True) -> FastAPI:
    app = FastAPI(title="Illustrious AI MCP Server", version="1.0.0")
    app.state.app_state = state

    def get_state(request: Request) -> AppState:
        return request.app.state.app_state

    @app.on_event("startup")
    async def startup_event():
        if auto_load and CONFIG.load_models_on_startup:
            if state.sdxl_pipe is None:
                sdxl.init_sdxl(state)
            if state.ollama_model is None:
                ollama.init_ollama(state)

    @app.get("/status")
    async def server_status(state: AppState = Depends(get_state)):
        return {
            "status": "running",
            "models": state.model_status,
            "gpu_available": torch.cuda.is_available(),
            "gpu_backend": CONFIG.gpu_backend,
        }

    @app.post("/generate-image")
    async def mcp_generate_image(request: GenerateImageRequest, state: AppState = Depends(get_state)):
        if state.sdxl_pipe is None:
            raise HTTPException(status_code=503, detail="❌ SDXL model not loaded. Please check your model path.")
        
        try:
            # Generate image with improved error handling
            image, status_msg = generate_image(
                state,
                request.prompt,
                request.negative_prompt,
                request.steps,
                request.guidance,
                request.seed,
            )
            
            if image is None:
                code = 507 if "out of memory" in status_msg.lower() else 500
                raise HTTPException(status_code=code, detail=status_msg)
            
            # Convert image to base64 with proper error handling
            try:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)  # Reset buffer position
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                buffered.close()  # Properly close the buffer
            except Exception as e:
                logger.error(f"Error encoding image to base64: {e}")
                raise HTTPException(status_code=500, detail=f"❌ Failed to encode image: {str(e)}")
            
            return {"success": True, "image_base64": img_base64, "message": status_msg}
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in image generation API: {e}")
            raise HTTPException(status_code=500, detail=f"❌ Generation failed: {str(e)}")

    @app.post("/chat")
    async def mcp_chat(request: ChatRequest, state: AppState = Depends(get_state)):
        if state.ollama_model is None:
            raise HTTPException(status_code=503, detail="Ollama model not available")
        messages = [{"role": "user", "content": request.message}]
        response = chat_completion(state, messages, request.temperature, request.max_tokens)
        return {"response": response, "session_id": request.session_id}

    @app.post("/analyze-image")
    async def mcp_analyze_image(request: AnalyzeImageRequest, state: AppState = Depends(get_state)):
        if state.ollama_model is None or not state.model_status["multimodal"]:
            raise HTTPException(status_code=503, detail="Ollama vision model not available")
        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        analysis = analyze_image(state, image, request.question)
        return {"analysis": analysis}

    @app.post("/switch-models")
    async def mcp_switch_models(request: SwitchModelsRequest, state: AppState = Depends(get_state)):
        result = {}
        if request.sd_model:
            result["sdxl"] = sdxl.switch_sdxl_model(state, request.sd_model)
        if request.ollama_model:
            result["ollama"] = ollama.switch_ollama_model(state, request.ollama_model)
        return {"models": state.model_status, **result}

    @app.get("/memory-report")
    async def memory_report(state: AppState = Depends(get_state)):
        guardian = get_memory_guardian(state)
        return guardian.get_memory_report()

    return app


def run_mcp_server(state: AppState, auto_load: bool = True) -> None:
    import uvicorn

    app = create_api_app(state, auto_load=auto_load)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

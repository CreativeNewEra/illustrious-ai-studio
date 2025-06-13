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
        
        # Enhanced API parameter validation
        try:
            # Validate prompt at API level
            if not request.prompt or not request.prompt.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="❌ Prompt cannot be empty. Please provide a valid text prompt."
                )
            
            # Validate parameter ranges at API level
            if request.steps <= 0 or request.steps > 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"❌ Invalid steps value: {request.steps}. Must be between 1 and 200."
                )
            
            if request.guidance <= 0 or request.guidance > 50:
                raise HTTPException(
                    status_code=400,
                    detail=f"❌ Invalid guidance value: {request.guidance}. Must be between 0.1 and 50."
                )
            
            if request.seed < -1 or request.seed >= 2**32:
                raise HTTPException(
                    status_code=400,
                    detail=f"❌ Invalid seed value: {request.seed}. Must be -1 (random) or between 0 and {2**32-1}."
                )
            
            logger.info(f"API: Generating image with prompt='{request.prompt[:50]}...', steps={request.steps}, guidance={request.guidance}, seed={request.seed}")
            
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
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"❌ Failed to encode image: {str(e)}. "
                        "Check disk space and permissions."
                    ),
                )
            
            return {"success": True, "image_base64": img_base64, "message": status_msg}
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in image generation API: {e}")
            raise HTTPException(
                status_code=500,
                detail=(
                    f"❌ Generation failed: {str(e)}. "
                    "Check your model path or GPU memory usage."
                ),
            )

    @app.post("/chat")
    async def mcp_chat(request: ChatRequest, state: AppState = Depends(get_state)):
        if state.ollama_model is None:
            raise HTTPException(
                status_code=503,
                detail="Ollama model not available. Is the Ollama server running?",
            )
        messages = [{"role": "user", "content": request.message}]
        response = chat_completion(state, messages, request.temperature, request.max_tokens)
        return {"response": response, "session_id": request.session_id}

    @app.post("/analyze-image")
    async def mcp_analyze_image(request: AnalyzeImageRequest, state: AppState = Depends(get_state)):
        if state.ollama_model is None or not state.model_status["multimodal"]:
            raise HTTPException(
                status_code=503,
                detail="Ollama vision model not available. Start the server or check your config.",
            )
        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {e}. Ensure the input is valid base64.",
            )
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

    class ProfileRequest(BaseModel):
        profile: str

    @app.post("/memory-profile")
    async def set_memory_profile(req: ProfileRequest, state: AppState = Depends(get_state)):
        guardian = get_memory_guardian(state)
        try:
            guardian.set_profile(req.profile)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"profile": guardian.config.get("profile")}

    class ThresholdsRequest(BaseModel):
        low: float | None = None
        medium: float | None = None
        high: float | None = None
        critical: float | None = None

    @app.post("/memory-thresholds")
    async def set_memory_thresholds(req: ThresholdsRequest, state: AppState = Depends(get_state)):
        guardian = get_memory_guardian(state)
        for level, val in req.dict(exclude_none=True).items():
            try:
                guardian.set_threshold(level, val)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        return guardian.get_memory_report()["thresholds"]

    return app


def run_mcp_server(state: AppState, auto_load: bool = True) -> None:
    import uvicorn
    from core.memory_guardian import start_memory_guardian, stop_memory_guardian

    app = create_api_app(state, auto_load=auto_load)

    # Start monitoring before launching the server
    start_memory_guardian(state)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

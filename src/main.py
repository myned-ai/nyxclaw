"""
nyxclaw — Voice-to-Avatar Server

FastAPI application for real-time voice-to-avatar interaction.
Combines Claw-based AI agents with Wav2Arkit model for synchronized facial animation.

Supports: sample_openclaw, sample_zeroclaw

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Or run directly:
    python main.py
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from auth import get_auth_middleware
from core.logger import get_logger, setup_logging
from core.settings import get_allowed_origins, get_settings
from routers import chat_router
from services import get_wav2arkit_service

logger = get_logger(__name__)


async def _preload_local_voice_stack(agent_type: str) -> None:
    if agent_type == "sample_openclaw":
        from agents.openclaw.openclaw_settings import get_openclaw_settings

        oc = get_openclaw_settings()

        if oc.stt_enabled:
            try:
                from services.stt_service import STTService

                stt = STTService(
                    stt_model=oc.stt_model,
                    vad_start_threshold=oc.stt_vad_start_threshold,
                    vad_end_threshold=oc.stt_vad_end_threshold,
                    vad_min_silence_ms=oc.stt_vad_min_silence_ms,
                )
                await stt.connect()
                await stt.disconnect()
                logger.info("Startup preload: STT warmed (OpenClaw)")
            except Exception as exc:
                logger.warning(f"Startup preload: STT warm failed (OpenClaw): {exc}")

        if oc.tts_enabled:
            try:
                from services.tts_service import TTSService

                tts = TTSService(
                    model_dir=oc.tts_onnx_model_dir,
                    voice_path=oc.tts_voice_path,
                    voice_name=oc.tts_voice_name,
                    noise_scale=oc.tts_noise_scale,
                    noise_w_scale=oc.tts_noise_w_scale,
                    length_scale=oc.tts_length_scale,
                )
                await tts.load()
                # Run a dummy synthesis to trigger ONNX JIT warmup —
                # eliminates ~1.5s cold-start penalty on the first real turn.
                async for _ in tts.synthesize("warmup"):
                    pass
                logger.info("Startup preload: TTS warmed (OpenClaw)")
            except Exception as exc:
                logger.warning(f"Startup preload: TTS warm failed (OpenClaw): {exc}")

    if agent_type == "sample_zeroclaw":
        from agents.zeroclaw.zeroclaw_settings import get_zeroclaw_settings

        zc = get_zeroclaw_settings()

        if zc.stt_enabled:
            try:
                from services.stt_service import STTService

                stt = STTService(
                    stt_model=zc.stt_model,
                    vad_start_threshold=zc.stt_vad_start_threshold,
                    vad_end_threshold=zc.stt_vad_end_threshold,
                    vad_min_silence_ms=zc.stt_vad_min_silence_ms,
                )
                await stt.connect()
                await stt.disconnect()
                logger.info("Startup preload: STT warmed (ZeroClaw)")
            except Exception as exc:
                logger.warning(f"Startup preload: STT warm failed (ZeroClaw): {exc}")

        if zc.tts_enabled:
            try:
                from services.tts_service import TTSService

                tts = TTSService(
                    model_dir=zc.tts_onnx_model_dir,
                    voice_path=zc.tts_voice_path,
                    voice_name=zc.tts_voice_name,
                    noise_scale=zc.tts_noise_scale,
                    noise_w_scale=zc.tts_noise_w_scale,
                    length_scale=zc.tts_length_scale,
                )
                await tts.load()
                async for _ in tts.synthesize("warmup"):
                    pass
                logger.info("Startup preload: TTS warmed (ZeroClaw)")
            except Exception as exc:
                logger.warning(f"Startup preload: TTS warm failed (ZeroClaw): {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for proper resource management.
    """
    settings = get_settings()

    # Setup logging based on settings
    setup_logging("DEBUG" if settings.debug else "INFO")

    # Startup
    logger.info("=" * 60)
    logger.info("NyxClaw Starting")
    logger.info("=" * 60)
    protocol = "wss" if settings.use_ssl else "ws"
    logger.info(f"WebSocket endpoint: {protocol}://{settings.server_host}:{settings.server_port}/ws")
    logger.info(f"Agent type: {settings.agent_type}")
    logger.info(f"Wav2Arkit model: {settings.onnx_model_path}")
    logger.info(f"Debug: {settings.debug}")
    logger.info(f"Auth: {'Enabled' if settings.auth_enabled else 'Disabled'}")
    if settings.auth_enabled:
        allowed_origins = get_allowed_origins()
        logger.info(f"Allowed Origins: {', '.join(allowed_origins)}")
    logger.info("=" * 60)

    # Initialize services (lazy loading - will connect on first request)
    get_wav2arkit_service()

    # Warm local ONNX STT/TTS resources once at startup to reduce first-turn latency
    await _preload_local_voice_stack(settings.agent_type)

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="NyxClaw",
    description="""
    Voice-to-avatar UI server for Claw-based agents.

    ## Features

    - **Claw Agents**: OpenClaw and ZeroClaw backends with local STT/TTS
    - **Facial Animation**: Wav2Arkit model for synchronized blendshapes
    - **WebSocket Streaming**: Real-time audio and animation data
    
    ## WebSocket Protocol
    
    Connect to `/ws` for real-time communication.
    
    ### Client → Server Messages
    
    - `{"type": "text", "data": "Hello"}` - Send text message
    - `{"type": "audio_stream_start", "userId": "user1"}` - Start audio streaming
    - `{"type": "audio", "data": "<base64>"}` - Send audio chunk (PCM16, 24kHz)
    - `{"type": "audio_stream_end"}` - End audio streaming
    - `{"type": "ping"}` - Heartbeat
    
    ### Server → Client Messages
    
    - `{"type": "audio_start", ...}` - AI started responding
    - `{"type": "sync_frame", "weights": {...}, "audio": "<base64>"}` - Synchronized frame
    - `{"type": "audio_end", ...}` - AI finished responding
    - `{"type": "transcript_delta", "text": "...", "role": "assistant"}` - Streaming text
    - `{"type": "transcript_done", "text": "...", "role": "user|assistant"}` - Complete transcript
    - `{"type": "avatar_state", "state": "Listening|Responding"}` - Avatar state change
    - `{"type": "interrupt"}` - User interrupted
    - `{"type": "pong"}` - Heartbeat response
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend development
settings = get_settings()
allowed_origins = get_allowed_origins() if settings.auth_enabled else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize authentication middleware
auth_middleware = get_auth_middleware()

# Include routers
app.include_router(chat_router)


@app.post("/api/auth/token")
async def get_auth_token(request: Request):
    """
    Generate authentication token for WebSocket connection.

    The widget should call this endpoint before connecting to WebSocket.
    Origin is validated and a signed token is returned.

    Returns:
        {"token": "base64-encoded-token", "ttl": 3600}

    Raises:
        403: If origin is not allowed or auth is disabled
    """
    if not settings.auth_enabled or not auth_middleware:
        raise HTTPException(status_code=403, detail="Authentication not enabled")

    # Get origin from request headers
    origin = request.headers.get("origin")
    if not origin:
        raise HTTPException(status_code=400, detail="Missing Origin header")

    # Generate token for this origin
    token = auth_middleware.generate_token_for_origin(origin)
    if not token:
        raise HTTPException(status_code=403, detail="Origin not allowed")

    return {"token": token, "ttl": settings.auth_token_ttl, "origin": origin}


@app.get("/inf")
async def root():
    """Root endpoint with server information."""
    settings = get_settings()
    response = {
        "name": "NyxClaw",
        "version": app.version,
        "status": "running",
    }

    if settings.debug:
        protocol = "wss" if settings.use_ssl else "ws"
        response["websocket"] = f"{protocol}://{settings.server_host}:{settings.server_port}/ws"
        response["agent_type"] = settings.agent_type

    return response


@app.get("/health")
async def health_check(response: Response):
    wav2arkit = get_wav2arkit_service()

    is_healthy = wav2arkit.is_available

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "details": "Critical service disconnected"}

    return {"status": "healthy"}


if __name__ == "__main__":
    settings = get_settings()

    # Configure logging BEFORE uvicorn starts
    # This ensures we control the handlers, not uvicorn
    setup_logging("DEBUG" if settings.debug else "INFO")

    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        reload=False,  # Reload doesn't work with app object, use uvicorn CLI for dev
        log_level="debug" if settings.debug else "info",
        log_config=None,  # Prevent uvicorn from overwriting our logging config
    )

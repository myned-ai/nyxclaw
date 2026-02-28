# avatar-chat-server

Real-time voice-to-avatar server. Multi-agent (OpenAI, Gemini, OpenClaw, ZeroClaw) with local STT/TTS pipeline via ONNX. Audio format: PCM16 @ 24kHz.

## Build & Run

```bash
uv sync                              # install deps
uv sync --extra local_voice          # include Piper TTS + faster-whisper STT
uv run python src/main.py            # run locally
docker compose up --build             # run in Docker
docker compose build --build-arg INSTALL_LOCAL_VOICE=true server  # Docker with voice
```

## Test & Lint

```bash
uv run pytest tests/                  # all tests
uv run ruff check src/                # lint
uv run ruff format src/               # format
```

## Code Conventions

- Use `asyncio.to_thread()` for blocking ops — Piper TTS and Whisper STT are synchronous and will block the event loop
- Private attributes prefixed with `_`
- Type hints: modern union syntax (`X | None`, not `Optional[X]`)
- Imports sorted by ruff isort; known first-party: `core`, `services`, `agents`, `routers`, `auth`, `wav2arkit`
- Graceful degradation: agents must work even if STT/TTS services are unavailable
- Thread safety: shared resources use class-level `threading.Lock()`

## Don't

- **No pip** — `uv` is the only package manager (lockfile is `uv.lock`)
- **No torch or silero-vad package** — VAD is pure ONNX via `_SileroVADIterator` (we eliminated the 2GB torch dependency)
- **No sherpa-onnx** — was removed, don't re-add
- **No HF_HOME** — all models live in `pretrained_models/` (Docker volumes mount there)
- **Don't over-engineer** — no extra files, abstractions, or flexibility that wasn't asked for
- **Don't add features not requested** — no docstrings/comments/types on code you didn't change
- **Don't assume requirements** — ask before guessing what the user wants

## Key Files

- `src/core/settings.py` — Pydantic BaseSettings, env var resolution, `resolved_onnx_model_path`
- `src/agents/base_agent.py` — Abstract agent interface (all agents inherit this)
- `src/chat/chat_session.py` — WebSocket session, audio playback, wav2arkit inference, barge-in
- `src/services/stt_service.py` — Silero VAD (ONNX) + faster-whisper transcription
- `src/services/tts_service.py` — Piper VITS ONNX text-to-speech

## Audio Pipeline

Before modifying STT, TTS, VAD, barge-in, or agent audio code, **read `STT_TTS_PIPELINE_HANDOFF.md`** first. It contains architecture diagrams, open problems (P1-P4), and exact line references.

## Git

- Branch naming: `feat/`, `fix/`, `docs/`
- Conventional commit style: `feat(scope): description`, `fix(scope): description`
- **Commit after every confirmed-working change** — always maintain a safe rollback point
- **Update memory notes** after significant changes — record what works, what doesn't, and why, so future sessions don't repeat mistakes

## External Libraries

When using faster-whisper, piper-tts, onnxruntime, or silero VAD APIs — web-search for current docs rather than relying on training data. These libraries update frequently.
